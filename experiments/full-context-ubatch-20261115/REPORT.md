# Connected full-context QLion backward, 2026-11-15

QLion and QLion2 single-sequence Gemma4/Lumen training now builds one differentiable window graph. Projection and attention-query tiles use the requested ubatch size. All query tiles read differentiable full-window K/V; shared-layer K/V stays connected. Gradients are combined before the window update. Optimizer periods count windows and gradient-accumulation groups, rather than token tiles. Other architectures and parallel training retain their existing backward paths.

There is no common 512-token cap. Native graph diagnostics and regression assertions verify 1024-token projections/queries for ub1024 and 512-token projections/queries for ub512. Training learning-rate settings are not changed.

Full-window K/V, masks and some activations remain allocated. Smaller ubatch bounds projection/query operations, not total activation storage. This does not implement a streaming backward pass with ubatch-sized peak memory, and it can use more memory than the old truncated path. FFN activation recomputation remains supported.

## Verification

- Fresh isolated Release CPU build in `/tmp/aikar-full-context-build`, with ccache disabled; production binaries were not rebuilt.
- `test-qat`: CPU reference tests passed.
- `test-logit-distillation`: CPU KL, finite differences, masks, cache, paired forward, tensor-parallel CPU checks and new full-context tests passed.
- GQA attention, flash on/off, all-token and last-token-only objectives: real ub512/1024 Q/K/V gradients passed on CPU and P620 CUDA. CUDA maximum relative L2 error was below 4e-7. Last-token loss reaches K/V rows in the first 512-token tile.
- CPU native dense/MoE, both optimizers: 16-update comparison on a short-prefix fixture passed. Final decoded projection-weight L2 difference was zero. First-update comparison on the long-prefix fixture also passed; first-momentum relative L2 errors were below 1.5e-4.
- P620 native dense/MoE, both optimizers: first-update and 16-update comparisons passed on a fixture with a 770-token unsupervised prompt and supervision after the 512-token boundary. Test LR was explicitly 1e-6, with constant schedule; production LR defaults were not changed.
- Recompute plus grad-accum=2, including a partial final accumulation group: both ubatches produced optimizer steps [1, 2]; momentum relative L2 difference was 0.00306542.
- A generated 12-layer Lumen fixture shares the layer-5 global K/V with layer 11 and uses FP4_KV global caches. Both ubatches produced nonzero layer-5 K momentum; relative L2 difference was 0.00204531.
- Python compilation and whitespace checks passed.

CUDA tests used the idle P620 (physical GPU 2), with CUDA0 explicitly selected and split mode none. The isolated CPU frontend used the existing `build/bin/libggml.so` backend via LD_PRELOAD. Commands and device/memory measurements are in the JSON artifacts. Fixtures were local; no downloads were needed.

## P620 long-prefix results

Relative L2 comparisons are after dequantization. Packed equality compares tensor bytes, applying missing adapter tensors from the base model. Each trajectory has 16 optimizer updates.

| Case | First momentum L2 | Max loss delta | Final weight L2 | Packed weights equal | Final momentum L2 | Final momentum cosine |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| dense-qlion | 0.0023740546 | 0 | 2.865505e-06 | False | 0.0035513993 | 0.99999369 |
| dense-qlion2 | 0.0023740546 | 0 | 0 | True | 0.0032613272 | 0.99999468 |
| moe-qlion | 0.0023302979 | 0 | 0 | True | 0.0036760471 | 0.99999324 |
| moe-qlion2 | 0.0023302979 | 0.00336551 | 0 | True | 0.093891472 | 0.99558459 |

The state comparisons are not bit-exact. In particular, MoE QLion2 momentum differs by about 9.4% in relative L2 even though the final packed projection weights match; momentum cosine is about 0.9956. The largest final weight relative L2 difference is about 2.9e-6. These short checks verify connected backward computation and bound observed differences; they do not establish equivalent long-training trajectories or quality.

An earlier aggressive LR=0.001 test, with broad attention selectors including quantized normalization weights, showed trajectory amplification after several updates (maximum loss delta about 0.1506 after 16 updates). That result was not treated as harmless rounding. No LR clamp, optimizer deadband, reduced ub1024 operation size, or common numerical tile was introduced to hide it.

## Reproduce

```sh
cmake -S . -B /tmp/aikar-full-context-build -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_OPENSSL=OFF
cmake --build /tmp/aikar-full-context-build --target llama-finetune-qlion test-logit-distillation test-qat -j 6
/tmp/aikar-full-context-build/bin/test-logit-distillation --full-context
CUDA_VISIBLE_DEVICES=2 LD_PRELOAD=/home/user/aikar-engine/build/bin/libggml.so /tmp/aikar-full-context-build/bin/test-logit-distillation --full-context-cuda
CUDA_VISIBLE_DEVICES=2 LD_PRELOAD=/home/user/aikar-engine/build/bin/libggml.so python3 examples/qlora_training/test_qlion_integration.py --bin-dir /tmp/aikar-full-context-build/bin --fixtures /mnt/openwebui/AIKAR/QLion2-test/artifacts --output-dir /tmp/aikar-full-context-final-cuda --full-context-only --gpu-layers 99 --gpu-index 2
```

Q8_0/Q4_0 native weight updates already avoid a persistent FP32 master weight. Block-local decoded values, momentum and residual are used to calculate target values and repack the quantized block. Residual feedback preserves updates below one integer code step; directly incrementing codes at a fixed scale does not replace that operation.
