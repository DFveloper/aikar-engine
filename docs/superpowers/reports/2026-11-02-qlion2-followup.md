# QLion2 CPU follow-up implementation and validation

Later implementation of the three remaining scopes is documented in [fixed-KV, compact routing, and core-loader validation](2026-11-03-qlion2-three-scopes.md). Remaining-work statements below describe the earlier implementation.

P620 access was restored on 2026-11-03. See the [P620 report](2026-11-03-qlion2-p620.md) for CUDA routing fixes, bounded workspace, actual hybrid 26B training, and the unmet time-ratio goal.

Date: 2026-11-02. This extends the [first-pass report](2026-11-02-qlion2-improvements.md). Source models were read only; generated artifacts and logs are under `/tmp/qlion-followup`. These are CPU correctness and memory results, not P620 performance validation.

## Changes

- Retained dense/MoE graphs preserve the dynamic path's supervised output rows. The previous full-output projection changed floating-point gradients and later expert routing. Compatible full-sequence, single-stream CPU graphs now reuse forward/backward/optimizer graphs with byte-identical multi-step weights; output-count or shape changes rebuild. GPU, microbatch accumulation, parallel streams, activation recomputation, MTP, and critical weighting retain the existing dynamic path. `--qat-static-graph` remains off by default.
- CPU fused routed accumulation computes one F32 gradient row per worker and writes Q8_0 directly, removing the whole F32 expert gradient temporary. It preserves the reference reduction order, intermediate quantization, and Q8_0 addition. Dense and compact modes use existing GGML operation codes. CUDA/Vulkan capability checks reject the CPU-only variants.
- Compact expert storage uses a routing upper bound over the accumulation group. Growing storage preserves values, IDs, and hash state; saturation or multiple gradient contributions fall back to dense Q8_0. Momentum/residual stay full selected-tensor states. Missing experts still receive dense zero-gradient momentum/decay updates. Skipping those updates would change the optimizer.
- `--qat-moe-gradient auto|dense|fused` defaults to `auto`: fuse when the F32 temporary is at least 16 MiB or the routing bound permits compact storage. The small fully routed CPU fixture was slower with forced fusion, so automatic selection keeps its dense reference. Explicit modes allow reproduction and comparison.
- Fixed activation broadcasting in the direct routed updater, Q2_0 decoding from Q8_0 accumulated gradients, and remapping the scalar loss seed when dynamic graph topology moves the loss node. Tests vary masks/views, repeated/strided routes, activation broadcasting, and gradient contributions.

## Validation

Release CPU build: `/tmp/qlion-improve-build`, using the build configuration in the first-pass report.

```sh
CCACHE_DISABLE=1 cmake --build /tmp/qlion-improve-build --target \
    llama-finetune-qlion llama-export-qlion llama-completion test-opt test-qat \
    test-qlion-adapter test-backend-ops -j 6
ctest --test-dir /tmp/qlion-improve-build \
    -R '^(test-opt|test-qat|test-qlion-adapter)$' --output-on-failure
/tmp/qlion-improve-build/bin/test-backend-ops test -b CPU \
    -o ACC_QLION_QAT_ID,ACC_QLION_QAT_EXPERTS
/tmp/qlion-improve-build/bin/test-backend-ops grad -b CPU \
    -o MUL_MAT_ID_GRAD,MOE_TRAINING_GRAD,SET_ROWS_GRAD,FLASH_ATTN_EXT_GRAD
PYTHONPATH=gguf-py python3 examples/qlora_training/test_qlion_integration.py \
    --bin-dir /tmp/qlion-improve-build/bin \
    --fixtures /mnt/openwebui/AIKAR/QLion2-test/artifacts \
    --output-dir /tmp/qlion-followup/integration-auto
```

- All three focused CTest targets pass; 12 fused accumulator backend cases and eight numerical gradient cases pass.
- All 54 native integration runs pass, including actual dense/MoE graph reuse, exact final weight comparison, adapter merge/runtime, checkpoint resume, and dense-versus-forced-fused microbatch training. The final `integration-auto/report.json` is a correctness rerun performed alongside other validation; its timings are not an isolated benchmark.
- `test-qat` checks exact dense/fused/automatic W/M/R trajectories over 12 microsteps for MXFP4, Q4_0, Q8_0, and Q2_0, including inactive momentum and weight decay. Compact capacity grows from 8 to 24 experts, then becomes dense: total state buffers are 3,008, 8,704, and 10,880 bytes in the synthetic case. A second gradient contribution also exercises fallback with accumulated values preserved. Zero-token and wide-row cases are covered.
- AddressSanitizer-instrumented GGML/core/CPU and `test-qat` pass with `ASAN_OPTIONS=detect_leaks=0`; LeakSanitizer is unavailable under this sandbox's ptrace restrictions.
- The first-pass argument-parser HTTP fixture restriction remains; this is not a claim that every repository test passed.

## Actual Gemma 4 26B update

Model: `/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/gemma-4-26B_q4_0-it.gguf`. Selected block-0 K, V, expert down, and fused expert gate-up weights total 434,700,288 bytes. Q6_K tied embeddings remain frozen.

```sh
/tmp/qlion-improve-build/bin/llama-finetune-qlion \
    -m /mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/gemma-4-26B_q4_0-it.gguf \
    --train-file /tmp/qlion-improve-results/large-one.jsonl \
    --optimizer qlion2 --quant-type q4_0 \
    --qat-targets blk.0.attn_k.weight,blk.0.attn_v.weight,blk.0.ffn_gate_up_exps.weight,blk.0.ffn_down_exps.weight \
    -c 256 -b 256 -ub 64 -ngl 0 -t 8 -tb 8 --dataset-threads 2 \
    --epochs 2 --val-split 0 --learning-rate 0.01 --lr-scheduler constant \
    --qat-out '' --qat-adapter-out /tmp/qlion-followup/26b-adapter.gguf \
    --qat-adapter-units experts --qat-moe-gradient fused \
    --save-every 0 --save-first-at 0 --verbose-loss --verbosity 5
```

Both optimizer steps completed: 81.840783 and 81.308400 seconds, with 10 supervised tokens per step. Elapsed process time including original hashing and export was 6:03.13. This is a synthetic two-step smoke test, not a quality or convergence evaluation.

| Measurement | First-pass run | Follow-up run |
| --- | ---: | ---: |
| Compute workspace bytes | 3,373,854,464 | 815,066,880 |
| Peak process RSS, KiB | 20,330,416 | 17,832,412 |
| Persistent Q8_0 gradient bytes | 821,100,544 | 821,100,544 |

Workspace fell 75.84%; peak process RSS fell 12.29%. The first pass used one cold step and the follow-up used two steps, so these runs do not establish a controlled steady-state speedup. In this configuration the routing bound reaches all 128 experts, so the persistent Q8_0 accumulator is still dense. The workspace saving comes from eliminating the full F32 expert gradient temporary; it does not prove all training state scales with actual routed experts.

The nonempty v2 adapter is 235,461,632 bytes (224.55 MiB), containing 235,090,944 payload bytes:

| Tensor | Changed units | Payload bytes |
| --- | ---: | ---: |
| K | 2,048 rows | 3,244,032 |
| V | 2,048 rows | 3,244,032 |
| Expert down | 67 / 128 experts | 74,714,112 |
| Fused expert gate-up | 69 / 128 experts | 153,888,768 |

`llama-export-qlion` successfully created `/tmp/qlion-followup/26b-merged.gguf`. With seed 42, prompt `hi`, two generated tokens, CPU loading, `--load-mode none --lazy-mode off --no-repack`, and matching completion options, merged-model and runtime-adapter stdout matched byte for byte. Both processes exited successfully. Independent GGUF inspection compared all 658 original/merged tensors: exactly the four selected tensors changed, every replacement unit matched its adapter payload, and all untouched units/tensors matched the original (`26b-verify.json`). Adapter metadata is recorded in `26b-adapter-info.json`; commands, process times, and logs are retained alongside it.

## Remaining limits and better next steps

- P620 cannot be measured because `nvidia-smi` cannot communicate with the driver. Throughput, peak VRAM, and the dense/MoE ratio target remain unverified. GPU implementations of the new fused variants need separate kernel work and numerical validation.
- Keep graph reuse constrained to compatible CPU shapes. General microbatch reuse needs a fixed KV/mask design and gradient-equivalence tests; padding the vocabulary output to force reuse already proved unsafe for exact trajectory equivalence.
- Actual-route-sized 26B storage needs a design beyond conservative capacity bounds. Preserve inactive momentum/residual semantics; allocating state only for experts seen in the current microbatch would be incorrect.
- A controlled warmup/steady-state benchmark on the target GPU is the next performance acceptance step. Keep the automatic small-tensor fallback: forced fusion is not uniformly faster. In the first isolated small-fixture comparison, forced dense/fused microbatch medians were 0.681028/0.7409025 seconds; this informed the automatic policy.
- Common-tool runtime application supports resident weights. Direct core-loader callers, mmap/lazy overlays, and backend repacking remain separate work. Optimizer resume still uses full checkpoints and sidecars, not adapters.
