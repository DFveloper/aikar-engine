# QLion2 improvements: CPU implementation and validation

Historical first-pass results. Subsequent fixes, nonempty 26B validation, and current limits are recorded in the [follow-up report](2026-11-02-qlion2-followup.md). Measurements below describe the first pass.

Date: 2026-11-02. Base commit: `cd0adf4a5`. This is a partial implementation of the complete A/B design, not a claim that the P620 performance or general MoE static-graph goals are complete.

## Implemented

- Native QLion intersects `--freeze-layers` with `--qat-targets`, validates selection before training, excludes tied embeddings/output by default, and reports actual optimizer registrations and state bytes. Explicit tied selection warns about dense vocabulary gradients and full replacement size. Unsupported unselected quantization types remain frozen, including the Q6_K embedding in the local 26B fixture.
- `--qat-adapter-out` captures the original file/tensor SHA-256 before updates, compares final bytes directly, and streams only changed replacements. v1 stores full tensors; optional v2 `rows`/`experts` stores changed units. No persistent hash cache is used. Full checkpoint/state resume stays on the existing path; final adapter-only saving is available with an empty `--qat-out` and disabled periodic checkpoints.
- `llama-export-qlion` shares validation with runtime application: architecture, original shape/type, file and unit hashes, payload hashes, sorted/ranged unit indices, conflict detection, and atomic publication. Compatible-base mode is explicit and warns about untouched tensors. Unsupported versions, malformed metadata, split GGUF, truncation, duplicates, and output/input aliases are rejected.
- Common CLI initialization applies disjoint adapters to resident model buffers before context creation, including duplicate physical copies of tied weights. mmap, lazy loading, CPU LES, and vocabulary-only loading are rejected; CPU repacking is disabled. Direct core-loader callers do not automatically receive this common-tool behavior.
- Experimental graph retention is opt-in for dense, full-sequence, single-stream optimizer steps. It refreshes inputs and loss masks, clears KV, and rebuilds/falls back across unsupported shapes, accumulation, parallelism, evaluation, and recomputation. Routed MoE uses dynamic graphs: the attempted retained path failed multi-step equivalence and was disabled.
- `qat_step` logs actual supervised tokens and optimizer-step duration with `--verbosity 5`. The timer excludes model loading/hashing. Memory reporting includes adaptive LR state.

## Fixtures and commands

Artifacts were read from `/mnt/openwebui/AIKAR/QLion2-test/artifacts`; generated outputs were written under `/tmp/qlion-improve-results`. No source artifact was modified.

| Fixture | GGUF elements, including constants | GGUF bytes | Experts / selected | Estimated active elements |
| --- | ---: | ---: | ---: | ---: |
| `gemma4-tiny-6l-q4.gguf` | 6,954,182 | 5,229,376 | dense | 6,954,182 |
| `gemma4-tiny-moe-q4.gguf` | 6,966,470 | 5,242,592 | 4 / 2 | 6,081,734 |

Both fixtures have six layers, Q4_0 matrix weights, Q8_0 normalization/embedding weights, F32 constants, and tied input/output embeddings. The MoE fixture keeps a shared dense FFN with intermediate size 384, versus 768 in dense; its expert FFN has size 96 and fused gate-up weights. The existing generation script is `QLion2-test/build_gemma4_moe_smoke.py`; it uses seed 42 and the same tokenizer. These fixtures are available locally; this change does not introduce a downloader or a second fixture generator.

Build and reproduce:

```sh
cmake -S . -B /tmp/qlion-improve-build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_MTMD=OFF -DCMAKE_BUILD_TYPE=Release
CCACHE_DISABLE=1 cmake --build /tmp/qlion-improve-build --target \
    llama-finetune-qlion llama-export-qlion llama-completion test-opt test-qat \
    test-qlion-adapter test-backend-ops -j 8
ctest --test-dir /tmp/qlion-improve-build -R '^(test-opt|test-qat|test-qlion-adapter)$' --output-on-failure
/tmp/qlion-improve-build/bin/test-backend-ops grad -b CPU \
    -o MUL_MAT_ID_GRAD,MOE_TRAINING_GRAD,SET_ROWS_GRAD,FLASH_ATTN_EXT_GRAD
PYTHONPATH=gguf-py python3 examples/qlora_training/test_qlion_integration.py \
    --bin-dir /tmp/qlion-improve-build/bin \
    --fixtures /mnt/openwebui/AIKAR/QLion2-test/artifacts \
    --output-dir /tmp/qlion-improve-results/integration5
```

## Correctness results

- `test-opt`, `test-qat`, and `test-qlion-adapter` pass. Adapter tests cover Q4_0, Q8_0, MXFP4 and Q2_0 byte layouts, unchanged omission, v1/v2, disjoint/overlapping replacements, malformed metadata, original/payload hashes, compatible metadata, stale snapshots, duplicate names, truncation, input aliases, and failed output publication.
- Eight focused backend gradient cases pass: K and V flash-attention gradients, Q flash-attention cases, a composite router/fused expert/shared FFN graph, repeated expert IDs with weight and activation gradients, and SET_ROWS with I32/I64 indices. K uses a larger four-point finite-difference interval to reduce F32 cancellation; the existing error tolerance is unchanged. This does not claim finite-difference agreement for discrete quantized weight codes.
- All 48 local integration commands pass. They cover dense/MoE training, whole-model/merged tensor byte equality, deterministic runtime/merged inference equality, unchanged base files and frozen tensors, K/V momentum with flash on/off, router/shared/expert momentum, explicit tied output, selection errors, optimizer checkpoint resume, partial accumulation/parallel batches, context changes, periodic evaluation, and safe graph fallback.
- Dense retained/dynamic weights match byte-for-byte over 12 updates. MoE retained execution diverged over multiple updates and is disabled; its fallback matches dynamic execution byte-for-byte.
- AddressSanitizer passes for the adapter checks with leak detection disabled. LeakSanitizer cannot run under this sandbox's process-inspection restrictions.
- `test-arg-parser` passes its local parsing phases, then fails at its existing `http://ggml.ai/` download test because HTTP access is unavailable. It is not reported as a passing full test. Python syntax compilation, the code-style workflow (163 model mappings), added-line ASCII checks, and changed-line/new-file whitespace checks pass; `ty` is not installed in this environment.

Detailed commands, fixture/data SHA-256, timings, exit codes, and log paths are in `/tmp/qlion-improve-results/integration5/report.json`.

## CPU measurements

Intel Xeon E5-2696 v3, CPU-only, four compute threads and two dataset workers; QLion2, LR 0.01 constant, no shuffle, context/batch/microbatch 256, one sequence, no accumulation, two epochs of the same 32-record dataset. Default targets exclude tied output. Each model executes 12 optimizer steps; the first two are excluded from timing summaries. There are 235 supervised tokens per epoch. Peak RSS comes from `/usr/bin/time` and includes the entire process lifetime.

The comparison goals were recorded before measurement: MoE/dense step median and peak RSS ratios at most 1.10. These are diagnostic CPU measurements, not P620 acceptance results.

| Configuration | Step median (s) | Step range (s) | Supervised tokens/s | Peak RSS (MiB) |
| --- | ---: | --- | ---: | ---: |
| Dense dynamic | 0.3926 | 0.3734-0.4936 | 94.31 | 98.05 |
| Dense retained | 0.4849 | 0.4663-0.5912 | 78.41 | 128.61 |
| MoE dynamic | 0.4524 | 0.4330-0.5382 | 84.60 | 111.71 |
| MoE retained requested, dynamic fallback | 0.4422 | 0.4195-0.5663 | 85.93 | 111.59 |

MoE/dense dynamic ratios are 1.1523 in step time and 1.1394 in peak RSS: both exceed the recorded tolerance. Dense graph retention also loses in this workload because it computes the full vocabulary projection for masked rows. It remains off by default. These results do not establish a speed or memory improvement, and no CPU or GPU performance goal is marked passed.

## 26B smoke test

The local `gemma-4-26B_q4_0-it.gguf` has 25.23B reported parameters, 30 layers, hidden size 2816, 128 experts with eight selected, and a Q6_K tied embedding. Four block-0 targets were selected: K, V, fused expert gate-up, and expert down. They contain 434,700,288 quantized bytes. CPU execution used eight threads, context/batch 256, microbatch 64, one sequence, and the first synthetic conversation.

One optimizer step completed with 10 supervised tokens and finite loss 3.77214670. The first run took 86.19 seconds for that step and reached 20,330,416 KiB peak RSS. This is a cold smoke step, not a steady-state benchmark. No quantized weight byte crossed an update boundary, so the v2 adapter correctly contained zero tensors and occupied 800 bytes.

A second run saved a complete checkpoint and optimizer sidecar. Independent inspection found finite, nonzero Q8_0 momentum in all four targets:

| Target | Nonzero momentum elements | Momentum L2 |
| --- | ---: | ---: |
| K | 5,684,034 | 0.08727 |
| V | 5,674,804 | 0.16990 |
| Expert down | 65,237,685 | 0.23112 |
| Fused expert gate-up | 106,238,637 | 0.34997 |

All 658 checkpoint tensors were independently compared with the original and matched in type, shape, and bytes. The duplicate 14.4GB checkpoint was then removed to free quota; the momentum diagnostics and sidecar were retained. Final full-model saving in that run failed under disk quota, and its truncated output was removed. This is not reported as a successful final full-model save. The existing full-model saving path was not redesigned here.

After freeing the generated duplicate, the 14.4GB export merge succeeded in 181.83 seconds with 58,580 KiB peak RSS. The copied base bytes were rehashed during streaming and matched the validated base hash. The resulting model completed CPU inference successfully (`hi;`, one generated token). The first merge attempt had hit disk quota and removed its temporary output without publishing a truncated file. This validates the actual 26B empty-adapter path; it does not substitute for a nonempty 26B replacement test.

This configuration still allocates an 821,100,544-byte dense Q8_0 accumulator; active-only accumulation for this actual MoE case is not complete. The CPU tests validate existing routed operations and their numerical gradients, not a new guarantee that all temporary buffers scale with active experts.

## Remaining work

- P620 is inaccessible: `nvidia-smi` cannot communicate with the NVIDIA driver. GPU throughput, peak VRAM, CUDA/Vulkan combinations, and the P620 dense/MoE target remain unverified. V100 execution remains outside scope.
- Investigate MoE retained/dynamic divergence before enabling retained graphs for routed models. A fixed maximum KV/mask design for general microbatch training is not implemented; unsupported cases intentionally retain the dynamic path.
- Profile and improve actual active expert accumulation and workspace, including the 26B dense accumulator above. Do not infer sparse memory behavior from adapter size or optimizer registration alone.
- The 26B smoke adapter is empty. Nonempty v1/v2 save/merge/runtime replacement is verified on the small fixtures; a meaningful multi-step 26B update and corresponding nonempty replacement still need validation.
- Common-tool runtime application is implemented; integration into every direct core-loader application, mmap/lazy overlays, and backend repacking of assembled replacements need separate design and tests.
