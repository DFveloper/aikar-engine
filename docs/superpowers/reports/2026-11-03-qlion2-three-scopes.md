# QLion2: three remaining implementation scopes

This implements the three requested scopes after the P620 follow-up. Step-speed optimization is excluded. The earlier 1.10 MoE/dense time-ratio target is not claimed as achieved.

## Fixed-KV microbatch graph reuse

`--qat-static-graph on` now supports compatible CPU/CUDA microbatch, accumulation, and multiple-sequence training. Graph reuse follows the existing KV capacity buckets; attention masks and other inputs are refilled on each microstep. A capacity-bucket transition rebuilds the graph. Supervised output rows retain the dynamic path's selection. A change in graph shape, supervised output count, optimizer/accumulation phase, or train/eval mode rebuilds the graph. Q8 accumulation reset flags follow the current accumulation group.

An initial full-capacity mask experiment passed CPU comparisons but changed CUDA quantized updates at context 512 despite equal logged losses. It was removed. The final implementation preserves the dynamic path's KV geometry and CUDA reduction order and reuses a graph while its KV bucket is stable. This avoids changing the training trajectory or adding a separate KV-cache interface. The failed experiment is recorded in `initial-full-kv-cuda-512-failure.json`; the byte-equality assertion was retained.

MTP, critical weighting, activation recomputation, source embeddings, and actual-route compact planning use the dynamic path. The existing `n_ctx == n_batch` eligibility check remains. Reuse does not attach gradients through earlier microbatch KV rows or pad the vocabulary output head.

Dense and MoE fixtures with context 256 and 512, microbatch 64, and repeated supervised rows actually reuse graphs. Each static result matches every tensor of its dynamic reference byte for byte: 86 dense tensors and 128 MoE tensors. The integration suite also checks accumulation, parallel final batches, changing lengths, periodic evaluation, full-sequence reuse, and optimizer resume. Both CPU and CUDA exercise compatible reuse.

## Actual-route Q8 expert accumulation

`--qat-moe-gradient compact` runs a forward routing probe before building the backward graph. It sizes expert Q8 storage from the actual expert-ID union across the accumulation group. Growing the union preserves accumulated gradients; the next group starts with its own actual capacity. Unsupported/multiple gradient contributions retain the dense fallback. GPU gradients with CPU expert state also use the dense reference path: computing their outer products on CPU changed the quantized trajectory. This fallback preserves GPU reduction order and emits an explicit log message. Default `auto` behavior is unchanged.

The probe preserves graph execution flags, snapshots inputs before forward allocation can reuse them, and uses explicit tensor-copy bindings. It allocates the backward topology with minimal expert slots, then executes only the forward prefix, excluding backward split copies that could overwrite route buffers. Retaining the backward topology keeps forward fusion and buffer-lifetime decisions consistent with replay; a forward-only topology could select different near-tied experts on CUDA. Replay keeps the same forward backend placement and applies the normal backward placement policy. After collecting routes, it builds the backward graph with the exact capacity and restored inputs. The probe does not execute an optimizer update. Compact planning disables retained graphs because route buffers and backward metadata can change between microsteps.

Periodic evaluation and additional gradient accumulation are regression cases on CPU, CUDA, and both directions of mixed CPU/CUDA placement. They compare all 128 final model tensors with dense accumulation byte for byte. Mixed placement selects K0, V0, and expert down5 on CUDA while the other weights remain on CPU. The reverse placement keeps expert down5 on CPU and the other weights on CUDA, and verifies the explicit dense fallback. Temporary state metadata is excluded when resetting reusable forward bindings, including when the planned expert buffers grow or reset.

CUDA computes and quantizes one expert gradient at a time, using one expert's F32 and Q8 workspace. It updates all selected experts, including zero-gradient momentum/decay updates for inactive experts. Full selected-tensor momentum/residual states remain allocated. Host route inspection synchronizes the stream, so these routed-gradient operations are excluded from CUDA graph capture. A Q8 reset copies blocks directly, matching CPU behavior; broadcast activations use the correct expert-independent source offset.

CPU tests cover MXFP4, Q4_0, Q8_0, and Q2_0; CUDA tests cover MXFP4, Q4_0, and Q8_0. Twelve microsteps include duplicate and strided routes, changing input lengths, a union growing from two to three experts, group reset, inactive experts, zero gradients, and nonzero decay. Dense and compact weight, momentum, and residual trajectories match byte for byte on each backend. Tests assert exact compact capacity and absence of the full F32 expert-gradient operation. Native tiny-MoE training also matches dense accumulation after twelve microsteps. Six CUDA compact-operation reference cases pass.

This is a memory option with an extra forward pass and host synchronization. It does not promise a faster step. Backends beyond CPU/CUDA have no new native compact kernel and may use scheduler fallback.

## Core-loader adapter integration

Validation and replacement handling now live in `src/llama-qlion-adapter.cpp`, shared by the exporter and core model loader. `llama_model_params.qlion_adapters` accepts a null-terminated path list; `qlion_allow_compatible_base` exposes the existing explicit compatibility mode. Common CLI tools pass their paths into this API. Rebuild consumers of the extended model parameter structure.

Resident tensors are assembled before backend upload/repacking. mmap uses a private writable mapping and copies replacement pages without writing to the base file. Lazy tensor ranges retain that mapping for the model's lifetime. v1 full replacements and v2 row/expert replacements retain the existing identity, shape, type, hash, alias, and overlap checks. Adapter inputs are consumed during loading; the mapping and backend buffers own the resulting bytes afterward.

`test-qlion-loader` links only `llama`, calls the core API directly, and emits logits for comparison with a fully trained model. Dense/MoE v1 and v2 adapters pass all 32 combinations of resident/mmap, lazy off/on, and repacking off/on. Actual Q4_0 CPU repacking is observed in loader logs. A generated PLE fixture adds a real lazy tensor; its nonempty v2 adapter replaces 50 embedding rows. All eight PLE combinations match reference logits, lazy-on logs confirm lazy reading, and the original file SHA-256 remains unchanged.

Split models, vocabulary-only and file-handle/embedded model loading, and the independent CPU LES disk cache remain unsupported. Optimizer resume still uses full checkpoints and optimizer sidecars. Windows copy-on-write mapping is implemented but was not exercised on this Linux host.

## Validation artifacts

Local artifacts are under `/tmp/qlion-three-checks`; they are not repository inputs. Fixtures come from `/mnt/openwebui/AIKAR/QLion2-test/artifacts` and require no download. The reproducible integration entry point is `examples/qlora_training/test_qlion_integration.py`; build `test-qlion-loader` together with the training, export, and completion tools.

- `integration-cpu/report.json` and `integration-cuda/report.json`: 68 CPU and 77 CUDA executions cover native training, reference comparisons, resume, target/freeze checks, and adapter export/runtime inference. Each suite additionally checks 40 direct-loader combinations.
- `static-check.log` and `static-cuda-check.log`: additional context-256/512 dense/MoE microbatch comparisons.
- `ctest-cpu.log`: `test-opt`, `test-critical-token-sft`, `test-qat`, and `test-qlion-adapter` pass.
- `cuda-unit-growth.log`: explicit `QAT_TEST_BACKEND=CUDA0` optimizer/state tests pass.
- `cuda-compact-op-final.log`: six compact CUDA reference operations pass.
- `asan-unit.log`, `asan-native-accum.log`, and `asan-native-eval.log`: AddressSanitizer optimizer and native compact training checks pass with `detect_leaks=0`, including accumulation-group changes and periodic evaluation.
- `asan-loader.log` and `asan-resident.log`: AddressSanitizer direct-core loader checks pass with `detect_leaks=0`. mmap/lazy/repacked and resident/repacked PLE logits match an ASAN reference load. Leak detection was not run.
- `lazy-check.log`, `ple-train.log`, and `repack.log`: lazy overlay, nonempty replacement, and actual repacking evidence.

These runs validate behavior and memory layout. Concurrent validation workloads mean timing and whole-device memory samples from fixture integration are not controlled performance comparisons. V100 and other CUDA architectures were not tested.

## Actual 26B compact run

The existing 26B source `/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/gemma-4-26B_q4_0-it.gguf` was loaded with only K0, V0, and expert down29 placed on P620 and selected for training. Context/batch was 256, microbatch 64, with `--qat-moe-gradient compact` and no full-model export. The final command and results are in `26b-compact-final.json`.

The down29 expert accumulator follows the actual route union: 42, 46, then 47 slots out of 128. Its allocated state grows from 88,469,248 to 99,003,008 bytes, including index/scratch metadata. Expert Q8 gradient bytes are 98,999,296 rather than the dense 269,615,104 bytes, a 63.3% reduction in this accumulator. The next accumulation group resets to 42 slots. Momentum (281,870,336 bytes) and residual (149,225,472 bytes) remain full across all three selected tensors.

Two optimizer steps completed with ten supervised tokens each. The first is QLion2 warmup and produces an empty adapter by itself; the second produces a nonempty v2 adapter with all three selected tensors, including 40 changed experts. Payload size is 51,093,504 bytes; total adapter size is 51,455,360 bytes. No full 14 GiB merged file was created. This is a functional/memory smoke, not a training quality result or controlled speed comparison.

Total elapsed time was 370.82 seconds including hashing and adapter publication. Step durations were 72.76 and 71.59 seconds. Peak process RSS was 15,966,504 KiB and sampled whole-device peak was 1,763,770,368 bytes (1.64 GiB). Device samples include concurrent fixture checks; they are not an isolated process memory measurement. The deterministic accumulator byte counts above describe the selected state directly.

The final nonempty 26B adapter was loaded through the core mmap path with backend repacking enabled and the same scoped GPU placement. Completion with prompt `hi` and two generated tokens exited successfully (`26b-core-runtime-final.json`, 101.42 seconds including hashing). No merged model was needed. Small-model tests separately establish exact reference-logit equality.
