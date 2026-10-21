# Pulsar CPU Optimization Implementation Plan

**Goal:** Improve CPU prompt/decode performance of the existing Lumen 3.6 Pulsar Q4_0 model while preserving the corrected inference behavior, and measure local DFlash and requested Q5_K_M MTP drafting.

**Architecture:** Reuse existing CPU_REPACK GEMM kernels for groups of routed expert rows. Improve the Q4_0 x86 dot-product formulation and signed Q8 argmax implementation without changing stored tensor formats. Use the existing lossless QKV converter as the activation-sharing experiment rather than introducing a graph cache with new invalidation rules.

**Design basis:** User approved audit items 1, 2 and 3 and explicitly requested DFlash and RachidAR/gemma-4-26b-A4B-it-assistant-gguf Q5_K_M experiments. Audit: /tmp/lumen-cpu-optimization-audit.md.

**Constraints:** Preserve prior signed-byte correctness fixes and existing edits. Do not commit or push. Keep timing jobs sequential. Keep model originals intact. Preserve first signed-maximum tie selection, full-byte Q8 clipping and non-AVX fallbacks. Use existing backend tests and native speculative code.

## Tasks

- [x] Save corrected baseline libraries and patch separately for matched comparisons.
- [x] Add routed MoE CPU_REPACK equivalence coverage for shared/per-expert input slots, skewed routing, partial four-row groups, multiple threads and non-256 Q4 reduction widths.
- [x] Add per-thread planner workspace for up to sixteen interleaved quantized activation rows and temporary GEMM output tiles; gather already quantized rows, call existing GEMM, scatter to expert slots, retain GEMV for tails/decode.
- [x] Optimize Q4_0 x86 GEMM using unsigned nibbles and a shared activation-sum correction; retain generic quant-type behavior and -128 correctness.
- [x] Vectorize signed Q8 scale selection in quantizer and repacker, preserving ties/zeros/extrema/NaN handling; benchmark directly against scalar selection.
- [x] Build a lossless fused-QKV variant in /tmp and compare against the original model as a second activation-sharing experiment.
- [x] Run backend equivalence tests, Q4/full-byte edge tests, AVX/AVX-512 syntax checks and matched model PPL; review source changes independently.
- [x] Run isolated baseline/candidate PP128/PP512/TG64 benchmarks and retain raw results. Adopt only measured useful paths.
- [x] Check local DFlash and requested Q5_K_M MTP metadata, prepare pure-CPU baseline/draft server trials with matched prompts and output limits, then measure latency, throughput, acceptance and output agreement.
- [x] Rebuild normal CPU libraries after successful verification; write a report identifying measured gains and all limitations/failures.

## Review focus

- Source/input data can be strided; gather addresses must use tensor strides and routing mappings.
- Four-row tails and thread partitions must not overwrite neighboring outputs or exceed planned scratch.
- Quantization padding must remain aligned across planner and runtime.
- Q4 unsigned correction must remain exact for -128 activation bytes and leave IQ4/MXFP4 paths unchanged.
- Drafts trained on a different Gemma target can be compatible but have low acceptance; measure with the actual pruned fine-tuned target.

## Execution ledger

- Baseline main CTest previously had six download/HTTP/fixture failures; these are recorded in /tmp/lumen-cpu-fix-validation.md and must not be represented as green.
- Corrected baseline libraries saved in /tmp/lumen-cpu-baseline-bin; initial patch saved in /tmp/lumen-cpu-before-optimization.patch.
- Ruling: work in the current checkout to preserve and extend the authorized existing fixes; keep baseline binaries separately and avoid commits.
- Ruling: use independent agents for disjoint x86 quantizer, Q4 kernel, and draft preparation tasks per dispatching-parallel-agents; root owns routed GEMM and shared tests. All timed work uses a single performance slot.

- Review found planner/runtime thread-count scratch mismatch; reproduced with OMP_THREAD_LIMIT=7, then made per-thread scratch independent of actual thread count and added a durable CTest regression. Both normal and limited-thread tests now pass.
- SIMD quantization is byte-identical to baseline and about 2.2-2.6x faster in isolated row microbenchmarks. Q4_0 GEMM is exact against independent scalar and prior kernel and about 1.56-1.73x faster in kernel tests.
- First end-to-end model check: PP512 69.80 tok/s, TG64 11.41 tok/s (18 threads, two repetitions, no warmup). Full matched comparison pending.

- Read-only independent review approved the bounded sixteen-row packing refinement and monotonic scratch layout; runtime tests/benchmarks for that refinement follow the draft slot.
- Local DFlash and HF Q5_K_M MTP both initialize on pure CPU. Initial full suite prefers MTP3: roughly 17.6-18.1 tok/s arithmetic/coding and 12.35 Korean vs 11 tok/s no-spec. DFlash Korean acceptance is poor and slower. Exact greedy batch-vs-single output differences are being diagnosed, not assumed harmless.
