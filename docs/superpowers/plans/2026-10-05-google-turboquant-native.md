# Google TurboQuant Native Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. Native execution in the current session is the established user preference; no commits or automatic deployment.

**Goal:** Reproduce the published TurboQuant MSE and QJL algorithms in native CPU/CUDA KV operations and measure Pulsar S precision and Paged KV throughput.

**Architecture:** Store separately versioned packed per-head K/V rows in I8 tensors. Share a context-owned FP32 parameter tensor containing independent head-wide Haar rotations, the independent Gaussian K residual projection, and dimension-specific Lloyd-Max codebooks. Native pack and attention operations consume these tensors without persistent uncompressed cache copies.

**Tech Stack:** Existing C/C++17 ggml, CPU backend, CUDA on V100, existing test executables, CMake builds with `-j36`.

**Spec:** `docs/superpowers/specs/2026-10-05-google-turboquant-native-design.md`.

## Global Constraints

- Algorithms 1 and 2 in arXiv:2504.19874v1 are the source of truth; do not claim identity with unpublished Google source or the incompletely specified 2.5/3.5-bit experiments.
- Full D256/D512 head transforms; Haar rotation from Gaussian QR, independent IID Gaussian QJL, spherical-density Lloyd-Max codebooks, original FP32 norms and FP32 query arithmetic.
- K total widths 3/4 use base widths 2/3 plus one residual bit. V widths 3/4 use all bits for MSE. No reconstructed-norm correction.
- Existing `turbo3`, `turbo4` and Q8_KV layouts remain readable and unchanged.
- Reuse existing test files; do not create `tests/*` files. No inline code comments unless requested.
- Preserve the fresh reference logits, other worktree changes and the production launcher. Do not commit, push, create PRs or deploy automatically.
- Execute native work in an isolated checkout, preserving the earlier root CUDA precision patch. Do not reuse or reset the unrelated `/tmp/aikar-paged-kv` checkout as the main implementation workspace.
- Checkpoint failed quality/performance gates as experimental; never describe an unmeasured candidate as passing.

## Review Focus

- Width/dimension conflicts: reject invalid CLI policies before allocating a cache, including explicit conventional types for selected layers.
- Seed/state mismatch: restoring packed bytes with different transforms must fail before modifying live cache data.
- Matrix orientation and indexing: GQA, multiple streams, noncontiguous queries and partial tiles must match the independent original-basis oracle.
- Input extremes: zero, tiny and large finite vectors must retain finite norms or return a defined failure; NaNs/infinities must not silently produce valid-looking rows.
- Physical/logical dimensions: byte strides, partial blocks, allocator copies and cache sharing must preserve complete row ownership without padding corruption.

## Task 1: CPU parameter generation and packed reference codec

**Files:** Create `ggml/src/ggml-turboquant.h`, `ggml/src/ggml-turboquant.cpp`; modify `ggml/src/CMakeLists.txt`, `tests/test-turbo-quant.cpp`.

**Interfaces:** Internal C-compatible entry points, exported from ggml-base for backend linking:
- `size_t ggml_turboquant_params_size(int32_t head_dim)` returns the parameter float count, zero for unsupported dimensions.
- `bool ggml_turboquant_params_init(float * parameters, int32_t head_dim, uint64_t seed)` initializes fixed-offset rotations, independent projection and codebooks for base widths 2/3 and V widths 3/4.
- `size_t ggml_turboquant_row_size(int32_t head_dim, int32_t total_bits, bool key)` returns the physical byte width or zero for invalid arguments.
- `bool ggml_turboquant_pack_row(const float * input, uint8_t * packed, const float * parameters, int32_t head_dim, int32_t total_bits, bool key)` writes one row and reports invalid input.
- `void ggml_turboquant_unpack_row(const uint8_t * packed, float * output, const float * parameters, int32_t head_dim, int32_t total_bits, bool key)` reconstructs in the original basis.
- `float ggml_turboquant_dot_row(const uint8_t * packed, const float * rotated_query, const float * projected_query, const float * parameters, int32_t head_dim, int32_t total_bits)` evaluates the packed K estimator.

- [ ] Add reference tests in the existing Turbo test executable: orthogonality, inversion, spherical Lloyd-Max centroid stationary conditions, actual row widths, original norm retention, canonical zero rows, invalid inputs and non-byte-aligned 3-bit packing.
- [ ] Run the focused executable after adding API declarations/minimal unavailable implementation; observe failure of the new algorithm checks before implementing the codec.
- [ ] Generate Haar rotations by double-precision Householder QR with diagonal-sign correction, using a deterministic Gaussian generator whose domain-separated streams produce K/V rotations and QJL independently. Solve codebooks once per head dimension with double-precision quadrature and Lloyd-Max iteration; cache the validated values independently of model data.
- [ ] Implement little-endian FP32 norms and low-bit-first contiguous index/sign packing. Compute norms safely in double and reject values whose stored FP32 norm cannot represent the result. Do not silently clip finite overflow.
- [ ] Independently reconstruct Algorithm 2 in the test using double accumulation and compare the packed dot estimator. For statistical validation, use fixed vector/query pairs and many independent projection seeds; assert bias within a predeclared confidence interval and report measured variance.
- [ ] Build `test-turbo-quant` with `-j36`; require all old and new CPU checks to pass. Record codebook resolution/convergence checks and the exact seeds.

## Task 2: Native ggml pack and attention operations

**Files:** Modify `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cpu/ops.h`, `tests/test-turbo-quant.cpp`.

**Interfaces:**
- `ggml_turboquant_pack(ctx, input, indices, cache, parameters, head_dim, total_bits, key)` returns a view of the I8 destination cache with a new side-effecting pack operation. Inputs use explicit vector/head/token/stream axes and physical row strides.
- `ggml_turboquant_attn(ctx, query, cache_k, cache_v, mask, sinks, parameters, head_dim, bits_k, bits_v, scale, max_bias, logit_softcap, n_kv_max)` returns an F32 tensor with the existing flash-attention output axes.
- Parameter tensor offsets and operation parameter layout are defined once in `ggml-turboquant.h`; constructors validate all dimensions/types/byte ranges.

- [ ] Add failing backend tests for CPU scatter writes and attention against the independent oracle, including GQA, local/causal masks, softcap, sinks, two streams, noncontiguous inputs and partial rows.
- [ ] Register two operation enums, names and symbols; update the existing operation-count assertions and explicit dispatch/support switches. Mark pack as a graph dependency/side effect consistently with `SET_ROWS`.
- [ ] Implement CPU pack via Task 1's codec. Implement attention with per-query transforms reused over rows, a bounded score/workspace buffer, model-order softcap/masking and original-basis inverse V rotation after the weighted sum.
- [ ] Add explicit CPU planner workspace sizes and thread ownership for rows/queries. Reject overlapping scatter destinations or define ordered ownership at construction rather than introducing a data race.
- [ ] Run the CPU backend cases and existing Turbo test suite; verify old paths still pass.

## Task 3: Ordinary native CUDA pack and attention

**Files:** Create `ggml/src/ggml-cuda/turboquant.cu`, `ggml/src/ggml-cuda/turboquant.cuh`; modify `ggml/src/ggml-cuda/ggml-cuda.cu`, its CMake registration if necessary, and `tests/test-turbo-quant.cpp`.

**Interfaces:** `ggml_cuda_op_turboquant_pack(ctx, dst)` and `ggml_cuda_op_turboquant_attn(ctx, dst)` implement the operations from Task 2 with the same packed layout and parameter tensor.

- [ ] Add failing CUDA backend cases against CPU-packed rows and the double original-basis oracle, excluding scalar boundary ties from exact-byte checks.
- [ ] Implement pack with head-wide rotation, original norm, original-basis residual, independent Gaussian projection and sign-bit writes. Use safe norm reduction and explicit byte ownership for 3-bit packing.
- [ ] Implement query rotation/projection once per query and head, consume packed K rows and merge base/residual contributions before softmax. Accumulate weighted V in its rotated basis and apply the inverse V rotation once.
- [ ] Support D256/D512, 3/4-bit combinations, multiple queries and streams, GQA, masks, sinks and partial tiles. Preserve source strides and the current CUDA stream. Account for bounded workspace using the existing CUDA pool.
- [ ] Build once with `-j36` and run the existing Turbo executable; require CPU/CUDA parity and record per-case maximum error. Do not substitute WHT or quantize Q to meet speed targets.

## Task 4: Explicit llama KV policy and cache lifecycle

**Files:** Modify `include/llama.h`, `common/common.h`, `common/arg.cpp`, `common/common.cpp`, `src/llama-cparams.h`, `src/llama-context.cpp`, `src/llama-memory.h` and implementation routes, `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`, `src/llama-graph.h`, `src/llama-graph.cpp`. Extend existing argument/state tests where their fixtures apply.

**Interfaces:** Add context/common fields for global/local TurboQuant widths and seed. Add `llama_kv_cache::set_turboquant_policy(...)` and context-owned immutable parameter tensors per required dimension. Existing memory wrappers forward the policy to their respective global/SWA caches.

- [ ] Add failing parser/context tests for `--kv-turboquant-global 3|4`, `--kv-turboquant 3|4`, a seed option, conflicting explicit conventional types, unsupported dimensions/backends and overlapping policies.
- [ ] Route selected cache layers to explicit I8 physical row allocation and Task 2's operations while preserving logical head dimensions. Retain local F16 for the global-only policy. Reject unsupported MLA, transposed V cache or context-shift modes until validated.
- [ ] Preserve existing slot indexing and construct per-head packed-row indices. Adapt stream views/copy offsets to byte strides. Log effective per-layer role/width, actual bytes and parameter/workspace overhead.
- [ ] Extend state serialization with a distinct version/policy marker and stored transform/codebook values. Validate compatibility before changing live data; old Turbo and conventional states continue through their existing readers.
- [ ] Validate clear, sequence remove/copy, shared layers, multi-stream copies and same/different-seed save/restore in the existing state test infrastructure. A mismatch must be rejected before partial restore.
- [ ] Build `llama-perplexity` and relevant existing argument/state tests with `-j36`, and run those tests.

## Task 5: Pulsar S quality and ordinary CUDA optimization

**Files:** Update `docs/pulsar-s-kv-20261005.md`; make focused optimizations in the new CUDA files only after correctness.

**Inputs:** The complete saved S F16 reference at `/home/user/lumen-kv-bench-20261005/Pulsar-S-f16.logits`, S model, corpus and matched settings recorded in the spec.

- [ ] Compare all-F16, global 3-bit and global 4-bit using 34 chunks, context/batch 512, zero CPU MoE layers and the same placement. Preserve full logs and effective cache-format output.
- [ ] Compare K-only/V-only variants using the same parameter identities; measure at least several predetermined seeds to identify seed variance rather than selecting the best seed for reporting.
- [ ] Report mean/99.9%/maximum KLD, RMS delta-p and top-1. Passing Q8-grade quality requires mean KLD <= 0.051338 and top-1 >= 92.341%; do not change the reference or acceptance threshold after seeing candidate results.
- [ ] Profile query transforms, cache packing and packed attention. Optimize launch count, shared parameter reuse and packed-row consumption without changing the published algorithms. Re-run correctness after each change.
- [ ] Benchmark matched PP/TG and total allocated cache/parameter/workspace memory. Record whether each quality/performance gate passes; retain failures as experimental.

## Task 6: Paged KV integration and final verification

**Files:** After reading that checkout's instructions, make only targeted changes to the existing `/tmp/aikar-paged-kv` Paged allocation, write and attention operations. Extend its existing `tests/test-backend-ops.cpp` cases and document exact source revision/diff and artifact hashes. Do not replace the validated Q8_KV staging package.

**Interfaces:** Existing Paged block-table and ownership infrastructure supplies physical row addresses to the same codec and completed score estimator from Tasks 1-3. Context parameter tensors remain shared per dimension; matrices are not duplicated per KV row or page.

- [ ] Add failing Paged backend cases for both profiles, block-boundary writes, partial pages, multiple sequences, local masks and allocator clear/reuse. Check packed bytes and decoded attention against the ordinary implementation.
- [ ] Extend the existing paged physical-stride calculations and scatter/attention kernels for the explicit TurboQuant policy. Avoid merging unrelated Paged changes into the root checkout.
- [ ] Build the separate candidate with `-j36`; require existing Q8_KV Paged cases and the new cases to pass.
- [ ] Re-run the fresh-logits quality comparison through the Paged implementation to detect indexing or masking drift.
- [ ] Measure single-request throughput using the prior 8K pool, 4K prompt, sampler parameters and three repetitions. Target PP > 1000 and TG around 100; report any failure with query-transform/packing timings.
- [ ] Run appropriate existing regression tests, `git diff --check`, launcher syntax checks and an inline code review. Record all skipped/failed checks and the production server state. No production replacement without user authorization.

## Handoff state

The written spec is approved. This plan is pending user review, as required by the writing-plans skill. No new TurboQuant implementation code has been written yet. The earlier precision patch and saved measurements are unchanged.
