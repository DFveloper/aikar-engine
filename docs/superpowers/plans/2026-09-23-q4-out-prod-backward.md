# Q4_0 OUT_PROD Backward Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute activation gradients directly from frozen Q4_0 weights without a full-weight F32 or F16 temporary.

**Architecture:** Add a Volta-compatible CUDA kernel that tiles destination rows by Q4_0 blocks, streams reduction rows, dequantizes in registers, and accumulates F32. Unsupported Q4_0 layouts use a two-dimensional bounded F16 workspace and `cublasGemmEx`; non-Q4_0 types keep the existing path.

**Tech Stack:** CUDA C++, Q4_0 block format, cuBLAS, ggml CUDA pool, `test-backend-ops`

**Spec:** `docs/superpowers/specs/2026-09-23-q4-fused-training-memory-design.md`

## Global Constraints

- Direct kernel target is CUDA V100 and Q4_0.
- Accumulation and destination are F32.
- Full frozen-weight F16/F32 conversion is forbidden for Q4_0.
- Fallback workspace is capped at 64 MiB independent of weight size.
- Do not add a new test file.
- Keep new source and comments ASCII-only.
- Do not commit, push, or create a PR without explicit user approval.

## Review Focus

- Transposed source 1 must preserve `OUT_PROD` semantics; Task 2 tests both layouts.
- Batch broadcasting with `dps2` and `dps3` must index the correct Q4 matrix; Task 2 adds 3D and 4D cases.
- `beta == 1` must accumulate exactly once; Task 2 extends the existing accumulate test.
- Edge destination columns and Q4 block-aligned rows must not read beyond tensor bounds; Task 2 adds 33-column and 96-row cases.
- Fallback allocation failure must return an error and never enter full dequantization; Task 3 injects a small workspace cap.

---

### Task 1: Add bounded Q4_0 workspace plumbing

**Files:**
- Modify: `ggml/src/ggml-cuda/out-prod.cuh`
- Modify: `ggml/src/ggml-cuda/out-prod.cu`
- Modify: `tests/test-backend-ops.cpp`

**Interfaces:**
- Produces: `bool ggml_cuda_out_prod_q4_0_supported(const ggml_tensor * dst)`.
- Produces: internal `constexpr size_t Q4_0_OUT_PROD_WORKSPACE_MAX = 64ull*1024*1024`.

- [ ] **Step 1: Extend existing Q4_0 test cases**

Register `test_out_prod(GGML_TYPE_Q4_0, GGML_TYPE_F32, ...)` for dense 2D, transposed B, broadcast dimensions, and `test_out_prod_accumulate(GGML_TYPE_Q4_0)`.

- [ ] **Step 2: Run the focused test to establish the reference**

Run:

```bash
cmake --build build --target test-backend-ops -j8
build/bin/test-backend-ops test -b CUDA0 -o OUT_PROD -p 'type_a=q4_0'
```

Expected: existing full-F32-dequant cases pass, providing the numerical reference before dispatch changes.

- [ ] **Step 3: Add explicit layout classification**

Return true only when source 0 is Q4_0, `nb[0]` is canonical, `ne00` is divisible by 32, source 1 and destination are F32, and the existing dimension/broadcast assertions hold. Keep layout classification separate from launch tuning.

- [ ] **Step 4: Add a test-only fallback workspace override**

Add internal CUDA test hooks through `ggml_backend_cuda_reg_get_proc_address`: `ggml_backend_cuda_test_set_out_prod_workspace_cap(size_t)` and a reset function. Production calls never set the override and use exactly 64 MiB. This makes minimum-tile and multi-tile paths deterministic in tests without changing operation parameters.

### Task 2: Implement the direct Q4_0 CUDA kernel

**Files:**
- Modify: `ggml/src/ggml-cuda/out-prod.cu`
- Modify: `tests/test-backend-ops.cpp`

**Interfaces:**
- Consumes: canonical Q4_0 layout from Task 1.
- Produces: `launch_out_prod_q4_0(ctx, src0, src1, dst, beta)`.

- [ ] **Step 1: Add edge-tile tests and force direct dispatch**

Add cases with `(m,n,k)` equal to `(32,33,257)`, `(96,17,129)`, and the model-like orientation `(1536,8,4096)`. Add a test dispatch assertion that these cases use the direct kernel.

- [ ] **Step 2: Run tests to verify RED**

Expected: the dispatch assertion fails because all Q4_0 inputs still enter the full-dequant path.

- [ ] **Step 3: Implement block decoding helpers**

Load `block_q4_0::d` once per block and decode nibble `q` as:

```cpp
const int v = ((q >> shift) & 0x0f) - 8;
const float w = __half2float(block.d) * v;
```

Use the repository's Q4_0 lane ordering exactly; validate it against `dequantize_block_q4_0` rather than assuming low/high nibble order.

- [ ] **Step 4: Implement the 2D kernel**

Assign one warp group to a 32-row Q4 block and a small destination-column tile. For every reduction row, load one Q4 block, load source-1 values, accumulate F32 registers, reduce lanes as required, and write the destination with:

```cpp
dst_value = beta == 0.0f ? sum : beta*dst_value + sum;
```

- [ ] **Step 5: Extend indexing to standard broadcasting**

Reuse `dps2`, `dps3`, `s02`, `s03`, `s12`, `s13`, `s2`, and `s3` semantics from the existing cuBLAS path. Do not allocate pointer arrays for the direct kernel.

- [ ] **Step 6: Dispatch before full dequantization**

At the top of the Q4_0 branch, call the direct launch and return. Leave F32 and non-Q4_0 behavior unchanged.

- [ ] **Step 7: Run correctness tests**

Run all Q4_0 `OUT_PROD` tests. Expected: every case passes NMSE `5e-4`, including accumulation, transpose, and broadcast.

- [ ] **Step 8: Run compute-sanitizer on edge tiles**

Run the focused 33-column and 96-row cases under `compute-sanitizer --tool memcheck`. Expected: zero invalid accesses.

### Task 3: Implement bounded tiled F16 fallback

**Files:**
- Modify: `ggml/src/ggml-cuda/out-prod.cu`
- Modify: `tests/test-backend-ops.cpp`

**Interfaces:**
- Produces: `ggml_cuda_out_prod_q4_0_tiled(...)`, which completes or fails through the existing CUDA error path.
- Guarantees: total Q4 dequant plus source-1 conversion workspace is at most the configured cap.

- [ ] **Step 1: Add a forced-fallback test layout**

Create a safe ggml Q4_0 view that the direct classifier rejects but the tiled copier can address. Set the test cap to 64 KiB so at least two reduction tiles execute. Compare with CPU at NMSE `5e-4`.

- [ ] **Step 2: Run the test to verify RED**

Expected: dispatch remains on full F32 dequant or lacks a bounded fallback counter.

- [ ] **Step 3: Compute bounded tile dimensions**

Choose destination rows in multiples of 32 and reduction rows so:

```cpp
weight_f16_bytes + src1_f16_bytes <= workspace_cap;
```

Reserve one CUDA pool allocation and split it into aligned weight and source-1 regions. Halve the reduction tile on allocation failure down to one row.

- [ ] **Step 4: Add tile conversion kernels**

Dequantize only `[m_begin:m_end, k_begin:k_end]` to F16 and convert the matching source-1 reduction tile to F16. Respect source strides and batch broadcasting.

- [ ] **Step 5: Accumulate with `cublasGemmEx`**

Use F16 A/B inputs, F32 destination, and `CUBLAS_COMPUTE_32F`. Set beta to the operation beta only for the first reduction tile and to 1.0 for later reduction tiles.

- [ ] **Step 6: Remove Q4_0 access to full dequantization**

The Q4_0 dispatch order becomes direct kernel, bounded tiled fallback, then status error. It must never reach `src0_f32_alloc.alloc(ctx.pool(), ggml_nelements(src0))`.

- [ ] **Step 7: Run fallback and failure tests**

Expected: multi-tile fallback passes, peak workspace stays within the injected cap, and a cap smaller than one legal tile triggers the expected CUDA allocation failure without entering full dequantization.

### Task 4: Validate backward training memory and correctness

**Files:**
- Modify only when verification identifies a defect in Tasks 1-3.

**Interfaces:**
- Verifies: direct/fallback Q4_0 backward and absence of full matrix conversion.

- [ ] **Step 1: Run backend regression tests**

Run:

```bash
cmake --build build --target test-backend-ops test-opt test-qat -j8
build/bin/test-backend-ops test -b CUDA0 -o OUT_PROD
build/bin/test-opt
build/bin/test-qat
```

- [ ] **Step 2: Run one deterministic QLoRA step**

Use the supplied model, rank 4, context/batch 4096, microbatch 512, and FA on. Record loss and adapter update norm.

- [ ] **Step 3: Trace allocations**

Expected: no approximately 1.50 GiB allocation originating at `ggml_cuda_out_prod`; any Q4_0 fallback workspace is at most 64 MiB.

- [ ] **Step 4: Run one deterministic QLion step**

Verify finite loss, unchanged parameter selection, QAT dependency-plan completion, and a finite update norm.

- [ ] **Step 5: Run final static checks**

Run `git diff --check` and inspect the complete CUDA diff. Save peak VRAM and step-time results for the integration report.
