# Q4_0 Forward Training Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep frozen Q4_0 weights quantized during CUDA training forward matrix multiplication on V100.

**Architecture:** Reuse the dormant source-0 precision slot in `MUL_MAT` and mark only frozen base-weight multiplications in training graphs. CUDA treats `GGML_PREC_Q4` as a memory-policy override and selects the existing MMQ implementation after normal capability checks, while ordinary inference keeps its existing speed heuristic.

**Tech Stack:** C++17, CUDA, ggml graph APIs, existing CUDA MMQ kernels, `test-backend-ops`

**Spec:** `docs/superpowers/specs/2026-09-23-q4-fused-training-memory-design.md`

## Global Constraints

- First supported target is CUDA, NVIDIA V100, and Q4_0 frozen weights.
- Do not change ordinary inference dispatch.
- Other backends and quantization types retain current behavior.
- Do not add a new test file.
- Keep new source and comments ASCII-only.
- Do not commit, push, or create a PR without explicit user approval.

## Review Focus

- A source-0 hint on a non-quantized tensor must not force MMQ; Task 1 adds an API rejection/dispatch test.
- A Q4_0 view with unsafe padding must retain the safe fallback; Task 2 adds a view case.
- `GGML_CUDA_FORCE_CUBLAS` must produce a clear unsupported result rather than ignoring the memory policy; Task 2 adds the compile-policy check.
- Q4_0 inference with more than 96 columns must retain the old heuristic when no hint is present; Task 2 tests hinted and unhinted graphs together.
- LoRA A/B and trainable QLion tensors must not receive the frozen-weight hint; Task 3 inspects a real optimizer graph.

---

### Task 1: Make source-0 precision an explicit ggml contract

**Files:**
- Modify: `ggml/include/ggml.h`
- Modify: `ggml/src/ggml.c`
- Modify: `tests/test-backend-ops.cpp`

**Interfaces:**
- Produces: `bool ggml_prec_set_src(struct ggml_tensor * tensor, enum ggml_prec prec, int idx)` accepting `idx == 0` and `idx == 1` for `MUL_MAT` and `MUL_MAT_ID`.
- Produces: source-0 precision in operation parameter slot 2 and source-1 precision in slot 3.

- [ ] **Step 1: Add a precision field to `test_mul_mat`**

Add `ggml_prec src0_prec = GGML_PREC_DEFAULT` to the constructor and variables. In `build_graph`, apply:

```cpp
if (src0_prec != GGML_PREC_DEFAULT) {
    GGML_ASSERT(ggml_prec_set_src(out, src0_prec, 0));
}
```

Add Q4_0 cases with `n = 128` for default and `GGML_PREC_Q4`, plus an F32 case that verifies the hint does not make the graph unsupported.

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```bash
cmake --build build --target test-backend-ops -j8
build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT -p 'src0_prec'
```

Expected: construction aborts because `ggml_prec_set_src(..., 0)` currently returns false.

- [ ] **Step 3: Extend the source precision API**

Change the `MUL_MAT` and `MUL_MAT_ID` branch to accept both source indices:

```cpp
if (idx > 1) {
    return false;
}
ggml_set_op_params_i32(a, 2 + idx, (int32_t) prec);
```

Update the public comment to document source 0 as the stored matrix and source 1 as the runtime activation.

- [ ] **Step 4: Run API and CPU regression tests to verify GREEN**

Run:

```bash
cmake --build build --target test-backend-ops -j8
build/bin/test-backend-ops test -b CPU -o MUL_MAT -p 'src0_prec'
```

Expected: all new cases pass numerically and CPU ignores the optional rank hint safely.

- [ ] **Step 5: Review the diff**

Run `git diff --check` and inspect only the three files in this task. Do not commit.

### Task 2: Honor the Q4 source-0 hint in CUDA dispatch

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modify: `ggml/src/ggml-cuda/mmq.cuh`
- Modify: `ggml/src/ggml-cuda/mmq.cu`
- Modify: `tests/test-backend-ops.cpp`

**Interfaces:**
- Consumes: source-0 precision in operation parameter slot 2.
- Produces: `bool ggml_cuda_can_use_mmq(enum ggml_type type, int cc)` for capability-only checks.
- Preserves: `ggml_cuda_should_use_mmq(...)` as the speed heuristic.

- [ ] **Step 1: Add hinted versus unhinted dispatch coverage**

Add internal CUDA test hooks through `ggml_backend_cuda_reg_get_proc_address`: `ggml_backend_cuda_test_reset_dispatch()` and `ggml_backend_cuda_test_last_mul_mat_dispatch()`. Run two otherwise identical Q4_0 multiplications with 128 columns and assert:

```cpp
GGML_ASSERT(default_dispatch == GGML_CUDA_TEST_DISPATCH_CUBLAS);
GGML_ASSERT(hinted_dispatch  == GGML_CUDA_TEST_DISPATCH_MMQ);
```

Add a non-contiguous Q4_0 view case and assert it does not select MMQ when `bad_padding_clear` is true.

- [ ] **Step 2: Run the CUDA test to verify RED**

Run the focused CUDA command from Task 1. Expected: both 128-column cases select cuBLAS on V100.

- [ ] **Step 3: Split capability from performance policy**

Move the type, shared-memory, and compiled-architecture checks into:

```cpp
bool ggml_cuda_can_use_mmq(enum ggml_type type, int cc);
```

Keep batch-size and device performance thresholds in `ggml_cuda_should_use_mmq`. Preserve the existing result of `ggml_cuda_should_use_mmq` for all unhinted calls.

- [ ] **Step 4: Add the forced-Q4 branch**

Before the ordinary MMQ heuristic, read:

```cpp
const enum ggml_prec src0_prec = (enum ggml_prec) ggml_get_op_params_i32(dst, 2);
const bool force_q4 = src0_prec == GGML_PREC_Q4 && src0->type == GGML_TYPE_Q4_0;
```

Select `ggml_cuda_mul_mat_q` when `force_q4`, the layout is safe, source and destination are F32, and `ggml_cuda_can_use_mmq` succeeds. Under `GGML_CUDA_FORCE_CUBLAS`, return backend failure for a forced policy instead of allocating the full F16 matrix.

- [ ] **Step 5: Run correctness and dispatch tests**

Run:

```bash
cmake --build build --target test-backend-ops -j8
build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT -p 'type_a=q4_0.*n=128'
```

Expected: hinted and unhinted results pass NMSE `5e-4`; only the hinted case uses MMQ.

- [ ] **Step 6: Run all CUDA `MUL_MAT` tests**

Run `build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT`. Expected: PASS.

### Task 3: Mark only frozen base multiplications in training graphs

**Files:**
- Modify: `src/llama-graph.h`
- Modify: `src/llama-graph.cpp`
- Modify: `src/llama-context.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: `bool llm_graph_params::training` and includes it in graph reuse identity.
- Produces: `const bool llm_graph_context::training`, initialized from `llm_graph_params::training`.
- Consumes: `training` in `llm_graph_context::build_lora_mm` and `build_lora_mm_id`.

- [ ] **Step 1: Add optimizer-graph hint assertions**

Extend the existing optimizer graph test to walk forward nodes and classify base versus adapter multiplications. Assert:

```cpp
if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_Q4_0 &&
    !(node->src[0]->flags & GGML_TENSOR_FLAG_PARAM)) {
    GGML_ASSERT(((int32_t *) node->op_params)[2] == GGML_PREC_Q4);
}
```

Also assert adapter and inference graph operations retain `GGML_PREC_DEFAULT`.

- [ ] **Step 2: Run `test-opt` to verify RED**

Run `cmake --build build --target test-opt -j8 && build/bin/test-opt`. Expected: the Q4 hint assertion fails.

- [ ] **Step 3: Carry explicit training graph identity**

Add `training` to `llm_graph_params`, its aggregate construction, and `allow_reuse`, and copy it into an immutable `llm_graph_context` member. Pass `true` from `opt_epoch_iter` and false from inference/decode paths. During optimizer graph-size measurement, pass true because its topology and operation parameters must match the allocated training graph.

- [ ] **Step 4: Mark the base operation**

Retain the base multiplication pointer before LoRA additions:

```cpp
ggml_tensor * base = ggml_mul_mat(ctx0, w, cur);
if (params.training && w->type == GGML_TYPE_Q4_0 && !(w->flags & GGML_TENSOR_FLAG_PARAM)) {
    GGML_ASSERT(ggml_prec_set_src(base, GGML_PREC_Q4, 0));
}
ggml_tensor * res = base;
```

Apply the same rule to `MUL_MAT_ID`. Do not mark LoRA A/B or QAT fake-quant operations.

- [ ] **Step 5: Run optimizer and inference identity tests**

Run:

```bash
cmake --build build --target test-opt llama-finetune-qlora llama-finetune-qlion -j8
build/bin/test-opt
```

Expected: optimizer hints pass and inference graph reuse remains valid.

### Task 4: Add a bounded forward fallback

**Files:**
- Create: `ggml/src/ggml-cuda/mul-mat-q4-tiled.cuh`
- Create: `ggml/src/ggml-cuda/mul-mat-q4-tiled.cu`
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modify: `tests/test-backend-ops.cpp`

**Interfaces:**
- Produces: `ggml_cuda_mul_mat_q4_0_tiled(ctx, src0, src1, dst, 64ull*1024*1024)`.
- Guarantees: combined weight and activation F16 tiles never exceed the cap.

- [ ] **Step 1: Add a forced-fallback view test**

Use a Q4_0 view that fails the direct MMQ layout checks but can be copied tile by tile. Set `GGML_PREC_Q4` and assert it uses the tiled dispatch hook rather than cuBLAS full conversion.

- [ ] **Step 2: Run the test to verify RED**

Expected: the hinted view enters the existing full-conversion cuBLAS path.

- [ ] **Step 3: Implement bounded conversion tiles**

Tile the output-row and reduction dimensions. Dequantize only the selected Q4_0 weight rectangle to F16, convert only the matching source-1 rectangle to F16, and keep their combined allocation at or below 64 MiB.

- [ ] **Step 4: Accumulate with cuBLAS**

Use `cublasGemmEx` with F16 inputs and F32 compute/output. Use beta zero for the first reduction tile and one for later tiles.

- [ ] **Step 5: Route forced unsupported layouts to the bounded path**

Dispatch order for a forced Q4_0 operation becomes existing MMQ, bounded tiled F16, then the existing CUDA error path. It must never call the full-matrix `ggml_cuda_mul_mat_cublas` conversion.

- [ ] **Step 6: Run direct and fallback tests**

Expected: both pass NMSE `5e-4`, and the test hook reports no workspace above 64 MiB.

### Task 5: Measure the forward memory result

**Files:**
- Modify only when a failure identifies a defect in Tasks 1-3.

**Interfaces:**
- Verifies: no complete frozen Q4_0 matrix is converted to F16 during forward.

- [ ] **Step 1: Build the CUDA trainers**

Run `cmake --build build --target llama-finetune-qlora llama-finetune-qlion test-backend-ops test-opt -j8`.

- [ ] **Step 2: Trace CUDA pool allocations for one rank-4 step**

Run the established context 4096, batch 4096, microbatch 512, FA-on command with the supplied Q4_0 model. Record every allocation over 256 MiB and its backtrace.

Expected: no approximately 768 MiB allocation from `ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>` for the tied matrix.

- [ ] **Step 3: Record numerical and timing deltas**

Compare loss, adapter update norm, wall time, and peak VRAM against the existing FA-on baseline of 9685 MiB and 33.53 seconds. Report speed loss separately from correctness.

- [ ] **Step 4: Run final checks**

Run `git diff --check`, the focused CUDA test, all `MUL_MAT` CUDA tests, and `test-opt`. Save results for the integration report.
