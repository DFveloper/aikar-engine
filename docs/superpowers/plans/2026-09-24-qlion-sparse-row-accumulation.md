# QLion Sparse Row Accumulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace full-shape Q8 accumulation for pure `GET_ROWS_BACK` QLion parameters with reusable compact Q8 row state so the supplied 4.63B Gemma4 model completes microbatch accumulation on a 16 GB V100.

**Architecture:** QLion parameter registration allocates only momentum and residual. The first optimizer graph classifies each parameter from its pending gradient operations and lazily selects either the existing dense Q8 accumulator or a compact sparse-row state. New CPU and CUDA operations deterministically merge repeated rows into Q8 state and apply one normal QLion update per unique row at the accumulation boundary.

**Tech Stack:** C++17, ggml graph operations, ggml optimizer, CPU backend, CUDA backend, CMake/CTest, NVIDIA V100

**Spec:** `docs/superpowers/specs/2026-09-24-qlion-sparse-row-accumulation-design.md`

## Global Constraints

- Classify from gradient graph semantics, never tensor names or model shapes.
- Only a single pure `GGML_OP_GET_ROWS_BACK` contribution may use sparse accumulation.
- Mixed `OUT_PROD + GET_ROWS_BACK` and ordinary dense gradients keep full Q8 accumulation.
- Preserve Q8 momentum, Q4 residual, QLion equations, error feedback, and one update per accumulation period.
- Do not modify Flash Attention, activation recompute, tied-output storage, or tied low-rank accumulation.
- Row ID 0 is valid and `-1` is the empty sentinel.
- Capacity exhaustion must grow safely or fail before execution; no gradient may be dropped.
- Do not commit, push, create a PR, or write PR text.
- Keep repository text and code comments ASCII-only.

## Review Focus

- Hash collisions must preserve the exact row-to-slot mapping and keep load factor below 0.5.
- Duplicate IDs within and across microbatches must produce one deterministic row update.
- Sparse state growth during an active period must preserve accumulated Q8 rows, IDs, and count before rehash.
- Reset must remove stale rows while retaining allocated capacity for the next period.
- Mixed tied gradients must continue through `ggml_acc_qlion_qat_tied` and never be misclassified as pure rows.

---

### Task 1: Define Sparse Operation Contracts with Failing Tests

**Files:**
- Modify: `examples/qlora_training/test-qat.cpp`
- Inspect: `examples/qlora_training/qat.cpp`

**Interfaces:**
- Consumes: existing `ggml_opt_step_qlion_qat_rows`, Q8 quantization helpers, and `qat_tensor_state_step` reference logic.
- Produces: executable expectations for `ggml_acc_qlion_qat_rows` and `ggml_opt_step_qlion_qat_sparse_rows`.

- [ ] **Step 1: Read the test quality rules before editing tests**

Run:

```bash
sed -n '1,320p' /home/user/.codex/plugins/cache/openai-curated-remote/superpowers/6.4.1/skills/test-driven-development/writing-good-tests.md
```

Expected: the complete test guidance is available and no production file has changed.

- [ ] **Step 2: Add a low-level sparse accumulation test that does not compile yet**

Add `test_native_sparse_rows_accumulation()` for Q4_0 and MXFP4 with `cols=64`, `rows=7`, `capacity=8`, and three ordered microbatches:

```cpp
const int32_t ids_a[] = { 1, 3, 1 };
const int32_t ids_b[] = { 3, 4 };
const int32_t ids_c[] = { 1, 5 };
```

Build an explicit dense per-row reference that performs the existing sequential Q8 rule after each microbatch. Assert exact Q8 sparse accumulator bytes, exact active row IDs in first-seen order `{1, 3, 4, 5}`, and exact updated weight, momentum, and residual after one sparse optimizer step.

- [ ] **Step 3: Add reset, row-zero, collision, and partial-period cases**

Use a deliberately small hash table whose selected IDs collide under the planned integer hash. Verify row ID 0, repeated IDs, a two-microbatch step with a configured maximum of four, and a second period containing only row 2. Assert that rows from the first period do not change during the second period.

- [ ] **Step 4: Build to verify RED**

Run:

```bash
cmake --build build --target test-qat -j36
```

Expected: compilation fails only because `ggml_acc_qlion_qat_rows` and `ggml_opt_step_qlion_qat_sparse_rows` are not declared.

### Task 2: Add Graph Operations and CPU Kernels

**Files:**
- Modify: `ggml/include/ggml.h`
- Modify: `ggml/src/ggml.c`
- Modify: `ggml/src/ggml-cpu/ops.h`
- Modify: `ggml/src/ggml-cpu/ops.cpp`
- Modify: `ggml/src/ggml-cpu/ggml-cpu.c`
- Modify: `ggml/src/ggml-backend-meta.cpp`

**Interfaces:**
- Consumes: tests from Task 1 and existing QLion block update/quantization helpers.
- Produces:

```cpp
ggml_tensor * ggml_acc_qlion_qat_rows(
        ggml_context * ctx,
        ggml_tensor * accumulator,
        ggml_tensor * state_ids,
        ggml_tensor * state_hash,
        ggml_tensor * state_count,
        ggml_tensor * micro_slots,
        ggml_tensor * micro_next,
        ggml_tensor * grad,
        ggml_tensor * ids,
        bool reset,
        bool rehash);

ggml_tensor * ggml_opt_step_qlion_qat_sparse_rows(
        ggml_context * ctx,
        ggml_tensor * weight,
        ggml_tensor * grad,
        ggml_tensor * ids,
        ggml_tensor * count,
        ggml_tensor * momentum,
        ggml_tensor * residual,
        ggml_tensor * params);
```

- [ ] **Step 1: Add the new op enum values, names, symbols, and shape assertions**

Add `GGML_OP_ACC_QLION_QAT_ROWS` and `GGML_OP_OPT_STEP_QLION_QAT_SPARSE_ROWS` beside the existing QLion ops. The accumulation result is a view of the compact Q8 accumulator. Validate contiguous tensors, `grad->ne[0] == accumulator->ne[0]`, `ggml_nrows(grad) == ggml_nelements(ids)`, hash/count I32 types, and `capacity <= weight rows` at the step builder.

- [ ] **Step 2: Implement deterministic CPU row indexing and accumulation**

In thread 0, reset or rehash state, process source IDs in input order with open addressing, allocate new slots in first-seen order, and build `micro_slots` plus `micro_next`. Synchronize the CPU threadpool, then parallelize Q8 blocks across the microbatch unique rows. Each block must dequantize the prior Q8 value, add all duplicate F32 occurrences in source order, and requantize once.

- [ ] **Step 3: Implement the CPU sparse optimizer step**

Parallelize over `count * blocks_per_row`, read compact Q8 gradient blocks, and call the same QLion block update helper used by the dense and existing row paths. Each compact slot maps to exactly one parameter row.

- [ ] **Step 4: Wire CPU dispatch, task counts, op metadata, and graph cloning**

Add both ops to CPU dispatch and support switches. Add generic backend metadata handling and every op-name or duplication switch required by the compiler. Do not add Vulkan execution in this task; unsupported non-CPU/non-CUDA backends must continue selecting their existing dense path later in optimizer classification.

- [ ] **Step 5: Run the focused CPU test to verify GREEN**

Run:

```bash
cmake --build build --target test-qat -j36
QAT_TEST_BACKEND=none build/bin/test-qat
```

Expected: all existing QAT tests and the new CPU sparse-operation cases pass.

### Task 3: Add Lazy Gradient State and Graph Classification

**Files:**
- Modify: `ggml/include/ggml-opt.h`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `examples/qlora_training/test-qat.cpp`

**Interfaces:**
- Consumes: sparse graph operations from Task 2.
- Produces: graph-driven dense/sparse state selection, reusable capacity, growth, and period-length control.

- [ ] **Step 1: Add failing dynamic-optimizer integration tests**

Create a helper that follows the production dynamic path: initialize QLion without a static graph, register the quantized parameter, call `ggml_opt_prepare_alloc`, `ggml_opt_alloc`, and `ggml_opt_eval` for each microbatch.

Add these assertions:

- pure `GET_ROWS_BACK`, `opt_period=2`: compact Q8 accumulator rows equal the graph-derived capacity and are less than parameter rows;
- no full-shape F32 row gradient or dense QLion accumulation op appears in the allocated graph;
- mixed `OUT_PROD + GET_ROWS_BACK`: accumulator shape equals the full parameter and existing tied accumulation op executes;
- ordinary dense parameter: full accumulator shape and existing dense operation remain unchanged;
- a larger second graph grows sparse capacity while preserving the first microbatch state;
- a second period reuses capacity and resets IDs/count;
- current period length 2 with configured maximum 4 performs one correctly scaled update after two microbatches.

- [ ] **Step 2: Run the integration tests to verify RED**

Run:

```bash
cmake --build build --target test-qat -j36
build/bin/test-qat
```

Expected: new integration cases fail because registration still allocates full gradient state and graph classification has no sparse-state branch.

- [ ] **Step 3: Split optimizer state ownership**

Change `ggml_opt_qat_register_param()` and canonical promotion so the existing per-parameter buffer contains only Q8 momentum and Q4 residual. Add a per-parameter gradient-state record:

```cpp
enum class qat_grad_state_kind { unclassified, dense, sparse_rows };

struct qat_grad_state {
    qat_grad_state_kind kind;
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_tensor * accumulator;
    ggml_tensor * row_ids;
    ggml_tensor * row_hash;
    ggml_tensor * row_count;
    ggml_tensor * micro_slots;
    ggml_tensor * micro_next;
    int64_t capacity;
    int64_t hash_capacity;
};
```

Free, reset, and canonical-promotion paths must own these buffers independently. Preserve the existing public gradient-accumulator accessor by returning the selected Q8 accumulator after classification and `nullptr` before classification.

- [ ] **Step 4: Classify pending gradients before combining them**

In the QLion pending-gradient builder, select sparse state only when `pending.size() == 1`, `pending[0]->op == GGML_OP_GET_ROWS_BACK`, and its sources/types/layout satisfy the existing row-step contract. Run the tied mixed specialization before this pure-row check. Any additional contribution or unsupported backend selects dense state.

- [ ] **Step 5: Allocate and grow sparse state from the graph bound**

Use `min(parameter->ne[1], configured_opt_period * ggml_nelements(rows_back->src[1]))` as the required bound. Choose a power-of-two hash capacity at least twice the row capacity. Reuse existing state when sufficient.

For growth, synchronize the owning backend, allocate the larger context/buffer, copy compact Q8 rows, IDs, and count, leave the new hash empty, mark the next accumulation op `rehash=true`, then release the old state. Do not read row IDs or count back to CPU on the CUDA path.

- [ ] **Step 6: Add current-period override semantics**

Track `configured_opt_period` and `current_opt_period`. Add a boundary-only API that accepts `1..configured_opt_period` when `opt_i == 0`. Use `current_opt_period` for loss scaling and step selection, then restore the configured value after the step. Update high-level epoch iteration to set a shorter final period when the remaining physical batch count is smaller than the configured maximum.

- [ ] **Step 7: Build sparse accumulation and step nodes**

For intermediate microbatches, return the sparse accumulation node. On the period boundary, feed its compact Q8 result, row IDs, and count into `ggml_opt_step_qlion_qat_sparse_rows`. Keep dense and tied branches byte-for-byte equivalent aside from lazy allocation plumbing.

- [ ] **Step 8: Run integration and dense regression tests to verify GREEN**

Run:

```bash
cmake --build build --target test-qat test-opt -j36
build/bin/test-qat
build/bin/test-opt
```

Expected: sparse integration tests pass, dense QLion reference bytes remain unchanged, tied mixed classification stays dense, and optimizer tests pass.

### Task 4: Add CUDA Sparse Accumulation and Step

**Files:**
- Modify: `ggml/src/ggml-cuda/opt-step-qlion-qat.cuh`
- Modify: `ggml/src/ggml-cuda/opt-step-qlion-qat.cu`
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modify: `examples/qlora_training/test-qat.cpp` only if backend-independent assertions need adjustment

**Interfaces:**
- Consumes: exact graph-op contracts and CPU reference behavior from Tasks 2 and 3.
- Produces: CUDA implementations with identical row ordering and Q8 state transitions.

- [ ] **Step 1: Run the CUDA test to verify RED**

Run:

```bash
QAT_TEST_BACKEND=CUDA0 build/bin/test-qat
```

Expected: the new sparse operations are reported unsupported or fail dispatch on CUDA while CPU tests remain green.

- [ ] **Step 2: Implement deterministic CUDA index preparation**

Use one CUDA thread to process current IDs in input order, perform open-addressed lookup/insertion, allocate compact slots, and fill persistent row IDs plus `micro_slots` and `micro_next`. This avoids duplicate insertion races and makes slot order deterministic. Validate IDs and set a device error flag if the table is exhausted or an ID is outside the parameter row range.

- [ ] **Step 3: Implement parallel Q8 row accumulation**

Launch one warp per `(microbatch unique row, Q8 block)`. Traverse the occurrence links in source order, sum F32 gradients, dequantize the prior compact Q8 block unless reset/new, add the sum, and requantize once. On reset, clear hash, count, IDs, and accumulator asynchronously on the same stream before indexing.

- [ ] **Step 4: Implement parallel sparse optimizer step**

Launch over `capacity * blocks_per_row`, skip slots `>= count`, map each valid slot to its parameter row, and call the existing templated QLion apply helper. Distinct slots update distinct rows, so execution order cannot change numerical results.

- [ ] **Step 5: Wire CUDA dispatch and support checks**

Add declarations, dispatch cases, and `supports_op` validation for Q4_0 and MXFP4 weights with Q8 compact gradients. Unsupported CUDA-like backends such as MUSA must return false unless their implementation is present.

- [ ] **Step 6: Run CPU/CUDA equivalence tests to verify GREEN**

Run:

```bash
cmake --build build --target test-qat -j36
build/bin/test-qat
QAT_TEST_BACKEND=CUDA0 build/bin/test-qat
```

Expected: CPU and CUDA produce exact compact Q8 accumulator, parameter, momentum, and residual bytes for the deterministic fixtures. If backend quantization is not byte-identical, record and enforce the strict dequantized tolerance in the test message.

### Task 5: Add Memory Instrumentation and Production Regression Checks

**Files:**
- Modify: `ggml/include/ggml-opt.h`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `src/llama-context.cpp`
- Modify: `examples/qlora_training/finetune_qat.cpp`
- Modify: `examples/qlora_training/test-qat.cpp`

**Interfaces:**
- Consumes: selected gradient-state records from Task 3.
- Produces: dense/sparse byte reporting and a graph-level memory regression assertion.

- [ ] **Step 1: Add failing memory-accounting assertions**

Assert that the pure-row fixture reports nonzero sparse bytes, zero dense bytes, capacity-derived Q8 bytes, and no tensor matching full parameter shape with type F32 or Q8. Assert that the mixed fixture reports full dense bytes and zero sparse bytes.

- [ ] **Step 2: Add minimal state inspection APIs**

Expose aggregate dense accumulator bytes, sparse accumulator bytes including IDs/hash/counters, sparse state count, and per-state capacity. Keep tensor pointers private except the existing gradient-accumulator accessor.

- [ ] **Step 3: Update trainer memory output**

Replace the single gradient byte total with explicit fields:

```text
gradient_dense_q8_0=...
gradient_sparse_q8_0=...
gradient_sparse_metadata=...
```

Print the selected class and capacity once when each state is first allocated. Do not print per-step diagnostics.

- [ ] **Step 4: Verify the focused memory tests**

Run:

```bash
cmake --build build --target test-qat llama-finetune-qlion -j36
build/bin/test-qat
```

Expected: memory assertions pass and the trainer reports separate dense/sparse state totals.

### Task 6: Build and Verify the Full Change

**Files:**
- Review all modified production and test files.
- Do not create source or test files outside existing test infrastructure.

**Interfaces:**
- Consumes: complete CPU/CUDA implementation and instrumentation.
- Produces: verified production behavior and before/after memory measurements.

- [ ] **Step 1: Build all affected binaries and tests**

Run:

```bash
cmake --build build --target test-opt test-qat test-arg-parser llama-finetune-qlion llama-finetune-qlora -j36
```

Expected: all targets build with CUDA enabled.

- [ ] **Step 2: Run the focused test suites**

Run:

```bash
build/bin/test-opt
build/bin/test-qat
QAT_TEST_BACKEND=CUDA0 build/bin/test-qat
build/bin/test-arg-parser
```

Expected: all suites pass with no new warning or skipped sparse CUDA case.

- [ ] **Step 3: Reproduce the fixed V100 production configuration**

Run the supplied command with `-c 128 -b 128 -ub 64`, CUDA0 only, Flash Attention on, and activation recompute on. Sample GPU memory at 50 ms. Stop only after at least two optimizer steps.

Expected:

- model and optimizer state load succeeds;
- no 9,203.26 MiB scheduler allocation occurs;
- first forward/backward and optimizer step complete;
- second accumulation period completes without stale rows;
- `per_layer_token_embd.weight` reports compact sparse state;
- tied `token_embd.weight` reports dense state;
- peak remains below 16,144 MiB usable VRAM.

- [ ] **Step 4: Measure sequence scaling in fresh processes**

Repeat accumulation runs for sequence lengths 128, 256, 512, and 1024 with recompute on. Record dense state, sparse Q8 rows, sparse metadata, scheduler buffer, process peak, retained post-step memory, and success or exact failure allocation.

- [ ] **Step 5: Run broad project tests in the configured build**

Run:

```bash
ctest --test-dir build --output-on-failure
```

Expected: all configured tests pass. If unrelated pre-existing failures occur, record their exact names and output without changing unrelated code.

- [ ] **Step 6: Review source and formatting**

Run:

```bash
git diff --check
git status --short
git diff --stat
git diff -- ggml/include/ggml.h ggml/include/ggml-opt.h ggml/src/ggml.c ggml/src/ggml-opt.cpp ggml/src/ggml-cpu ggml/src/ggml-cuda examples/qlora_training/test-qat.cpp examples/qlora_training/finetune_qat.cpp src/llama-context.cpp
```

Expected: no whitespace errors, no diagnostic-only allocator changes, no generated binaries, and only the approved production/test/spec/plan files are modified.
