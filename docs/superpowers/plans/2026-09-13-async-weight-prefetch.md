# Async Weight Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlap host-to-device transfer of the next streamed training split with CUDA computation of the current split by extending the existing ggml backend scheduler.

**Architecture:** Add a weight-streaming configuration and runtime state to `ggml_backend_sched`. The scheduler retains graph ownership and split execution, but async mode reserves two non-aliasing weight slots, packs the next split into bounded pinned host memory, uploads it through a second backend instance for the same CUDA device, and coordinates upload and compute with events. Unsupported configurations use the current synchronous streaming path.

**Tech Stack:** C++17, ggml backend scheduler, CUDA backend abstraction, ggml events, pinned host buffer types, llama optimizer context, common argument parser

**Spec:** `docs/superpowers/specs/2026-09-13-async-weight-prefetch-design.md`

**Implementation note:** CUDA directly registers mmap-backed host weight buffers when possible. The two pinned host arenas remain the portable fallback.

## Global Constraints

- Extend the existing trainer and backend scheduler; do not add a model-specific trainer.
- Preserve behavior when weight streaming or async prefetch is disabled.
- Keep new code and comments ASCII-only.
- Use graph splits, not parsed transformer layer names, as execution units.
- Keep dynamic `MUL_MAT_ID` expert copies on the synchronous path.
- Do not tile a single tensor.
- Do not add a new test file under `tests/`.
- Do not commit without explicit user approval for that commit.

---

### Task 1: Add the scheduler configuration contract

**Files:**
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: `ggml_backend_sched_weight_streaming_params`.
- Produces: `bool ggml_backend_sched_set_weight_streaming(ggml_backend_sched_t, ggml_backend_sched_weight_streaming_params)`.
- Preserves: a bool compatibility overload is not required because all callers are in this branch and are updated in Task 4.

- [ ] **Step 1: Extend the existing scheduler replacement test with configuration assertions**

Add a subcase in `test_scheduler_replacement()` that configures disabled, synchronous, and async-requested policies while the scheduler is reset. Assert that configuration after graph allocation returns false.

- [ ] **Step 2: Build to verify the new type and signature are missing**

Run:

```bash
cmake --build build --target test-opt -j18
```

Expected: compile failure naming `ggml_backend_sched_weight_streaming_params`.

- [ ] **Step 3: Add the public configuration structure and scheduler state**

Use:

```cpp
struct ggml_backend_sched_weight_streaming_params {
    bool enabled;
    bool async_prefetch;
    size_t staging_bytes;
};
```

Store requested and effective policy separately in `ggml_backend_sched`. The setter rejects calls unless `sched->is_reset` is true and no graph has been reserved. Disabled policy clears effective async state.

- [ ] **Step 4: Run the focused test**

Run:

```bash
cmake --build build --target test-opt -j18
build/bin/test-opt
```

Expected: exit 0 on CUDA and CPU backends.

### Task 2: Reserve bounded double-buffered streamed weights

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Consumes: scheduler streaming configuration from Task 1.
- Produces: per-split staged byte accounting, largest streamed tensor size, effective slot count, and a normal reservation failure.

- [ ] **Step 1: Add a failing bounded-slot scheduler test**

Construct a small graph in `tests/test-opt.cpp` with at least three host weight inputs assigned to a non-CPU backend. Configure a per-slot limit that holds one weight but not two. Verify graph splitting creates multiple streamed splits and two copy indices do not alias. Add a second case whose manual limit is smaller than one weight and assert reservation returns false.

- [ ] **Step 2: Run the test and confirm current allocation ignores the budget**

Run `cmake --build build --target test-opt -j18 && build/bin/test-opt`.

Expected: the new split-count, non-aliasing, or undersized-budget assertion fails.

- [ ] **Step 3: Generalize streamed split grouping**

Replace the fixed `512 MiB` grouping constant with `staging_bytes / slot_count`. Track the aligned byte sum of unique streamed host weights in each split. Start a new split before the next unique weight exceeds the per-slot budget.

- [ ] **Step 4: Add two streamed copy slots without enabling pipeline parallelism**

Separate streamed-weight slot count from `sched->n_copies`. Allocate two split-local tensor copies for streamed weights when effective async mode is active, while ordinary graph inputs retain existing copy semantics. Select streamed slot by split sequence, not `sched->cur_copy`.

- [ ] **Step 5: Return normal reservation errors**

During split analysis, record the largest aligned streamed tensor. If a nonzero user budget is smaller, log both sizes and return false from reservation instead of asserting or reaching CUDA allocation.

- [ ] **Step 6: Run focused tests**

Run `build/bin/test-opt` and confirm all backend/optimizer cases pass.

### Task 3: Add the upload stream and pinned host arenas

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: two async upload slots, each owning pinned host storage and upload/compute events.
- Consumes: `ggml_backend_dev_init`, `ggml_backend_dev_host_buffer_type`, and existing event APIs.

- [ ] **Step 1: Add an async-versus-sync numerical test**

Run the same small multi-split graph twice in `tests/test-opt.cpp`, once with synchronous streaming and once with async requested. Copy output tensors to host and assert values match within the existing backend tolerance. Repeat the async graph enough times to reuse both slots.

- [ ] **Step 2: Verify the async test fails before runtime support exists**

Expected: effective async mode is false or the slot reuse assertion fails.

- [ ] **Step 3: Initialize async resources lazily during reservation**

For the selected non-CPU backend, check device properties for async execution, host buffers, and events. Create a second backend with `ggml_backend_dev_init(device, nullptr)`. Allocate two pinned buffers from `ggml_backend_dev_host_buffer_type(device)` and create upload-complete and compute-complete events per slot.

- [ ] **Step 4: Implement deterministic cleanup and fallback**

Synchronize compute and upload backends before freeing events, pinned buffers, or the upload backend. In automatic mode, any unsupported capability or two-slot allocation failure logs a reason and restores synchronous single-slot policy. A manual limit below one tensor remains an error rather than fallback.

- [ ] **Step 5: Add pinned packing metadata**

For each prefetched split, assign aligned offsets in one host arena for dense streamed weights. Copy bytes from original host tensors into the pinned arena, then enqueue `ggml_backend_tensor_set_async()` from each packed offset into its destination tensor.

- [ ] **Step 6: Run focused tests**

Run `build/bin/test-opt` on CUDA0, CUDA1, and CPU. CUDA devices with async capabilities must report async effective mode; CPU must report synchronous fallback.

### Task 4: Pipeline next-split upload with current-split compute

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Consumes: two device slots, pinned arenas, upload backend, and events from Task 3.
- Produces: `prepare_split(split_id)` and ordered double-buffered execution inside `ggml_backend_sched_compute_splits()`.

- [ ] **Step 1: Add a forced slot-reuse regression case**

Use at least five streamed splits so slots follow `0,1,0,1,0`. Make every split consume distinct weight values. Assert repeated runs produce the synchronous reference result; an early overwrite changes the result and fails the comparison.

- [ ] **Step 2: Verify failure with upload-only implementation**

Temporarily requesting async without compute-complete waits must fail the repeated numerical comparison or a debug ownership assertion.

- [ ] **Step 3: Extract synchronous split copying**

Move current input-copy logic into a helper that preserves user inputs and the `MUL_MAT_ID` used-expert path. Keep this helper as the fallback and for dynamic MoE splits.

- [ ] **Step 4: Implement async dense split preparation**

Before packing slot S, wait for S's compute-complete event if it has been used. Pack dense host weights into pinned slot S, enqueue H2D writes through the upload backend, and record S's upload-complete event. Mark the split prepared only after successful submission.

- [ ] **Step 5: Implement pipelined execution**

Prepare split 0. For each split N: make the compute backend wait for N's upload event, submit compute N, record its compute-complete event, then prepare the next eligible dense split in the other slot. Execute dynamic MoE or unsupported splits through the synchronous helper after synchronizing required predecessors.

- [ ] **Step 6: Verify numerical equivalence and slot safety**

Run `build/bin/test-opt`. Expected: synchronous and async output comparisons pass across repeated slot reuse.

### Task 5: Add automatic budget selection and qlora CLI

**Files:**
- Modify: `common/common.h`
- Modify: `common/arg.cpp`
- Modify: `src/llama-context.h`
- Modify: `src/llama-context.cpp`
- Modify: `src/llama-ext.h`
- Modify: `examples/qlora_training/finetune_qlora.cpp`

**Interfaces:**
- Produces: `common_params::layer_staging_mib` with default zero.
- Produces: `--layer-staging-mib N` for qlora.
- Replaces: `llama_opt_set_weight_streaming(ctx, bool)` with a configuration-bearing internal call.

- [ ] **Step 1: Add failing CLI and lifecycle checks**

Verify help lacks the option. Extend the scheduler replacement test to suspend and recreate a scheduler with the same requested staging policy.

- [ ] **Step 2: Add CLI parsing and validation**

Accept integer MiB values greater than or equal to zero. Reject the option unless `--grpo-phase-offload` is active. Zero requests automatic sizing; a positive value becomes `staging_bytes` with overflow-checked MiB conversion.

- [ ] **Step 3: Preserve configuration across optimizer suspend/resume**

Store the complete streaming parameters in `llama_context`. Apply them both in initial optimizer scheduler construction and `opt_create_backend_sched()`.

- [ ] **Step 4: Implement automatic sizing**

During the first measurement reservation, query `ggml_backend_dev_memory`. Compute fixed graph bytes by subtracting the measured largest live synchronous staging allocation. Use a default total cap of 1024 MiB and safety margin `max(512 MiB, 10% total VRAM)`. Repeat reservation with the chosen slot count and verify final planned bytes plus the margin fit current free memory.

- [ ] **Step 5: Enable async prefetch in phase-offload training**

The phase path requests weight streaming with async prefetch and passes the CLI staging value. Legacy GRPO and ordinary QLoRA paths remain disabled.

- [ ] **Step 6: Build and verify help**

Run:

```bash
cmake --build build --target llama-finetune-qlora test-opt -j18
build/bin/llama-finetune-qlora --help | rg -- '--layer-staging-mib'
```

Expected: both commands succeed and help documents zero as automatic.

### Task 6: Add transfer metrics and full verification

**Files:**
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `src/llama-context.cpp`

**Interfaces:**
- Produces: per-step streamed bytes, packing time, upload wait time, compute time, split counts, and overlap estimate.

- [ ] **Step 1: Add scheduler metric accessors**

Return a value structure after synchronization. Reset counters at graph compute start and accumulate monotonic host timings around packing, submission, waits, and compute synchronization boundaries.

- [ ] **Step 2: Log effective configuration and metrics**

Log once after first reservation and once after each optimizer epoch. Include effective mode, total and per-slot MiB, largest tensor MiB, prefetched split count, synchronous split count, streamed MiB, and overlap estimate.

- [ ] **Step 3: Run static and focused regression checks**

Run:

```bash
git diff --check
cmake --build build --target llama-finetune-qlora test-opt test-qat -j18
build/bin/test-opt
build/bin/test-qat
```

Expected: all commands exit zero.

- [ ] **Step 4: Run deterministic sync and async smoke tests**

Use rank 4, context 1024, microbatch 512, fixed seed, one GRPO step, and direct IPC rewards. Save separate adapters. Compare tensor metadata and values within the existing QLoRA numerical tolerance.

- [ ] **Step 5: Run two async phase-offload steps**

Expected: optimizer iteration increases across both suspend/resume cycles and the final adapter saves normally.

- [ ] **Step 6: Probe the target configuration**

Use context 16384, microbatch 512, and rank 64. Record selected staging budget, planned CUDA buffer, peak VRAM, streamed bytes, packing time, upload wait time, compute time, and step wall time. Stop after one complete step unless the user requests a longer run.

- [ ] **Step 7: Compare against synchronous streaming**

Run the same target probe with async disabled. Report speedup, overlap ratio, and peak VRAM. Async mode must not exceed its selected staging budget or change optimizer results outside tolerance.
