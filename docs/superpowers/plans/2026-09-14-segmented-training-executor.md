# Segmented Training Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute an existing ggml optimizer graph in bounded forward and reverse segments so rank-64 QLoRA at context 16384 fits a 16 GiB V100.

**Architecture:** Keep the normal forward and autograd graph metadata, but plan dependency-safe contiguous segments with measured device budgets. Execute forward segments while checkpointing live boundaries on the host, then walk reverse segments while rematerializing forward values and accumulating the existing parameter gradients. The current trainer, loss, optimizer, QAT, checkpoint, and GRPO coordinator remain authoritative.

**Tech Stack:** C++17, ggml graph/autograd APIs, ggml backend scheduler, CUDA backend abstraction, existing QLoRA trainer and Python IPC wrapper

**Spec:** `docs/superpowers/specs/2026-09-14-segmented-training-executor-design.md`

## Global Constraints

- Keep new source and comments ASCII-only.
- Do not add a model-specific trainer.
- Do not add a new file under `tests/`.
- Preserve the existing full-graph optimizer path.
- Do not update parameters until every backward segment succeeds.
- Implement synchronous execution before asynchronous prefetch.
- Use dependency analysis, not tensor names, for correctness.
- Plan against the complete 16384-token KV prefix.
- Do not commit without explicit user approval for that commit.

---

### Task 1: Make optimizer allocation failure recoverable

**Files:**
- Modify: `ggml/include/ggml-opt.h`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `src/llama-context.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: `enum ggml_status ggml_opt_alloc(...)`.
- Preserves: current successful full-graph behavior.

- [ ] **Step 1: Add an allocation-failure case to `tests/test-opt.cpp`**

Use a test buffer type with a deliberately insufficient allocation limit. Assert that optimizer allocation returns `GGML_STATUS_ALLOC_FAILED`, leaves input tensor buffers unset, and does not execute an optimizer update.

- [ ] **Step 2: Build and run the focused test to verify RED**

Run `cmake --build build --target test-opt -j8 && build/bin/test-opt`.

Expected: the new test aborts or cannot compile because the optimizer allocation APIs do not return status.

- [ ] **Step 3: Propagate scheduler reservation failures**

Change optimizer allocation to return `GGML_STATUS_ALLOC_FAILED` when `ggml_backend_sched_alloc_graph()` fails. In `llama_context::opt_epoch_iter()`, return before `res->set_inputs()` and before `ggml_opt_eval()` when allocation failed.

- [ ] **Step 4: Run the focused test to verify GREEN**

Run `cmake --build build --target test-opt -j8 && build/bin/test-opt`.

Expected: allocation failure returns normally and all existing optimizer tests pass.

### Task 2: Add dependency and liveness planning

**Files:**
- Modify: `ggml/include/ggml-opt.h`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: `ggml_opt_segmented_params`.
- Produces: internal `ggml_opt_segment_plan` containing forward node ranges, live-ins, live-outs, backward node ranges, incoming gradients, and outgoing gradients.
- Produces: `ggml_opt_plan_segments(ggml_opt_context_t, ggml_cgraph *, size_t)`.

- [ ] **Step 1: Add a liveness test with skip connections**

Build a four-stage graph in `tests/test-opt.cpp` where stage four consumes values from stages one and three. Request at least three segments. Assert that the stage-one tensor remains a live boundary until stage four and that parameters are classified as external state rather than activation checkpoints.

- [ ] **Step 2: Run the test to verify RED**

Expected: compile failure because segmented planner types and APIs are absent.

- [ ] **Step 3: Implement producer, consumer, and boundary analysis**

Map every graph tensor to its producing node and last consuming node. Build contiguous forward ranges and calculate each range's activation live-ins and live-outs. Exclude parameters, immutable leaf weights, optimizer state, and persistent backend tensors from checkpoint ownership.

- [ ] **Step 4: Derive reverse dependencies from the autograd graph**

For each forward range, collect backward nodes whose dependency closure reaches tensors produced in that range. Record boundary gradient tensors separately from parameter gradient tensors. Reject overlapping ownership with an actionable planner error.

- [ ] **Step 5: Run planner tests to verify GREEN**

Run `cmake --build build --target test-opt -j8 && build/bin/test-opt`.

Expected: skip-connection boundaries and reverse ownership assertions pass.

### Task 3: Add measured automatic segment sizing

**Files:**
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Consumes: dependency ranges from Task 2.
- Produces: `ggml_backend_sched_measure_graph()` returning required bytes per backend without retaining allocations.
- Produces: a segment plan whose largest measured device allocation is at most `device_budget`.

- [ ] **Step 1: Add a budget-driven split test**

Construct a graph whose complete allocation exceeds a synthetic budget while each half fits. Assert that automatic planning produces two or more segments and reports the measured maximum.

- [ ] **Step 2: Run the test to verify RED**

Expected: planner returns one segment or measurement API is missing.

- [ ] **Step 3: Add non-retaining scheduler measurement**

Measure a graph view with the real backend allocator, record per-backend required bytes, synchronize, then reset without retaining its compute buffers. Return failure instead of asserting when one node cannot fit.

- [ ] **Step 4: Implement cut search**

Start from the largest remaining range. Measure it; if it exceeds budget, move the end to an earlier low-live-boundary cut and measure again. Stop with the exact operation and byte requirement if a single-node segment exceeds budget.

- [ ] **Step 5: Account for worst-prefix attention shapes**

Build the measurement graph with positions and KV visibility corresponding to the end of `n_ctx_train`. Store the measured prefix in the plan signature so a shorter-prefix plan cannot be reused for a longer window.

- [ ] **Step 6: Run the focused test to verify GREEN**

Run `cmake --build build --target test-opt -j8 && build/bin/test-opt`.

Expected: every planned segment is within budget and single-node overflow returns normally.

### Task 4: Add host checkpoint storage

**Files:**
- Modify: `ggml/include/ggml-opt.h`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: internal `ggml_opt_checkpoint_store` keyed by original tensor pointer.
- Produces: synchronous `save_tensor()` and `restore_tensor()` operations.
- Consumes: live boundary sets from Task 2.

- [ ] **Step 1: Add a boundary round-trip test**

Execute a producer segment, save two differently shaped live-outs, release its scheduler allocation, restore them for a consumer segment, and compare the final output to a full-graph reference.

- [ ] **Step 2: Run the test to verify RED**

Expected: consumer tensors have no valid data after the producer allocation is reset.

- [ ] **Step 3: Allocate bounded host checkpoint slots**

Allocate exact tensor byte ranges with backend alignment. Prefer `ggml_backend_dev_host_buffer_type()` and fall back to aligned CPU backend buffers. Track layout, type, shape, and generation for every saved tensor.

- [ ] **Step 4: Implement synchronized save and restore**

Copy device live-outs to the checkpoint store before scheduler reset. Restore live-ins only after their destination tensors are allocated. Reject stale generations and shape mismatches with `GGML_STATUS_FAILED`.

- [ ] **Step 5: Run the test to verify GREEN**

Expected: segmented forward output exactly matches the reference within backend tolerance.

### Task 5: Execute segmented forward graphs

**Files:**
- Modify: `ggml/include/ggml-opt.h`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `src/llama-context.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: `ggml_opt_eval_segmented_forward()`.
- Consumes: segment plan, scheduler measurement, and checkpoint store.
- Produces: the same loss tensor values as normal forward evaluation.

- [ ] **Step 1: Add a three-segment forward equivalence test**

Use a graph with normalization, two matrix multiplications, a residual edge, and cross-entropy loss. Compare logits and loss between full and segmented execution.

- [ ] **Step 2: Run the test to verify RED**

Expected: segmented evaluation entry point is absent.

- [ ] **Step 3: Execute each graph view with a fresh scheduler allocation**

For every segment, reset the scheduler, allocate the segment graph, restore live-ins, set external inputs, compute synchronously, save live-outs, and reset the allocation. Return immediately on any failed phase.

- [ ] **Step 4: Integrate forward selection in `llama_context::opt_epoch_iter()`**

Use segmented forward only when optimizer segmented params are enabled. Leave normal inference, ordinary training, and the existing full-graph call unchanged.

- [ ] **Step 5: Run the test to verify GREEN**

Run `cmake --build build --target test-opt -j8 && build/bin/test-opt`.

Expected: logits and loss match and at least three scheduler resets occurred.

### Task 6: Execute reverse segments with rematerialization

**Files:**
- Modify: `ggml/include/ggml-opt.h`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: `ggml_opt_eval_segmented_backward()`.
- Consumes: forward checkpoints, reverse plan, incoming boundary gradients, and existing parameter gradient accumulators.
- Produces: outgoing boundary gradients and accumulated parameter gradients.

- [ ] **Step 1: Add gradient equivalence tests**

Force three segments in the Task 5 graph. Compare every trainable parameter gradient and the first segment input gradient against the full backward graph.

- [ ] **Step 2: Run the test to verify RED**

Expected: segmented forward produces values but parameter gradients remain zero or missing.

- [ ] **Step 3: Rematerialize one reverse segment**

Restore its forward live-ins, run the forward nodes required by the reverse dependency closure, restore incoming boundary gradients, then execute only that segment's backward nodes.

- [ ] **Step 4: Save outgoing gradients and accumulate parameter gradients**

Copy boundary gradients to host before reset. Add parameter gradient contributions to the existing authoritative accumulators exactly once per segment. Track completion so an error cannot trigger a partial optimizer step.

- [ ] **Step 5: Handle persistent KV writes**

Synchronize before reset and allow only idempotent writes to the same layer and position during rematerialization. Add a KV-like state test that verifies a forward plus reverse replay produces the same state and gradients as full execution.

- [ ] **Step 6: Run the test to verify GREEN**

Expected: all input and parameter gradients match the full graph within existing backend tolerance.

### Task 7: Preserve optimizer, QAT, and failure atomicity

**Files:**
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `tests/test-opt.cpp`
- Modify: `examples/qlora_training/finetune_qlora.cpp`

**Interfaces:**
- Consumes: completed segmented backward result from Task 6.
- Preserves: existing AdamW, gradient accumulation period, QAT residual state, and checkpoint serialization.

- [ ] **Step 1: Add update-equivalence and failure-atomicity tests**

Compare one full-graph and segmented AdamW update. Inject failure into the middle reverse segment and assert parameters, optimizer step, and QAT state remain unchanged.

- [ ] **Step 2: Run the test to verify RED**

Expected: segmented execution either does not update or permits a partial update.

- [ ] **Step 3: Gate optimizer execution on complete backward traversal**

Keep optimizer nodes outside segment execution. Run them only after all parameter gradient accumulators are complete. On failure, discard the window's boundary checkpoints and clear uncommitted accumulators.

- [ ] **Step 4: Preserve QAT ordering**

Accumulate QAT gradient dependencies across reverse segments, then execute quantized optimizer and residual-feedback nodes in their existing order after traversal completion.

- [ ] **Step 5: Run optimizer and QAT tests to verify GREEN**

Run `cmake --build build --target test-opt test-qat -j8 && build/bin/test-opt && build/bin/test-qat`.

Expected: update equivalence passes and injected failure leaves state unchanged.

### Task 8: Enable segmented phase offload and progress reporting

**Files:**
- Modify: `common/common.h`
- Modify: `common/arg.cpp`
- Modify: `src/llama-context.h`
- Modify: `src/llama-context.cpp`
- Modify: `src/llama-ext.h`
- Modify: `examples/qlora_training/finetune_qlora.cpp`
- Modify: `/mnt/openwebui/AIKAR/Merge/RL.py`

**Interfaces:**
- Produces: `--train-device-budget-mib N`, where zero selects the automatic budget.
- Enables: segmented execution by default with `--grpo-phase-offload`.
- Produces: forward and backward segment IPC progress messages.

- [ ] **Step 1: Add CLI and lifecycle tests**

Verify help output, zero-budget parsing, explicit budget parsing, and optimizer scheduler suspend/resume preserving segmented parameters.

- [ ] **Step 2: Run tests to verify RED**

Expected: new CLI option is absent and segmented params are lost during scheduler recreation.

- [ ] **Step 3: Add configuration and automatic budget selection**

For zero, query the selected CUDA device and use free VRAM minus `max(1536 MiB, 15% total VRAM)`. Reject values that leave no executable segment. Enable this mode only for phase-offload unless explicitly requested by a trainer option.

- [ ] **Step 4: Add structured progress messages**

Emit plan, forward segment, backward segment, recompute, transfer, optimizer, and peak-memory fields. Update `RL.py` to show the active phase and ETA without changing reward IPC behavior.

- [ ] **Step 5: Run tests to verify GREEN**

Build qlora and verify CLI, suspend/resume, and progress parsing tests.

### Task 9: Validate numerical behavior and the target V100 run

**Files:**
- Modify only if a verification failure identifies a defect in an earlier task.

**Interfaces:**
- Verifies: all spec success criteria.

- [ ] **Step 1: Run static and unit verification**

Run `git diff --check`, `cmake --build build --target llama-finetune-qlora test-opt test-qat -j8`, `build/bin/test-opt`, and `build/bin/test-qat`.

- [ ] **Step 2: Run deterministic small-model equivalence**

With rank 4, context 1024, microbatch 256, fixed seed, and identical direct IPC rewards, compare full and segmented LoRA updates within the existing optimizer tolerance.

- [ ] **Step 3: Run two phase transitions**

Complete two rollout/training cycles and verify LoRA state, optimizer step, checkpoint save, and scheduler recreation continue correctly.

- [ ] **Step 4: Run the target memory test**

Use the user's rank-64, context-16384, batch-512, microbatch-512 V100 configuration. Complete all 128 training microbatches and one optimizer update. Record forward and backward segment counts, host checkpoint bytes, peak device bytes, and wall time.

- [ ] **Step 5: Verify the target acceptance criteria**

Confirm no CUDA OOM, no allocator assert, no partial update, and peak device allocation below the selected budget with its reserve margin intact.

### Task 10: Prepare asynchronous prefetch extension points

**Files:**
- Modify: `ggml/include/ggml-opt.h`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `docs/superpowers/specs/2026-09-14-segmented-training-executor-design.md`

**Interfaces:**
- Produces: stable segment lifecycle hooks `prepare`, `execute`, `retire`.
- Does not enable: asynchronous transfers in this implementation.

- [ ] **Step 1: Add lifecycle ordering assertions to existing segmented tests**

Record the synchronous order and assert every segment follows `prepare -> execute -> retire`, with no slot reuse before retirement.

- [ ] **Step 2: Run the test to verify RED**

Expected: execution works but has no explicit lifecycle contract.

- [ ] **Step 3: Refactor synchronous execution behind lifecycle hooks**

Move existing save, restore, allocation, compute, synchronization, and reset operations behind the three hooks without changing behavior. Keep slot count fixed at one.

- [ ] **Step 4: Run final verification**

Run the complete Task 9 static, unit, numerical, and target checks again. Expected: identical synchronous results and explicit hooks ready for a later two-slot implementation.
