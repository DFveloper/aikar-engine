# Q4_0 Training Memory Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate and validate the Q4_0 forward, backward, output-compaction, and optional recomputation changes on the supplied V100 model.

**Architecture:** Execute the four component plans in dependency order, then run a controlled configuration matrix. Use allocator backtraces and graph tensor shapes to attribute every large allocation rather than relying only on `nvidia-smi` peak values.

**Tech Stack:** CMake, CUDA V100, llama QLoRA/QLion trainers, gdb or cuda-gdb allocation traces, existing test binaries

**Spec:** `docs/superpowers/specs/2026-09-23-q4-fused-training-memory-design.md`

## Global Constraints

- Model: `/mnt/openwebui/AIKAR/Lumen-3.2-MoXXf-2B/Lumen-3.2-Flare-base-Q4_0.gguf`.
- Primary run: rank 4, context/batch 4096, microbatch 512, flash attention on.
- Fixed seed and identical sample for every numerical comparison.
- Recomputation is off by default and tested both off and on.
- Report speed or numerical tradeoffs separately.
- Do not commit, push, or create a PR without explicit user approval.

## Review Focus

- CUDA pool retention can hide which optimization removed an allocation; Task 2 uses fresh processes per run.
- First-run CUDA module and graph allocations can distort peaks; Task 1 defines warmup and measurement boundaries.
- QLoRA and QLion have different optimizer-state ownership; Task 3 records buffer categories separately.
- A lower `nvidia-smi` peak with missing updates is invalid; Task 2 requires update-norm checks.
- Recompute speed loss can make an otherwise correct default undesirable; Task 4 keeps it opt-in and reports throughput.

---

### Task 1: Establish reproducible commands and telemetry

**Files:**
- Modify: `examples/qlora_training/README.md`
- Create outside `tests/`: `scripts/bench-q4-training-memory.sh`

**Interfaces:**
- Produces: one command wrapper that launches each configuration in a fresh process and writes machine-readable peak, scheduler, loss, update-norm, and time records.

- [ ] **Step 1: Capture the exact current training command**

Copy the already validated dataset, seed, optimizer, rank, context, batch, microbatch, and one-step arguments into shell arrays. Do not embed credentials or external paths other than the supplied model and local test dataset.

- [ ] **Step 2: Add fresh-process telemetry**

Record pre-load, post-model, post-optimizer, and peak CUDA memory plus scheduler buffer size. Write one TSV row per run with configuration flags and exit status.

- [ ] **Step 3: Run the unchanged baseline twice**

Expected reference: approximately 9685 MiB peak and 33.53 seconds with FA on. Investigate variance above 128 MiB or 10 percent before comparing optimized runs.

### Task 2: Run the QLoRA feature matrix

**Files:**
- Modify only when a component verification fails.

**Interfaces:**
- Consumes: all four component implementations.

- [ ] **Step 1: Run Q4 fused kernels with compaction and recomputation off**

Record peak, scheduler allocation, loss, update norm, and time. Target peak is at most 7.5 GiB.

- [ ] **Step 2: Run the same configuration with recomputation on**

Target peak is about 6 GiB or less. Loss and update norm must remain within the established backend tolerance.

- [ ] **Step 3: Run microbatch 128 controls**

Repeat off/on at the same logical context and batch. Confirm scheduler scaling remains near-linear and no removed full-weight temporary reappears.

- [ ] **Step 4: Trace all allocations above 256 MiB**

Expected: persistent model allocation is present; no 768 MiB full F16 tied-weight conversion; no 1.50 GiB full F32 Q4 `OUT_PROD` conversion; logits allocations match active output rows.

### Task 3: Run QLion correctness and memory checks

**Files:**
- Modify only when a component verification fails.

**Interfaces:**
- Verifies: QLion dependency callback and QAT state remain correct with fused kernels and compaction.

- [ ] **Step 1: Run a small deterministic QLion off/on comparison**

Use a context that fits the current baseline, fixed seed, and one update. Compare QAT state count, dependency count, loss, gradient norm, and update norm.

- [ ] **Step 2: Run the largest fitting QLion configuration**

Record persistent QAT parameters, momentum, residual, gradient accumulator, scheduler, KV, and CUDA pool categories separately.

- [ ] **Step 3: Verify the prior gradient bug scenarios**

Run tied tensor, routed expert, row update, zero-gradient, and accumulation cases from `test-qat`. Confirm the fused frozen-weight path changes only activation gradients and not QLion state-gradient ownership.

### Task 4: Run the complete regression suite and report

**Files:**
- Modify: `examples/qlora_training/README.md`
- Modify: `docs/superpowers/specs/2026-09-23-q4-fused-training-memory-design.md` only to record measured deviations from targets

**Interfaces:**
- Produces: final table of baseline versus each optimization and off/on recomputation.

- [ ] **Step 1: Run static and build checks**

Run:

```bash
git diff --check
cmake --build build --target llama-finetune-qlora llama-finetune-qlion test-backend-ops test-opt test-qat -j8
```

- [ ] **Step 2: Run operation and optimizer tests**

Run CUDA `MUL_MAT`, `OUT_PROD`, and flash-attention backward tests, followed by `test-opt` and `test-qat`.

- [ ] **Step 3: Assemble the memory table**

Include baseline, fused only, fused plus compaction, and fused plus compaction plus recomputation. Show peak MiB, scheduler MiB, largest temporary, active rows, seconds per step, loss delta, and update-norm delta.

- [ ] **Step 4: Update user documentation**

Document automatic Q4_0 fused behavior, bounded fallback, output compaction, `--activation-recompute on|off`, supported first-version scope, measured speed cost, and unsupported-mode errors.

- [ ] **Step 5: Perform final diff review**

Inspect all changed files for accidental inference behavior changes, Unicode in new source/comments, unbounded allocations, stale checkpoint claims, and unrelated edits. Do not commit.
