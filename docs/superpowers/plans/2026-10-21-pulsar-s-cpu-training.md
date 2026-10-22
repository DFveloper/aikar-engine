# Pulsar S CPU optimization implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution after review. Keep changes uncommitted.

**Goal:** Improve CPU fine-tuning first and validate inference alongside it.

**Architecture:** Profile existing CPU QLoRA, then optimize one existing operation without changing training semantics. Keep source-matched baseline and candidate builds and independent outputs.

**Tech Stack:** C++17, ggml CPU, CMake, existing GGUF Python tools.

**Spec:** `docs/superpowers/specs/2026-10-21-pulsar-s-cpu-training-design.md`

## Global Constraints

- Preserve model precision, training objective, trainable parameters, effective batch, and checkpoint compatibility.
- Preserve originals, existing services, user changes, and generic CPU fallbacks.
- Do not commit, push, create a PR, or launch an unattended full curriculum.

## Review Focus

- Quantized backward: noncontiguous gradients must retain their byte strides.
- MoE: repeated expert IDs must retain separate route outputs.
- Accumulation: preexisting destination values must be preserved.
- SIMD: tails and unaligned input must remain supported.
- Fusion: Q-only adapters must not become joint QKV adapters.

### Task 1: Establish current-source baseline

**Files:** Separate ignored build directories and `/tmp/pulsar-cpu-training-20261021` results.

**Interfaces:** Produces baseline binaries, fixed input subset, commands, timings, profiles, and recorded supported options for Task 2.

- [x] Configure Release CPU-only current-source build with native ISA and existing tests.
- [x] Run fixed real-data QLoRA diagnostic with attention adapters and actual optimizer updates; record effective settings and errors.
- [x] Profile backward and optimizer operations; record inference prefill/decode independently.
- [x] Select a small bottleneck improvement from measured evidence. If none is justified, report the evidence without speculative kernel changes.

### Task 2: Optimize the selected existing operation

**Files:** Existing CPU operation implementation and its existing tests, selected by Task 1 evidence.

**Interfaces:** Consumes Task 1 workload and profile; produces a focused patch and candidate binary with the same settings.

- [x] Add a regression or numerical comparison in the existing relevant operator test before implementation; exercise strides, repeated routes, accumulation, and tails where relevant.
- [x] Run the test on baseline and verify its numerical behavior. Use the measured performance baseline as the failing performance criterion; do not add flaky timing assertions to CI.
- [x] Implement one optimization using existing helpers. Preserve operation order where practical and retain fallback for unsupported shapes/types.
- [x] Run affected backend tests plus optimizer/QAT coverage. Remove the experiment if correctness fails or the representative workload does not improve.

### Task 3: Compare and report

**Files:** `docs/pulsar-s-cpu-training-20261021.md` and raw result directory.

**Interfaces:** Consumes baseline/candidate binaries and fixed inputs; produces reproducible results and limitations.

- [x] Run matched training cases at least three times sequentially after warmup and report median/range and RSS.
- [x] Compare saved adapter tensors and losses from identical initialization; run resume validation if affected.
- [x] Compare inference prefill/decode and numerical outputs using the retained change.
- [x] Review the full diff against the repository CPU/backend, security, testing, and general checklists; record failures and unverified cases explicitly.

## Execution notes

The user approved the specification and asked to proceed and continue. Execution is inline to keep the profiling and code decisions together. Baseline investigation can proceed while this written plan is reviewed; product implementation waits for plan review under the brainstorming workflow. Exact kernel changes require measured evidence and are not predetermined by a preference for QKV fusion.

## Follow-up: actual stage1 window

The user authorized fixing the actual-window failure, testing again, and adjusting CPU microbatch/recomputation when memory permits.

- [x] Trace the SET_ROWS assertion to inactive mixed-input branches activated by recomputation graph reordering.
- [x] Preserve inactive-node COMPUTE flags and add a failing-then-passing selected-input regression, including two dynamic evaluations and optimizer-update equivalence.
- [x] Pass the focused five-test CTest selection and review the fix.
- [x] Add a separate CPU stage1 launcher with configurable microbatch/recomputation and output/checkpoint guards.
- [x] Complete one actual stage1 window with microbatch512, recomputation off, and 18 threads; stop after its checkpoint.
- [x] Inspect the saved checkpoint and report measured time, memory, and stage1 ETA limitations.

Ruling: use microbatch512/recompute-off as the CPU launcher's default because this is the configuration selected for complete-window validation. The microbatch256 probe looks promising but is not a complete-window comparison. Original GPU scripts, model and dataset remain unchanged.
