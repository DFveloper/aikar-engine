# Pulsar CPU LES and FA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement CPU LES and cache-aware bounded attention, then serve the IQ3 Quality model without SWA-full using measured settings.
**Architecture:** Reuse PAGED_ATTN and MAP_CUSTOM3; add optional expert-priority metadata to MUL_MAT_ID. Own asynchronous prediction/prefetch state per llama context. Keep exact routing and original expert output slots.
**Tech Stack:** C++17, ggml CPU, CMake, existing backend tests, native AVX2 host.
**Spec:** docs/superpowers/specs/2026-10-24-pulsar-cpu-les-fa-design.md

## Global Constraints

- No SWA-full in any new serving configuration.
- No commits, pushes, PRs, new quantization types, or new attention operators.
- Preserve the real expert selection, expert scales, and final summation order.
- Scalar attention reference remains reachable with use_ref; unconfigured LES leaves default behavior unchanged.
- Worker generations, bounded byte preparation, cancellation, and teardown must prevent stale publication and use-after-free.
- ASCII code/comments; reuse existing tests and helpers; no timing thresholds in correctness tests.

## Review Focus

- Partially populated pages and chunks with no visible keys must not produce NaNs or cross-sequence reads.
- Quantized cache rows and noncausal visibility must agree with the reference.
- MMID priority cannot omit duplicate or unprepared real selections or change original output slots.
- Stale asynchronous jobs cannot publish current-generation readiness or touch freed model buffers.
- Prefix reuse beyond the sliding window must reset or restore safely without full SWA allocation.

### Task 1: Establish isolated baseline and tests

Files: existing tests/test-backend-ops.cpp, tests/CMakeLists.txt; new tests/test-cpu-les.cpp for the new scheduler component.
- [x] Build native CPU server, bench, and backend tests in the managed worktree; run a focused paged reference baseline.
- [x] Write small literal fixtures for CPU LES prediction and priority ordering; demonstrate the missing interface before implementation.
- [x] Record test and benchmark commands in .superpowers/pulsar-cpu-les/progress.md.

### Task 2: CPU LES and original-slot scheduling

Files: new src/llama-cpu-les.h/.cpp; src/CMakeLists.txt; context, cparams, graph, common params/args and include/llama.h; ggml/include/ggml.h; ggml/src/ggml.c; CPU repack.cpp and ggml-cpu.c.
Interfaces: llama_cpu_les owns layer descriptors, prepare(), synchronize(), stats, and a MAP_CUSTOM3 callback. ggml_mul_mat_id_set_expert_priority(result, priority) attaches an optional contiguous F32 per-expert priority snapshot as src[3].
- [x] Implement deterministic prediction and generation-safe bounded asynchronous cache preparation; tests cover hit/miss, budget rejection, drain and destruction.
- [x] Add optional priority-aware execution to generic and repacked MMID. Compute prepared groups before others while preserving output indices; verify numerical equivalence with irregular selections.
- [x] Integrate the callback into Gemma4 MoE graph construction and context lifetime; expose --cpu-les and --cpu-les-prefetch-mib (default 16 MiB, positive when requested).
- [x] Verify prefill/default fallback, incompatible configurations, and LES off/on output/logit agreement.

### Task 3: Cache-aware paged CPU attention

Files: ggml/src/ggml-cpu/ops.cpp; CPU work sizing only if required; tests/test-backend-ops.cpp.
- [x] Add focused boundary/noncausal/GQA/reference tests before changing the kernel; record the existing behavior.
- [x] Preserve the scalar implementation for use_ref. Add per-thread blocked attention with bounded scratch, SIMD helpers, and K/V reuse; keep every validity and visibility rule.
- [x] Run F16/Q8_KV plus other existing supported cache cases, page boundaries, multiple sequences, and empty visible partitions.
- [x] Measure short/long decode and prefill; avoid enabling slower tile settings.

### Task 4: Integrate, review, measure, and serve

Files: source changes copied into the user's checkout only after review; scripts/serve-pulsar-iq3-temporary.sh; a dated result report.
- [x] Build all relevant test targets, run the appropriate CPU suite, and report any preexisting/model-dependent failures.
- [x] Run matched sequential actual-model experiments with LES off/on and scalar/optimized attention. Record distributions and actual outputs, not projected speedups.
- [x] Have a fresh reviewer inspect the whole diff for correctness, lifetime, bounded sizes and spec compliance; resolve important findings and reverify.
- [x] Adopt only measured winning defaults; keep correct slower experimental modes opt-in. Restart the requested temporary service without SWA-full and verify health, alias, Korean/arithmetic, streaming and safe prefix reuse.
