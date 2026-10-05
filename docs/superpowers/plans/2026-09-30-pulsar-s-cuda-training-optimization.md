# Pulsar S CUDA Training Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve the CUDA QLoRA training graph's throughput and temporary memory use on V100.

**Architecture:** Preserve the existing graph and add narrowly guarded CUDA fusion for the dominant LoRA gradient accumulation path. Keep fallback behavior unchanged and use existing backend/QAT tests plus a short real-training benchmark for validation.

**Tech Stack:** C++, CUDA, ggml backend tests, QLoRA training binary.

**Spec:** `docs/superpowers/specs/2026-09-30-pulsar-s-cuda-training-optimization-design.md`

## Global Constraints

- Preserve numerical behavior within existing CUDA tolerance.
- Do not change dataset, optimizer, LoRA, or checkpoint semantics.
- Do not add a new test file under `tests/*`; reuse existing test targets.
- Do not commit or push changes.

## Review Focus

- Alias and in-place accumulation: fusion must not overwrite an input before it is read.
- Non-contiguous or non-F32 accumulation: use the existing fallback.
- V100 shared-memory and launch limits: compile and run on compute capability 7.0.
- Sparse/routed gradients: no change to route selection or duplicate-route handling.
- Numerical drift: compare loss and adapter tensors against the unfused path.

### Task 1: Add a failing fusion eligibility regression

**Files:**
- Modify: `tests/test-backend-ops.cpp` or the existing CUDA optimizer test target selected by the repository build.

- [ ] Add a test that constructs an eligible `OUT_PROD` followed by in-place F32 accumulation and asserts the CUDA graph uses the fused path's result.
- [ ] Run the focused test and confirm it fails before the implementation.

### Task 2: Implement guarded CUDA fusion

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modify: the smallest existing CUDA helper/header needed for the fused launch.

- [ ] Add the minimal fused launch path for eligible `OUT_PROD -> ADD` graphs.
- [ ] Keep fallback dispatch for all other types, shapes, aliases, and layouts.
- [ ] Run the focused test and confirm it passes.

### Task 3: Validate optimizer and training behavior

**Files:**
- Modify only if required by the focused regression.

- [ ] Run `test-qat` on CPU and CUDA0.
- [ ] Run the relevant backend optimizer tests.
- [ ] Run a short real-data CUDA comparison with fusion enabled and disabled.
- [ ] Measure tokens/s and peak VRAM on V100; retain the change only if it improves one without violating numerical checks.

### Task 4: Document measured result

**Files:**
- Modify: `examples/qlora_training/pulsar-s-v100.md`

- [ ] Record the benchmark command, baseline, optimized throughput, peak VRAM, and validation result.
