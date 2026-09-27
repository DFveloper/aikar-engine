# Pulsar S F16 and Q8_KV Fusions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Pulsar S CUDA decode launches with three contained fusions while preserving F16 and Q8_KV behavior.

**Architecture:** Extend existing CUDA top-k, RMSNorm, graph-matching, and SET_ROWS code paths. Add no public operator or format; all changes are internal CUDA specializations selected by strict graph and shape checks.

**Tech Stack:** C++, CUDA, ggml graph fusion, `test-backend-ops`.

**Spec:** `docs/superpowers/specs/2026-09-26-pulsar-s-f16-q8kv-fusions-design.md`

## Global Constraints

- Preserve routing, sampling, KV format, and model graph semantics.
- Support F16 and Q8_KV cache destinations only.
- Do not add Q4_KV or multi-token expert fusion.
- Reuse existing source and test files.
- Do not commit, push, or create a PR.

## Review Focus

- 116 experts requires ceiling register-slot allocation and must never select padded lanes.
- Router scalar multiplication must preserve zero bias and operand order.
- F16 cache rows flatten heads, while Q8_KV cache rows address each head separately.
- D=512 Q8_KV must write all eight 64-value blocks without crossing head boundaries.
- Fusion matchers must reject incompatible views, indices, types, and aliases.

---

### Task 1: Add 116-Expert Fused Top-K

**Files:**
- Modify: `tests/test-backend-ops.cpp`
- Modify: `ggml/src/ggml-cuda/topk-moe.cu`

**Interfaces:**
- Consumes: Existing `topk_moe_cuda<n_experts, has_bias>` and fusion matcher.
- Produces: Correct fused routing for 116 experts and top-8.

- [ ] Add 116-expert backend test coverage with one decode row.
- [ ] Run the focused test before implementation and record the unfused baseline.
- [ ] Change expert slot allocation to ceiling division, instantiate 116, and admit only 116 in the special-count check.
- [ ] Rebuild and run focused top-k tests.

### Task 2: Fuse Router RMSNorm, Scale, and Multiply

**Files:**
- Modify: `tests/test-backend-ops.cpp`
- Modify: `ggml/src/ggml-cuda/norm.cu`
- Modify: `ggml/src/ggml-cuda/norm.cuh`
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`

**Interfaces:**
- Produces: Internal `ggml_cuda_op_rms_norm_scale_mul_fused` entry point and strict graph match.

- [ ] Add whole-graph tests for 2816 columns and one/four rows.
- [ ] Run focused tests before implementation and record the three-op baseline.
- [ ] Extend RMSNorm output scaling and add the fused CUDA entry point.
- [ ] Add strict graph, type, shape, edge, bias, contiguity, and alias checks.
- [ ] Rebuild and run focused RMSNorm tests.

### Task 3: Fuse V RMSNorm and Cache Write

**Files:**
- Modify: `tests/test-backend-ops.cpp`
- Modify: `ggml/src/ggml-cuda/set-rows.cu`
- Modify: `ggml/src/ggml-cuda/set-rows.cuh`
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`

**Interfaces:**
- Produces: Internal `ggml_cuda_op_rms_norm_set_rows` entry point for Pulsar F16 and Q8_KV layouts.

- [ ] Add F16 and Q8_KV whole-graph tests for D256/H8 and D512/H2.
- [ ] Run focused tests before implementation and record the unfused baseline.
- [ ] Implement per-head RMSNorm with direct F16 or Q8_KV destination writes.
- [ ] Add strict graph, view-layout, index, type, dimension, and alias checks.
- [ ] Rebuild and run focused cache-write tests.

### Task 4: Integrated Verification

**Files:**
- Verify all modified files.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: Build, backend-test, and model-level evidence.

- [ ] Run the relevant CUDA backend operation suite.
- [ ] Run F16 and Q8_KV Pulsar S decode smoke tests.
- [ ] Inspect the final diff for scope and matcher safety.
- [ ] Record any unavailable profiling or long-context checks explicitly.

