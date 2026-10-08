# Turbo KV Quality and Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reproduce the historical Pulsar S local-F16/global-Turbo3 quality result and improve ordinary CUDA Turbo3/Turbo4 KV speed without violating the measured quality gates.

**Architecture:** Keep the existing Turbo block formats and CPU reference quantizers. Add a reproducible perplexity comparison report first, then optimize the existing CUDA Flash Attention Turbo read path behind the current type dispatch. Paged Turbo KV remains outside this plan until ordinary Turbo meets the quality and speed gates.

**Tech Stack:** C++17, CUDA, llama.cpp `llama-perplexity`, `llama-bench`, existing GGML CUDA attention kernels and backend tests.

**Spec:** `docs/superpowers/specs/2026-10-05-turbo-kv-quality-performance-design.md`

## Global Constraints

- Preserve the existing `block_turbo3_0` and `block_turbo4_0` layouts and CPU reference quantization.
- Historical quality uses fixed-corpus logits from `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar-logits`.
- The primary candidate is `-ctlk f16 -ctlv f16 -ctgk turbo3 -ctgv turbo3`.
- Require Mean KLD `<= 0.0001` and Same top p `>= 99.9%` for the primary candidate unless the reproduction establishes a stricter historical value.
- Do not overwrite deployed binaries, model files, or the existing logits file.
- Do not add a new test file under `tests/*`; reuse existing test targets and fixtures.
- Do not commit or push changes without explicit user approval.

## Review Focus

- Reference logits generated with different model metadata or tokenizer: Task 1 records model hash, token count, and command-line identity.
- Local/global cache policy accidentally reversed: Task 1 checks both mixed-policy directions and reports effective types.
- Turbo3/Turbo4 packed block boundaries at D256 and D512: Task 2 runs existing backend attention cases for both dimensions.
- SM70 dispatch falling back to a slower or unsupported path: Task 2 compares kernel selection and matched `llama-bench` timings.
- Small batch and long-context regressions hidden by one benchmark shape: Task 3 measures prompt/decode cases at batch 1, 12, 64 and long context.

### Task 1: Reproduce the historical quality baseline

**Files:**
- Modify: `scripts/bench-lumen-kv.sh` only if a reusable perplexity mode is needed
- Create: `docs/turbo-kv-quality-20261005.md`

**Interfaces:**
- Consumes: existing model, corpus and saved logits paths under `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar`.
- Produces: a checked-in command record and measured baseline table for Tasks 2-3.

- [ ] Verify the saved logits and model are readable without changing them.
- [ ] Run FP16, local-F16/global-Turbo3, local-F16/global-Turbo4, all-Turbo3, all-Turbo4 and Q8_KV with `llama-perplexity --kl-divergence`.
- [ ] Record Mean KLD, 99.9% KLD, maximum KLD, delta-probability statistics, Same top p, token count, commit, binary path and GPU.
- [ ] Run the existing Turbo quantization test target and confirm the baseline binary is current.
- [ ] Write the results to `docs/turbo-kv-quality-20261005.md` without claiming a pass if the historical metrics cannot be reproduced.

### Task 2: Optimize the ordinary CUDA Turbo attention path

**Files:**
- Modify: `ggml/src/ggml-cuda/fattn-common.cuh`
- Modify: `ggml/src/ggml-cuda/fattn-vec.cuh` only if dispatch gating is required
- Modify: `ggml/src/ggml-cuda/fattn.cu` only if kernel selection requires a narrow SM70 guard
- Test: existing `test-backend-ops` attention cases and `test-turbo-quant`

**Interfaces:**
- Consumes: existing Turbo3/Turbo4 block layouts and vector Flash Attention template dispatch.
- Produces: the same attention output contract for Turbo K/V with reduced dequantization and load overhead.

- [ ] Add or extend a focused backend operator case that covers Turbo3/Turbo4 K/V at D256 and D512 against the existing dense reference.
- [ ] Run the case before the kernel change and record the current numerical tolerance and timing.
- [ ] Implement the smallest CUDA change that reuses per-block scale/centroid data and uses coalesced packed loads while retaining FP32 accumulation.
- [ ] Keep unsupported dimensions, mixed types and non-SM70 devices on their existing dispatch paths.
- [ ] Run the focused backend cases on CUDA0 and the existing CPU/reference path; require no tolerance regression.
- [ ] Run `test-turbo-quant` and build the affected CUDA targets.

### Task 3: Measure and select Turbo3 or Turbo4

**Files:**
- Modify: `scripts/bench-lumen-kv.sh` if it needs a reproducible matched benchmark mode
- Modify: `docs/turbo-kv-quality-20261005.md`

**Interfaces:**
- Consumes: Task 1 quality baseline and Task 2 optimized binary.
- Produces: matched ordinary CUDA performance table and recommendation.

- [ ] Benchmark FP16, Q8_KV, Turbo3 and Turbo4 at batch 1, 12, 64 and the deployed long-context lengths with fixed warmup and repetitions.
- [ ] Record prompt tok/s, decode tok/s, peak VRAM, effective KV types and kernel/backend logs.
- [ ] Compare optimized and unoptimized Turbo results and reject changes with a quality regression or a material speed regression.
- [ ] Select the better Turbo type only if it meets the quality gates and improves the target workload.
- [ ] Leave the deployed `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/start.sh` unchanged until the result is reviewed.

### Task 4: Final verification and handoff

**Files:**
- Modify: `docs/turbo-kv-quality-20261005.md`

**Interfaces:**
- Consumes: the selected ordinary CUDA implementation and all previous measurements.
- Produces: a complete reproducibility report with limitations and rollback information.

- [ ] Run the existing CUDA backend operator and Turbo quantization checks after the final build.
- [ ] Re-run the selected quality matrix from Task 1 using the final binary.
- [ ] Re-run the matched performance matrix from Task 3 and preserve raw JSONL output outside the source tree.
- [ ] Document exact binary, commit, model, arguments, GPU, quality values, speed values and any unsupported cases.
- [ ] Report whether paged Turbo KV should be split into a separate follow-up design; do not implement it in this plan.
