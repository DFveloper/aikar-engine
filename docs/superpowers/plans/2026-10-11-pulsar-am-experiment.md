# Pulsar Attention Matching Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test Attention Matching on actual Pulsar Q8 paged KV before deciding whether to integrate it with Hadamard and Q4/IQ4.

**Architecture:** A disposable checkout gets a minimal attention-bias extension and a throwaway capture/replay executable. An offline Python worker fits compacted caches; replay measures actual full-model continuation quality. Production prefix storage, persistence, codecs and launch settings remain unchanged until results support integration.

**Tech Stack:** Existing C++17/GGML CPU/CUDA code, SM70 V100, installed NumPy/SciPy/PyTorch, existing backend tests. No product dependency installation.

**Spec:** [AM design](../specs/2026-10-11-paged-kv-attention-matching-design.md) and [conditional codec design](../specs/2026-10-11-paged-kv-q4-iq4-262k-design.md).

**Status:** Canceled at the user's request on 2026-10-11. Task 1 stopped after workspace setup, CPU baseline verification and capture-helper self-checks; no AM model capture, fitting or replay occurred. The AM-only build was interrupted. Reuse the isolated baseline for codec/capacity work instead of continuing this plan.

## Global Constraints

- Experiment first, integration only if quality/feasibility results justify it.
- Full layers only: 5 layers, 2 KV heads, D512; SWA unchanged.
- One sequence, text-only immutable prefix, exact recent tail; initial old-region retained fraction 0.5, then 0.25.
- One head/chunk at a time; original chunk length 2048; reservoir at most 2048 queries/head; host query cap 40 MiB; solver scratch cap 64 MiB excluding separately measured CUDA runtime overhead; packed snapshots/staging cap 4 GiB host memory.
- Preserve root working tree changes, production processes and launcher. No commits, pushes or PR text. Do not create files under `tests/`; modify the existing test file only inside the experiment checkout.
- Do not call the sparse-table probe a native compact-prefix kernel or a production speedup.

## Review Focus

- Synthetic rows retain post-RoPE keys; continuation position IDs must remain the original positions (Task 3).
- An old bias must not survive snapshot restoration and contaminate the Q8 control (Task 3).
- Calibration and validation queries must be disjoint, including duplicate sample identities (Task 2).
- Value fitting can produce large/nonfinite values despite a small calibration residual; reject before importing (Task 2).
- The model callback may expose strided tensors; copying as contiguous bytes corrupts Q/K/V (Task 1).

## File and interface map

- Disposable checkout `/tmp/pulsar-am-experiment/engine`: source baseline plus a reviewed copy of current tracked root modifications. Keep its starting patch and hashes in `/tmp/pulsar-am-experiment/artifacts/`.
- Experimental edits: `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cuda/pagedattn.cu`, CUDA/Vulkan support guards, `src/llama-kv-cache-paged.{h,cpp}`, `src/llama-graph.cpp`, and `tests/test-backend-ops.cpp`.
- Throwaway files outside the repo: `/tmp/pulsar-am-experiment/am-probe.cpp`, `fit_am.py`, `run_experiment.py`, and a small standalone CMake file linking the checkout's built libraries. Use private headers only in this Linux experiment executable; do not add a public llama API.
- Probe consumes a JSON job and writes JSON results plus packed snapshots/arrays. Modes: `capture`, `codec`, `replay`. CLI: `am-probe --job JOB.json`. Required job keys: `mode`, `model`, `artifact_dir`; codec jobs omit `model` and add `type`, `head_dim`, `input`, `direction` (`encode` or `decode`). Reject unknown modes and missing/incompatible fields.
- Capture/replay jobs also specify `prompt_file`, `followups_file`, `n_ctx`, `n_batch`, `n_ubatch`, `n_blocks`, `seed` and `n_predict`; replay adds `capture_manifest` and `candidate_manifest`. Tokenize with the real GGUF tokenizer and apply its chat template using existing common helpers. Follow-ups are a JSON list of prompt strings, expected facts and unique case IDs. Use seed 42, batch/microbatch 256, block size 32 and 128 predicted tokens for initial controls; derive context/pool capacity from the full original + repeat-prefill token count before allocating.
- Capture artifacts: `manifest.json`, `original.state`, `tokens.i32`, `queries.f32`, `query_ids.i32`, and per-Full-layer packed cache files/page maps. Record type, dimensions, all byte strides, layer/head IDs, positions, attention scale, prompt hash, source patch hash and binary hash.
- Fit output: `candidate.json` plus per-Full-layer canonical packed K/V rows and F32 beta. Record original protected interval, selected original indices per head, packed row count, codec, solver settings and held-out residuals. No new production state format.

## Task 1: Capture trustworthy Pulsar cache and queries

**Interfaces:** `am-probe --job JOB.json` in capture mode produces the artifacts above from one Q8 paged context; it must not create an F16 reference context.

- [ ] Use the git-worktree skill to isolate the experiment. Save/apply the root tracked patch without moving or resetting root files. Configure Release/SM70 builds with CUDA and existing tests/examples; keep the original build untouched.
- [ ] Add capture self-checks named `capture_strides`, `capture_q8_decode` and `capture_geometry` in the throwaway executable: stride-aware row reads agree with a contiguous copy, Q8 packed decode agrees with the existing GGML traits, exactly layers 5/11/17/23/29 are selected, and all positions/head IDs are valid. Run these before collecting model data.
- [ ] Implement capture using existing eval-callback/kv-quality patterns. Use `Qcur_pos` for unrotated post-RoPE queries. Read actual stored K/V through paged layer tensors and byte strides; do not reconstruct the Q8 baseline from unquantized projection outputs.
- [ ] Snapshot after the original prompt. Collect calibration queries from a repeat-prefill control and held-out queries from distinct follow-up prompts, restoring the original snapshot before each branch. Cap the reservoir deterministically at 1536 calibration plus 512 validation queries/head. Keep greedy reference continuations for later teacher forcing.
- [ ] Run a 4K-token capture with batch/microbatch 256, one slot, block size 32, matching Q8_KV, no rotation, graphs disabled, and enough physical/logical capacity for the repeat-prefill pass. Confirm all artifact hashes, cache geometry, positions and budget counters; report callback overhead separately.

## Task 2: Fit and validate AM without the serving model resident

**Interfaces:** `fit_am.py --manifest MANIFEST --retained-fraction 0.5 --output DIR` consumes capture artifacts and produces `candidate.json`. The probe's codec mode is the only encoder/decoder; Python does not invent IQ4/Q4 codecs.

- [ ] Add throwaway self-checks named `duplicate_mass`, `future_concat`, `disjoint_queries`, `bad_fit_rejected` and `budget_rejected`: duplicate-key fixtures recover their multiplicity through beta; adding fixed future tokens preserves the expected weighted output; train/validation IDs are disjoint; singular/nonfinite fits reject cleanly; over-budget matrices reject before allocation. Compare the float64 mathematical fixture with maximum absolute error <= 1e-10 before introducing Q8 storage error.
- [ ] Implement stable scaled logits using the actual exported attention scale. Rank keys by RMS attention, select per-head keys, fit nonnegative mass weights with stable row offsets, then fit values with regularized/rank-aware least squares. Start Q8 only; encode/decode selected keys before fitting and encode/decode fitted values before held-out evaluation. Include the 0.5 and 0.25 fractions and an uncompressed identity candidate.
- [ ] Process one 2048-row old-region chunk/head at a time. Protect the first 64 prompt tokens and the most recent 1024 tokens; preserve template/system tokens when their protected span is larger. Measure/report actual overall retained fraction. Use positive constrained fitting with residual/convergence checks, not an unconstrained solve renamed NNLS. Never explicitly invert a normal matrix.
- [ ] Run CPU references and, if runtime SM70 support is confirmed, GPU fitting with installed PyTorch. Measure solver peak separately from CUDA context overhead. Check candidate files contain finite decoded K/V and beta, legal selected indices and exact unchanged protected rows.
- [ ] Report held-out attention NMSE and log-mass error against the original stored cache. A good reconstruction result permits replay; it does not by itself authorize integration.
- [ ] Construct the identity candidate from original packed bytes in original order, beta zero and unchanged maps. Do not refit/requantize identity rows; this control isolates capture/import errors from AM error.

## Task 3: Replay actual model continuations with experimental beta

**Interfaces:** `void ggml_paged_attn_set_bias(struct ggml_tensor * a, struct ggml_tensor * beta)` attaches an optional F32 tensor in `src[7]`, shape `[block_size, heads_kv, physical_pages]`. Null bias retains the old op. The experimental paged `layer` gets a nullable bias tensor and owned experiment-only buffers.

- [ ] Extend the existing backend fixture with `test_paged_attn_bias` for zero/nonzero beta and grouped heads, with an independent dense biased attention oracle. Test zero-beta equivalence, scale 0.37, masks, split merging and CPU/CUDA results; require the existing paged maximum NMSE of 1e-6. Establish a failure before changing kernels.
- [ ] Add the optional bias to CPU and every CUDA paged decode path, after score scaling and before online softmax. Preserve the old null-bias dispatch. Reject nonnull bias on Vulkan. Ordinary appended rows reset beta to zero; experimental page copies copy beta together with payloads.
- [ ] Implement replay in the throwaway executable: synchronize, validate candidate geometry/budgets, retain a recovery snapshot, pack synthetic Full prefix rows into ordinary low logical table indices, keep protected prefix and exact tail at original positions, remove obsolete Full rows and attach bias. Per-head keys may differ, but packed lengths agree across Full layers/heads. Leave all SWA cache tensors/maps intact.
- [ ] Keep the original position counter by retaining the nonempty exact tail. Run queries only after the compacted original interval. Reject partial edits, a second sequence, shared-page mutation and writing an experimental state to a production state file. Restore the original state and detach/zero bias between every control/candidate branch.
- [ ] Run `build-am/bin/test-backend-ops -b CPU -o PAGED_ATTN,PAGED_ATTN_WRITE` and the corresponding `-b CUDA0` command. Require no failed supported cases and report skips. The replay identity candidate must reproduce the control within the same numerical tolerance used by the existing paged oracle before evaluating compressed candidates.
- [ ] Check full-model continuation positions, direct KL, top-1 agreement and greedy answer changes using identical teacher-forced tokens. Native replay is required; a Python attention-only surrogate cannot establish full-model quality.

## Task 4: Run staged quality/latency experiments and decide

**Interfaces:** `run_experiment.py --probe BIN --artifacts DIR --stage smoke|extended` writes raw per-case JSON and `summary.json`; candidate labels are `Q8`, `AM-Q8-0.5`, `AM-Q8-0.25`, and `AM-identity`.

- [ ] Smoke stage: six deterministic cases each at 4K and 8K, covering exact name/number retrieval, dispersed multi-needle retrieval, Korean instructions and code facts. Store expected facts and token arrays with results. Replay at least 128 continuation tokens where appropriate. Do not use these follow-up queries to fit their own candidate.
- [ ] Use the approved provisional long-control gate for screening: candidate-versus-Q8 mean KL <= 0.01, top-1 agreement >= 99%, and no critical deterministic retrieval regression. Report uncertainty and individual failures, not just means. A smoke pass is promising, not an application-quality certificate.
- [ ] If 0.5 is promising, extend to at least 24 independent cases at 8K/16K/32K, including held-out layouts and multi-turn continuations. Expand sample size when the <= 1 percentage point task-accuracy gate remains inconclusive. If 0.25 fails, keep it as a documented rejected ratio. If 0.5 fails, test 0.75 as a diagnostic and report that the intended 2x old-region target failed.
- [ ] Measure initial prefill, repeat/query collection, fit, upload, full-model decode time and peak/live/allocated memory separately. Use warmups and three repetitions for timing, with capture disabled during replay measurements. Compute amortized break-even only when measured per-token savings are positive. The sparse-table reader is not evidence for native packed-prefix speed.
- [ ] Write `docs/pulsar-am-experiment-20261011.md` with model/build/patch hashes, raw artifact locations, actual retained counts, quality metrics, failures and timing. Classify the outcome as rejected, inconclusive, or promising for opt-in integration. No production default change follows from this experiment.
- [ ] If promising, resume the already approved native AM/codec designs and write their concrete integration plans using these measurements. If rejected/inconclusive, explain the result and continue independent paged memory/prefill/decode optimization without enabling AM. Keep all experiment code explicitly throwaway until an integration plan is reviewed.

## Completion evidence

- Root source/launcher untouched except approved documentation; root changes match the saved baseline.
- Supported CPU/CUDA biased-attention cases and identity replay pass with real output logs.
- Raw model-level quality and cost measurements exist, or an explicit failed/inconclusive feasibility result identifies why they could not be obtained.
- No claims of 262K quality, native compaction speed or production readiness are made from the short-length experiment.

Recommended execution: native in this session. Capture, fitting and replay depend on the same cache geometry and artifact contract; one implementer can keep those consistent while retaining a final independent review.
