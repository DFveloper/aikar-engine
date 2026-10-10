# Global CE/KL RCO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. Proposed execution: inline in the current session, without subagents.

**Goal:** Generate and evaluate a Gemma 4 E2B GSQ+RCO mixed-precision GGUF with genuine global CE/KL precision gradients and a verified total-file byte cap.

**Architecture:** Reuse native Gemma graph construction and GGML backward/recomputation. Keep packed candidates on disk, stream candidate contractions, and retain only small precision optimizer state. Preserve existing GSQ and the separate reconstruction proxy.

**Tech Stack:** C++17, native GGML CPU/CUDA, GGUF codecs/writer, existing Python/PyTorch numerical oracle.

**Spec:** docs/superpowers/specs/2026-10-09-rco-global-ce-kl-design.md

## Global Constraints

- Source: /home/user/aikar-engine/tools/llama-optimize. Artifacts: /mnt/openwebui/AIKAR/GSQ-RCO.
- No original BF16 overwrite, commits, pushes or upstream submissions. Preserve all pre-existing uncommitted work.
- Service interruption is authorized; record restart configuration, stop only before GPU work, and restore and health-check after experiments, including failures.
- V100 16 GB; candidate lazy loading, CPU storage, measured temporary-buffer reuse and existing recomputation. No full-model optimizer moments.
- CE and full-vocabulary teacher KL are separate objectives. Layerwise reconstruction cannot satisfy global RCO acceptance.
- Exact total-file byte cap includes destination metadata, alignment, fixed tensors and selected packed payloads. No rounded byte feasibility claims.
- Raw corpus: first 12,000 token IDs calibration, discard 512, last 5,096 held-out. No cross-split windows; report actual scored tokens and single-conversation correlation.
- GSQ/PTQ producer identity is distinct from codec identity. Copy selected GSQ packed bytes verbatim.
- No new tests/* files. Extend tool self-test, test_reference.py and test_linear.py as applicable.
- All new source/comments use ASCII. No unsolicited core refactors or new quantization types.

## Review Focus

- Cached graph or checkpoint reused with different tokens/candidates: reject hash/shape/config mismatch (Tasks 2, 5, 7).
- Shared K/V reused across layers: preserve and test early-layer derivative dependencies (Tasks 3, 5).
- Missing codec backward, BF16 derivative support or routed adapter: explicit unsupported error; no gradient detachment (Tasks 3, 4).
- Same codec from multiple producers or stale GSQ payload: keep candidates distinct and verify hashes through output (Tasks 2, 8).
- Infeasible byte cap, DP state overflow, output collision or interrupted service: fail safely and restore runtime (Tasks 2, 7, 8).

## File responsibilities

- Existing optimize.h/optimize.cpp: GSQ and RCO geometry remain authoritative; add precision softmax VJP helpers.
- Existing main.cpp: command dispatch, capture/GSQ/proxy behavior, GGUF packing helpers and numerical self-test.
- New tool-local candidate-store.h/.cpp: validated packed candidate index, lazy decoding, exact byte accounting, payload preservation.
- New tool-local global-loss.h/.cpp: global objective adapter and candidate gradient contractions; no independent Gemma forward implementation.
- Existing src/llama-context.h/.cpp and src/llama-graph.h/.cpp: minimal opt-in graph hooks for training weight replacement and loss/backward evaluation.
- Existing src/models/gemma4.cpp: only training attention dependency adjustments if required by gradient tests.
- Existing GGML optimizer interfaces: expose existing backward/recompute behavior only where the adapter cannot use it directly.
- Existing test_reference.py: independent CE/KL, interpolation, STE and geometry oracle.
- New tool-local prepare_data.py and evaluate_corpus.py: immutable token split and multi-window evaluation orchestration.
- Existing run_experiment.py and README.md: experiment resources, reproducible commands and truthful mode documentation.

### Task 1: CE/KL and precision-gradient numerical harness

**Interfaces:** `loss_result { double value; std::vector<float> logits_gradient; }`; `loss_result token_loss(const std::vector<float> & logits, const std::vector<int32_t> & targets, const std::vector<float> & teacher_log_probs, size_t vocabulary, bool kl)`; `std::vector<double> precision_vjp(const std::vector<double> & probabilities, const std::vector<double> & option_gradients, double temperature)` in global-loss/optimize files.

- [ ] Extend self-test JSON and test_reference.py with shifted CE/KL from a two-layer nonlinear graph, real candidate interpolation and an upstream PyTorch oracle. Assert finite nonzero gradients and matching token normalization.
- [ ] Run the new oracle; confirm failure because the global loss outputs/helpers are absent.
- [ ] Implement CE/KL value/logit derivatives and precision VJP with stable vocabulary normalization. Construct the small GGML backward graph in self-test; do not use probability-only objectives.
- [ ] Check smooth-forward finite differences at several step sizes, compare analytic gradients with PyTorch, and validate hard-forward STE VJP separately. Use relative error < 1e-3 or absolute error < 1e-5 for FP32; report measured errors rather than relaxing thresholds silently.
- [ ] Run `python3 tools/llama-optimize/test_reference.py /tmp/rco-global-audit-build/bin/llama-optimize /tmp/llama-gsq-original /tmp/llama-rco-original`; all old and new checks must pass.

### Task 2: Packed candidate store and file-budget preflight

**Interfaces:** `candidate_store(source_model, manifest_path)`; `read_packed(group, option) -> std::vector<uint8_t>`; `decode_tile(group, option, first_row, row_count) -> std::vector<float>`; `costs() -> std::vector<std::vector<uint64_t>>`; `fixed_bytes() -> uint64_t`. A manifest group carries tensor name, full dimensions, operation kind and optional expert ID; each option carries producer, codec, payload hash/path and raw/aligned bytes.

- [ ] Add failing checks for duplicate producer/codec identity, shape/hash mismatch, same-codec different producers, infeasible total cap, DP overflow and output collision. Use existing self-test/oracle infrastructure.
- [ ] Extract existing candidate creation/GSQ byte import into the store without altering GSQ numerics or proxy behavior. Reuse GGML codecs and current assign function.
- [ ] Implement destination-GGUF metadata accounting and fixed bytes; derive tensor cap from requested total bytes before graph allocation. Retain the DP hard state limit and explicit failure.
- [ ] Verify lazy tile decode against whole-tensor decode, manifest roundtrip, exact GSQ bytes and predicted destination size. Reject unsupported routed operations explicitly while preserving manifest extension fields.

### Task 3: Opt-in native training graph access

**Interfaces:** internal `llama_training_graph_hooks` with `replace_weight(ggml_context *, ggml_tensor *) -> ggml_tensor *` and `build_loss(llm_graph_result &) -> ggml_tensor *`; internal `llama_context::evaluate_training_graph(const llama_batch &, const llama_training_graph_hooks &, float * loss) -> int`. Hooks live only for the synchronous call; ordinary inference uses no hooks.

- [ ] Add a tool command `global-check MODEL TOKENS CONFIG REPORT GPU_LAYERS` that initially fails explicitly without training graph support.
- [ ] Wire the internal hooks through graph parameters and cache identity. Apply replacement consistently to dense matmul, embeddings if selected and routed matmul extension points; absent hooks preserve existing inference.
- [ ] Attach the supplied loss before allocation, build gradients through existing backward builder, and use existing scheduler/recompute infrastructure. Do not invoke Adam on model weights.
- [ ] Retain current-window K/V graph dependencies and cross-layer shared K/V on the Gemma training path. Reject unsupported ops and unsafe cached graphs explicitly. Document each core interface change.
- [ ] Compare BF16-reference forward logits, masks and CE with native inference on the same short tokens. Test hooked/unhooked graph rebuild and inference-after-training behavior.

### Task 4: Stream candidate derivative contractions

**Interfaces:** `global_rco_adapter(candidate_store &, llama_context &, memory_budget)` with `evaluate(tokens, objective, teacher_cache, alpha, noise, temperature, hard_assignment) -> {loss, precision_gradient, diagnostics}`. It consumes Task 2 storage and Task 3 graph hooks.

- [ ] Fail a two-tensor global graph test whose early candidate affects later CE/KL; verify that a disconnected candidate cannot pass.
- [ ] Load selected hard-forward weights and compute per-option VJPs from global adjoints, using candidate differences and streamed tiles. Route only the small probability adjoints into precision_vjp.
- [ ] Keep frozen tensors and large candidate state on CPU/disk; reuse bounded work buffers. Cast frozen BF16 tensors where the existing backward path requires it and record compute precision.
- [ ] Verify equivalent contractions against resident-candidate oracle; run with a deliberately small buffer cap. Verify explicit failure for missing backward support and unsupported routed adapters.

### Task 5: Immutable corpus and full teacher cache

**Interfaces:** `prepare_data.py --binary --model --source --output --context` writes token arrays and split manifest; `teacher-cache MODEL TOKENS CACHE REPORT GPU_LAYERS` writes window-indexed FP32 reference log-probabilities with token/model hashes.

- [ ] Test duplicate BOS, escape processing and mismatched teacher/student IDs via existing oracle script. Inspect unmatched tool delimiters before offering any structured conversion.
- [ ] Generate separate raw-token split artifacts preserving original bytes. Record 17,608 source tokens, 12,000/512/5,096 split boundaries and scored-token counts.
- [ ] Build teacher cache sequentially without student/backward residency; process vocabulary reductions in bounded chunks. Store full-vocabulary probabilities, never unreported top-k KL.
- [ ] Verify cache replay, reference KL approximately zero, teacher CE parity and rejection of wrong tokens/context/model hashes.

### Task 6: Actual Gemma precision-gradient acceptance

**Files:** global-loss.*, test_reference.py, global-check report; reuse the source GGUF and Task 5 token artifacts.

- [ ] Run CPU short-window global-check for early/middle/late candidate tensors, including K/V-producing layers and shared-KV consumers. Measure raw and projected gradients separately.
- [ ] Check smooth precision finite differences with fixed tokens/noise and several step sizes; validate STE surrogate VJP separately. Check candidate sensitivity against full model CE/KL and compare loss with native inference.
- [ ] Before GPU checks, record the running server's exact launch/restart configuration and health endpoint. Stop the authorized service and repeat CUDA checks on V100, measuring memory; restore service in failure cleanup.
- [ ] Increase complete window size only after measured memory and gradient correctness pass. If native dependency/recompute support requires a larger redesign, document the exact failing op/path and stop for design review.

### Task 7: Global RCO optimizer and resumable search

**Interfaces:** CLI `rco-global MODEL MANIFEST TOKENS CACHE OUTPUT TOTAL_BYTE_BUDGET STEPS BACKEND REPORT`; objective, context, seed and memory cap are recorded in the manifest/config. Checkpoints contain alpha, Adam moments, step, RNG state and candidate/token/config hashes.

- [ ] Add a deterministic multi-step global-loss test and checkpoint replay check; verify failure before command implementation.
- [ ] Reuse seeded Gumbel sampling, exact assign, current manifold operations and GGML FP32 Adam. Execute backward, tangent projection, Adam, retraction and first-moment transport in pinned upstream order.
- [ ] Accumulate token-normalized gradients across calibration windows; implement exponential temperature annealing and no weight decay. Log objective and constraint diagnostics per step.
- [ ] Verify feasible hard assignments and expected-cost residuals, checkpoint resume equivalence and explicit stale checkpoint rejection. Record actual loss trace without requiring monotonic stochastic loss.

### Task 8: GSQ candidate expansion and verified output

**Interfaces:** candidate generator accepts PTQ, effective IMatrix PTQ, GSQ-RTN and GSQ-GPTQ producers; materialization consumes Task 7 selection and Task 2 packed payloads.

- [ ] Expand calibration captures and existing GSQ training across intended dense tensors; run existing full-Hessian initializer and direct Q2_0 packer unchanged. Check GSQ prior/packing numerical regression after expansion.
- [ ] Run PTQ-only RCO and GSQ+RCO with compatible fixed tensors, token corpus and total-file cap. Keep candidates from different producers distinct even at equal cost.
- [ ] Stream selected payloads into a new partial GGUF, verify metadata/hash/size before publication, report unused capacity, and reload through native engine. Test infeasible budgets and no-clobber behavior.
- [ ] Preserve final GSQ+RCO GGUF, its manifest and reproducible training checkpoint. Store all paths in the Korean completion report.

### Task 9: Fair multi-window quality/resource evaluation

**Interfaces:** `evaluate_corpus.py --binary --models-manifest --tokens --teacher-cache --output --context`; model manifest includes producer, file bytes/hash, fixed tensors and budget. Reuse existing capture/reconstruction and native bench paths.

- [ ] Compare BF16, PTQ, IMatrix PTQ, GSQ-RTN, GSQ-GPTQ, RCO and GSQ+RCO only after Task 6 and Task 8 pass. Reject overlapping evaluation windows and inconsistent IDs/masks/context.
- [ ] Replace the 512-token total evaluation limit with bounded multi-window orchestration; compute token-weighted CE, exp(CE), full KL and explicitly defined reconstruction MSE. Include tails or report dropped-token counts.
- [ ] Match file size where feasible, report each actual mismatch, and avoid treating Q2_0's ignored IMatrix input as a distinct effective baseline. Evaluate comparison outputs sequentially if disk cannot hold all arms.
- [ ] Extend run_experiment.py to distinguish process VRAM and total device VRAM, collect peak RSS and sample intervals, wall time and native prefill/decode throughput under recorded settings.
- [ ] Restore and health-check the existing service. Write raw JSON results and a Korean report covering the user's 11 requested items, exact gradient/manifold evidence, actual GGUF paths and unimplemented limitations.

### Task 10: Final verification and documentation

- [ ] Update README with actual commands, global/proxy distinctions, candidate contract, data correlation, memory settings and routed-operation extension boundary. Full Lumen testing remains deferred.
- [ ] Build CPU and CUDA targets; run existing quantization and GSQ/RCO oracle tests plus new global-gradient and GGUF integration checks. Run `git diff --check` and inspect every core hook for the no-hook inference path.
- [ ] Review spec coverage against concrete reports. Declare completion only with actual Gemma global CE/KL optimization, byte-feasible GSQ+RCO output/reload and held-out evaluation evidence. No synthetic-only completion claims.

## Execution handoff

Design approved. This plan still needs user review before product-code changes, as required by writing-plans. Proposed execution is inline because native graph access, candidate contractions and recomputation share interfaces and depend on sequential correctness checks. No subagents or commits are proposed.
