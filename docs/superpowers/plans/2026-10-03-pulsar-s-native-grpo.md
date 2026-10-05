# Pulsar S Native GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline execution, or superpowers:subagent-driven-development if the user selects delegated execution. Track steps with checkboxes. Repository instructions forbid unauthorized commits and automated submissions.

**Goal:** Implement native outcome-supervised GRPO and verify Pulsar S training on CUDA0 within a 16 GiB V100, with GRPO phase offload removed.

**Architecture:** Reuse the existing GGUF/QLoRA model, optimizer, sparse-target loss infrastructure, and activation recomputation. Add a sparse GRPO loss with CPU/CUDA backward support and a response-aware group update API. Keep one resident model/context and store the frozen reference adapter in host memory.

**Tech Stack:** C++17, ggml C API, CUDA, existing Python stdlib IPC driver, existing CMake/test runners.

**Spec:** `docs/superpowers/specs/2026-10-03-pulsar-s-native-grpo-design.md`

## Global Constraints

- Target `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf` on the V100 CUDA0 device with 16384 MiB capacity.
- Preserve all pre-existing uncommitted changes and keep comments concise and ASCII.
- No commits, pushes, PR descriptions, or automated upstream submissions.
- No new test files under `tests/`; extend existing optimizer/backend tests.
- One GPU model, one context/KV allocation, and one trainable adapter; no GRPO scheduler suspension, temporary rollout runtime, or model reload.
- Initial memory configuration: rank 16, alpha 8, `attn_output.weight`, LoRA QAT q4_0, tensor type recipe Q4_0_XL, F16 KV, FlashAttention on, activation recomputation on, context/batch 8192, microbatch 256, seed 3407.
- Initial group size 4 and maximum completion 512. If 256 cannot fit, explicitly measure 128 and 64 and report the working configuration.
- Population reward standard deviation, denominator `std + 1e-8`, clipping epsilon 0.2, beta 0.04, and one update iteration by default.
- Frozen original reference scores and original rollout token IDs survive every update iteration and checkpoint resume.
- Raw rewards remain supplied by IPC. A production reward provider has not been specified.
- Prepare external launch artifacts inside the repository; request filesystem approval before installation outside writable roots.

## Review Focus

- Malformed or non-finite reward/CLI inputs must fail before modifying parameters; Task 4 pins this behavior.
- Immediate EOG and tokens with long text pieces must retain sampled IDs and completion-label alignment; Task 4 pins this behavior.
- Partial prompt/completion microbatches must normalize by response length and update exactly once per group; Task 3 pins this behavior.
- Reference scoring errors and interrupts must restore the current adapter and never overwrite a stage checkpoint; Task 4 pins this behavior.
- Missing or stale checkpoint reference companions must fail rather than silently reset the reference; Task 5 pins this behavior.

## File Map

- `common/common.h`, `common/arg.cpp`: remove phase-offload configuration and add objective options.
- `ggml/include/ggml.h`, `ggml/src/ggml.c`: sparse GRPO forward/backward tensor constructors and autodiff registration.
- `ggml/src/ggml-cpu/ops.h`, `ops.cpp`, `ggml-cpu.c`: CPU implementations and dispatch.
- `ggml/src/ggml-cuda/cross-entropy-loss.cuh`, `.cu`, `ggml-cuda.cu`: CUDA implementations and backend support/dispatch.
- `ggml/include/ggml-opt.h`, `ggml/src/ggml-opt.cpp`: optimizer loss mode, metadata inputs, and whole-group accumulation.
- `include/llama.h`, `src/llama-ext.h`, `src/llama-context.h`, `.cpp`: optimizer settings, internal response/group API, and variable-length training loop integration.
- `examples/qlora_training/finetune_qlora.cpp`: rollout records, frozen reference, raw reward normalization, checkpoint metadata, and coordinator.
- `examples/qlora_training/grpo_example.py`: raw reward IPC and configurable training/objective settings.
- `examples/qlora_training/README.md`, `pulsar-s-v100.md`: accurate GRPO behavior and measured launch instructions.
- `examples/qlora_training/pulsar-s-grpo.sh`: new configurable launch artifact prepared in the repository.
- `tests/test-opt.cpp`, `tests/test-backend-ops.cpp`: numerical correctness, accumulation, backend parity, and regressions.
- Existing September 13 phase-offload spec/plan: add a superseded notice, preserving their historical contents.

## Task 1: Remove GRPO phase offload while preserving SFT memory management

**Interfaces:** Remove `common_params::grpo_phase_offload` and the `--grpo-phase-offload` option. `run_grpo_mode` no longer accepts `common_params * rollout_params`. Remove unused `llama_opt_suspend`/`llama_opt_resume` wrappers and methods; keep `opt_create_backend_sched` and `ggml_opt_set_backend_sched` where existing SFT/evaluation calls need them.

- [ ] Record `git status --short` and the initial diff so later edits can be separated from user changes. At execution time, read the worktree skill and use an isolated workspace that includes the required existing changes, without resetting the original workspace.
- [ ] Capture baseline trainer help and build the existing trainer/optimizer test targets: `cmake --build build --target llama-finetune-qlora test-opt test-backend-ops -j4`. Save build output separately from user training logs.
- [ ] Remove phase-offload option/field, temporary rollout setup, adapter-map copying helpers, GRPO phase emitters, suspend/resume calls, and branches that force host model placement.
- [ ] Remove exclusively unused lifecycle APIs after checking every consumer with `rg`. Preserve user-added `opt_sched_train` handling and SFT/evaluation scheduler recreation. Keep generic scheduler replacement tests that still exercise shared infrastructure.
- [ ] Add superseded notices to the old phase-offload documents and remove active documentation that presents the deleted option as supported.
- [ ] Rebuild `llama-finetune-qlora` and `test-opt`. Verify trainer help lacks the removed option and passing it fails with an unknown-option error. Verify source searches show no active GRPO phase-offload configuration or runtime.

## Task 2: Implement sparse GRPO loss and CPU/CUDA gradients

**Interfaces:** Add `GGML_OP_GRPO_LOSS` and `GGML_OP_GRPO_LOSS_BACK`. Add `ggml_grpo_loss(ctx, logits, targets, metadata, temperature, epsilon, beta)` returning a scalar sum and `ggml_grpo_loss_back(ctx, grad, logits, targets, metadata, temperature, epsilon, beta)` returning the logits gradient. `targets` is I32 `[nrows]`; `metadata` is F32 `[4,nrows]` with old log-probability, reference log-probability, advantage, and normalization weight. Masking uses weight zero and target -1. Logits are F32 `[vocab,nrows]`. No metadata input has a gradient.

- [ ] In `tests/test-backend-ops.cpp`, add GRPO operation cases with independent double-precision host loss and gradient calculations. Cases use positive/negative advantages, ratios below/inside/above clip bounds, beta zero/positive, temperature 0.8/1.0, masked rows, and weights from unequal response lengths. Check analytical gradients against finite differences away from clip discontinuities; check CPU/CUDA agreement at `atol=2e-5, rtol=2e-4` for these small fixtures.
- [ ] Run the existing backend test runner with its operation filter and confirm the new constructors/operations are missing before implementation. First read its actual `--help` to use the existing filter syntax.
- [ ] Implement constructors, shape/type checks, op names, and autodiff in ggml core. Use stable log-sum-exp of `logits / temperature`. Return a sum of already normalized token losses, never a mean over graph rows.
- [ ] Implement the CPU operation and backward using existing sparse cross-entropy work partitioning. Compute unclipped/clipped branches by advantage sign; at branch boundaries use a consistent subgradient. With `d = log_pi_ref - log_pi`, KL derivative with respect to log_pi is `beta * (1-exp(d))`. Multiply the resulting log-probability derivative by `(one_hot - softmax) / temperature` and the supplied normalization weight/upstream gradient.
- [ ] Implement CUDA kernels inside the existing cross-entropy files. Reuse warp reductions and scalar reduction scratch allocation. Do not allocate dense labels or a persistent vocabulary-sized softmax tensor. Add CUDA support checks and dispatch for both operations.
- [ ] Verify forward/backward masked rows return zero even with placeholder scores. Add unsupported-type/shape validation at construction. Detect non-finite resulting loss in the coordinator before claiming a successful group.
- [ ] Build and run the new cases on CPU and CUDA0 and run existing sparse cross-entropy cases. Record tolerances and results.

## Task 3: Integrate response-aware optimizer accumulation

**Interfaces:** Add `GGML_OPT_LOSS_TYPE_GRPO`, objective fields in `ggml_opt_params`/`llama_opt_params`, and `ggml_opt_grpo_metadata(opt_ctx)` returning the metadata tensor from Task 2. Extend the internal current-period setter for a bounded whole-group period without changing SFT defaults. Add internal `llama_opt_grpo_sequence` with token pointer, total token count, prompt token count, completion old/reference score pointers, and advantage. Add `llama_opt_grpo_group(ctx, sequences, n_sequences, stats)` returning success/failure and filling group loss/update statistics. Expose these tool-specific types and entry point in `src/llama-ext.h`.

- [ ] Add cases to `tests/test-opt.cpp`: split a fixed logits/metadata fixture into unequal physical batches, accumulate a group, and compare parameters with an unsplit SGD update. Assert exactly one optimizer iteration, no period-based rescaling, and unchanged old/reference inputs. Exercise an additional update on the same frozen scores.
- [ ] Run the new optimizer cases and confirm missing GRPO mode/API failure before implementing them.
- [ ] Build the new optimizer loss mode with sparse targets, normalization weights, and Task 2 metadata; do not apply SFT `1/opt_period` scaling. Add all needed defaults to every aggregate initializer found with `rg` to preserve other tools.
- [ ] Calculate the actual number of forward/backward microbatches across all response sequences and configure one accumulation period at the group boundary. A partial final microbatch contributes its actual row count. No optimizer step occurs between response sequences. Keep persistent gradient accumulators across dynamic graph allocations.
- [ ] Extend `opt_epoch_iter` for an internal optional GRPO metadata span and actual last-batch token count. Fill metadata using supervised-row positions with labels shifted by one input token; use weight `1/(G*completion_length)` for real completion targets and zero for prompt/sentinels. Reuse KV training and activation recomputation. The group API uses variable-length sequences directly rather than an SFT packed dataset.
- [ ] Add a scoring-only internal path to obtain teacher-forced completion log-probabilities with training-compatible LoRA QAT settings, without changing optimizer moments/iteration. Compare old rollout scores before the first update; reject a material mismatch instead of training on inconsistent probabilities.
- [ ] Test zero-weight prompt-only chunks, partial microbatches, response boundaries, group sizes 2/4, and repeated iterations. Verify SFT tests retain original optimizer scaling and update cadence.

## Task 4: Replace the IPC coordinator with token-preserving GRPO

**Interfaces:** Define a tool-local `grpo_rollout` holding prompt/completion IDs, immutable old/reference scores, decoded text, reward, advantage, and termination flags. Replace text-only `generate_response` with a success/failure function returning this record. Add `--grpo-clip` (0.2), `--grpo-beta` (0.04), and `--grpo-iterations` (1). `REWARD` now means finite raw scores normalized by C++ once.

- [ ] Add integration cases using the existing test runner and temporary scripted IPC inputs: reward values `[0,1]` create opposite advantages; equal rewards and beta zero leave adapter values and optimizer iteration unchanged. Malformed count, NaN/Inf, and trailing nonnumeric rewards return ERROR and a nonzero exit status.
- [ ] Implement config validation before loading the model: group size at least 2, finite positive temperature, positive completion/iteration limits, epsilon in `(0,1)`, finite beta at least zero, and compatible batch/context settings. Reject incompatible critical-token/MTP modes.
- [ ] Implement stable categorical sampling at the configured temperature and capture original sampled IDs/log-probabilities. Use `llama_vocab_is_eog`, include terminal IDs in records, and decode text with size-aware token-to-piece conversion. Respect context boundaries and fail explicitly on decode errors.
- [ ] Snapshot initial reference A/B tensors to host once. Implement an RAII reference-scoring guard that saves/restores current device A/B values around scoring using the single resident model/context. Synchronize and clear KV at adapter changes. Inject a scoring failure in a small-model fixture and assert current values are restored.
- [ ] Receive raw scores, normalize with population standard deviation plus `1e-8`, and retain signed advantages. Remove SFT sample construction, response duplication, window reward averaging, reward TLS use, and the mini-epoch update call from GRPO.
- [ ] Call Task 3 group updates for the configured iteration count. Skip beta-zero groups with exactly zero advantages before any optimizer update/weight decay. Report group/optimizer steps separately; drive the learning-rate schedule using actual optimizer updates and account for iterations/skips.
- [ ] Compute/log policy loss, KL, clip fraction, lengths, reward mean/std, skipped groups, and optimizer iteration using compact token scores. Preserve READY/PROMPT_REQ/GEN/REWARD_REQ/PROGRESS/CHECKPOINT/DONE/ERROR transport tags.
- [ ] Run deterministic two-group small-model integration tests, including immediate EOG, long decoded pieces, prompt too long, and failure after reference installation. Verify positive/negative gradient directions with beta zero.

## Task 5: Preserve reference identity across resume

**Interfaces:** Use `<checkpoint>.grpo-reference.gguf` for the frozen adapter companion and objective/reference metadata stored with the GRPO checkpoint. Store reference adapter tensor names/shapes/types, base identity, temperature, beta, epsilon, and iteration count. Add a checkpoint schema/version distinction from legacy reward-SFT GRPO checkpoints.

- [ ] Add integration checks for missing companion, wrong base identity, tensor mismatch, objective mismatch, legacy checkpoint, and interrupted companion write. All must fail before a training update.
- [ ] Save companions through temporary paths and final rename, then publish the corresponding checkpoint. Never label an incomplete pair resumable. Store/reuse the original reference rather than the current adapter.
- [ ] Validate metadata and load frozen reference host tensors on resume. Allow legacy adapters only as explicit `--lora` initialization; reject legacy `--resume` files that falsely imply this GRPO state.
- [ ] Resume a deterministic two-group fixture and confirm reference values/scores match the original run. Log the existing optimizer-moment restart limitation explicitly and restore supported scheduler/counter fields without claiming exact optimizer continuation.

## Task 6: Driver, launch artifact, and 16 GiB validation

**Interfaces:** Python driver sends raw rewards, passes objective flags, and supports stage1 memory controls. New `pulsar-s-grpo.sh` resolves model/recipe/output/binary paths from configurable variables and invokes the driver. It runs in the foreground so an enclosing tmux session can manage it.

- [ ] Update `grpo_example.py` to send raw reward values with sufficient precision, accept signed/scientific-notation loss logs, and fail on unexpected subprocess EOF/nonzero exit. Replace its blocking timeout claim with a real timeout mechanism or remove the misleading timeout behavior. Document its example heuristic as demonstration only.
- [ ] Expose rank/alpha/targets, microbatch, LoRA QAT recipe, F16 KV, recomputation, GPU device, context, seed, and objective flags through driver arguments. Keep the C++ coordinator responsible for reward normalization.
- [ ] Write `examples/qlora_training/pulsar-s-grpo.sh` with stage1's Pulsar S settings, group size 4, max completion 512, epsilon 0.2, beta 0.04, and one iteration. Validate paths and reject a conflicting active trainer. Use distinct GRPO output/log paths; do not overwrite SFT stage artifacts. Verify shell syntax with `bash -n` and Python syntax without generating files in the external model directory.
- [ ] Build CUDA `llama-finetune-qlora`, `test-opt`, and `test-backend-ops`. Run relevant optimizer/loss/backend tests and one SFT regression fixture. Use CPU build only if CUDA execution is unavailable; report that this would leave GPU acceptance incomplete.
- [ ] Run a short Pulsar S GRPO test with the exact stage1 model/recipe/rank/targets/context/batch/microbatch, at least two groups, and deterministic smoke-test raw rewards. Save outputs under a temporary validation directory. Run GPU commands with approved escalation when needed.
- [ ] Sample `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader` during generation, reference scoring, and backward, and retain trainer buffer logs. Force a sufficiently long prompt/response fixture to exercise a full 256-token backward microbatch at the 8192-context configuration; do not infer full-microbatch fit from tiny completions.
- [ ] If allocation fails, diagnose the actual graph/buffer failure before trying 128 and 64 explicitly. Confirm at least two groups complete, loss/gradient inputs are finite, nonzero rewards update the adapter, and peak device memory stays below 16384 MiB. Report context, microbatch, LoRA targets, group size, measured peak, and throughput.
- [ ] Update README and Pulsar V100 documentation with the measured working invocation and reward-provider boundary. Run `git diff --check`, review the full task diff against the initial user diff, and verify active source/help no longer references phase offload.
- [ ] If external installation is desired, prepare the exact launcher first and request filesystem approval for copying it to `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar`; do not make long training start automatically as part of installation.

## Completion Evidence

Report the changed trainer behavior, numerical test results, SFT regression result, actual Pulsar S two-group run result, measured VRAM peak/configuration, checkpoint limitations, and production reward integration status. Do not claim the 16 GiB requirement is met based solely on a successful build or a memory estimate.

## Self-Review

The six tasks cover phase removal, exact objective/gradient, response boundaries and accumulation, fixed-policy scoring, raw reward handling, reference-preserving resume, driver updates, and measured hardware acceptance. New low-level operations are limited to CPU and CUDA; other backends are outside the first measured target and must not claim native support. No task authorizes commits, spawning agents, or external writes without the corresponding user selection/approval.
