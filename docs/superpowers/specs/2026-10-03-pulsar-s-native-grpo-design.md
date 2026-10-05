# Native GRPO for Pulsar S on a 16 GiB V100

## Agreed scope

Replace reward-weighted SFT in `llama-finetune-qlora --grpo-mode` with outcome-supervised GRPO. Remove GRPO phase offload. Target `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf` and the single-GPU training configuration in `stage1.sh`.

The user approved native GRPO and requested deletion of phase offload. The reward provider is not yet specified. Keep the stdin/stdout reward interface and accept finite raw scores; do not invent a production reward function. The example heuristic is only a smoke-test provider.

Existing uncommitted CUDA, context, test, and documentation changes belong to the user and must be preserved. No commits, pushes, or upstream submissions are authorized.

## Current defects

The existing loop samples response text, tokenizes that text again, packs responses into SFT windows, averages rewards over windows, duplicates samples to fill a context window, and calls the cross-entropy optimizer. The Python example maps normalized rewards into positive weights. This does not implement the GRPO objective and loses response boundaries and negative advantages.

## Implementation approach

Extend the existing native optimizer and sparse target loss infrastructure. Keep the GGUF base, LoRA loading, CUDA training kernels, activation recomputation, KV training, and checkpoint infrastructure. Do not introduce a Transformers trainer or convert the model.

A separate Python/Transformers trainer would require a different model representation and a new memory plan. Replacing rewards with signed cross-entropy weights would still omit the probability ratio and clipping. Neither approach meets this request.

## One resident model and context

Load the quantized base onto CUDA0 once. Keep one persistent context, one trainable LoRA adapter, and its optimizer state. Generate responses sequentially using that context. Clear KV between independent responses and before each teacher-forced sequence.

Capture the initial LoRA A/B tensors in host memory as the frozen reference. For reference scoring, synchronize pending work, snapshot current trainable A/B values to host, copy the frozen reference values into the same device tensors, score responses without backward or optimizer updates, and restore current A/B values. Restore the current adapter on every exit path. Keep the adapter attached with the same scale and clear KV and graph reuse state when changing its values.

For a new adapter, the reference is the initial zero-B adapter. For `--lora`, it is the supplied adapter at GRPO initialization. GRPO checkpoints must preserve the original reference snapshot and objective settings in a companion file and reject incomplete or mismatched resume inputs. Restarting from a checkpoint must not silently redefine the reference. Exact restoration of Adam moments remains outside the existing checkpoint guarantee and must be reported explicitly.

No second model, second KV cache, GPU reference adapter, or per-step model reload is allowed. Scheduler lifetime remains stable across rollout and training. Existing SFT/evaluation scheduler recreation has separate callers and must be preserved where needed.

## Rollout records and sampling

Each response record contains the original prompt token IDs, sampled completion token IDs, old-policy log-probabilities, frozen-reference log-probabilities, raw reward, advantage, and terminal/truncation status. The text sent to the reward provider is decoded from these IDs; it is never tokenized back into training labels.

Require at least two generations, positive finite temperature, positive token limits, and valid context and batch settings. Sample from the complete categorical distribution at the configured temperature. Use the same temperature in differentiable current-policy scoring and in old/reference scoring so the probability ratio describes the sampled policy. Do not apply top-k or top-p in this implementation.

Include a sampled end-of-generation token in the completion and its loss, even when it is omitted from the displayed response text. Use the vocabulary EOG predicate. Keep the first completion token aligned with the last prompt position. Prompt length plus completion input length must fit the context. Return an actionable error for an oversized prompt; do not silently truncate it or shift the context.

Old log-probabilities are immutable throughout all update iterations on a response group. Reference scores are immutable for the run. Check rollout scoring against teacher-forced training-mode scoring before updates, including LoRA QAT behavior, since numerical or execution-mode differences can invalidate an intended on-policy ratio.

## Objective

For each prompt group of size G, compute population mean and standard deviation of finite raw rewards. Set A_i = (r_i - mean) / (std + 1e-8). Equal rewards produce exactly zero advantages. Do not shift or clip advantages to a positive interval.

For each completion token t:

    log_ratio = log_pi_current(i,t) - log_pi_old(i,t)
    ratio = exp(log_ratio)
    surrogate = min(ratio * A_i, clamp(ratio, 1-epsilon, 1+epsilon) * A_i)
    delta = log_pi_reference(i,t) - log_pi_current(i,t)
    kl = exp(delta) - delta - 1
    loss(i,t) = -surrogate + beta * kl

The group loss is `(1/G) * sum_i ((1/completion_length_i) * sum_t loss(i,t))`. Prompt and padding positions have zero weight. Sequence boundaries must be retained; do not average rewards across packed windows or replicate responses to fill a window. KL remains active for equal-reward groups when beta is positive. With beta zero and equal rewards, skip the optimizer update, including weight decay and moment drift, and report the skipped group.

Support clipping epsilon, nonnegative KL beta, and update iterations per group. Proposed defaults are epsilon 0.2, beta 0.04, and one update iteration. Multiple iterations reuse the same old-policy and reference scores. Reject non-finite loss or gradient inputs with an error; do not silently clamp the objective's exponential into a different algorithm.

## Loss graph and optimizer integration

Add a GRPO-specific loss mode with sparse target IDs and per-output old/reference log-probabilities, advantages, and normalization weights. Reuse existing sparse cross-entropy reduction patterns to compute stable target log-softmax without allocating dense labels or another vocabulary-sized probability tensor.

Implement CPU and CUDA forward/backward support in the existing loss-related files. Backward must differentiate the clipped surrogate and KL with respect to current log-probability and then logits, including the temperature factor. Old scores, reference scores, advantages, and normalization weights have no gradients. Preserve activation recomputation and existing quantized backward optimizations.

Accumulate gradients for all response microbatches before one optimizer update per group iteration. Account for partial microbatches explicitly. Do not divide the objective again by the SFT optimizer period. Use zero-weight sentinels only where the existing graph requires an output, never as a real completion label. Prompt-only work must not trigger an optimizer step.

## Memory target

The first measured configuration follows stage1:

- V100 CUDA0, split mode none, all model layers and experts on the device.
- Pulsar S LD Q4_0_XL base and the existing Q4_0_XL tensor type recipe.
- LoRA rank 16, alpha 8, targets `attn_output.weight`, LoRA QAT q4_0.
- F16 K/V training cache, FlashAttention on, activation recomputation on.
- Context and batch 8192, microbatch 256, seed 3407.
- Sequential response processing; initial group size 4 and maximum completion 512.

Only compact token records and adapter/reference snapshots accumulate on the host. Group size must not enlarge the GPU training microbatch or create a batched rollout KV cache. Compare peak device memory with stage1 and keep it below the device's 16384 MiB capacity with successful allocation and update execution.

If microbatch 256 cannot fit the additional GRPO graph, measure 128 and 64 and publish the measured working setting. Do not claim an 8192/256 configuration fits without evidence. Do not silently adjust user-supplied settings.

## Phase-offload removal

Remove `grpo_phase_offload`, its CLI option, temporary rollout runtime, adapter-map copying, and GRPO calls to scheduler suspend/resume. Remove suspend/resume entry points if they have no remaining consumers. Preserve scheduler helpers used by existing SFT/evaluation memory management and preserve their tests. Mark old phase-offload design/plan documents as superseded so they do not prescribe reintroducing the deleted feature.

## Driver and execution

Update `grpo_example.py` to send raw rewards and expose objective and memory settings. Add a Pulsar S launch example using stage1's model, recipe, LoRA targets, and GPU settings. Keep model and output paths configurable. Record reward, reward standard deviation, advantage, policy loss, KL, clip fraction, completion lengths, skipped groups, and optimizer iteration in logs.

The external Pulsar directory is outside the writable sandbox roots. Prepare launch artifacts in the repository first; installation into that directory requires filesystem approval. Long training is not required for validation: execute short deterministic GPU smoke tests and save their artifacts separately from the user's stage checkpoints.

## Verification and acceptance

Reuse existing test files and test runners.

1. Check loss and logits gradients against an independent double-precision implementation and finite differences. Cover positive/negative advantages, both clipped branches, non-unit temperature, KL gradients, masked targets, unequal completion lengths, and partial microbatches.
2. Verify CPU and CUDA loss/gradient agreement within explicit tolerances.
3. Verify prompt/completion alignment, terminal tokens, group normalization, and original token preservation.
4. Verify an equal-reward beta-zero group does not change parameters or optimizer iteration, and a nonzero group updates once per iteration.
5. Verify multiple responses and update iterations use fixed old/reference scores and the reference adapter stays unchanged.
6. Build the trainer and run existing relevant optimizer/QLoRA checks, including an SFT regression smoke test.
7. Run Pulsar S on CUDA0 using the stage1 memory settings. Complete at least two GRPO groups, check finite metrics and parameter updates, and sample peak VRAM through generation, reference scoring, and backward.
8. Confirm the removed phase-offload option is rejected and no GRPO phase-offload execution remains.
9. Verify checkpoint and reference-snapshot resume validation. Report optimizer restart semantics accurately.

The deliverable is a built and tested native GRPO trainer plus a measured Pulsar S launch configuration. A code-only change or an unmeasured memory estimate does not satisfy the 16 GiB requirement.
