# Global CE/KL RCO and GSQ integration

Status: design approved by the user on 2026-10-09; implementation plan review is pending.

## Outcome and authorization

Produce a Gemma 4 E2B mixed-precision GGUF using global model CE or teacher KL gradients, exact packed GSQ/PTQ candidates, and an actual file-byte budget. Preserve the existing GSQ implementation and its numerical checks. A reconstruction proxy is a separate mode and cannot satisfy global RCO acceptance.

Source: /home/user/aikar-engine/tools/llama-optimize.
Experiment artifacts: /mnt/openwebui/AIKAR/GSQ-RCO.
The user permits service interruption. Stop the GPU service only when a GPU experiment needs its memory, record its launch configuration first, and restore and health-check it after experiments, including failures. Do not overwrite the source BF16 model, commit, push, or submit upstream contributions.

## Observed state

- Existing uncommitted PoC implements GSQ forward/backward, Lion, a full-Hessian GPTQ Python initializer, direct Q2_0 packing, RCO projection/retraction/first-moment transport, GGML Adam with zero decay, and exact-byte discrete assignment.
- `rco_proxy` independently minimizes captured linear reconstruction MSE. It does not backpropagate global CE/KL.
- Its candidate manifest imports exact GSQ Q2_0 bytes, but creation and search are coupled, all candidate dequantizations are resident, and only selected dense 2D tensors are supported.
- Its evaluation path permits at most 512 input tokens. Prior results use 45 predicted tokens and are smoke results only.
- BF16 GGUF: 9,311,286,336 bytes, 601 tensors, 35 blocks, 4,647,450,147 stored parameters. Per-layer token embedding alone is 4,697,620,480 bytes. E2B is not a 2 GB checkpoint.
- V100 has 16,384 MiB; the current server occupies about 16,047 MiB. Approximately 77 GiB host RAM is available. Experiment filesystem currently has about 16 GiB free; avoid seven full simultaneous model copies.
- Local pinned references: GSQ `03fc16484c369e3127225615d5e03e8d3a6043e3`, RCO `9a1e09c07d468109cbe60a1b87d5036034a79d10`.

## Approach selection

Recommend a native GGML global training adapter using the existing model graph, scheduler, sparse CE, backward builder and activation recomputation. Precision controls and optimizer states remain small FP32 host arrays; manifold calculations retain existing FP64 host behavior. Candidate storage and final output use the existing GGUF codecs and writer. Any additional internal graph hook must be opt-in and leave ordinary inference unchanged.

Alternatives: a PyTorch graph reconstructed from GGUF would provide convenient autograd but duplicate Gemma forward semantics; an HF autograd bridge would require a matching HF checkpoint, exact tensor/tokenizer mapping, and another model-loading path. Neither is the default because the native engine already has training infrastructure and GGUF is the available source.

Implementation must first prove that the native adapter can retain every required dependency. If this requires a larger core redesign, report the exact blocker and obtain design review rather than silently detaching gradients or changing the objective.

## Global training contract

1. Tokenize once with the source GGUF vocabulary. Cache token IDs, window boundaries, masks and hashes; both teacher and student consume identical arrays.
2. Run the BF16 source teacher separately with no backward state. Cache full-vocabulary reference distributions by window on disk or host RAM. Teacher log-probability storage precision is recorded; FP32 is the correctness reference, and any lower-precision cache must be measured against it.
3. Student forward follows native Gemma semantics including per-layer embedding/projection, SWA/global masks, shared K/V, normalization, softcap and tied output embedding. Frozen BF16 tensors are cast or streamed where required by backward type support.
4. CE is next-token mean negative log-likelihood over scored tokens. KL is teacher-to-student full-vocabulary KL over the same shifted positions. Expose `ce` and `kl` objectives separately as upstream does; do not invent an unreported weighted combination or top-k approximation.
5. For each step, replay seeded Gumbel noise and compute soft probabilities at the upstream exponential temperature schedule. Solve the exact-byte multiple-choice assignment for hard forward weights. STE uses the softmax of those same perturbed logits.
6. Backpropagate model loss to candidate probabilities and then precision logits. The per-option derivative contracts the global weight gradient with the candidate weight, or an equivalent candidate delta from a fixed reference. Release each candidate after its contraction.
7. Project the accumulated precision gradient using the current budget normal, perform Adam with upstream hyperparameters and zero decay, retract expected byte cost to the target, then project Adam's first moment onto the new tangent plane. Do not transport the second moment.
8. Final assignment maximizes the learned discrete selection scores under the total-byte cap. This solves the assignment score objective, not an exact discrete minimization of model CE/KL.

The teacher can retain BF16 source weights while V100 computation uses FP16, with FP32 reductions. Report compute/cache precision explicitly. CPU validation uses FP32 where necessary. Quantization codecs must not be confused with optimizer or computation precision.

## Memory and graph correctness

Keep packed candidates on disk or memory-mapped CPU storage. Dequantize only the active tensor or tile. Do not allocate optimizer state for frozen full-model weights. Reuse existing recomputation and weight-streaming infrastructure where its semantics fit this graph.

The training attention graph must keep differentiable current-window K/V dependencies, including reused K/V from earlier layers. Inference KV cache writes can terminate gradients and are not accepted as an equivalent training graph without numerical proof. Chunking vocabulary reductions preserves global normalization; chunking sequence computation must not truncate causal dependencies or backward paths within an evaluation/training window.

Start with one complete short window, measure memory, then increase context within the measured budget. Windowed full-model loss retains upstream calibration semantics. Layerwise loss and truncated layer backpropagation remain explicit proxy modes only.

## Candidate store and GGUF accounting

Separate candidate generation/import, search and materialization. Each candidate records source model identity, tensor name/shape, codec, producer (PTQ, IMatrix PTQ, GSQ-RTN, GSQ-GPTQ), packed payload path/hash, raw bytes, aligned bytes and supported operation kind.

Import GSQ packed payloads verbatim. Training dequantizes those bytes; output copies the selected bytes verbatim. Never requantize optimized GSQ weights. Preserve multiple producers at the same codec/byte cost. Validate dimensions and codec compatibility before training. Extend existing core validation rather than rewriting GSQ.

Total size is destination GGUF metadata plus every aligned tensor payload, including fixed tensors. Compute destination metadata with the real writer, subtract fixed bytes from the configured total-file budget, and reject an infeasible cap before training. With fixed embeddings, budgets below the fixed-size lower bound are infeasible and must be reported.

Current sparse DP prunes dominated score states and has a one-million-state cap. Retain exact feasibility; do not silently round byte costs if the cap is exceeded. Report solver failure explicitly or introduce a separately reviewed exact solver. Report selected payload bytes, predicted file bytes, actual bytes and unused capacity. Retraction equality applies to expected cost; discrete file size is bounded by the cap and need not equal it.

Write a new partial output, verify metadata and candidate payload hashes, check exact file size before publication, then reload with native inference. Failures cannot overwrite the source or pre-existing outputs. Process one comparison output at a time if disk space is insufficient.

## Data findings and split

Found /mnt/openwebui/AIKAR/Lumen-Lumen3.txt: 50,337 bytes, SHA256 `c97214c3fb18a6032c9246b17467f667b0d0550fc0a61904558e8d3c70f1f6a4`, 17,608 tokens with `--no-bos --no-escape` and special parsing enabled. It has one user turn and one long model turn; 21 tool-call openings, 23 tool-response openings, and 21 tool-response closings. Its apparent unmatched delimiters require inspection before structured conversion; embedded quoted tool text may contain delimiters.

Gemma vocabulary contains `<bos>`, `<|turn>`, `<turn|>`, and tool-call/response delimiters. GGUF has no chat template. Token recognition alone does not validate chat formatting. Default CLI escape handling changes this source, and automatic BOS insertion duplicates its existing BOS.

Preserve the original. Use a separately recorded raw-token corpus first, without injecting a Lumen system template or assuming an absent Gemma chat template. A sequential disjoint split within this single conversation is suitable for a restricted experiment, not independent-conversation generalization. Use the first 12,000 tokens for calibration, discard 512 boundary tokens, and use the remaining 5,096 as held-out data. Windows cannot cross split boundaries. Report actual scored tokens after window packing and next-token shifting; do not count discarded tokens. Additional independent data can expand evaluation later without changing the fixed initial split.

## Validation and completion gates

A. Rebuild current code and rerun `test_reference.py` against pinned GSQ/RCO plus existing quantization tests. Extend existing self-test/test scripts for invalid budgets, duplicate candidate identity and packing preservation; do not add files under tests/.

B. Validate CE and KL logits derivatives and smooth precision interpolation with finite differences on a small complete graph. Use deterministic noise, FP64 reference where supported, several step sizes and relative/absolute errors. Hard STE is a surrogate derivative: finite differences of a discontinuous hard assignment cannot validate it. Verify its VJP against the smooth softmax surrogate at the hard-forward evaluation point, separately from smooth-forward finite differences.

C. On Gemma, compare the training adapter's logits and CE/KL with native inference on identical tokens before quantization. Check precision gradients in early, middle and late layers, including shared-KV effects. Require finite nonzero global gradients, a deterministic candidate sensitivity test and a replayable optimization trace; report failures rather than declaring synthetic success.

D. Generate PTQ and GSQ candidate stores, run global RCO, materialize a byte-feasible GGUF, verify imported GSQ payload preservation and reload it. Log objective, raw/projected gradient norms, tangent residual, retraction residual, temperature, RNG state and selected costs per step. Save resumable small optimizer checkpoints.

E. Only after C and D pass, compare BF16, PTQ, IMatrix PTQ, GSQ-RTN, GSQ-GPTQ, RCO and GSQ+RCO on identical held-out IDs/context/masks. BF16 is the quality reference; report actual size mismatch for fixed-codec baselines rather than implying equal size. Q2_0 currently ignores IMatrix values, so it cannot be advertised as a distinct IMatrix baseline without an effective codec.

Report token-weighted CE, exp(CE) PPL, full teacher KL, reconstruction MSE with its precise definition, GGUF bytes, training wall time, process peak RSS, per-process and total-device VRAM, prefill and decode throughput. Do not mix other process VRAM with this experiment's allocations. Preserve raw reports and model identities so sequentially deleted comparison outputs can be reproduced.

## Files and extension boundary

Expected implementation locations: tools/llama-optimize/main.cpp, optimize.h, optimize.cpp, CMakeLists.txt, README.md, test_reference.py, and experiment runner. Add focused tool-local files only if they reduce coupling. Minimal opt-in internal graph/training hooks may affect src/llama-context.*, src/llama-graph.* and GGML optimizer interfaces; explain each changed interface before editing it.

Candidate-store grouping must carry operation kind and expert identity independently of Gemma layer numbers. Define separate dense matmul and routed expert matmul adapters; reject unsupported routed training explicitly. Local Lumen 3.5 Pulsar S metadata reports 23,091,066,806 stored parameters, 116 experts, top-8 routing, 60 expert tensors and a 13,121,388,640-byte GGUF. Full Lumen experiments are deferred until Gemma acceptance passes.

## Current completion status

No global RCO implementation, new trained GGUF, global precision-gradient evidence, training loss trace or seven-arm quality evaluation has been produced in this audit. Existing proxy outputs cannot fill those requirements. The next stage is user review of this design, followed by a concrete implementation plan and inline execution review as required by the architectural workflow.
