# Pulsar paged KV Attention Matching experiment

Date: 2026-10-11
Status: Canceled at the user's request on 2026-10-11 due to concern about AM cost. No compaction implementation, model-level timing or quality result exists. Resume the approved codec/capacity design with serving overlap and expert-cache assessment; retain this document for reference.

## Purpose and evidence

The user requested consideration and, if compatible, implementation of [Fast KV Compaction via Attention Matching, v2](https://arxiv.org/html/2602.16284v2). It reduces KV entry count, whereas Q4/IQ4 reduce bytes per entry. The paper matches attention outputs and mass using selected keys, fitted values and a per-entry bias. It keeps original logical positions and leaves SWA unchanged in its Gemma3 experiment. It did not evaluate quantization; reported GPU timings do not establish V100 latency or Pulsar quality.

The [official code](https://github.com/adamzweiger/compaction/tree/2020f6d37940fe013e7e7f1ebf409e5d107c5670) was inspected at that commit. Relevant files are `compaction/algorithms/highest_attention_keys.py`, `compaction/algorithms/base.py`, `models/cache.py`, and `models/gemma3/modeling_gemma3.py`. Its configurable score selection, stable exponentials, nonnegative weight fitting and value regression provide the reference. Some defaults differ from paper variants: the key score defaults to max, and zero NNLS iterations use a clamped unconstrained fit. Select RMS and validate constrained residuals explicitly. Its MIT license requires retaining attribution if code is adapted.

## Compatibility in this engine

Pulsar has separate GQA K/V, not MLA, so the representation can be supported. Current paged attention computes scaled QK and FP32 online softmax. It has no fitted bias input, compact-prefix storage, per-head selected-key metadata or compaction transaction. Adding these changes the cache contract and requires a separate implementation unit after the quantization foundation.

Orthogonal rotation preserves dot products before quantization: `(Q Hk) (K Hk)^T = Q K^T`. Value regression can use the V-rotated basis and attention output can use the existing inverse transform. Quantization adds error to both keys and fitted values. This algebra establishes compatibility, not quality equivalence.

Implement an opt-in experiment first, with Full layers only and SWA unchanged. Start with 2x reduction of the selected old region, one sequence, a text-only immutable completed prefix, uniform retained length across Full layers/heads, and an exact recent tail. Report the overall retained fraction including protected tokens. Different heads may select different keys; packed row indices do not represent their original token positions. OMP, nonuniform head budgets, multimodal compaction and multi-slot deployment require later evidence.

## Representation and execution contract

- Keep original logical position/seen-token counters separate from packed prefix length. Appended tokens use original positions for RoPE and causal/SWA masks. Do not renumber keys or run RoPE a second time on stored rotated keys.
- Store Full prefix K/V in the existing page payload format, with F32 beta per KV head/entry and explicit packed-prefix length/end position. Exact suffix entries use beta zero. The prefix descriptor must be per sequence/layer; it cannot be inferred from ordinary logical page masks.
- Iterate packed prefix rows directly, then ordinary exact suffix pages. Do not scan all removed logical positions, and do not reconstruct a dense attention mask. Add beta after QK scaling and before the online softmax maximum/exp; shared GQA queries use their owning KV head's beta.
- Merge prefix/suffix and split partials with the same stable maximum, exponential mass and weighted-value accumulators. Independently normalizing the two outputs and averaging them is incorrect.
- Compacted entries summarize a completed interval. Allow queries only after that interval; reject prefix edits, truncation inside the interval and incompatible partial-state operations. The first experiment must not modify an actively executing batch or shared pages.
- Server prompt reuse must validate the full immutable original prefix and its boundary. A cache match ending inside a compacted interval must fall back to re-prefill rather than reuse synthetic entries as ordinary token KV. Do not enable idle RAM caching or checkpoints for the experiment until its serialized descriptor is supported.
- Reuse sequence ownership/refcounts for immutable prefix descriptors before any multi-slot rollout. Save/load needs a distinct version, all prefix boundaries/biases/policies and alias rules. State from the codec-only design must remain identifiable.

## Reference capture and bounded compaction

Use the existing eval callback to sample post-RoPE/post-K-rotation Full queries, together with owning KV heads and exact positions. Reserve disjoint calibration and validation query sets. Context-prefill queries are a cheaper initial probe; repeat-prefill is a later quality control, not an automatic second 262K prefill.

Use installed NumPy/SciPy/PyTorch in an offline experimental worker; do not add Python or a linear algebra library to the serving binary. The installed PyTorch binary includes SM70, but GPU availability, memory use and runtime kernels still require a real check. CPU fitting is an available correctness reference. Export only one bounded KV chunk/head at a time and keep original packed snapshots in host memory for recovery; never materialize all layers' 262K F32 KV on GPU.

Initial probe bounds: chunks of 2048 original rows, retained lengths 1024 then 512, up to 2048 reference queries per head, one Full head at a time. Candidate/solver GPU scratch has a hard 64 MiB ceiling; account for CUDA context/workspace separately. Query reservoir is at most 40 MiB host memory for 10 Full KV heads at D512/F32. Reject rather than allocate an unbounded score matrix. Snapshot/staged packed payloads use a separate 4 GiB host cap.

For each chunk: obtain original attention targets from actual stored/dequantized cache; select keys by RMS attention; encode/decode selected keys using the candidate cache codec; fit nonnegative mass weights using those stored keys and stable row offsets; fit values in the selected rotation basis; encode/decode fitted values; validate held-out mass and attention output. Use constrained residual checks and regularized/rank-aware solves; never explicitly invert normal equations. Reject nonfinite/ill-conditioned fits or a failed quality gate before replacing any source pages.

The initial experiment uses a disposable checkout and a throwaway capture/replay executable under `/tmp`. A minimal biased-attention patch in that checkout is sufficient to test actual continuation logits; it is not the integrated packed-prefix implementation. Pack synthetic prefix rows into ordinary low logical table indices, preserve a nonempty exact tail at original positions, and attach experiment-only beta tensors. This sparse-table probe can establish quality but does not establish the integrated kernel's speed or allocator savings. Do not persist/import its synthetic state as ordinary production state. Restore original snapshots between trials and delete the bias attachment before replaying a control.

The experimental patch and executable are implemented only after the written design and implementation plan reviews. Reuse the backend test file; no new test file is required. Production does not depend on the external worker. Native compact-prefix iteration, versioned persistence, streaming and multi-slot integration are conditional on the experiment.

## 262K capacity meaning

This experiment can maintain 262144 logical positions with fewer than 262144 physical KV entries; it does not provide exact retention of every entry. The codec-only 8192-page route separately targets all 262144 entries.

At 4096 pages x 32, the Full pool remains allocated at 1320 MiB for Q8_KV or 720 MiB for Q4/IQ4. F32 beta for all 131072 rows across 10 Full KV heads is 5 MiB. Compaction frees live pages but does not shrink an already allocated pool; memory savings require configuring a smaller pool or a future dynamic allocator.

Reserve an 8192-token exact tail and at least 1024 rows for pending work. At logical length 262144, retained old entries must fit within 121856 of 253952 older tokens, approximately 47.98%, before page rounding and fixed-prefix exclusions. A nominal 50% ratio alone does not fit this reserve. Choose the budget from actual free pages and protected tokens; trigger compaction before pool exhaustion and process incoming prefill in bounded chunks.

The first implementation establishes correctness at short lengths before streaming to 128K/262K. Streaming compaction changes later prefill states; compare its task quality independently from one-shot compaction. Current full-context prefill cannot be run first and then compacted if it already exceeds the pool.

## Quality and performance gates

Compare matched paged Q8 without compaction, Q4/IQ4 without compaction, AM with Q8, and AM with each selected Hadamard/4-bit codec. Use the same logical prompt/continuation and report the original physical-budget limits honestly. An exact short-context F16 control helps separate codec and compaction errors.

Reuse the provisional task-accuracy gate in the companion codec design and add tests for future-token concatenation, beta-zero equivalence, compact-prefix versus dense biased oracle, split merging, logical positions, repeated compaction and failed-import rollback. Mass/output reconstruction alone cannot establish application quality. A calibration query must not appear in the held-out score.

Record initial PP/TTFT, query collection time, compaction wall time, host transfers, decode speed, memory peak and multi-turn amortized latency. Estimate break-even from measured one-time compaction cost divided by per-token decode time saved; if there is no positive saving, there is no decode break-even. This design offers no promise that compaction accelerates the first prefill.

Rollout order: offline mathematical/codec probe; native compact-prefix reader and persistence; one-sequence inference quality; bounded streaming capacity; shared-prefix/multi-slot lifecycle. Keep the feature opt-in unless all relevant gates pass. If the initial probe fails quality or feasibility, retain its results and the independent 4-bit/262K improvements, and do not enable Attention Matching in serving.
