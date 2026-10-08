# REAP pruning audit - 2026-10-08

This is the historical pre-fix audit snapshot. The follow-up fixes and current behavior are documented in [Pruning bug fixes](prune-bug-fixes-20261008.md).

## Verdict

- Historical REAP implementation correct: **no**. The conditional-average formula, gate, norm and precision are correct, but strided Top-K IDs were read as contiguous. Contributions were assigned to the wrong expert after the first token of each batch.
- Comparison of the intended importance metrics fair: **no**. Both historical outputs remove the same count and use the same baseline logits, but REAP statistics are corrupted and the legacy score is identically zero for this scaled Gemma 4 model.
- Current stale-report/cache-from-another-model evidence: **not found**. Report provenance, cache compatibility and filename bugs were found separately.
- Genuine superiority of the intended aikar metric over correct REAP: **not established**. The reported quality numbers compare the two actual masks. They do not establish this algorithmic conclusion.

The audit fixes ID collection and diagnostic/reporting issues. It does not change either scoring formula, add frequency weighting to REAP, or silently replace the legacy score. Full calibration and KL/PPL reruns are still needed to evaluate corrected REAP.

## Exact formulas and forward tensors

For collector routes j, activation vector y_j and effective gate w_j:

```
REAP_e = reap_sum[e] / reap_count[e], or 0 if count is 0
reap_sum[e] = sum_{j: id_j=e} w_j * sqrt(sum_d y_j[d]^2)
reap_count[e] = number of those output measurements
```

After the stride fix, id_j is the actual selected expert and the validated measurement count equals `reap_selection_count`. The denominator is not total calibration tokens. Each selected route counts even when its gate or norm is zero. Top-K contains distinct expert IDs, so each route corresponds to one selection by a token.

For Gemma 4, router logits are:

```
z(t) = W_router * (router_scale * RMSNorm(attn_out(t)) / sqrt(2816))
p_e(t) = softmax_over_128_experts(z(t))_e
I(t) = Top8(p(t))
g_e(t) = p_e(t) / max(sum_{i in I(t)} p_i(t), 6.103515625e-5), e in I(t)
```

There is no selection bias, routing correction/grouping, input bias, or additional router weight scaling on this model (`w_scale=1`). Soft masks set disabled logits to negative infinity. REAP uses the `ffn_moe_weights_norm` DIV output used in the actual forward pass. It does not use logits or pre-TopK probabilities.

`y_j` is `ffn_moe_down_scaled-{layer}`, shape `[2816, 8, batch_tokens]`, after GELU/gated expert FFN and down projection including the expert's down scale, before `ggml_mul(experts, weights)`. It is not the intermediate width-704 activation, weighted output, residual, or post-aggregation RMSNorm. The same tensor receives the `ffn_moe_down` callback name and then is renamed `ffn_moe_down_scaled`, which matters for the legacy collector.

The exact existing aikar (`router-output`) formula is:

```
A_e = weighted_output_sum[e] / selection_count[e], or 0 if count is 0
weighted_output_sum[e] += gate * output_norm
```

Only the `ffn_moe_down-` name updates its numerator. It has no variance, layer normalization, damage estimation, balancing, calibration loss weighting, or explicit global frequency factor. It is not `P(selected) * E[gate * norm | selected]`. If output capture works and selection counts have no view duplicates, it is the same conditional average as REAP. With two routing callbacks per selection and one output callback, it becomes approximately REAP/2, which alone does not change ordering.

For this actual scaled Gemma model, the down tensor name is overwritten before execution. Consequently every one of 3,840 legacy output/weighted sums and scores in the full cache is zero. Ties are broken by original expert ID, and all layers prune IDs 0-11. This also occurs in HEAD's original collector, so it is not a new scoring change introduced by this audit. The audit warns about zero scores and preserves this legacy behavior.

## Routing defect and deterministic real-model trace

`ggml_argsort_top_k` returns a view of the full argsort. Here its shape is `[8,26]`, strides `[4,512,13312]` bytes. The second token starts 128 IDs after the first, not 8 IDs after it. The old flat copy read lower-ranked experts from the first token as if they were later tokens' selected experts.

Example on the actual model:

```
token 1 wrong recorded IDs: [15,83,49,111,41,89,19,27]
token 1 actual Top8 IDs:    [105,60,41,111,17,49,124,19]
```

This explains the suspicious historical full-cache counts: all 3,840 experts have between 49,369 and 49,718 measurements, despite those not being trustworthy per-expert selection counts. The count-equality check could not catch this: both counters used the same wrong IDs. Sums alone cannot reconstruct the original routes or repair that cache.

A temporary standalone callback harness, not normal production logging, captured actual logits, actual post-softmax values, Top-K, normalized gates and pre/post-gate outputs on 26 deterministic CPU calibration tokens. Independent calculations verified all 208 routed rows in layer 0. First-token example:

```
layer=0 token=0 expert=103
router_logit=3.43051815032959
softmax_probability=0.06398146415206775
selected_probability=0.06398146599531174
final_gate=0.24339544773101807
pre_gate_output_norm=3.5553914641961737
REAP_contribution=0.8653660972870676
weighted_output_norm=0.8653660967998998
```

Maximum absolute difference between `gate * pre_gate_norm` and actual weighted norm was `1.0767887914653329e-8`. There is no gate squaring. The corrected small calibration has 70 zero-count experts in layer 0 and selected counts ranging from 0 to 15, with exactly 208 total selections. All 30 layers pass the count/measurement check. The exported three-token trace is in `verified-tokens.csv` and `verified-tokens.json`, including full router arrays.

## Precision, rare experts and overhead

FP32/FP16/BF16 activations are converted to double before squaring, norm reduction and accumulation. Scalar sums use double; counts use uint64_t. REAP selection and measurement count overflow, invalid/negative gates, non-finite activations/norms and accumulated overflow raise errors. FP32 tiny values are squared in double, avoiding FP32 reduction underflow. Information already lost to the model's low-precision inference cannot be recovered by profiling. Double addition can lose extremely small contributions on sufficiently enormous runs; no compensated summation is used. Legacy selection-count/probability accumulators retain their existing unchecked overflow behavior.

Zero-count score is 0, with expert ID resolving zero-score ties. A once-selected expert gets its single contribution as its score. Rare experts can therefore have large, high-variance scores without confidence correction or smoothing. This conditional averaging policy is unchanged.

Additional REAP persistent scalars are 32 bytes per expert, about 120 KiB for 30x128 experts; norm scratch is 16 KiB. Existing routing IDs/gates remain batch-sized. GPU ID collection retains one full-span download, then compacts rows using the actual stride. No expert activations are retained in production and no extra forward pass is added. Norm work is O(tokens x TopK x output_width) per layer, with existing callback synchronization/device reads. No isolated before/after performance benchmark was run.

Shared dense MLP experts are neither ranked nor removed. Supported pruning remains Gemma 4 26B A4B with homogeneous routed-expert counts; hard rewriting remains restricted to Q4_0 expert weights. Generic graph variants with weight-before-FFN, post-normalization gate scaling, or separate output biases are not certified by this collector and other architectures are rejected by model inspection.

## Actual ratios, selections and correlations

At requested 10%, `floor(128 * .10 + 1e-12)=12` experts are removed per layer, subject to the same maximum layer ratio and at least Top-K survivors. Actual expert ratio is **9.375%**, not exactly 10%. Neither metric has different protected routed experts or excluded layers. Both hard outputs have 116 routed experts per layer, with the shared path unchanged.

| Layer | Original experts | REAP pruned | aikar pruned | Common pruned |
|---|---:|---:|---:|---:|
| 0 | 128 | 12 | 12 | 1 |
| 1 | 128 | 12 | 12 | 2 |
| 2 | 128 | 12 | 12 | 0 |
| 3 | 128 | 12 | 12 | 1 |
| 4 | 128 | 12 | 12 | 4 |
| 5 | 128 | 12 | 12 | 1 |
| 6 | 128 | 12 | 12 | 1 |
| 7 | 128 | 12 | 12 | 0 |
| 8 | 128 | 12 | 12 | 3 |
| 9 | 128 | 12 | 12 | 0 |
| 10 | 128 | 12 | 12 | 0 |
| 11 | 128 | 12 | 12 | 0 |
| 12 | 128 | 12 | 12 | 1 |
| 13 | 128 | 12 | 12 | 1 |
| 14 | 128 | 12 | 12 | 1 |
| 15 | 128 | 12 | 12 | 1 |
| 16 | 128 | 12 | 12 | 2 |
| 17 | 128 | 12 | 12 | 0 |
| 18 | 128 | 12 | 12 | 1 |
| 19 | 128 | 12 | 12 | 0 |
| 20 | 128 | 12 | 12 | 2 |
| 21 | 128 | 12 | 12 | 3 |
| 22 | 128 | 12 | 12 | 2 |
| 23 | 128 | 12 | 12 | 1 |
| 24 | 128 | 12 | 12 | 0 |
| 25 | 128 | 12 | 12 | 1 |
| 26 | 128 | 12 | 12 | 3 |
| 27 | 128 | 12 | 12 | 0 |
| 28 | 128 | 12 | 12 | 1 |
| 29 | 128 | 12 | 12 | 0 |

Across all layers: 360 removed by each, 33 common, 327 REAP-only and 327 aikar-only. Pruned overlap is **9.1667%**; Jaccard is **4.8035%**. Retained overlap is 3,153/3,480 = **90.6034%**.

Score-based Spearman correlation is **undefined**, because aikar scores are constant within every layer. The mean per-layer correlation of deterministic ordinal ranks including the expert-ID tie-breaker is `0.003754081975218208`; this is not a meaningful correlation between importance scores.

The 3,840-row `historical-experts.csv` exports requested scores, ordinal ranks, prune flags, frequency, gate/norm means and counts. `historical-layers.csv` lists the two exclusive removal sets for every layer. These faithfully describe the historical corrupted cache and actual outputs; they are not corrected importance measurements. Reconstructed masks from that cache match both reports exactly. An independent GGUF reader verified all 30 layers' surviving router rows and expert scales against the source and respective report mappings.

Both outputs are 13,121,388,640 bytes versus source 14,329,791,488 bytes: actual reduction 1,208,402,848 bytes (**8.4328%**). Reported removed tensor bytes are 1,208,403,360; the 512-byte difference is metadata/alignment overhead, not a pruning-count difference.

## Calibration and evaluation equivalence

The historical full cache identity is:

```
model: Lumen-3.1-Pulsar-Q4_0_XL.gguf
architecture: gemma4
model_sha256: 43441aa63748088031dc8d2da81db308fbc738242d2cecf4a7847857e82878c6
dataset: ../train_ko.jsonl
dataset_sha256: 68a18d726b942884672ca4c2a48e900825f1f7946d324a68e0c79cda93de13ca
total_tokens: 793610
processed_calibration_tokens: 793230
assistant_loss_tokens: 677280
context: 4096
metric: reap
requested_ratio: 0.10
actual_ratio: 0.09375
```

The current `prune.sh` specifies seed 3407, batch/ubatch 512, threads/dataset-threads 36 and GPU layers 30. The historical cache did not record these settings, so the script alone cannot prove the exact past invocation. Historical aikar calibration settings also cannot be recovered from its report. Its all-zero scores make its ID-based mask independent of the dataset.

For the same-cache diagnostic export, both scores use exactly the same cached model, tokens, order, norm/gate observations and ratio. That establishes count/mask reconstruction, not trustworthy REAP calibration. Dataset loading preserves record order; existing serial/parallel tokenization tests cover it. Calibration is teacher-forced, without sampling or stochastic token selection.

The user confirmed evaluations target `Lumen-3.5-Pulsar_S-LD-Q4_0_XL.gguf` (REAP) and `Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf` (aikar), using the same `Lumen-3.1-Pulsar-logits` file. Its header records context 512, vocabulary 262144 and 34 chunks; its 17,408 stored input token IDs hash to `77e9d9f994a12d305bacc40e4fa7fa9573f10866fec277a6f04b07219237dce9`. The KL evaluator reads those tokens from this file. Equal reported base PPL is consistent with that common baseline. Exact evaluated-model execution commands/KV precision/build settings were not retained in the supplied evidence, so their equivalence cannot be certified.

Reported REAP vs aikar results remain 16.815479 vs 15.729477 PPL and 0.609482 vs 0.518127 mean KL. The audit does not rerun or alter them. They show that the historical ID-based mask outperformed the historical corrupted REAP-derived mask on this evaluation. The ID-assignment bug invalidates attributing that result to faithful REAP. Even correct conditional contribution scores do not directly optimize KL/PPL, especially after Gemma's aggregate RMSNorm and interaction with other experts; that is a possible issue to evaluate after fixing instrumentation, not a demonstrated cause here.

## Report, state, path and cache findings

Historical reports contain only byte counts and original-to-new mappings. Filename, checkpoint identity, metric and calibration identity do not affect their bytes. Two different output filenames using the same mapping/shape can therefore legitimately produce the same report SHA256. This is identical content, not a cryptographic collision.

Current hashes:

```
3.1 report: 096fa143a6378e408ff8656972dcdfa529a0c78c1381fd2b58596ff2743fce2b
3.5 report: 69267ee3a62f4dc49b93ac232346b9f1b305880a64ac232f17ac18d56872a020
```

They have distinct inodes, are not symlinks, and match their current GGUF router/scales. Report objects and collectors are local and recreated per run. No reused global report, cache entry from another model, ignored metric CLI argument or current wrong model handle was found. The cache checks source model and expert-tensor SHA256, architecture/shapes, dataset hash and context. The model hash cache uses canonical path, inode/device, size, mtime and ctime. The original historical identical-report incident cannot be conclusively reconstructed from files that have since changed.

Verified independent bugs:

1. Strided Top-K IDs flattened incorrectly: corrupts REAP and frequency assignment.
2. Legacy scaled-output name prevents output capture: all-zero router-output ranking. Deliberately left unchanged, with a warning.
3. Cache compatibility omitted execution settings and collector version: incorrect statistics could be reused silently. New REAP/frequency cache use requires `routing_stats_version:2`; execution metadata is saved and known differences are rejected. Automatic incompatible REAP/frequency caches recalibrate, explicit old ones fail. Legacy migration remains limited to router-output.
4. Ratio filenames round to integral percent. `--ratios .10,.102` previously claimed two profiles but wrote one `profile-010.json`. The audit rejects collisions before writing profiles.
5. Reports lack model/metric/calibration provenance and did not check final stream flush. Both are corrected.
6. Source/output aliases could overwrite the source; a source matching a staging filename could even be deleted by failure cleanup. Final, staging, report and temporary paths are checked before opening files or arming cleanup.

Remaining limitations identified, not hidden:

- Model and report publication use two renames. A failure between them can leave a new GGUF with an older report. Output SHA256 in new reports makes this detectable; publication is not a two-file transaction.
- Cache keys still do not include a tokenized-input/template digest, backend version, full build identity or every inference environment setting. Dataset/model hashes and execution options improve provenance but are not a universal reproducibility guarantee.
- Old router-output caches remain usable for compatibility and may have incomplete execution provenance. Old profiles/models are not automatically repaired; regenerate them from corrected calibration before assessing REAP.

New hard reports include `model_path`, `model_sha256`, `model_architecture`, `metric`, requested/actual ratio and a deterministic calibration fingerprint (model/dataset hash, processed/evaluated counts, seed, context and batches). Unknown legacy fields are null. CLI publication adds final output path/hash and profile path/hash, reusing the already computed output hash. No timestamp is added.

## Patches, tests and next reproducible comparison

Audit changes are in:

- `common/moe-prune.h`, `common/moe-prune.cpp`: strided ID reader and optional calibration profile provenance; formulas/ranking unchanged.
- `tools/aikar-prune/main.cpp`: collector integration, cache version/settings checks, ratio collision rejection, zero-score warning, provenance and source-alias guards.
- `tools/aikar-prune/hard-prune.cpp`: report identity, flush verification and source-alias guards.
- `tests/test-moe-prune.cpp`: existing test file expanded; no new tests source file.
- `docs/moe-pruning.md`, this report and adjacent diagnostic exports.

Deterministic tests cover the hand example (1.6, 0.5, zero), ranks, lack of total-token normalization/gate squaring, rare-specialist versus global-contribution rank reversal, actual GGML softmax/Top-K normalization, strided multi-token ID views, FP16/BF16/FP32 norm paths, existing router-output ordering unchanged, provenance round-trip/report fields and source aliases. Separate CLI checks cover old-cache rejection, three-metric profile counts, known seed mismatch rejection and filename collisions. A real CPU calibration and independent real-token trace validate the fixed collection path. CPU and CUDA/Vulkan builds run the existing `test-moe-prune` target.

Use a fresh output directory/cache. Do not mark the old cache version as 2: its sums cannot be repaired.

```sh
build/bin/aikar-prune analyze \
  --model /mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar-Q4_0_XL.gguf \
  --dataset /mnt/openwebui/AIKAR/train_ko.jsonl \
  --metric reap --ratios .10 --output-dir reap-corrected \
  --ctx-size 4096 --batch-size 512 --ubatch-size 512 \
  --threads 36 --dataset-threads 36 --n-gpu-layers 30 --seed 3407 \
  --no-evaluate

for metric in reap router-output frequency; do
  build/bin/aikar-prune profiles \
    --importance-cache reap-corrected/importance-cache.json \
    --metric "$metric" --ratios .10 --output-dir "comparison-$metric"
done
```

This full GPU command requires sufficient free GPU memory. The audit used a separate small CPU calibration without interrupting the serving process. No full 793,230-token recalibration, hard rewrite of production models, or corrected KL/PPL benchmark was performed. Fixing the legacy all-zero collector should be a separately reviewed behavior change before claiming a meaningful aikar-versus-REAP algorithm comparison.
