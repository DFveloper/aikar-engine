# Pruning fixes verified on 2026-10-08

This follows the historical [REAP audit](reap-pruning-audit-20261008.md). The historical caches, models, reports and comparison tables were preserved. The pruning formulas were not changed to favor one metric.

## Corrected collection

For a routed expert, both current `router-output` (aikar) and REAP compute:

```text
score[e] = sum(selected tokens, effective_gate[e,t] * L2(pre_gate_output[e,t])) / selected_count[e]
```

Zero-count experts score zero. This is a conditional average, not a frequency-weighted contribution. The existing aikar formula already had this denominator. Its historical implementation missed Gemma's scaled down-projection callback and counted a reshaped routing view twice. Consequently the audited Gemma cache contained all-zero aikar scores; the resulting expert-ID tie-break ranking was not evidence of a better importance formula.

Both metrics now use the final pre-gate down projection, including expert-specific scaling, and count only the exact original normalized-gate callback. Top-K expert IDs are copied using the tensor's actual row stride instead of treating the argsort view as contiguous. Pending gates are cleared when new IDs arrive. Duplicate reshaped callbacks are rejected before synchronization.

For the supported Gemma 4 path, the gate is the router softmax probability divided by the sum of the selected Top-K probabilities, with the graph's denominator clamp of `6.103515625e-5`. No gate is applied twice. The expert tensor has shape `[embedding_dimension, selected_experts, tokens]`; the norm excludes gate multiplication, aggregation and subsequent RMS normalization. Shared dense experts are unchanged.

FP32, FP16 and BF16 output reads accumulate norms and scalar sums in double precision. Counts use `uint64_t`; invalid gates, non-finite outputs and count/sum overflow fail rather than creating misleading scores. Extra persistent REAP storage is 32 bytes per routed expert. Norm reads use bounded scratch space and require no second calibration forward pass.

## Cache and profile validity

All three metrics now require collector version 3. Versions 1 and 2 contain collection errors and cannot be repaired by re-ranking their saved sums. Explicit obsolete or inconsistent caches fail. Automatic obsolete or malformed caches trigger fresh calibration and are replaced only after successful collection. Legacy baseline checkpoints are not migrated.

Validation checks selection/measurement count equality, per-layer Top-K totals, finite nonnegative scalar sums, consistency of aikar and REAP measurements, processed/total token counts, and baseline loss/count bounds. Cache identity includes model and dataset hashes, tokenized record order and field masks, context, seed, batch sizes, execution settings, build/compiler/backend information and relevant routing/backend environment settings. Cache-hit verification loads vocabulary and tokenizes without an inference pass. This records the available build identity; it is not a cryptographic attestation of every loaded library.

Generated profiles carry collector version and calibration/token/execution fingerprints. Common validation rejects obsolete generated profiles for both soft masking and hard conversion. Existing generated profiles must be regenerated from fresh calibration, not edited to bypass version checks.

## Reports and output safety

Reports now identify source model path/hash, architecture, metric, requested/actual ratio and calibration fingerprints. The CLI also records final output path/hash and profile path/hash. Timestamps are omitted for deterministic comparison. A matching report hash alone was not proof of stale state: historical reports omitted these identity fields and could serialize identical pruning plans.

Different ratios that map to the same rounded profile filename fail before calibration. Non-finite ratios fail early. Source-model aliases through final, temporary, report or recovery paths are rejected.

Both low-level hard conversion and CLI publication use the same model/report publication helper. Existing files are backed up with hard links. The old report is removed before replacing its model. Publication failure restores the previous pair; failed rollback or interruption leaves the report absent and retains recovery backups. Publication refuses outstanding recovery paths. This prevents a stale report from silently describing a new model; two filesystem paths are not a single atomic transaction.

`selection_frequency` now means selections divided by processed calibration tokens. The previous share of all selected Top-K slots remains available as `routing_slot_fraction`.

Pruning commands disable CUDA graph capture before backend initialization. This retains the previously verified fix for callback-segment MMVQ activation-cache OOM. It changes only the pruning process. This follow-up did not interrupt the serving process or repeat the earlier GPU calibration check.

## Verification

- Rebuilt CPU and CUDA/Vulkan `aikar-prune` and `test-moe-prune`; the existing pruning test passed in both builds.
- Added tests to the existing test file for exact synthetic conditional scores, ordering, zero counts, Top-K normalization, strided expert IDs, no gate squaring or frequency weighting, duplicate-view exclusion, count overflow, profile provenance, dataset fingerprint sensitivity, source aliases and report-failure preservation.
- Ran direct callback tests for both scaled and unscaled expert outputs and duplicate routing views.
- Ran publication fault injection for success, report-rename failure, failed rollback, process interruption and failed first publication. A low-level conversion regression reproduced old model replacement on report failure and passes after the fix.
- Exercised all three metrics through both CPU and installed CUDA/Vulkan CLI profile generation. Rejected obsolete caches, inconsistent statistics/losses, missing execution provenance, changed seed/tokenized input, invalid ratios and filename collisions. A compatible explicit cache was reused without model forward passes. A malformed automatic cache triggered fresh calibration; its statistics matched the first deterministic run exactly.
- Collected fresh statistics from the actual Pulsar source model on one small deterministic Korean chat record: 27 dataset tokens, 26 processed tokens, 30 layers, 128 experts and Top-K 8. There were 706 experts with nonzero aikar scores. Every layer's selection and output measurement counts matched; aikar and REAP sums matched exactly. At requested ratio 0.10, both removed 12 experts per layer (360 total, actual ratio 0.09375), with identical ranks and masks. Frequency remains a separate metric.

The tiny run establishes collection correctness, not model quality. The full 793,230-token calibration, production hard rewrite, and PPL/KL evaluation were not rerun in this follow-up. The historical quality comparison should be repeated with new version-3 statistics and identical calibration/evaluation inputs. Existing supported architecture and hard-layout limits remain Gemma 4 26B A4B with homogeneous routed expert counts and Q4_0 QAT expert tensors.

## Files

This follow-up changed `common/moe-prune.{h,cpp}`, `tools/aikar-prune/{main.cpp,dataset.h,dataset.cpp,hard-prune.h,hard-prune.cpp}`, the existing `tests/test-moe-prune.cpp`, and pruning documentation. The implementation plan is in `docs/superpowers/plans/2026-10-08-prune-bug-fixes.md`. Other pre-existing workspace changes were preserved.
