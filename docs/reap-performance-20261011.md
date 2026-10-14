# REAP performance - 2026-10-11

## Findings

REAP analysis and one REAM layer do different work. REAP collects routing and output norms for every MoE layer in one full forward pass. Sequential REAM calibration runs only the prefix through its current target layer and collects output norms only for that layer. Early REAM layers therefore have substantially less model and collector work. Comparing their throughput does not measure the relative cost of pruning versus merging.

Two avoidable costs existed in REAP:

- The output collector downloaded one expert output row at a time. An F32 tensor shaped `[2816, 8, 512]` required 4,096 synchronous device reads per layer. REAM's callback already downloaded its target layer output in one call.
- `--no-evaluate` skipped ratio evaluations but still produced full baseline vocabulary logits and calculated CPU NLL. Gemma 4 has 262,144 vocabulary entries, making this baseline work expensive even when only pruning statistics are needed.

## Changes

The shared output collector now downloads at most 4 MiB at a time, reducing the example above to about 12 device reads. Host tensors are read directly. Tensor strides, F32/F16/BF16 conversion, double squaring and accumulation order, norm checks, counters, gate multiplication, and legacy statistics are preserved.

`analyze --no-evaluate` now collects every layer's statistics without baseline PPL. It reuses the existing Gemma 4 calibration graph through the last MoE layer, skipping the final vocabulary head. Output capacity is limited to one placeholder per batch, and CPU NLL work is skipped. Calibration records, windows, input tokens, batches and context remain unchanged.

Statistics-only caches record `baseline.ppl_evaluated: false`, zero evaluated counts and zero NLL. Analysis JSON writes null PPL, CSV leaves baseline PPL blank, and the text summary says `not evaluated`. Profiles still record the actual processed calibration tokens. The default evaluation mode retains baseline and ratio PPL. Old caches without the new flag default to PPL available. Evaluation rejects an explicit statistics-only cache and recalibrates an automatic one. Profile generation accepts statistics-only caches.

## Measurements

The V100-SXM2-16GB test used the current `Lumen-3.6-Pulsar-L-LD-Q4_0_XL.gguf`, the first complete record of `train_ko.jsonl`, context 4096, batch/ubatch 512, CPU/dataset threads 36, GPU layers 99, CPU MoE layers 1, seed 3407 and disabled CUDA graphs. Both runs processed 12,559 tokens across the same context windows. A fresh importance cache was used for each run.

| Path | Calibration seconds | Tokens/s |
| --- | ---: | ---: |
| Original REAP with baseline PPL | 90.0176 | 139.517 |
| Optimized REAP without PPL | 30.0503 | 417.933 |

This is a 3.00x improvement in calibration, excluding model hashing, loading, tokenization and report generation. Every raw expert statistic in all 30 layers matched exactly, including selection and measurement counts and double saliency sums. The pruning mappings matched exactly. This is one complete record, not a full-dataset timing or a new pruned production model.

A separate actual-GPU collector benchmark used deterministic F32 outputs shaped `[2816, 8, 512]`. Original collection took about 59-61 ms; bounded bulk collection took about 26-28 ms. All per-expert results matched direct CPU collection exactly. These are collector timings, not end-to-end speedups. Splitting dtype loops and using pinned host memory did not show a clear additional benefit and were excluded.

## Verification

- Built `aikar-prune` and `test-moe-prune`.
- Existing `test-moe-prune` now checks bounded device read sizes/counts and exact host/device results for padded F32/F16/BF16 tensors spanning multiple download chunks. The original collector failed the read-count assertion; the new collector passes.
- CTest `test-moe-prune` passed.
- Synthetic 30-layer calibration matched every original statistic and pruning mapping with PPL skipped. Default baseline evaluation matched the original NLL and statistics exactly.
- Separate CLI checks covered null PPL reporting, profile generation from statistics-only caches, explicit cache rejection and automatic cache recalibration with evaluation enabled.
- `git diff --check` passed.

The user's existing REAP process was stopped with SIGINT before GPU validation. The installed `aikar-prune` command is a symlink to `build/bin/aikar-prune`, so the optimized build is available without reinstalling. Full production calibration was not restarted, and no GGUF was overwritten.
