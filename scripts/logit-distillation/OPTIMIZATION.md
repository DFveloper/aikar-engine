# CPU distillation optimization validation

Date: 2026-11-04. Initial distillation implementation committed as dea3f28b8. The subsequent CPU optimizations and production scripts are working-tree changes. Existing unrelated changes are preserved.

This records the earlier Korean-only optimization experiment. THROUGHPUT.md and README.md describe the current 1:1 mixed input and attention targets restricted to blocks 11, 17, 23 and 29.

The workflow continues to use pure teacher top-512 plus tail forward KL, with no ground-truth CE term. The machine is a Xeon E5-2696 v3; all measurements use CPU and eight threads unless specified.

| Measurement | Reference | Optimized |
| --- | ---: | ---: |
| KL forward/backward benchmark, vocabulary 262144, 16 rows | 156.680 ms | 16.559 ms |
| Benchmark graph tensor allocation | 420009808 bytes | 33643360 bytes |
| Real training compute workspace, context 256 | 1733842944 bytes | 928012288 bytes |
| Four-window teacher extraction processing | 42.595 s | 36.196 s |
| Two-window validation KL after one epoch | 2.39438766 | 2.39455098 |

The fused CPU loss kernel is about 9.46 times faster in this isolated benchmark. Training workspace drops 46.48%. Extraction processing drops 15.02%, and the resulting cache is byte-identical. An 18-thread extraction run took 39.810 seconds, so the supplied default remains eight threads. Measurements are single local runs, not repeated statistical benchmarks.

The full model remains dominated by other forward/backward operations. The optimized two-training-window run took about 62 seconds against about 60 seconds previously, with concurrent work during the optimized measurement. These runs do not establish an end-to-end training speed improvement. Small KL differences are expected from fused floating-point reductions.

Extraction now reuses row encoder storage, encodes teacher rows in parallel with deterministic output ordering, and avoids repeated per-vocabulary error-string construction. Cache loading reuses buffers and reads sequential rows without redundant seeks. The fused KL operation uses stable separate full-vocabulary and tail normalizers, and row-sized scratch per worker. Full logits and logit gradients are still allocated.

Validation includes double-precision reference comparisons, finite-difference gradients at two temperatures, masked rows, extreme logits, cache corruption checks, and backend operation tests. A real interrupted extraction resumed from three complete windows, discarded its torn fourth window, and produced a cache byte-identical to uninterrupted extraction. Checkpoint configuration validation also rejects a changed teacher-cache path, even if that file happens to contain identical bytes.

A real full-model checkpoint was saved after optimizer step one, then loaded with its optimizer state to finish step two. Resumed validation KL was 2.39455098, identical to uninterrupted training. SHA-256 hashes of both trained tensor payloads also matched uninterrupted training. The backend support and forward tests passed all three KL shapes; small-shape numerical gradient tests also passed. The large-vocabulary numerical gradient case is skipped by the existing backend harness for speed, while the dedicated benchmark and real training exercise that vocabulary size.

Production wrappers and the storage plan are documented in README.md. Default training selects all 2929 training windows, updates attention Q/K/V/output across all layers, and retains full model plus optimizer checkpoints. The default checkpoint budget requires about 137.64 GiB free after the 3.73 GiB cache; the current model volume has about 21 GiB free. The full production run has not been started and full recovery quality is unmeasured.

The earlier REPORT.md records the original smoke test and its original graph implementation. This file describes the later optimized implementation.
