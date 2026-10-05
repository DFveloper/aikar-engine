# Paged attention optimization - 2026-10-04

Target: `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/start.sh`. GPU: Tesla V100 SXM2 16 GB, CUDA0. Build: `-j36`.

## Configuration and implementation

- Full-attention physical pool remains 8192 blocks x 16 tokens = 131072 tokens (1320 MiB). Logical context remains 262144. Q8_KV format and block size are unchanged.
- 16 slots with GPU embedding/output failed at compute-buffer allocation. Default is 12 active slots. Extra requests queue; 16 requests are still accepted.
- Remove the CPU override for the tied embedding/output tensor, moving 396 MiB to GPU. The multimodal projector still runs on CPU.
- SWA allocation decreases from 1830.47 to 1405.08 MiB as slots decrease from 16 to 12. Total KV allocation decreases from 3150.47 to 2725.08 MiB: 425.39 MiB saved without reducing the full-attention pool.
- CUDA paged writes quantize each K/V once, then write all sequence destinations. Warp reductions reduce shared scratch space.
- Attention partitions adapt to query count and GPU SM count. D256/D512 use the warp path for decode as well as prefill.
- Reuse attention graphs while refreshing page tables, write slots, and queries. The attention bound is fixed within a 256-token bucket; rebuild when that bucket changes.

## Identical 16-request workload

Old: 16 active slots, CPU embedding/output. New: 12 active slots, GPU embedding/output, 4 requests queue. Temperature 0; no prompt cache; identical prompts and generation counts. Long prompts contain 7929-7930 tokens per request.

| Metric | Old | New |
| --- | ---: | ---: |
| Short-request generated tokens / total wall time | 116.97 tok/s | 187.80 tok/s (+60.6%) |
| Long-request prompt tokens / total wall time | 868.29 tok/s | 903.74 tok/s (+4.1%) |
| Short workload completion | 17.51 s | 10.91 s |
| Long workload completion | 146.12 s | 140.38 s |
| Peak GPU memory | 16053 MiB | 16049 MiB |
| Short workload maximum time to first token | 1.33 s | 7.80 s |

The generated-token rate includes prefill and queue time; it is not the single-sequence decode rate. Queueing improves aggregate throughput here but delays first output for the last four requests. Peak memory is nearly unchanged because saved SWA memory funds GPU embedding/output. Only 96 MiB remained free in these measured workloads, so more demanding workloads can still exhaust memory.

With the same 12-slot GPU settings, the kernel/graph changes alone improved short aggregate throughput by 10.1% and the long workload by 4.7%. These are single runs, not statistical confidence intervals.

## Validation and reproducibility

- CUDA PAGED_ATTN/PAGED_ATTN_WRITE: 54 cases, including F16/Q8_KV, D128/256/512/1024, causal/noncausal, SWA/full, partial warp groups, and 1/7/10/12/16/32/33-token batches.
- Argument parser passes.
- Multimodal chat returns the newspaper headline `MEN WALK ON MOON`.
- Q8_KV and F16 fragmented state restore pass, including copy-on-write, allocation rollback, and repeated page reuse. Restored versus clean paged logits have max_abs=0.
- The ordinary-KV reference has different logits (also seen before optimization); this fixture does not require numerical agreement with ordinary attention and is not proof of model quality.
- Actual production chat: 12 simultaneous 10566-token conversations (126792 prompt tokens, 96.7% of the physical token budget) all recalled unique first and last secrets and answered 17+25 correctly. A clean versus crowded response was identical. Follow-up name recall passed after slots were reused. No gibberish was reproduced in these cases; this does not prove absence for all prompts.
- Final 16-slot GPU-output startup still fails allocating the 348.49 MiB compute buffer. Production start.sh now uses the durable runtime, 12 slots, and the original 8192-block full pool. Health and loaded library paths were verified.

Durable runtime and libraries: `/home/user/aikar-engine/build-paged-opt/bin`. Launch sets LD_LIBRARY_PATH to that directory. Source snapshot: `build-paged-opt/sources.tar.gz`; bounded optimization diff: `build-paged-opt/optimization.patch`. Source worktree: `/tmp/aikar-paged-kv`; the snapshot survives /tmp cleanup. Existing main-root build and unrelated training work were not replaced.

To rebuild from the snapshot, extract it into a separate directory and configure CUDA architectures 61;70, then build llama-server with `-j36`. Existing paged files and integration are included. The bounded diff describes only this optimization relative to the pre-existing paged implementation.

## Remaining optimization opportunities

1. Prefill remains the dominant cost. Reuse KV loads across query tiles or adapt the existing flash-attention infrastructure for paged access; this requires a separate numerical comparison and review.
2. Reducing physical bytes per KV token requires a different KV format or representation. This change preserves Q8_KV and full pool capacity; savings come from the SWA slot allocation.
3. CUDA graphs stay disabled to avoid activation-cache memory growth. More memory headroom could allow measured graph capture or 16 GPU-output slots, but current startup testing does not support that configuration.

## Single-request decode measurement

Actual chat, temperature 0, reasoning disabled, 50-token prompt, 512 generated tokens: 39.24, 39.36, 39.43 tok/s; mean 39.34 tok/s. Relative to the user-reported 100 tok/s baseline, this is 60.66% lower. The 100 tok/s configuration was not reproduced in this session, so this comparison does not isolate the effect of the kernel changes. The 10566-token single chat decoded at 34.82 tok/s, but its 13-token output is too short for a stable throughput estimate. Aggregated concurrent throughput improved; single-request 100 tok/s performance has not been recovered.

A second correctly formatted completion (27 prompt tokens, 256 generated tokens) decoded at 45.99 tok/s, 54.01% below 100 tok/s. Korean chat after the near-capacity workload also produced a coherent answer. A bare completion without chat formatting produced repetitions; applying the model chat template restored coherent output. That bare completion is not used to judge KV correctness.

## Follow-up: fused GQA prefill

The CUDA prefill path shares loaded and dequantized K/V across two query heads for D256 and four query heads for D512 when the GQA ratio permits. Each head retains its own FP32 online softmax and output accumulator. Packed Q8_KV writes, page mappings, sparse valid masks, causal/SWA bounds, and the physical pool remain unchanged. Default dispatch uses this path for batches of at least 128 tokens; smaller batches retain the existing path. `GGML_CUDA_PAGED_FUSED=0` disables it, while a nonzero value forces it for supported batches larger than eight tokens.

A direct fused write/read experiment reconstructed new packed K/V from the input and wrote cache rows inside attention. Although its original 54-case operator matrix passed on V100, repeated quantization made it slower. It is retained as an experiment artifact, not deployed. The final path keeps one packed write kernel and fuses K/V loading and conversion with multiple query heads inside attention.

V100 operator measurements with CUDA graphs disabled, Q8_KV, and two interleaved sequences starting at position zero:

| Shape | Separate attention | Direct fused write/read | Fused GQA attention |
| --- | ---: | ---: | ---: |
| D512, 2 KV heads, 32 queries | 40.00 us | 111.30 us | 41.54 us |
| D256, 8 KV heads, 32 queries | 31.61 us | 66.20 us | 32.26 us |
| D512, 2 KV heads, 1280 queries | 8079.14 us | 21771.76 us | 4747.48 us |
| D256, 8 KV heads, 1280 queries | 5076.79 us | 10789.30 us | 3528.56 us |

These are operator timings, not end-to-end PP rates. The small-batch results motivate the 128-token default threshold. The existing test-backend-ops performance matrix now includes these four shapes for F16 and Q8_KV. Source and test changes are maintained in the production source worktree `/tmp/aikar-paged-kv`; the main workspace's paged CUDA file mirrors that implementation.

The final write kernel uses `__fdiv_rn` for the Q8_KV scale and reciprocal. The CUDA build uses `-use_fast_math`; approximate division produced a sporadic one-byte mismatch against CPU reference quantization. A deterministic half-integer boundary fixture reproduces this with grouping disabled and passes after the precise division change. Boundary inputs are confined to exact-byte write tests; attention tests retain random K/V. The corrected kernel passed 58/58 paged attention/write cases on both V100 and P620, with grouping forced for batches larger than eight tokens to cover grouped split output as well as the default 128-token threshold.

The separate deployment package is `/home/user/aikar-engine/build-paged-fused`. It contains the runtime and libraries in `bin`, the complete source snapshot in `sources.tar.gz`, and a bounded `optimization.patch` relative to the previous production snapshot. The original `build-paged-opt` runtime remains available for rollback. Disabling `GGML_CUDA_PAGED_FUSED` disables head grouping; it retains precise Q8_KV quantization and does not restore the old source byte-for-byte.

Final matched production workload: 12 active slots, 16 simultaneous requests, GPU embedding/output, physical pool 8192 x 16, batch/ubatch 1280, chunk budget 64, temperature zero, no prompt caching. Both runs evaluated 126870 long-prompt tokens and generated 512 long-output tokens; the short workload generated 2048 tokens. The final runtime includes the precise Q8_KV write fix.

| Metric | Previous production | Final fused GQA |
| --- | ---: | ---: |
| Long prompt tokens / total wall time | 912.59 tok/s | 1234.41 tok/s (+35.3%) |
| Long prompt + generated tokens / total wall time | 916.28 tok/s | 1239.39 tok/s (+35.3%) |
| Long workload completion | 139.02 s | 102.78 s |
| Short generated tokens / total wall time | 190.31 tok/s | 201.08 tok/s (+5.7%) |
| Short workload completion | 10.76 s | 10.18 s |
| Long workload maximum time to first token | 137.91 s | 101.93 s |
| Peak GPU memory | 16049 MiB | 16049 MiB |

All 16 short and all 16 long greedy response strings matched the previous runtime. Multimodal inference returned the newspaper headline `MEN WALK ON MOON`. This remains a single matched run per configuration, not a confidence interval or a general quality proof. A preceding grouped run without the precise-write fix completed the long workload in 97.15 s; the final corrected result above is the reported deployment measurement. Reports are in `build-paged-fused/validation/fused-baseline-20261004.json` and `fused-final-20261004.json`.

Deployment completed: `start.sh` now selects `build-paged-fused/bin/llama-server`; its previous script is preserved as `start.sh.before-paged-fused-20261004`. Production health returned `ok`, 12 slots were verified, and `/proc` mappings confirmed the new package's CUDA/ggml/llama libraries. Final random-input retesting passed 32/32 targeted V100 cases and 58/58 full P620 cases. No commits or external submissions were made.
