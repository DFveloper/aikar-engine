# QLion2 P620: MoE gather and Q4 residual state kernels

Date: 2026-11-04. The baseline includes the [microgradient fusion](2026-11-04-qlion2-microgradient-fusion.md) and all earlier optimizations. Measurements isolate the additional changes below, on Quadro P620, SM61, CUDA 12.4.131, driver 580.178.04, Release build.

## Implementation

The shared CUDA OUT_PROD_ID and compact expert accumulation path previously issued two device-to-device copies per routed token, one each for activations and upstream gradients. It now uploads one expert-grouped route list, then gathers both vectors in one kernel per nonempty expert. The host route builder, within-expert token order, GEMM dimensions, beta, and arithmetic stay the same. Broadcast activations, independent expert/token strides, empty experts, and repeated expert IDs remain supported. A stream synchronization after uploading the host route list protects its lifetime. This is a reduction in per-token copy submission, not a fully device-resident routing redesign.

Fresh Q4_0 residual and Q8_0 momentum quantization now compute the reciprocal of the warp-uniform scale in each lane. This removes the lane-zero branch and subsequent broadcast while preserving division and rounding. The residual format, subnormal rule, error feedback, and optional state-scale reuse policy are unchanged. The routed apply kernel still uses 39 registers with no spills.

A separate XOR-shuffle packing candidate matched all hashes but showed no useful timing benefit in its probe (median after/before 0.999). It was not installed. State scale reuse is tested as a separate existing option; it is not enabled by this change.

## Isolated measurements

The gather harness exercises complete OUT_PROD_ID calls with 11 experts and three routes per token, and includes gathering, zeroing, GEMM, and stream completion. It measures host wall time after one warmup, averages three calls per round, and reports medians of three alternating baseline/candidate rounds. All 72 output hashes match across two shapes, three token counts, broadcast/non-broadcast activations, and contiguous/strided token and ID storage.

| Expert matrix | Tokens | Before (ms) | After (ms) | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 96 x 35 | 1 | 0.046716 | 0.036839 | 21.1% |
| 96 x 35 | 37 | 0.622041 | 0.091771 | 85.2% |
| 96 x 35 | 256 | 4.614670 | 0.146441 | 96.8% |
| 1408 x 2816 | 1 | 4.028320 | 4.064034 | -0.9% |
| 1408 x 2816 | 37 | 8.353598 | 8.100362 | 3.0% |
| 1408 x 2816 | 256 | 14.131413 | 12.156826 | 14.0% |

The large single-token case is slightly slower; there are few copies to amortize the route upload and gather launch. Large matrix zeroing and GEMM also limit the improvement. This probe does not imply a universal per-operation gain.

The residual/momentum reciprocal probe compares 60 routed update configurations over three alternating rounds, covering Q4_0/Q8_0/MXFP4, 96 x 5 and 1408 x 2816 expert matrices, broadcast activations, direct and GEMM routes, and enabled/disabled state-scale reuse. All 180 weight, momentum, and residual hash triples match after ten updates. For large matrices, median after/before ratios across configurations are 0.9887 for Q4_0, 0.9932 for Q8_0, and 0.9884 for MXFP4. These are approximately 1% full routed-update improvements; they do not measure residual quantization alone. Small matrix results are near parity and include regressions. The larger MoE benefit comes from gather submission.

## Complete training

Each setting uses one complete epoch, seed 1234, context/batch/microbatch 256, one parallel sequence, Q4_0/Q8_0 selected weights, all layers offloaded, four CPU threads, two dataset threads, constant learning rate 0.001, weight decay 0.01, and fixed tau with no warmup. The MoE fixture has approximately 7M parameters. Accumulation=4 repeats the 32-record synthetic dataset four times. The dense control has 200,402,386 parameters. The compact and fast rows explicitly select the existing --qat-moe-gradient compact and --qat-fast-state-scale options respectively.

Each row is the median of three matched runs per variant, with order reversed in the middle round. Wall time includes loading, complete training, last-step optimizer checkpoint serialization, and final-model serialization; hashing is outside the timing interval. Steady step time excludes the first two logical updates. Normal runs are not profiled.

| Setting | Wall before (s) | Wall after (s) | Wall reduction | Step before (s) | Step after (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| moe_a4_fixed | 3.073 | 2.318 | 24.6% | 0.370076 | 0.247479 |
| moe_a4_compact_fixed | 3.737 | 3.080 | 17.6% | 0.504880 | 0.372199 |
| moe_a4_fast_fixed | 3.095 | 2.191 | 29.2% | 0.375065 | 0.239979 |
| moe_a1_fixed | 0.982 | 1.075 | -9.5% | 0.060968 | 0.062433 |
| 200m_a4_fixed | 15.520 | 15.413 | 0.7% | 2.050342 | 2.036294 |

Default MoE accumulation=4 reduces median complete wall time by 24.6% and steady step time by 33.1%. Compact mode reduces these by 17.6% and 26.3%; the existing fast-state-scale option by 29.2% and 36.0%. Comparisons are always within the same setting, not across quantization policies.

The accumulation=1 row does not use OUT_PROD_ID gathering and has a 9.5% slower wall median in these three pairs. Five additional alternating control pairs give 0.982 s baseline versus 1.053 s candidate wall medians, with overlapping ranges (0.974-1.043 s versus 0.948-1.077 s). Their steady step medians are 0.062528 s versus 0.061903 s, a 1.0% reduction. No accumulation=1 end-to-end gain is claimed; its wall-time limitation remains visible. The dense 200M control is essentially unchanged.

All 15 primary pairs and five additional control pairs match model tensor shapes/types/bytes, optimizer tensor shapes/types/bytes, printed epoch losses, steps, and supervised-token counts. NVML sampling every 10 ms reports unchanged median peak GPU memory: 200 MiB for MoE accumulation=4, 194 MiB for MoE accumulation=1, and 1392 MiB for the dense control. This change targets work submission, not persistent state memory.

## Profiling

nvprof captures the contiguous interval after step 2 through step 4, covering two optimizer updates and eight microbatches per run. The final model tensor hashes from all four profiled runs match their corresponding normal runs. GPU activity and API time sums are separate measurements and must not be added to wall time.

| Metric | Default before | Default after | Compact before | Compact after |
| --- | ---: | ---: | ---: | ---: |
| GPU activity sum (ms) | 408.623 | 296.337 | 492.445 | 381.053 |
| GPU activity calls | 98020 | 13876 | 103600 | 19456 |
| Kernel launches | 12096 | 12480 | 16504 | 16888 |
| cudaMemcpyAsync calls | 85452 | 924 | 86262 | 1734 |
| cudaMemcpyAsync API time (ms) | 529.126 | 29.951 | 499.952 | 41.987 |
| Stream synchronizations | 1643 | 1739 | 2451 | 2547 |

Default-mode device-to-device copies fall from 84,672 calls / 118.625 ms to 48 calls / 0.579 ms. Compact mode falls from 84,678 / 118.510 ms to 54 / 0.601 ms. Each candidate launches 384 gather kernels, taking 5.913 ms in default mode and 5.938 ms in compact mode. The production trace confirms 28 registers per gather thread; compilation reports zero spills.

The net cudaMemcpyAsync call reduction is 84,528 per capture; the new route-list uploads add 96 host-to-device transfers and synchronization points. GPU activity sums fall 27.5% in default mode and 22.6% in compact mode. Host ID reads, per-expert GEMMs, and synchronization remain. Moving route grouping fully onto the device is not part of this patch.

## Validation and limits

- CPU/CUDA QAT suites pass, including trajectory and compact expert state checks.
- All eight CUDA OUT_PROD_ID backend tests pass against CPU. Five new fixtures cover broadcast inputs, padded token and ID strides, repeated IDs, and one active expert.
- All 28 supported selected CUDA backend tests pass; six unsupported dense routed accumulation cases are skipped. test-opt passes and git diff --check is clean.
- CPU/CUDA operation support CSVs include the new fixtures. No public API, type, or CLI is added.
- The evidence is specific to P620 and the synthetic fixtures. No full 26B model or training-quality evaluation is claimed. Other GPU backends were not benchmarked.

Artifacts are under /tmp/qlion-moe: gather-summary.json and three rounds of CSVs, residual probe CSVs and compilation reports, frozen baseline/candidate binary hash manifests, per-run training tensor hashes and logs, timing-summary.json, comparisons.json, and nvprof exports. Generated model dumps are deleted after hashing. One 200M run failed at the session write quota and is excluded; its log is under invalid/checkpoint-quota. Earlier CPU profiler archives were losslessly compressed to /tmp/qlion-craftsman/train/cpu/200m_a4/{baseline,candidate}/cpu.er.tar.gz to release space, retaining their text reports. Post-run integrity checks confirm both binary snapshots and all four original input files are unchanged.
