# Training efficiency audit, 2026-11-16

Target: 60 hours / 6,057 windows = 35.661 seconds/window, with both V100 GPUs limited to 150 W. Context 16,384, microbatch 768, accumulation 6, eight Q/O targets, data order, FP4 forward values and STE gradients are preserved.

Artifacts: `/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit`.

## Changes

- Fill ordinary contiguous causal masks by ranges instead of checking every token pair. The complete 1 GiB pair of masks has the same hash. CPU mask construction decreased from 1.190 to 0.160 seconds in the isolated warm benchmark.
- Mark the existing FP4 dequantization cast as detached instead of allocating another detached F32 copy. Forward bytes and identity STE gradients match on CPU and CUDA. The actual graph has 128 rather than 138 CPY nodes; their logical tensor sizes decrease by 640 MiB. This is not a claim of 640 MiB peak HBM savings.
- Add direct peer AllReduce for two peer-accessible GPUs when `GGML_CUDA_P2P=1`. The previous internal fallback staged transfers through pinned host memory despite NVLink availability. Preserve the original BF16 rounding before F32 addition. Unsupported types/sizes retain the existing fallback.
- ACC reuse now uses the existing allocator's ownership, last-consumer, view and layout checks, with `GGML_TRAIN_ACC_ALLOC_INPLACE=1`. Only src0 is eligible. CPU ACC skips a self-copy when the allocator aliases src0. Dense and MoE 120-window/20-update probes match all 81/82 tensors and losses. Actual-model 12-window/two-update validation also matches all 26 tensors and losses. A two-window partial run reduced times from 70.932/46.564 to 66.874/42.663 seconds and reduced the scheduler buffer by 80 MiB per GPU.
- Correct saved data-window counting to use ceil(context/microbatch), 22 rather than 21. The captured production checkpoint is optimizer step 7 / data window 42. Original checkpoint artifacts are retained.

## Evidence and remaining work

The instrumented baseline's two training windows plus one validation window transferred 450 GB H2D, 380 GB D2H and 6.06 TB D2D. Profiling adds overhead; these are transfer counts, not a production throughput measurement.

Nested layer and MoE tile recomputation repeats some expert projections. Removing all recomputation would violate the HBM budget. RAM checkpoint experiments reduced replay nodes from 153,241 to 79,261, but exceeded HBM capacity. Limiting them to eight layers fit memory, but changed the second forward loss from 1.62849301 to 1.61049610 before the first optimizer update. All RAM checkpoint changes were removed from the working sources and excluded from deployment. The precise cause remains unresolved.

Explicit ACC views were also rejected: the first version increased the actual model's scheduler buffer from 7,108 to 9,684 MiB per GPU and failed allocation. Flattening the view chains did not resolve capacity. Both versions were removed from working sources; artifact snapshots remain for diagnosis.

## Validated 12-window result without allocator ACC reuse

The unprofiled baseline averages 80.227 seconds/window; causal-mask, FP4-copy and peer-transfer changes average 58.071 seconds/window, a 27.6% reduction. All 12 logged losses and both optimizer-update losses match. All 26 saved weight and optimizer tensors are byte-identical. Sampled device memory peaks decreased from 15,058/15,666 MiB to 14,798/15,428 MiB. These are sampled peaks, not allocator high-water marks.

The candidate's runner exit status was not captured, but its complete 12-window logs, final checkpoint marker and saved optimizer step 2 are present. Results are validated directly from those artifacts. The baseline runner exited successfully.

## Final result with allocator ACC reuse

| Measurement | Baseline | Optimized |
| --- | ---: | ---: |
| Mean time inside each window | 80.227 s | 53.992 s |
| Training loop, including transitions | 978 s / 12 = 81.500 s/window | 664 s / 12 = 55.333 s/window |
| Process wall time, including startup, validation and export | 1089.901 s | 764.801 s |
| Sampled GPU 0 peak | 15,058 MiB | 14,686 MiB |
| Sampled GPU 1 peak | 15,666 MiB | 15,316 MiB |
| Reported compute workspace | 12,713,029,632 bytes | 12,604,047,360 bytes |

All 12 window losses, both update losses, validation loss and all 26 saved weight/optimizer tensors match. The optimized process exited successfully. The clean deployment build passed the complete existing CPU and paired CUDA regression suites.

The loop time improves by 32.1%. Extrapolating 55.333 seconds/window across 6,057 windows gives 93.1 hours before validation/checkpoint overhead. Another 35.6% reduction is needed for 35.661 seconds/window. This probe covers six windows of each alternating data type; it is not a measurement over the full dataset.

The clean deployment build also passed a two-window training plus validation profile against the original baseline: all 24 saved optimizer tensors and losses match.

| Recorded copy volume in the same two-window profile | Baseline | Optimized |
| --- | ---: | ---: |
| H2D | 450.25 GB | 114.18 GB |
| D2H | 379.60 GB | 43.53 GB |
| D2D | 6064.48 GB | 460.53 GB |

H2D falls 74.6%, D2H 88.5% and D2D 92.4%. The collector does not record P2P transfers in these categories, so these numbers are not total interconnect traffic. Kernel-launch counts remain almost unchanged at 1.172 million across the profile. GPU launch overhead, host synchronization and nested MoE replay remain substantial. Runtime API totals are summed CPU-thread durations, not wall time.

## Production

The production launcher was restarted from the preserved merged step-7 model and optimizer sidecar, with dataset window 42, warmup 50, CPU accumulation and both 150 W limits. Trainer PID 3973394 loaded all four libraries from the isolated `validated-release` directory. Native logging confirms scheduler start step 7, 6,057 training windows, microbatch 768, context 16,384 and accumulation 6. The first resumed window completed with finite loss 5.30778623; GPU execution is active. Its 78-second cold first-window progress is not the 12-window benchmark's steady mean. A long-run mean after this checkpoint is not yet measured.

## Next performance work

The remaining target gap is 19.67 seconds/window including transitions. Prioritize fixing nested MoE replay lifetimes with an exact forward/state reference, then reducing the approximately 1.172 million launches and host synchronization calls. The failed RAM checkpoint path must not be reused without resolving its forward-value change. Graph construction/allocation consumes about 3.9 seconds/window and input preparation about 0.3 seconds; optimizing only those cannot close the gap. Preserve the measured 150 W/numerical baseline during further experiments.

## Further MoE trials: rejected, 2026-11-16

Four isolated two-window trials completed successfully, but all failed numerical comparison with the verified two-window deployment profile. All 24 saved optimizer tensors were compared; the eight momentum tensors differed in every trial. The reference profile previously matched the original baseline, including saved state.

| Candidate | First window loss | Second window loss | Result |
| --- | ---: | ---: | --- |
| Verified reference, tile 256 | 2.60183626 | 1.62849301 | Reference |
| MoE tile 512 | 2.60183626 | 1.63272089 | Reject |
| MoE tile 1024 | 2.59544474 | 1.62568724 | Reject |
| Tile 1024, expert projections kept at 256 | 2.59544474 | 1.62568724 | Reject |
| Tile 512, ACC reuse disabled by removing its environment variable | 2.60183626 | 1.63272089 | Reject |

Tile 512 reduced replay nodes from 153,241 to 95,641 and its two timed windows took 63.574/39.355 seconds. Tile 1024 reduced replay nodes to 77,881 and took 61.391/39.269 seconds. These short, cold-window timings are diagnostic; the reference comparison was profiled, and no sustained speedup is established. Larger tiles fit memory but do not preserve this training computation. Disabling ACC reuse does not remove the loss difference. Preserving expert projection shapes alone does not remove it either; the precise cause remains unresolved.

An initial overlapping ACC-ablation launch failed during model allocation and was excluded. A repeat with `GGML_TRAIN_ACC_ALLOC_INPLACE=0` still enabled ACC because the allocator tests variable presence; it was excluded as an ablation. The final ACC-disabled trial actually removed the variable and reproduced the same loss difference.

The experimental MoE options were removed from working sources. Their reproducible patch and comparisons are in `efficiency-audit-tools/moe-experiment.patch` (apply with `git apply --unidiff-zero`) and `efficiency-audit-tools/moe-tile-results.json`. Production retains the validated 256-token MoE path, peer transfers, mask/copy improvements and ACC reuse. No additional optimization was accepted in this round; the validated loop result remains 55.333 seconds/window, short of the 35.661-second target.

The step-11 adapter and optimizer were preserved before releasing the paused trainer. `production-step11.gguf` was merged against the exact step-7 base; all five adapter tensor payloads match the merged model byte for byte. The sidecar records optimizer/scheduler step 11, dataset window 66 and accumulation 6. Production was launched from this pair using the existing isolated `validated-release` libraries and both 150 W limits. Resume metadata is in the artifact directory's `production-step11-resumed.json`; the checkpoint check is also recorded in `efficiency-audit-tools/step11-checkpoint-validation.json`.

Resume verification passed: trainer PID 213937 loaded all four validated libraries, the scheduler starts at step 11, and no experimental MoE environment variables are present. The first resumed window completed with finite loss 1.48817137 at training elapsed 45 seconds. This single window is not a new steady throughput measurement. Both GPU limits remain 150 W.
