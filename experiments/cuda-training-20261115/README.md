# CUDA training optimization, 2026-11-16

## Current attention backward optimization

Production restarted at 2026-11-16T04:35:10.267450+00:00 with trainer PID 2418380. The original train.sh launched it, and every original argument matches. Final Compute Sanitizer memcheck reports zero errors.

The user authorized small floating point differences after gradient and loss validation. The accepted path is Volta D256 bounded SWA backward with batch 1, <=1280 visible keys, a query count divisible by 16, and no requested sink gradient. Global FP4 attention and STE are unchanged. Production arguments and data are unchanged.

The kernel fuses QK, softmax, dP and softmax gradient. FP32 score/dP tiles stay in shared memory; FP16 P/dS still use global memory. Batched dQ contracts only visible keys. The representative backward improves from 5.2185 to about 2.16 ms. dQ relative L2 is 8.87e-7; dK/dV are byte-identical. Eight final random-input boundary cases pass, including all-masked rows, finite mask bias, Q-only and sinks. CUDA asynchronous dependency regression tests pass. See `fattn-fusion-validation.json`.

The final actual-model two-window diagnostic takes 172.087 seconds versus the immediate baseline's 175.499 seconds (1.94% reduction). Logged training and validation losses match. Eighteen of 24 optimizer state tensors match; six Q8 momentum tensors differ by 0.5-2.7% relative L2 after dequantization. These states match the previously validated broad prototype exactly. Two windows do not establish long-run convergence equivalence. The comparison script's exact-state assertion fails as expected; `results-fattn-fusion-final.json` retains `state_bytes_equal: false` and records the user's numerical-drift authorization explicitly.

Complete removal of P/dS global matrices was also implemented and tested, then rejected for performance. Full QKV backward holds these matrices in shared memory but needs repeated dP computation and atomic dK/dV accumulation: 4.568 ms versus the accepted 2.16 ms. Q-only variants keep probabilities in registers and dS in shared memory. Their best time is 1.386 ms versus the accepted Q-only path's 1.328 ms. Reusing two dQ accumulator tiles improved these candidates; staging Q/dO in shared memory regressed them. All final synthetic gradient cases pass, but full-fusion candidates were not run in production or actual-model loss diagnostics. Rejected sources and measurements remain in `fattn-full-validation.json` and `rejected-fattn-full-*.cu`; production source contains only the accepted partial fusion.

The 60-hour goal remains unmet. The two-window measurement is not a steady-state production ETA. Current run identity is recorded in `production-host-kernel-restart.json`; earlier PIDs in historical sections are obsolete.

## Accepted idle reduction build

The allocator caches tensor allocation sizes within each placement pass, including the buffer ID, and resets the cache at the next pass. Meta tensor slices use hash lookup; split-state cache hits reuse the existing iterator, and slice vectors move into their map. Tensor allocation order and numerical operations are unchanged. Temporary phase timers were removed from the final build.

The second-window scheduler allocation/initialization falls from 3.671 to 3.196 seconds (12.9%); including scheduler graph splitting, allocation falls from 3.776 to 3.302 seconds. The shared GPU idle span around that transition falls from approximately 5.4 to 4.8 seconds in 200 ms nvidia-smi samples. These are single-transition observations, not a steady-state utilization guarantee.

Two-window actual-model elapsed time falls from 176.283 to 175.499 seconds (0.44%). All 24 optimizer tensors are byte-identical, and training/validation losses match. See `idle-reduction-validation.json`. The 60-hour target remains unmet.

## Historical accepted build and live run

The original production script was restarted at 2026-11-16 00:28:21 UTC. At that historical restart, trainer PID was 1470870, launcher PID 1470866 and telemetry PID 1470867. Every original argument matches, including warmup 50 and CPU gradient accumulation. `production-host-kernel-restart.json` is the current run identity; earlier PIDs below are historical.

The final actual-model two-window diagnostic takes **208.394 seconds versus 298.700 seconds**: 30.2% less elapsed training time, or 1.433x throughput. Both training losses, validation loss and all 24 optimizer state tensors are byte-identical. All 32 dense/MoE integration runs pass, and all 1,840 optimizer state tensors match the original integration baseline. The asynchronous CUDA collective/host-copy/dependency suite passes. See `final-performance-validation.json`, `results-recompute-fixed.json` and `integration-recompute-fixed-baseline.json`.

The accepted recompute dependency cutoff uses forward node order to reject ancestors that precede the checkpoint boundary. CPY destination self references are ignored when verifying order; other forward references disable the cutoff. This preserves clone creation and graph order. Actual recompute preparation falls from 23.187 seconds to 0.202 seconds on the first window, and from 21.498 seconds to 0.163 seconds on the second. First graph allocation/preparation falls from 30.189 seconds to 7.033 seconds. The unchanged replay graph has 2,014 regions and 153,243 nodes. No probe logging remains in accepted source.

The 60-hour target is **not met**. The two-window diagnostic alone projects about 175 training-only hours; this is not a steady-state ETA and excludes validation/saves. Remaining CPU preparation and GPU compute still cause idle intervals; complete removal of GPU pauses has not been established. The main improvements are CPU-gradient CUDA acceleration and removal of repeated recompute dependency scans. Production telemetry is `production-host-kernel-gpu.csv`.

## Historical experiments

This directory records local optimization of the private AIKAR training fork. Production arguments, data, optimizer, precision and hardware settings remain unchanged. No commit or push was made.

## Accepted implementation

- `acc.cu`: sparse slice accumulation avoids touching unrelated data for in-place updates; out-of-place updates copy once before updating the slice. Irregular layouts retain the existing path.
- `fattn-back.cu`: skip clearing outputs written with beta zero; retain clearing for atomic accumulation.
- `allreduce.cu`: replace CPU event waits at reused communication slots with cross-stream dependencies on both devices; record completion for every chunk.
- `ggml-backend.cpp`: queue both Meta shard reads before synchronizing host copies.
- `ggml-backend-meta.cpp`: submit the two distinct CUDA backends concurrently, joining host submission before collectives. Unsupported asynchronous copy layouts use the synchronous fallback. Other backend configurations retain sequential submission.
- `mmq.cu`: select Volta Q4 routed tile width from tokens per expert.
- `out-prod.cu`: reduce Volta Q4 expert backward column tiles from 64 to 32 and reuse shared workspace while preserving accumulation order.

`ggml-opt.cpp` retains its original graph placement. Rejected cache tiling and graph-reordering changes are excluded. The host outer-product path below supersedes the earlier GPU gradient-placement experiments.

## Controlled asynchronous submission measurement

Actual model, same Nsight instrumentation, one training window and one validation window, diagnostic warmup zero:

| Build | Training seconds | GPU 0 active | GPU 1 active | Largest individual idle gap |
| --- | ---: | ---: | ---: | ---: |
| Previous accepted kernels | 169.421 | 76.9% | 78.4% | 4.91 s |
| Async collectives and expert tiles | 162.139 | 77.7% | 78.2% | 4.94 s |
| Also concurrent CUDA submission | 152.098 | 81.9% | 82.4% | 2.62 s |

Active fractions count the union of device kernels/copies from the first Q4 matmul through the last validation kernel, excluding startup and initial graph construction. The asymmetric 4.9/3.3-second gaps disappear. The remaining 2.62-second shared gap occurs between training and validation, with little CUDA API activity. This measurement does not establish continuous production occupancy.

All 24 optimizer state tensors and training/validation losses match bytes across these runs. Standalone realistic expert forward is about 2.2x faster; expert backward improves 14-17%, with identical output bytes.

## Remaining production bottleneck

A single-window diagnostic performs a partial optimizer update and misses the gradient-only accumulation path. The production configuration accumulates six windows. Instrumentation of this path measured a 465-node CPU split at 52.682 seconds: 224 F32 OUT_PROD nodes, 217 ADD nodes and eight Q8 copies. CPU placement of Q8 gradient copies pulls subsequent gradient matrix products onto the CPU. Graph allocation/preparation adds further host time.

The accepted two-window diagnostic takes 298.700 seconds, including the first gradient-only window at approximately 196 seconds. This is not a steady-state throughput estimate. The previous production restart observed its first two windows at approximately 196 and 103 seconds.

Rejected experiments:

| Experiment | Two-window seconds | Reason excluded |
| --- | ---: | --- |
| Generic CUDA SGEMM gradient placement | 248.731 | Seven momentum tensors differ |
| Ordered F32 CUDA, initial version | 259.521 | Eight momentum tensors differ |
| CPU F32 / quantized cache tiling | about 299 | No actual-model improvement despite isolated wins |
| Corrected fused ordered CUDA placement | 250.140 | Four attention-output momentum tensors differ |

The last experiment preserves sequential fused FMA accumulation and leaves the first already-GPU target alone. It matches 20/24 state tensors, with maximum relative L2 difference 0.05648 in the other four. Its identical losses do not establish equivalent training. See `results-grad-fused-ordered.json`. These changes were removed before restarting production.

## Validation and live run

Previous checks passed: CPU ACC 28/28, backend checks 26/26 on P620 and each V100, V100 116-expert cases 4/4, mixed-size asynchronous collective stress and CUDA-to-host shard copies. CPU Meta-to-CPU scheduler copying is a pre-existing unsupported case, reproduced with the old libraries; this work does not fix it.

Final build and whitespace checks pass. The final two-device CUDA unit check and all 32 full-context integration runs pass. Latest outcomes are in `unit-final-accepted.log`, `integration-final-accepted.log`, and the copied integration JSON reports (exit code zero). Runtime identity and observations are recorded in `production-final-restart.json`; the launch uses the original external `train.sh`. Historical launcher PID was 901390 and trainer PID was 901394; the trainer command matches every original argument. Its first window completed in 196 seconds with loss 4.36751747 and accuracy 4.585979%; the initial progress ETA was 330:42:01, which is not a steady-state estimate. Earlier restart JSON files are historical. `production-final-gpu.csv` belongs to this restart; older telemetry spans diagnostics and idle periods.

## 60-hour target

6057 training windows require less than 35.661 seconds per window before validation (73 windows), checkpoints and startup. The target is not met. The one-window asynchronous diagnostic projects 255.90 training-only hours; the accepted two-window diagnostic projects 251.28 hours. Neither is a reliable steady-state production ETA. Long shared GPU idle intervals from CPU gradient work remain unresolved.

## Host gradient acceleration follow-up

The CPU F32 outer-product fallback keeps two rows in SIMD registers, preserving sequential FMA order. The actual two-window run improved from 298.700 to 287.016 seconds, with all 24 optimizer tensors byte-identical.

The optional CUDA registry callback now receives the original CPU input tensors, preserving graph placement and CPU Q8 quantization. It accumulates each output in registers in the original K order, seeded by the existing accumulator when fused with ADD. Only active CUDA backends are eligible. Two eligible devices split output rows and use independent nonblocking streams; host completion waits before CPU quantization consumes the result. Both devices allocate workspace before either changes the host accumulator, so allocation failure falls back to CPU without applying a slice twice. The CPU reference path and unsupported layouts retain their original behavior.

The single-device actual-model run took 257.981 seconds versus 298.700 seconds. Training/validation losses and all 24 optimizer tensors were identical. See `results-host-kernel.json`.

The corrected standalone benchmark allocates persistent input buffers across iterations and checks finite output. The large accumulated matrix measured about 174 ms on CPU and 39 ms across the two V100s; all 92 MB of output were byte-identical. K=32,33,127,512 and odd output rows passed on P620 (`host-out-prod-tails.json`). Earlier benchmark runs that reused input memory are invalid and superseded by these finite-output checks.

The broad host graph reordering experiment is rejected: 547 splits, no actual-model speed gain, and optimizer-state drift in repeated dense accumulation. `rejected-host-async-backend.cpp` preserves its source. Its Nsight run also hit a temporary storage quota and produced an incomplete state file; that state was removed and is not a reference. Only rejected experiment outputs were cleaned.

Final dual-device actual-model validation took 252.688 seconds (15.4% less time, 1.182x throughput). All 24 state tensors and training/validation losses match. The asynchronous CUDA collective/host-copy/dependency unit suite passes. Evidence and live production identity are recorded in `final-host-kernel-validation.json`, `results-host-kernel-dual.json` and `production-host-kernel-restart.json`. Production trainer PID is 1353073, started through the original external train.sh, with exact original argv. Earlier PIDs are historical. Original production arguments, warmup, targets, data, hardware settings and external launch script are preserved. The 60-hour target remains unproven; graph preparation and activation recompute still require host work.


## Recompute graph search experiment history

Production after the dual-device host kernel completed its first two windows in 252 seconds (first window 150 seconds), with the same first losses. Sampling still found shared idle intervals of 26-35 seconds between windows. These align with CPU graph preparation; they are not eliminated by accelerating the CPU outer products.

A guarded forward-order cutoff in recompute dependency search passes 32 integration runs and matches all 1,840 optimizer state tensors against the original integration artifacts. The actual-model diagnostic is 253.919 seconds, which does not improve on 252.688 seconds; this cutoff is not accepted on performance grounds. A prepare-only instrumented probe is checking why it failed before deciding whether another minimal change is justified. Production is temporarily stopped during this experiment, under the prior restart authorization. Historical PID 1353073 is stopped; the next live restart will be recorded separately.

The corrected cutoff excludes CPY self references. The original guard rejected node 215 because its CPY destination references itself, disabling the optimization. A prepare-only probe counted 108,395,970 memo entries without the cutoff. The corrected implementation is accepted after the final actual-model and integration comparisons above. The prepare-only probe was intentionally terminated before training; its exit -15 is expected and is not a validation result. Earlier claims that ggml-opt.cpp is unchanged describe historical builds only.

Production confirmation: trainer 1470870 completed its first window in 127 seconds with loss 4.36751747 and accuracy 4.585979%. The second window is running. Live run observations and a sampled utilization snapshot are recorded in production-host-kernel-restart.json. Sampling is not an exact CUDA timeline and excludes initial model setup; it does not prove elimination of every idle interval.


## Additional kernel optimization, 2026-11-16

The latest production restart is recorded in `production-host-kernel-restart.json`; trainer PID 1637626 supersedes the historical PIDs above. It uses the original external train.sh with exactly matching argv, including warmup 50. Production arguments, data, precision and GPU settings are unchanged.

The earlier valid CUDA trace ranks FlashAttention, Q4 routed matmul and MoE backward highest. CONCAT contributes 7.480 aggregate GPU seconds across 57,480 calls. The new Nsight attempt exited 139 and collected no CUDA events, with an unsupported-driver warning. It is excluded from performance conclusions; see `kernel-profile-limitations.json`.

Two additional changes are accepted:

- MoE Q4 backward: all eight warps scan weight scales and reduce the maximum. The MMA accumulation and scaling are unchanged. V100 up/down shapes improve from 3.194/4.864 ms to 2.970/4.524 ms. All 6,488,064 output elements match bytes.
- CONCAT: reuse the existing copy kernel with uint4 when source row lengths, pointers and outer strides are aligned to 16 bytes. Other layouts keep the existing path. The 256x512x32 plus 256x256x32 F32 case improves from 0.192 to 0.083 ms (2.33x), with 6,291,456 identical output elements. This reduces copy instructions and address calculations; it does not remove the necessary HBM input/output traffic.

The actual two-window diagnostic improves from the previous accepted 208.394 seconds to 204.765 seconds (1.74% less time). MoE alone takes 207.744 seconds. All 24 optimizer tensors match full byte strings against the original CPU reference; training and validation losses match. Diagnostic warmup zero and two-window limits are only for controlled comparisons. They are not production settings or a steady-state ETA.

Validation: 187/187 CONCAT backend cases, 2/2 V100 116-expert MoE cases, asynchronous CUDA collective/host-copy/dependency tests, build and whitespace checks pass. Measurements and hashes are in `kernel-optimization-validation.json`, `moe-cooperative-scale-results.json`, `concat-vector-results.json`, `results-vector-concat-final.json` and `kernel-final-source-hashes.json`.

The 60-hour goal remains unmet. It requires less than 35.661 seconds per training window before validation/checkpoints. FlashAttention and Q4 matrix multiplication remain the largest kernel targets, with host graph preparation and ordered CPU-gradient transfers also remaining. No claim is made that every multi-second GPU idle interval has disappeared.


## FP4-aware attention follow-up, 2026-11-16

The accepted production restart is `production-host-kernel-restart.json`, trainer PID 1785741, started at 2026-11-16 01:43:50 UTC through the original external train.sh. Original argv matches exactly; warmup 50 and all production settings are preserved.

Lumen4Exp global KV uses FP4_KV (E2M1 with one E4M3 scale per 16 values). Full-sequence training quantizes the F32 master to FP4, restores F32 and applies STE, then shares an F16 attention input across query tiles. SWA does not use this global FP4 fake quantization. STE controls gradients and can coexist with a future packed FP4 forward kernel. V100 has no native FP4 Tensor Core instructions. Existing D256 sparse dispatch selects Q8 KV, so enabling it unchanged would alter precision.

Accepted CUDA changes are limited to Volta D256 attention: keep Q in shared memory for the 64-column configuration, and skip fully negative-infinity masked K/V tiles after a block-wide mask check. Surviving tiles retain their original accumulation order. The relevant kernel stack drops from 480 to 64 bytes. FP4 quantization and the STE backward path are unchanged.

| Standalone shape | Before ms | Tuned ms | Output |
| --- | ---: | ---: | --- |
| D256, 16K KV, window 1024 | 6.369 | 1.297 | byte-identical |
| D256, 16K KV, dense causal | 6.372 | 5.529 | byte-identical |
| D256, 4K KV, window 1024 | 1.649 | 0.551 | byte-identical |
| D256, all masked | 0.454 | 0.102 | byte-identical |
| D512 global, FP4-restored inputs | 10.622 | 10.622 | byte-identical |

Nsight independently measures the D256 SWA kernel at 6.732 ms before and 1.327 ms after. These are isolated kernel measurements, not full training speedups. New mask regressions pass 6/6 and existing selected attention regressions pass 19/19. The asynchronous CUDA collective/host-gradient suite passes. Q4 DP4A unpack reuse was tested on six full/sharded/fused expert shapes with identical outputs but no significant speedup; that source change was removed.

The actual-model two-window diagnostic takes 191.805 seconds versus the previous accepted 204.765 seconds (6.33% less time), and the original 298.700 seconds. All 24 optimizer state tensors and training/validation losses match exactly. See `fa-final-validation.json` and `results-fa-mask-final.json`.

A separate forward benchmark runs CUDA FP4 quantization, F32 restoration, STE arithmetic and F16 conversion on every invocation: 16.736 ms including attention, versus 10.622 ms with prepared FP4-restored F16 inputs. Full outputs match bytes between CPU-reference FP4 inputs and the CUDA STE graph. The 6.114 ms difference includes quantization that a packed FP4 kernel would still need; it is not an achievable speedup prediction. Production shares conversion across 64 query tiles for a 16K window, so this experiment overstates conversion frequency relative to production. A packed FP4 forward requires measuring repeated unpack against saved F16 traffic, and preserving STE backward; it is not implemented here.

The 60-hour goal remains unmet. The diagnostic averages 95.903 seconds per training window including initial graph work; the target requires less than 35.661 seconds before validation and checkpoint costs. This is not a steady-state production ETA. Remaining global attention, Q4 matmul, backward work and host preparation require further measurement.


## SWA backward range experiment, 2026-11-16

The dense Volta backward computes five GEMMs over 16K keys for each 256-query tile even when SWA permits only 1024 keys. The new internal range hints restrict score, probability, dV, dP and dK work to the aligned union of visible keys. Forward inputs and computation remain unchanged. dQ retains the original full-key GEMM reduction length with zero padding, preserving its numerical order. Full-size K/V gradient outputs are zeroed before writing the visible range. Other backends retain the original computation.

The D256/16K/256-query standalone backward improves from 13.508 to 5.221 ms (2.59x). All 68,157,440 Q/K/V gradient floats match bytes, including the original dense dQ. See `attn-back-range-benchmark.json`. CPU/CUDA full-context and asynchronous collective/host-gradient tests pass. The unit graph uses D32, so the D256 standalone and actual-model comparisons are necessary to validate the optimized Volta path.

Forward K/V cropping was rejected: it changed training losses. Its source is retained only as an experiment artifact. The unrelated F16 VIEW gradient fix was removed from the accepted candidate. A first backward-only actual-model run takes 183.727 seconds versus 191.805 seconds for the previous accepted build, with identical logged losses. Its optimizer-state output hit the /tmp user quota and is invalid; it cannot establish state equality. A repeat writes the state through a workspace symlink. Final acceptance and restart are recorded separately below.

The requested 179h to 60h reduction requires approximately 2.98x overall throughput. This isolated backward result does not establish that target. Q4 routed/dense matmul, global attention, other backward work and host preparation remain substantial.

Final repeat exits zero: 184.020 seconds for two windows. All 24 optimizer tensors and training/validation losses match the original reference byte-exactly. This is 4.06% less time than the immediately preceding accepted 191.805-second build, and 1.623x throughput relative to the historical 298.700-second reference. These include initial graph work and are not steady-state ETA predictions. See `results-attn-range-final.json` and `attn-range-validation.json`. Production restarted through the original external train.sh at 2026-11-16T02:36:39.891567+00:00, trainer PID 2047719, with exact original argv. The 60-hour goal remains unmet.


## Q4 register and load optimization, 2026-11-16

The Volta routed Q4 backward now loads two packed bytes and one scale per four reconstructed values. It retains the packed data and scale in registers, writes two half2 pairs directly to the existing shared MMA tile, and preserves scaling, FP16 conversion, MMA and output accumulation order. No full dequantized weight is written to HBM. The fused kernel drops from 100 to 81 registers per thread; shared memory stays 20,928 bytes and stack stays zero. Source-level redundant scale/packed loads and shared stores are reduced; HBM transaction counters were not collected, so a fourfold physical HBM reduction is not claimed.

A narrow Volta forward dispatch uses J32 instead of J24 for Q4 non-fallback shapes with 17-24 average expert tokens. It reuses the same weight tile across more output columns. It retains the existing dot implementation. The experiment with two independent integer DP4A accumulators was rejected: four forward shapes slowed by 1-3%. The broader J32 fallback experiment showed no up-projection improvement and was narrowed.

Fresh alternating saved-library/reference and final-library measurements:

| Shape | Reference ms | Final ms | Speedup | Output |
| --- | ---: | ---: | ---: | --- |
| Q4 routed backward, cols2816 rows352 | 2.968 | 1.966 | 1.510x | byte-identical |
| Q4 routed backward, cols352 rows2816 | 4.528 | 2.722 | 1.664x | byte-identical |
| Q4 routed forward, cols2816 rows352 | 0.797 | 0.796 | unchanged | byte-identical |
| Q4 routed forward, cols352 rows2816 | 1.005 | 0.943 | 1.066x | byte-identical |

All use 116 experts, 8 selected experts and 256 tokens. See `q4-register-final-micro-results.json`. These are isolated CUDA0 shapes, not whole-model speedup predictions. Existing 116-expert tests pass 4/4; scale/zero/masked-gradient tests pass 11/11; asynchronous collective/host-gradient tests pass. Actual-model validation is recorded separately.

The backward scope source audit found only eight selected attention weights registered as parameters. Frozen expert weights do not receive weight gradients; their input gradients remain necessary to propagate loss to earlier trainable parameters. The dependency pass prunes paths that do not depend on a parameter, and attention passes Q/K/V/SINKS need flags to the CUDA implementation. See `backward-scope-audit.json`; this is a source/log audit, not a newly collected complete per-node runtime census.

Actual-model validation exits zero: two windows take 176.272 seconds versus the previous accepted 184.020 seconds (4.21% less time). All 24 optimizer tensors and training/validation losses match exactly against the original reference. See `results-q4-register-final.json` and `q4-register-validation.json`. This includes initial graph work and does not establish a steady-state ETA or the 60-hour target. Production restarted at 2026-11-16T03:20:21.825941+00:00 through the original external train.sh, trainer PID 2147497, with exact original argv.
