# Mixed-data CPU throughput

## V100 extraction overlap measurement

On 2026-11-09, two V100-SXM2-16GB GPUs with tensor split 1,1 extracted the same two windows from the webtext dataset using `gemma4-v2-Q8_0.gguf`, context/batch 16384, microbatch 4096, eight CPU threads and top-512 at T=1. Both runs used `GGML_CUDA_P2P=1`, `GGML_CUDA_ALLREDUCE=none` and `GGML_CUDA_DISABLE_GRAPHS=1`.

| Extraction | Loop seconds | Input tok/s | Supervised tok/s |
| --- | ---: | ---: | ---: |
| Serial decode and encoding | 82.377 | 397.781 | 397.575 |
| CPU encoding overlapped with next decode | 78.502 | 417.416 | 417.200 |

The runs processed 32768 input tokens and 32751 supervised rows. The complete cache files were byte-identical, with SHA-256 `8d720d0dc34afba556c734036e60679ff74fb4df6f6c4ea748e3e70087057332`. This short comparison measured a 4.9% throughput increase; it does not establish sustained throughput for the entire corpus. The overlap adds up to 4 GiB of host staging memory for these dimensions. Raw logs are retained in `/mnt/data/lumen-teacher-logits/webtext-25m-benchmaxx/overlap-benchmark/`; the compared caches are temporary files `/tmp/logit-before.dat` and `/tmp/logit-after.dat`.

A follow-up removed staging for the final microbatch of each window, saving up to 4 GiB of host copying per window. Its cache remained byte-identical to the serial reference. Follow-up measurements observed GPU0 at approximately 397 MHz and GPU1 at 997 MHz; they are not directly comparable to the earlier rates. With the same two input windows, F32 internal AllReduce measured 352.887 input tok/s, the direct-final-microbatch path with peer reduction measured 354.928, microbatch 1024 measured 368.325, and CUDA graphs enabled measured 354.099. The microbatch-1024 cache differed from the reference, so the running extraction retained microbatch 4096, peer reduction and disabled CUDA graphs. These short runs do not establish a speed improvement from removing the final copy. Logs are retained beside the earlier benchmark logs.

## Extraction across window boundaries

On 2026-11-10, the same two-window V100 test with microbatch 4096, peer reduction and disabled CUDA graphs measured 76.940 s (425.890 input tok/s) when encoding stopped at window boundaries and 75.981 s (431.266 input tok/s) when the last microbatch encoding overlapped the next window. This is a 1.3% increase in one short comparison, not a sustained-throughput guarantee. Both complete caches were byte-identical. The change preserves the cache format by deferring each window header until the previous encoding finishes. Host staging and full-vocabulary device-to-host transfers still cause short GPU idle intervals. Logs are retained in the same `overlap-benchmark` directory.

## GPU top-k extraction

On 2026-11-10, the same two windows and numerical settings measured 75.666 s (433.061 input tok/s) with CPU overlap and 61.937 s (529.054 input tok/s) with `GGML_DISTILL_GPU_ENCODE=1`, a 22.2% throughput increase in this short comparison. GPU clocks remained approximately 397/997 MHz. Both complete caches were byte-identical over 32751 supervised rows, including all IDs, FP16 log-probabilities and FP32 tail values. This does not guarantee bit identity for every distribution: FP64 normalization reductions have a different summation order.

The graph computes top-k IDs and F64 normalization statistics on the device, reducing host transfers from 4 GiB to about 24.1 MiB per full microbatch. CPU sorting and cache packing overlap the next decode. GPU0 utilization samples every 200 ms showed utilization below 10% in 13.9% of CPU-path samples and 1.3% of compact-path samples; the longest consecutive low-utilization sample run decreased from approximately 2.2 s to 0.4 s. These sampled utilization figures are approximate, not a kernel timeline. Logs, utilization CSVs and numerical comparison results are in the existing `overlap-benchmark` directory under `compact-*`.

CPU and both V100 CUDA devices pass compact-encoding reference checks with vocabularies 13, 257, 4097 and 262144; k=1 and up to 512; one and three rows; T=0.5, 1 and 2; random logits, cutoff ties, signed zero, extreme tails and NaN rejection. The TOP_K_STATS backend comparison passes 12 cases on CPU and CUDA0. Existing distillation gradient, tensor-parallel and cache checks also pass. Another 89 CUDA TOP_K regression cases pass. Resuming one complete CPU window followed by a torn next window with device encoding preserves the first window and reproduces the corresponding device-encoded remainder exactly.

## Historical CPU measurements

These measurements used the earlier Korean multi-turn mixture. After the data audit, the default keeps only the first Korean user/assistant pair. The current 634-row mixture has 4950283 bytes, 4493 training windows and a 6609698365-byte teacher cache at context 512. Rates below are historical measurements, not a benchmark of the new corpus. See DATA_AUDIT.md and run benchmark.sh for a new measurement.

Date: 2026-11-04. CPU: Xeon E5-2696 v3, eight threads, no GPU. Context=batch=microbatch=512, parallel sequences=1, gradient accumulation=1, teacher top-512 plus tail at T=1. Training uses pure KL, QLion2, learning rate 1e-5 and static graph reuse.

## Dataset and targets

All 317 nonblank JSONL records from /mnt/openwebui/AIKAR/train_ko.jsonl are retained. Reservoir sampling without replacement selects 317 of 120342 records from /mnt/openwebui/AIKAR/train.jsonl, with seed 42. The deterministic shuffled output has 634 records and 6138194 bytes. This is a 1:1 conversation ratio, not a token ratio. Both extraction and training use the same mixed file, split and masks.

Training targets only attention Q/K/V/output weight names under blk.11, blk.17, blk.23 and blk.29. The runtime registers exactly 12 tensors: Q, K and output for each block. There are no separate V tensors in this model. These blocks reuse global KV from block 5, so their stored K projections are unused in the forward graph. Payload inspection confirms that the eight Q/output tensors change, while the four unused K tensors remain identical. No local layers or block 5 weights are selected. Model metadata marks global blocks as 5, 11, 17, 23, 29, using zero-based indices.

Frozen Q4_1 MoE weights still need to propagate activation gradients between selected attention blocks. The CPU MUL_MAT_ID_BACK support gate now permits Q4_1, using the existing block dequantization kernel. A direct dequantized matrix reference test passes at one and four threads; ten backend cases pass with multiple dimensions, duplicate expert IDs and strided inputs. Q4_1 FFN weights are not optimizer targets.

## Measured rates

| Stage | Input presentations | Supervised presentations | Loop seconds | Input tok/s | Supervised tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher extraction, four windows | 2048 | 1892 | 77.418 | 26.454 | 24.439 |
| Training, two windows / two updates | 1024 | 959 | 290.502 | 3.525 | 3.301 |

Training supervised tok/s is calculated from the cache's 959 training rows divided by the measured training time. The measured binary logs input throughput; the final instrumentation also prints the supervised count and rate. Extraction processing includes inference, top-k encoding and cache writes. Training includes forward/backward, optimizer, cache reads and graph preparation; epoch-end validation, initial model/dataset loading and final saving are excluded. Input presentations include packed prompts, padding and overlapping windows. This short run includes initial graph setup and is not a sustained full-dataset benchmark.

Raw extraction process wall time including loading/preparation is 111.73 seconds, corresponding to 18.33 input tok/s. Raw logs are under benchmark-20261104T103530Z: extract.log, extract.time, train-selected.log and train-selected.time. The earlier train.log records an abandoned all-layer attempt that exposed the missing Q4_1 support gate; its failure is not used as a throughput measurement.

Training exits successfully after two optimizer updates, validation KL 2.13099790 and saving a separate full-model checkpoint plus optimizer state. Whole-process training wall time, including loading, validation and serialization, is 409.87 seconds (2.498 input tok/s). Peak RSS is 21438512 KiB (20.45 GiB); compute workspace is 5634437120 bytes (5.25 GiB). The temporary benchmark checkpoint is removed after inspection; runtime logs and the small teacher cache remain.

## Production plan

The mixed corpus packs to 5562 training windows, with 2513367 supervised training presentations per epoch. Held-out validation uses 32 sampled windows and 14705 supervised positions. Cache size is 7799262856 bytes (7.26 GiB). The default three epochs produce 16686 optimizer updates and retain up to 17 checkpoint sets; the conservative storage budget is about 234.69 GiB after extraction. Full production training has not been started.

The model-directory copies of extract.sh, train.sh, production-extract.sh and production-train.sh all prepare/reuse the mixture by default. Override DATA to explicitly select a different existing file. The shared targets.sh sets the four-block attention selection. Run bash benchmark.sh to repeat extraction and training sequentially with those defaults. Small benchmark checkpoints are temporary; source models are never output destinations.

## Interleaved V100 training fusion (2026-11-12)

Two V100-SXM2-16GB GPUs use tensor split 1,1, F32 peer-copy reduction, disabled CUDA graphs, context/batch 16384 and QLion2 on the same eight Q/output targets. Each comparison uses the first training and first held-out validation window from the Korean CPT, Benchmaxx and Agentic caches: 49152 input and 24322 supervised training presentations, three optimizer updates and three validation windows. The native optimizer and loss are unchanged. Loading, validation and checkpoint export are excluded from training throughput.

| Configuration | Training seconds | Input tok/s | Supervised tok/s | Validation KL |
| --- | ---: | ---: | ---: | ---: |
| Microbatch 128, recompute off | 332.076 | 148.014 | 73.242 | 3.32645095 |
| Microbatch 512, recompute off | 141.645 | 347.009 | 171.711 | 3.31111950 |
| Microbatch 1024, recompute on, F32 expert GEMM | 115.657 | 424.979 | 210.293 | 3.29828215 |
| Microbatch 1024, recompute on, fused Q4_0 expert dX | 77.476 | 634.412 | 313.928 | 3.29769152 |

The selected configuration is 4.29x faster than the original microbatch-128 run in this short comparison. Sampled peak VRAM is 14938/15546 MiB. Changing microbatch also changes the truncated attention-backpropagation boundaries. These three updates do not establish recovery quality or sustained full-epoch throughput.

Volta Q4_0 routed dX now reads quantized weights directly and fuses tile decoding, gradient gather, Tensor Core multiplication and scatter. Per-route power-of-two scaling preserves small gradients during F16 conversion; accumulation and output remain F32. Route counting also computes gradient scales, prefix construction also emits 64-route tiles, and all-zero gradient tiles write zeros without multiplication. The selected 128x64 output tile reuses shared storage between MMA inputs and output fragments. No decoded expert-weight or gathered-gradient matrix is materialized. Other shapes, types and devices retain the existing implementation. `GGML_CUDA_DISABLE_FUSION=1` selects the F32 reference path.

CPU comparisons pass 59 cases; each V100 passes 39 supported cases, including 19 new cases for duplicate IDs, non-contiguous ID views, partial tiles, zero weights/gradients and gradient magnitudes from 1e-20 to 1e20. The scaled cases use 1e-6 NMSE tolerance. The final partial nvprof trace records the fused WMMA kernel; trace buffer warnings make its timing percentages approximate. Throughput above comes from separate unprofiled runs.

Raw logs, commands, small-fixture preparation code, source snapshots and `results.json` are retained in `/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/logit-distillation/interleaved/optimization-20261112`. The previous CUDA library is preserved in `build/interleaved-optimization-20261112/reference/libggml-cuda.so.0`. The production launcher keeps all 6057 training and 73 validation windows and saves adapters separately from the base model.

### QLion2 stability follow-up

The three-step throughput fixture did not establish long-run training quality. In the optimized full-data run, the same first three held-out windows gave KL 4.24073812 for the base and 5.58460577 at step 132 (+31.7%). Restoring the saved adapter and optimizer state reproduced 5.58460577 exactly. That run was stopped and its checkpoint pair retained. From identical step-132 weights, momentum, residuals and dataset position, one further update gave KL 5.59254155 with the fused backward and 5.59364272 with the previous F32 backward library; this is a local comparison, not a proof of trajectory equivalence.

The logged `lr` is the global cosine schedule over 6057 steps. QLion2 additionally multiplies it by `RMS(old_momentum)/(RMS(old_momentum) + tau)` independently for each 32-value block. At step 132, tensor medians of this factor ranged from 0.0264 to 0.4220 with tau=0.001, despite the global LR remaining near 1e-5. The logged `loss` is the epoch aggregate, so successive entries also reflect changing source composition.

A separate optimizer defect was demonstrated: with zero old momentum, QLion2's first effective update is zero, but requantizing the existing weights with a freshly computed scale changed Q4 weights by approximately 5.1% relative L2. Existing legal block scales need not be the scales produced by the reference max quantizer. CPU, CUDA and the QAT reference now compare projections using the existing and freshly computed scales, retaining the candidate with smaller squared error. An unchanged target preserves the original weight bytes. Error feedback and momentum updates continue; this does not freeze learning. The regression checks zero effective LR, small accumulated updates and a subsequent representable weight change. It passes on CPU and both V100s, while the preserved previous CUDA library fails the zero-update check.

The checkpoint loader also exposed an existing tensor-parallel limitation: writes had to cover complete split chunks, although adapter payloads and optimizer state are streamed in fixed byte chunks. Partial writes now splice the intersecting byte ranges, matching the existing partial-read mapping. Tests cover F32, Q4_0 and Q8_0, three split axes, repeated/segmented splits, boundary crossings and untouched bytes on CPU and both V100s.

Investigation logs and machine-readable results are in `runs/20261112T001619Z-476684/diagnostics` under the model's interleaved directory. Validation subsets are diagnostic samples, not the complete 73-window validation set.

After this repair, a fresh full-manifest trial at the unchanged global LR 1e-5 gave fixed-three-window KL 4.24073812 at step 1 (identical to base), 3.88225259 at step 8 and 3.87908629 at step 16 (-8.53% versus base). The saved adapter and complete optimizer state record step/window 16. At step 16, medians of the per-block LR factor across the eight target tensors range from 0.1952 to 0.7654. These short checks do not establish full-epoch quality.

A post-repair repeat of the same balanced throughput fixture measured 531.899 input tok/s (49152 tokens / 92.409 seconds). Both V100s were at 795 MHz with a 150 W power limit; the historical 634.412 tok/s sampling recorded 997 MHz during GPU work. The hardware conditions differ, so these rates do not isolate the optimizer repair's cost. No GPU power or clock settings were changed during this follow-up.

The model directory now contains `resume-interleaved-guarded.py` and an archived recovery trainer source/build script in `logit-distillation/interleaved/recovery`. The launcher resumes the adapter after capturing the original base, restores momentum/residual/schedule/dataset position, evaluates all 73 held-out windows initially and every 128 steps, and saves checkpoints at those periodic validation steps. It stops before saving if validation KL is nonfinite or more than 5% above the best observed full-validation value in that process. Its initial reference is the resumed step-16 model, not the original untrained model. The final epoch validation uses the same guard. The original step-132 pair and the repaired step-16 pair are retained separately.

### Full-validation regression after step 128

The guarded continuation completed full 73-window validation at step 128 with KL 2.50966557, improving on step 16's 3.88862539. At step 256, KL increased to 3.58475849 (+42.84% from the best), so the guard exited with code 3 before saving the rejected state. The complete step-128 adapter/state pair remains in `runs/20261112T034747Z-597538-repaired`.

Recovery diagnostics resume that step-128 pair at global LR 2e-6 / minimum 2e-7, retaining QLion2, its momentum/residuals, the 6057-step cosine schedule and the original data position. Nine fixed validation windows include all three sources; their values must be compared with the same nine-window baseline, not directly with the full 73-window metric. These trials test an LR-overshoot hypothesis; reducing LR is not yet proof of long-term stability.

The guarded launcher now accepts explicit LR, validation interval, prior best validation loss and prior best checkpoint arguments. Defaults for subsequent recovery runs are LR 2e-6, minimum 2e-7 and full validation every 64 steps. A provided previous-best KL initializes the guard, so restarting cannot silently reset the protection reference to a worse model. Periodic checkpoint pruning retains the best validated adapter/state pair as well as the latest pair. The loss guard and checkpoint-retention tests pass, including regression after an improving baseline, NaN and invalid inputs.

The nine-window lower-LR trial gave KL 2.27852773 before updates, 2.29565244 after eight updates, and 2.30749240 after sixteen (+1.27%). This shows neither an immediate large regression nor an improvement. The trial checkpoint is retained for diagnostics; production recovery instead rolls back to the full-validated step-128 model, uses LR 2e-6 / minimum 2e-7, carries best full KL 2.50966557 forward, and validates/saves every 64 steps (next at 192). Long-run stability remains unconfirmed. The new run and exact settings are linked from `recovery/latest-run.json`.

### Automatic LR backoff

The LR 2e-6 recovery validated at KL 2.50966557 (step 128), 2.58474046 (step 192), and 2.64130132 (step 256). The last value is +5.25% above the best and triggered the unchanged 5% guard. The step-128 pair remains the best full-validated model.

The model directory's guarded Python launcher now supervises repeated attempts. A confirmed validation-guard exit after training rolls back to the lowest full-validation loss with an available complete checkpoint pair and halves both base LR and cosine minimum. The prior best value is carried into the next native process. Other errors, failed saves, a failed initial restore validation and user cancellation do not trigger LR retries. The controller retains the V100 lock across attempts, atomically records status/history, and forwards termination signals to the trainer. LR reduction defaults to factor 0.5 with base-LR floor 1e-9; the floor and factor are configurable.

Eleven CPU-only fake-trainer integration tests pass, covering two successive automatic backoffs, LR/minimum scaling, checkpoint/data-position rollback, best-loss carryover, incomplete/mismatched checkpoints, non-guard failures, cancellation and the floor. The actual automatic run starts from step 128 at LR 1e-6 / minimum 1e-7. Its stable controller/status path is recorded in `recovery/latest-run.json`. This automation preserves the validation guard; it does not demonstrate long-run stability.


### Learning stagnation investigation (2026-11-13)

The automatic LR 5e-7 retry reached step 960 with full held-out KL 2.58078093, above the unchanged best step-128 KL 2.50966557. Continuing to lower LR did not establish learning progress. That controller was stopped; all checkpoints and logs remain available. Step 128 remains the quality baseline.

Inspecting the eight dense targets at step 960 found median effective QLion2 updates around 1.5e-8 to 2.87e-7, below the Q4 residual precision for most blocks. An opt-in F32 error-feedback residual now preserves small updates with compensated accumulation. Model weights and momentum remain quantized and the fused quantized backward GEMM remains in use. F32 mode also considers a least-squares scale for the existing quantization codes. CPU and both V100 devices pass independent accumulated-update checks for Q4/Q8 weights and F32/Q8 gradients. The legacy Q4 residual mode remains the default. Native recovery can migrate Q4 residuals, saves their actual type, and byte-checks restored state.

This numerical fix alone did not improve quality: a nine-window, three-source diagnostic went from KL 2.27852773 at step 128 to 2.33354283 at step 144. Restoring the F32 checkpoint reproduced the latter loss exactly. A second experiment started error feedback from zero and shuffled only the unconsumed training windows within each source, preserving source interleaving and all validation data. Its step-144 KL was 2.28590945, still above the baseline. These short diagnostics are not evidence of improved learning. A standalone cache test verifies that source shuffling preserves token/label/teacher associations and the consumed prefix; checkpoint metadata enforces the same permutation on resume.

The original eight Q/output targets read shared global K/V from block 5, whose K projection was frozen. An experimental K/V phase trains only `blk.5.attn_k.weight` while retaining the eight previously calibrated tensors in exported adapters. The model has no separate V projection. The phase initializes only the newly selected optimizer state and preserves the restored model and dataset position. Its additional backward workspace exceeded V100 memory at microbatch 1024; the next diagnostic uses 512 with the same 16384-token context. This phase remains experimental until held-out validation demonstrates a benefit. Diagnostic commands, raw logs and checkpoints are under the model directory's `logit-distillation/interleaved/stagnation-20261113`.

Fourteen CPU-only supervisor tests pass, including retention of F32-residual, source-shuffle, shared-K/V and microbatch options across automatic retries. The controller accepts an explicit microbatch override so a restart retains the measured memory configuration.


The microbatch-512 shared-K/V trial completed eight updates without OOM or CPU graph nodes. Its same-configuration nine-window KL improved from 2.27972300 (step 128) to 2.27797969 (step 136), approximately 0.076%. At step 132, all eight carried tensors were byte-identical to the prior best adapter; 4558 of 2883584 K weights changed, with relative L2 delta 2.92e-6. The new target's momentum was nonzero in approximately 97% of elements. This is a small subset improvement, not a new full-validation best. A bounded LR 1e-4 comparison resumes its saved F32 state to test whether the new phase can learn faster.


Continuing the K/V phase from step 136 to 144 at LR 1e-4 produced nine-window KL 2.27802040, effectively flat. Because those updates use subsequent training windows, they do not isolate the effect of LR. A matched comparison restarts step 128 and trains the same eight shuffled windows at LR 1e-4. The diagnostic continuation selects the lower-loss matched candidate, compares original and candidate weights on all 73 held-out windows at the same microbatch 512, and starts the existing guarded supervisor only if full validation improves. A failed comparison preserves the original best and stops; it does not launch another unproductive LR-reduction loop.


### Larger training microbatch (2026-11-13)

The shared-K/V run now uses microbatch 768 instead of 512, retaining context/batch 16384 and LR 1e-5. A window contains 21 microbatches of 768 tokens and one of 256 tokens. Native single-sequence, full-window QLion distillation now counts microbatches with ceiling division and weights partial batches by token count so the final fragment contributes the correct loss and gradient. Divisible configurations retain their previous weighting. Other partial-batch training modes remain rejected. The recovery launcher retains microbatch 768 across LR retries.

A synthetic actual-model test masks the first 768 tokens of a 1024-token window and supervises only the final 256. It produces exactly one optimizer update at the complete-window boundary with nonzero K-projection momentum, verifying that the partial tail is consumed. Existing CPU/two-V100 QAT tests and fourteen supervisor tests pass. A two-window real-data test at context 16384 advances step 136 to 138, saves a complete adapter/state pair, and peaks at 13026/13634 MiB on the two GPUs with no CPU graph nodes. This short test does not establish the maximum memory requirement over every training window. Its saved pair resumes the normal guarded training run with microbatch 768; original quality-best checkpoint 136 is preserved. Raw commands, logs and peak samples are under `stagnation-20261113/ubatch-768` in the model directory.
