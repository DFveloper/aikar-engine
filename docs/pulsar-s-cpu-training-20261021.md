# Pulsar S CPU training optimization

## Implementation

The private-fork patch tiles CPU MUL_MAT_ID_BACK dequantization across up to eight blocks and uses existing vector multiply-add helpers. Few-route cases retain one-block tasks. It selects native buffers for the QLoRA training model to avoid the reproduced repacked embedding backward abort, preserving separate GRPO rollout policy. It applies batch thread counts and CPU threadpools at every optimization iteration instead of only QAT CPU accumulation initialization.

No original model or training data was changed. No commits, services or full curriculum runs were started. Temporary profiling instrumentation was removed before performance measurement.

## Environment

- Baseline source: 521084952; GCC 15.2.0, CMake 4.2.3, Ninja, Release, GGML_CUDA=OFF, GGML_NATIVE=ON.
- Xeon E5-2696 v3, 18 physical / 36 logical cores, AVX2/FMA/F16C, one NUMA node, about 121 GiB RAM. No explicit affinity.
- Original benchmark model: `/mnt/data/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf`, 13,121,388,640 bytes, gemma4, about 23.09B parameters. Already on /mnt/data.
- Model SHA256: `891d003f6c0abb22778e20464661c721c9d0e3a7c7765bce8fb0d831910ba1a2`.
- Diagnostic SHA256: `1ee2d37164e852265b7efcf104d28b2d94a1a1cbd83ace6f594d1acf5b2d4c18`.
- Raw scripts, commands, logs, adapters and summary: `/tmp/pulsar-cpu-training-20261021/`. Archive these temporary artifacts before clearing /tmp.
- Separate builds: `build-pulsar-cpu-baseline-20261021` and `build-pulsar-cpu-candidate-20261021`. Do not rebuild the frozen baseline from patched source.

## Diagnostic training results

The fixed reduced diagnostic comes from the first real curriculum row in `/mnt/openwebui/AIKAR/Lumen-3.2-Flare/stage1.jsonl`. It retains the first user message and assistant final content truncated to 600 characters, removing reasoning. It has 256 input tokens, 161 supervised tokens, 32 padding tokens, four microbatches and one optimizer update. Three fresh repetitions are not a multi-update convergence test.

Matched commands are in `run-training.sh` in the raw result directory: seed 3407, rank16, alpha8, attn_output.weight only, context/batch256, microbatch64, learning rate8e-6, weight decay0, one epoch, no validation, no recomputation, flash attention off, explicit --no-repack. Runs were sequential without concurrent CPU benchmarks or builds.

| Case | Window median seconds (range) | Three microbatches after first | Median peak RSS KiB |
| --- | --- | --- | --- |
| Baseline, 4 threads | 184 (183-184) | 137 (137-138) | 14751372 |
| Candidate, 4 threads | 120 (119-120) | 90 (89-91) | 14752564 |
| Candidate, 18 threads | 38 (38-38) | 28 (28-28) | 14759752 |

Matched four-thread speedup is 1.53x. The 18-thread case combines code and configuration effects and omits --no-repack to exercise automatic native buffers. All runs report loss 3.5146 +- 1.3159. All 60 adapter tensors in every saved run are byte-identical to baseline-1, with maximum absolute difference zero.

## Operator and inference results

Separate profiling attributed 114.275 seconds to MUL_MAT_ID_BACK over four microbatches, followed by OUT_PROD at 27.3383 seconds. Only operations above 10ms were logged. Hardware perf and debugger attachment were blocked by host policy.

Pulsar-shaped standalone Q4_0 MoE dX uses 2816 columns, 1408 rows, four experts, eight routes and 64 tokens. Four iterations exclude the first as warmup.

| Threads | Baseline median seconds (range) | Candidate median seconds (range) |
| --- | --- | --- |
| 1 | 1.510041 (1.508737-1.557604) | 0.823767 (0.809425-0.999618) |
| 4 | 0.603857 (0.594464-0.622189) | 0.246273 (0.235372-0.274634) |
| 18 | 0.142967 (0.135513-0.144292) | 0.068724 (0.066557-0.068977) |

Outputs are byte-identical locally; this is not a portable bit-identity claim.

Inference benchmark: same original model, CPU only, flash attention off, prompt128/512, generation32, threads4/18, three repetitions. Four-thread baseline/candidate median tokens/s: prompt128 19.648/19.292, prompt512 19.829/19.856, generation32 7.209/7.246. Eighteen-thread values: 68.260/70.371, 69.957/69.518, 11.162/11.201. No material regression observed; no inference kernel changed. Full ranges are in summary.json.

## Verification and limitations

Expanded scalar-reference dX tests cover Q4_0/Q8_0/MXFP4, tile tails, padded gradients and IDs, repeated routes, and 1/4/18 threads. CPU backend cases pass 30/30, including padded expert rows and offset IDs; their reference uses the same kernel and is not an independent arithmetic oracle. Focused CTest passes 5/5: test-opt, test-qat, test-quantize-fns, test-tiled-mulmat and its thread-limit variant. Final rebuild and diff whitespace checks passed.

Full suite, CUDA, adapter resume, full-model gradient dumps, inference logits/perplexity, nonunit inner gradient strides, full-context convergence and curriculum throughput were not validated by the reduced diagnostic.

Fused QKV converter dry-run found 25 fusable layers, skipped5/11/17/23/29, tensors658->608, zero extra weight bytes. No fused model was created: adapter names and parameterization can change, and training equivalence is unproven.

## Actual stage1 window and recomputation fix

The user requested one actual stage1 window. The test uses the stage1 script's model `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.5-Pulsar_S-LD-Q4_0_XL.gguf` and Benchmaxx.jsonl, rank64, alpha32, Q/K/V/O adapters, 8192 context/batch, microbatch64, Q4_0 QAT plus existing tensor overrides, activation recomputation, F16 KV-cache training and flash attention. It runs on 18 CPU threads. This intentionally uses the actual stage1 model rather than the earlier comparison model.

The full dataset packs 503 training and two validation windows from 1310 samples. Initial validation is omitted for timing, and save-every1 plus a watchdog limits the run to the first completed training checkpoint. It fails before that checkpoint: microbatch one completes in 17 seconds, then CPU SET_ROWS asserts `i1 >= 0 && i1 < ne1` at ggml/src/ggml-cpu/ops.cpp:5982. The debugger helper cannot attach under the host ptrace policy. The test and its stuck debugger were terminated. No original adapter or script was overwritten.

The invalid indices belong to the inactive mixed-embedding input branch, not the KV cache. Activation recomputation rebuilt the reordered graph with unconditional `ggml_build_forward_expand`, which activated branches deliberately excluded by `ggml_build_forward_select`. The fix preserves each node's COMPUTE flag by using `ggml_build_forward_order` for inactive nodes. A selected-input regression fails before the fix and passes after it. It covers two dynamic graph evaluations, checks that the unused branch remains inactive, and compares forward outputs, loss and optimizer updates with recomputation disabled. The focused CTest selection passes 5/5 after this fix.

The repaired original microbatch64/recompute-on configuration completes five microbatches before an intentional stop for configuration testing. Peak RSS is 23,724,028 KiB. A microbatch256/recompute-off probe completes two microbatches in 47/104 seconds cumulative, with peak RSS 26,958,812 KiB. These are partial probes, not complete-window comparisons.

The CPU script is `scripts/pulsar-stage1-cpu.sh`. It retains the stage1 model, data and training hyperparameters, selects CPU execution with 18 threads, microbatch512 and recomputation off, and exposes UBATCH and RECOMPUTE overrides. Original stage1 scripts and model files remain unchanged. The script refuses an existing output/checkpoint family and another active trainer. Outputs default to /mnt/data.

Run `./scripts/pulsar-stage1-cpu.sh` to start the full stage1 epoch, including initial validation and periodic evaluation. This full run is not started automatically. For another configuration, use for example `UBATCH=256 RECOMPUTE=off OUTPUT_PATH=/mnt/data/AIKAR/Lumen-3.1-Pulsar/stage1-cpu-ub256.gguf ./scripts/pulsar-stage1-cpu.sh`. MODEL_PATH, TRAIN_FILE, TENSOR_TYPE_FILE, THREADS and QLION_BIN can also be overridden. The candidate binary must be built before use.

The full first window with microbatch512/recompute-off completes all 16 microbatches in **5940 seconds (99 minutes)** of logged training time. Total watchdog wall time through checkpoint save is 5950.065 seconds. Peak RSS is **30,661,404 KiB (29.24 GiB)**. Timing includes brief test pauses and some overlap with test-only builds/checks, so it is an operational measurement rather than an isolated benchmark. Initial validation is omitted. Later microbatches slow down with increasing KV context: cumulative times at microbatches 4/8/12/16 are 707/1959/3704/5940 seconds.

The checkpoint is `stage1-ub512-off.gguf.epoch1.ckpt1.gguf` in the raw directory. Its 230 tensors are all finite. All 115 LoRA B tensors started at zero; 60 have nonzero values after this window, with 14,581,760 nonzero elements and maximum absolute value 7.999999746e-21. The tiny update matches the preserved initial warmup learning rate 8e-21. Metadata confirms training window1, context8192, dataset503 and scheduler step1. The watchdog stops the process after the successful save (SIGTERM, return code -15); it does not finish the epoch. A second window may begin graph setup during the watchdog's polling delay.

At 5940 seconds per window, **503 training windows extrapolate to 34.58 days**, before initial/periodic validation and other overhead. This is a single first-window estimate, not a measured whole-epoch duration; label density, data content, scheduling and optimizer state can change throughput. The earlier seven-day estimate from the reduced diagnostic is superseded. Microbatch512/off is verified for one complete actual window; microbatch256/off and microbatch64/on have only the partial probes above. This does not establish that 512 is the fastest configuration or validate long-run convergence or checkpoint resume.

The raw directory contains the exact command, log, watchdog status and `stage1-checkpoint-inspection.json`. The final focused CTest selection passes 5/5, and shell syntax and diff whitespace checks pass. Code review found no remaining material issue in the recomputation fix, regression and CPU launcher.
