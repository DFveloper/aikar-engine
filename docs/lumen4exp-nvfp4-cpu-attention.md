# Lumen4Exp direct NVFP4 CPU attention

These measurements describe the earlier row-scaled NVFP4 cache. Lumen4Exp now defaults to [FP4 KV with per-16 E4M3 scales](lumen4exp.md#default-fp4-kv-cache), which removes the FP32 row header. The direct CPU attention path supports both formats.

The CPU attention kernel now reads NVFP4 KV rows without materializing full-cache F32 and F16 copies. Each row keeps its FP32 master scale. AVX2/F16C converts packed values directly to F16, preserving the previous conversion order and rounding. The kernel reuses the existing online softmax, F16 accumulators and split-K reduction; up to eight query heads share each decoded KV row.

## Dispatch and memory

The global KV pool has 1,000,192 aligned cells, two independent cache layers, two KV heads per layer, K and V, and head dimension 512. Pure four-bit values require 1.9077 GiB. Each 512-value NVFP4 row stores 256 bytes of values, 32 bytes of sub-block scales and a four-byte FP32 master scale: 292 bytes total. Thus the allocation is `1000192 * 2 * 2 * 2 * 292 = 2336448512` bytes, or 2.17599 GiB (2,228.21 MiB). The additional 14.0625% is format metadata, not duplicated caches. Four server slots share this global pool. The local SWA F16 pool is separate.

With Flash Attention enabled, the model graph automatically selects this path for CPU-resident paired NVFP4 K/V, at least 512 cache cells, one query token and one stream. At 65,536 cache cells and above, prefill and batched decode also use direct attention to bound intermediate storage. Below that threshold, prefill, batched decode and shorter caches retain their previous conversion path because direct attention did not consistently improve measured throughput. GPU caches retain their previous conversion path. Mixed NVFP4/F16 attention operands are rejected by CPU backend support checks.

Cache precision, model weights and routing decisions are unchanged. The long-cache path removes whole-cache conversion intermediates during both decode and prefill; query/key masks and other graph outputs still require memory. Registers hold temporary values within an executing kernel and cannot replace persistent KV or tensors passed between kernels. At 8K active tokens, observed process RSS changed only by tens of MiB before the reservation change. The earlier mmap/repacking fix is responsible for the large reduction from approximately 30 GiB to 17 GiB.

For host NVFP4 KV, conversion and attention nodes are explicitly assigned to CPU. With `--no-op-offload`, attention for other host KV formats is also kept on CPU. This prevents GPU backend propagation from copying CPU KV and its full-cache conversion intermediates into a small GPU when projection weights are selectively offloaded.

The final real-server check retained the launcher's 1M context and four slots. CPU compute reservation fell from 8,860.29 MiB to 1,085.79 MiB (87.7%). A 652-token prompt and eight generated tokens completed successfully, with 55.20 prompt tok/s and 10.13 generation tok/s. RSS after that request was 18,753,404 KiB (17.88 GiB). The earlier 17.36 GiB RSS check used a much shorter prompt, so these RSS snapshots do not establish a memory regression or savings at the same workload. The reservation reduction is the directly comparable result. Production was not restarted.

## Measurement method

Hardware is an Intel Xeon E5-2696 v3, 18 physical cores, AVX2/F16C/FMA, and a Quadro P620 with 2 GiB VRAM. Model: `/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/Lumen-3.8-Pulsar-base.gguf`.

The whole-model profiler allocates a 1M-token context, one sequence, compact SWA, 18 threads, CPU repacking and `-lm none`. It progressively prefills fixed repeated token IDs to depths 128, 2048 and 8192, warms up three tokens and times 16 further fixed-token decode positions without per-node callbacks. It saves every vocabulary logit at those 16 positions. Four separate decode iterations use synchronized per-node callbacks to group bottlenecks. Callback timings include instrumentation overhead and should not be equated to unprofiled wall time. Other host workloads cause measurable run-to-run variation.

The baseline binary and libraries were preserved before the attention changes. Candidate and baseline consume the same tokens rather than allowing a sampling difference to alter subsequent inputs. This verifies a finite set of logits, not general model quality.

## Whole-model results and bottlenecks

| Active depth | Baseline, tok/s | SIMD candidate, tok/s | Final dispatch, tok/s |
| ---: | ---: | ---: | ---: |
| 128 | 11.715 | 10.167 | 11.003 |
| 2,048 | 7.751 | 8.685 | 9.377 |
| 8,192 | 6.183 | 7.522 | 8.717 |

At 8K the two candidate runs are 21.7% and 41.0% faster than the earlier baseline. Treat this as a noisy observed range rather than a stable promised gain. The final dispatch preserves the original short-context graph; its remaining short-context timing difference illustrates host noise. All 16 full vocabulary logits vectors at each of three depths are bitwise identical to baseline (48 positions, 12,582,912 float values).

The synchronized 8K decode profile below divides grouped time by four iterations. Percentages refer to that instrumented profile. Removing the two conversions saves about 43 ms/token in this profile; attention math, MoE and projection costs remain similar.

| Group | Baseline, ms/token | Final, ms/token | Final share |
| --- | ---: | ---: | ---: |
| Attention | 36.49 | 37.49 | 24.8% |
| MoE matmul | 32.66 | 32.02 | 21.2% |
| Attention projection | 28.26 | 27.58 | 18.2% |
| Dense FFN/router | 16.03 | 15.77 | 10.4% |
| Output projection | 14.22 | 15.50 | 10.2% |
| NVFP4 -> F32 | 17.65 | 0 | 0% |
| F32 -> F16 | 25.50 | 0 | 0% |

## Isolated attention measurements

A same-process alternating comparison uses a persistent 18-thread pool, head dimension 512, two KV heads, sixteen query heads, one query token, and synthetic NVFP4 rows with varying master scales. The baseline includes full NVFP4 -> F32 -> F16 conversion plus attention. Each reported time is a median, and outputs are bitwise identical for these inputs.

| KV tokens | Previous path, ms | Direct path, ms | Speedup |
| ---: | ---: | ---: | ---: |
| 2,048 | 3.938 | 1.058 | 3.72x |
| 8,192 | 17.550 | 3.357 | 5.23x |
| 65,536 | 300.802 | 28.709 | 10.48x |
| 1,000,000 | 4,809.115 | 439.812 | 10.93x |

The 1M result measures one synthetic attention operation, not full-model throughput at a 1M-token context. Full-model gains are smaller because MoE and projection operations remain.

## Validation and reproduction

The NVFP4 backend suite passed 30 cases; two mixed-type cases were correctly unsupported. `test-quantize-fns` passed. Backend coverage includes head dimensions 64/512, short/long KV, query counts 1/8/64, GQA, permuted cache views, multiple streams, masks, all-masked outputs, sinks, ALiBi and softcap. Additional cases broadcast KV across query streams. Quantization tests compare SIMD F16 output bytes with the original codec followed by F16 rounding, including very small, large and ordinary scales. The scalar fallback exists but has not been executed on a non-AVX2 machine.

Build with `CCACHE_DISABLE=1 cmake --build build --target llama-server test-backend-ops test-quantize-fns -j 6`.

Run `build/bin/test-backend-ops test -b CPU -o FLASH_ATTN_EXT -p 'type_K=nvfp4|type_V=nvfp4'` and `ctest --test-dir build -R '^test-quantize-fns$' --output-on-failure`.

Session artifacts are `/tmp/lumen-nvfp4-profile.cpp`, `/tmp/lumen-nvfp4-profile-{baseline,candidate}`, `/tmp/lumen-nvfp4-{baseline,simd,final}*`, `/tmp/lumen-nvfp4-kernel-bench.cpp`, `/tmp/lumen-nvfp4-kernel-results.txt`, and `/tmp/lumen-nvfp4-kernel-1m.txt`. These `/tmp` files are session-local.
