# Pulsar serving optimization against Strata

Date: 2026-10-07

## Selected configuration

- Tesla V100-SXM2-16GB, CUDA 12 build targeting sm_70.
- Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf.
- 12 slots, 262144 logical context, 4096 physical blocks x 32 tokens.
- Q8_KV K/V, fused CUDA paged attention, backend sampling, CPU mmproj.
- Batch and microbatch remain 1280; mixed prefill chunk changes from 64 to 1024.
- CUDA Graphs remain disabled. Enabling them with microbatch 1280 ran out of memory during the mixed request workload. Microbatch 256 avoided the observed failure but reduced PP throughput substantially.
- Frozen package: `/home/user/aikar-engine/build-pulsar-strata-optimized/bin`.
- Runtime script: `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/start.sh`.

## Matched measurements

Fixed token IDs, 1024/8192 input tokens, 128 generated tokens, ignore_eos, greedy sampling, and explicit slot clearing for cold requests. No profiler was active during the measurements in this table. Baseline has two repetitions; candidate has three.

| Workload | Original | Selected |
| --- | ---: | ---: |
| Four distinct cold 1024-token requests, generated tokens/s including PP and request wall time | 54.02 | 96.36 |
| Single TG after 1024-token prompt, tokens/s | 51.56 | 51.63 |
| Single TG after 8192-token prompt, tokens/s | 51.86 | 51.99 |
| Cold single 8192-token PP, tokens/s | 1434.07 | 1425.64 |

The cold mixed request throughput improved by 78.4%. Baseline runs: 52.83, 55.21. Candidate runs: 90.27, 96.41, 96.36.

This is mixed PP/TG request throughput, not pure batched decode throughput. The repeated-prompt runs retained a cold slot 0 alongside three cached slots, so they must not be labeled pure cached decode benchmarks. Single PP and TG are essentially unchanged; the chunk option applies only while generation is active. Larger chunks reduce repeated small mixed prefill batches and allow dense matrix multiplication to use its existing FP16 tensor-core path. They can increase the pause between tokens for a request that is already generating.

These are short local measurements. They do not establish a general speedup for other prompt distributions, 12 simultaneous requests, or very long contexts.

## Configuration sweep

| Mixed chunk | Microbatch | Cold 8192-token PP, tokens/s | Four cold requests, generated tokens/s including PP |
| --- | --- | ---: | ---: |
| 64 | 1280 | 1434.1 | 54.0 |
| 128 | 1280 | 1434.9 | 67.7 |
| 256 | 1280 | 1429.8 | 77.0 |
| 512 | 1280 | 1430.2 | 85.8 |
| 1024 | 1280 | 1420.4 | 93.9 |
| 1280 | 1280 | 1426.1 | 94.9 |
| 512 | 128 | 720.4 | 56.9 |
| 512 | 256 | 958.2 | 70.8 |
| 512 | 512 | 1130.7 | 82.2 |

## Strata comparison and rejected kernels

`../Strata/docs/NVIDIA_V100.md` and `../Strata/src/kernels/cuda/qsa_prompt_attn.cu` show Volta FP16 MMA for prompt attention, FP16 GEMM for dense projections, and DP4A for quantized MoE experts. V100 has no INT8 tensor cores. This engine already uses FP16 GEMM for sufficiently large dense quantized matrices and DP4A for small quantized expert batches.

Two paged attention prototypes were evaluated: a 32-row kernel reusing `mma.cuh`, and a 16-row WMMA kernel. Both retain the Q8_KV layout and FP32 accumulation. The first needed a correction to 16-token page mask indexing and a high/low query split. Its selected accuracy suite passed 40/40 against the existing CPU and FP64 references; the second passed 24/24 selected cases. Compiled SASS contains HMMA instructions. Both were slower in kernel benchmarks and end-to-end PP; the WMMA version measured about 551 tokens/s on the 8192-token cold prompt. Neither is deployed. The paged attention source was restored byte-for-byte to the pre-task version.

Prototype sources, build logs, independent numerical tests, benchmark driver, raw JSON and original runtime script are retained in `/tmp/pulsar-strata-tc-20261007`. No PR or commit was created. Existing workspace changes were preserved.

## Verification and rollback

- Restored Release build completed for ggml-cuda and llama-server.
- Separate Nsight Systems package validation recorded 2460 calls to CUTLASS sm_70 s884 FP16 tensor-core GEMM, plus native Volta s884 GEMMs. Profiled timings are excluded from the performance table.
- All six selected serial greedy outputs matched the original baseline at 1024 and 8192 prompt tokens.
- Startup defaults were updated to the frozen package and prefill chunk 1024; the production script passed `bash -n`.
- Live port 11435 passed health, 12-slot/config checks, chat, CPU mmproj white-image recognition, and required structured tool-call checks.
- `git diff --check` passed.

To restore the old mixed prefill behavior, start with `PULSAR_PREFILL_CHUNK_SIZE=64`. The old runtime script is backed up at `/tmp/pulsar-strata-tc-20261007/start.before.sh`.
