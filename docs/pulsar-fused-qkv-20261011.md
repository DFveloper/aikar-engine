# Pulsar 3.5 Fused QKV conversion and serving plan

Date: 2026-10-11

## Delivered files

Directory: `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar`.

- `convert_gguf_to_fused_qkv.py`: lossless Gemma4 Q/K/V row concatenation, no dequantization/requantization. Repo copy: `scripts/convert_gguf_to_fused_qkv.py`.
- `Lumen-3.5-Pulsar_S-LD-Q4_0_XL-FusedQKV25.gguf`: fuses the 25 layers with explicit Q/K/V. Leaves layers 5, 11, 17, 23, 29 separate because V reuses K. 608 tensors; no additional weight bytes.
- `Lumen-3.5-Pulsar_S-LD-Q4_0_XL-FusedQKV30.gguf`: all 30 layers fused, duplicating K rows for missing V. 603 tensors; 8,110,080 additional weight bytes (7.73 MiB).

The original model is retained. The engine already supports physical `blk.N.attn_qkv.weight` in `src/models/gemma4.cpp`; no C++ modification or rebuild was needed.

## Converter and verification

```sh
PYTHONPATH=/home/user/aikar-engine/gguf-py python3 convert_gguf_to_fused_qkv.py source.gguf fused25.gguf --verify
PYTHONPATH=/home/user/aikar-engine/gguf-py python3 convert_gguf_to_fused_qkv.py source.gguf fused30.gguf --missing-v duplicate --verify
```

Use `--dry-run` to inspect the conversion plan. Output must be a new filename. The converter writes to a temporary file, verifies when requested, then publishes without replacing any existing destination. Weight data is streamed one tensor at a time; fused tensors are temporarily materialized in RAM. Mixed Q/K/V quantization types, separate projection scales/biases, split GGUFs and big-endian files are rejected. This converter targets Gemma4; it does not claim general architecture support. The GGUF writer does not support empty metadata arrays. Verification of NaN metadata values is not supported.

Both real outputs passed comparison of all metadata keys/types/values, tensor shapes/types and every weight byte, including duplicated K rows. Small generated GGUF checks cover Q/K/V order, missing V, metadata including custom `GGUF.*` keys, unchanged tensors and refusal to overwrite an existing output. During development a NumPy uint32 alignment caused offsets beyond 4 GiB to overflow; passing alignment as a Python int fixed this. The final full-model verification checks the resulting offsets through all tensor data.

## Matched serving measurements

Tesla V100-SXM2-16GB; frozen `build-pulsar-strata-optimized/bin/llama-server`; 12 slots, logical context 262144, paged Q8_KV, 4096 blocks x 32 tokens, fused paged attention, CUDA Graphs disabled, backend sampling, CPU mmproj, batch/microbatch 1280 and mixed prefill chunk 1024. Three repetitions per model; medians below. All models use the same token IDs, 128 generated tokens, greedy sampling, explicit slot erasure and ignore_eos. No profiler was active.

| Model | 1K PP tokens/s | 8K PP tokens/s | TG after 1K tokens/s | TG after 8K tokens/s | Four cold requests aggregate tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1914.15 | 1437.15 | 51.50 | 51.94 | 96.90 |
| qkv25 | 1924.91 | 1436.13 | 51.95 | 52.49 | 97.71 |
| qkv30 | 1928.83 | 1430.52 | 51.92 | 52.36 | 97.84 |

Fused25 improves median TG by 0.88% after 1K and 1.07% after 8K; aggregate mixed request throughput improves by 0.83%. Fused30 improves median TG by 0.81% and 0.82%, and mixed throughput by 0.97%, while 8K PP falls by 0.46%. These small differences may fall within normal run variation. Neither conversion establishes a large end-to-end speedup or a robust winner. Prefer Fused25 for the next optimization experiments because it avoids redundant K projection work and has no added weight storage; Fused30 is available when all layers must use the fused format.

The aggregate metric includes prompt processing and wall time for four distinct cold 1K requests. It is not pure batched decode throughput. Cached-request probes are retained in raw data but excluded here because not every slot was fully cached.

All six serial synthetic completions per fused model match the corresponding original completions exactly. Parallel completions are not universally identical: Fused25 matches one of three four-request groups, Fused30 none. The original model itself also varies between repeated parallel groups, so batch scheduling is a confounder. Natural chat probes give the same arithmetic answer (391) and identical 128-token Python responses; Korean travel responses use different wording across all three models. Byte-identical weights do not guarantee identical floating-point execution or generated text after changing matrix shapes. These checks do not establish benchmark-quality equivalence on a wider evaluation set.

## Next optimization steps for start.sh

1. Remove redundant Q/K/V copies first. The existing fused Gemma4 graph calls `ggml_cont` on each Q/K/V slice before reshape and RMSNorm. Build correctly strided 3D views directly and consume them in RMSNorm instead. CUDA RMSNorm and fused RMSNorm/multiply already read `nb[1..3]` in `ggml/src/ggml-cuda/norm.cu`. This could remove 75 copy nodes for Fused25 or 90 for Fused30, without changing GGUF format. Validate CPU and CUDA support for the views, numerical error and real PP/TG behavior before adoption; node reduction is not a measured speedup.
2. Revisit CUDA Graphs only with bounded memory use. The production configuration occupies about 16 GiB of the 16 GiB V100. Prior matched tests in `docs/pulsar-strata-serving-20261007.md` hit OOM with graphs and microbatch 1280. Examine decode-only capture or a stricter graph/activation-cache memory bound, retaining large prefill batches. Current graph eviction uses idle time rather than an explicit byte budget. `GGML_CUDA_DISABLE_GRAPHS=0` still disables graphs because the code checks variable presence; it must be unset to enable them.
3. Then sweep batch/microbatch and mixed prefill chunk with the selected fused graph. Keep 1280/1280 and chunk 1024 as controls; compare microbatch 512/768/1024/1280 and mixed chunks 512/1024/1280. Measure TTFT, inter-token latency, aggregate throughput and peak memory at 1, 4 and 12 active slots. Smaller chunks may improve streaming latency at the cost of mixed throughput. Do not reduce the physical KV block budget without explicitly accounting for reduced shared context capacity.
4. If the five missing-V layers remain significant after profiling, consider a fused QK path that reuses the K output as V. Fused30's duplicated K is compatible with today's engine but computes those rows twice. This would be a separate engine change, not part of the delivered converter.

The original `start.sh` defaults are retained, and its original model server was restarted after tests. A future deployment can add a model override and validated batch/graph controls once the next optimization demonstrates a meaningful benefit.

## Artifacts

- Raw benchmark JSON, server logs and drivers: `/tmp/pulsar-fused-qkv-20261011`.
- Natural chat comparison: `/tmp/pulsar-fused-qkv-20261011/accuracy-summary.json`.
- Small converter checks: `/tmp/test_pulsar_fused_qkv.py`.

No commit, push or PR was created. Existing unrelated working tree changes were retained.
