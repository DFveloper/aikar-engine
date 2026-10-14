# Pulsar 3.5 paged KV quantization and 262K capacity

Date: 2026-10-11
Status: User approved the written design and direct execution. Codec, rotation and PKV4 changes are implemented, reviewed and applied to the root workspace. Numerical/state checks passed; quality/speed adoption gates failed, so both compressed codecs remain experimental and production defaults are unchanged. An8192-page pool allocated with12slots reserved;180224actual tokens completed before the user waived the rest of the long stress test. Full262Koccupation and longquality remain uncertified. On2026-10-11 the user canceled AM and restored codec/capacity optimization as the priority, with Decode Overlap and Expert Cache as measured follow-up designs.

## Goal and boundaries

Reduce memory waste and improve prefill/decode serving for Lumen 3.5 Pulsar on the V100 16 GB. The user permits KV compression subject to quality verification and requested reuse of local/global cache options, Hadamard Q4 and IQ4. This document specifies the first implementation unit: existing-format paged KV quantization, rotation, persistence, and matched quality/capacity measurements. The companion Attention Matching design is retained as a canceled experimental proposal. Kernel and allocator optimizations remain part of the serving roadmap below.

Keep the user's existing uncommitted changes, especially Gemma4 QKV views and backend tests. Use an isolated checkout for implementation. Reuse existing test files. No commits, pushes, PRs, or production launcher changes are part of this design.

## Verified constraints

- Model: 30 layers; 25 SWA layers with D256, 8 KV heads and window 1024; 5 Full layers with D512 and 2 KV heads; 16 query heads; logical context 262144.
- Current launcher: 12 slots, batch/microbatch 1280, mixed prefill chunk 1024, block size 32 and 4096 Full blocks. Physical Full capacity is 131072 tokens shared across slots, not 131072 per slot.
- Paged cache stores matching K/V types per domain in one tensor per layer. Constructor, GGML op, CPU and CUDA currently accept F16/Q8_KV only. Hadamard is explicitly rejected by the paged memory implementation.
- `-ctlk/-ctlv` already select SWA types; `-ctgk/-ctgv` select Full types. Existing Hadamard local/global flags reach the context. No new CLI is needed for the first unit.
- Q4_0 and IQ4_NL are existing CLI types and both use 18 bytes per 32 components. IQ4_XS is not currently a CLI KV type. Implement IQ4_NL first; IQ4_XS needs a separate codec/backend assessment.
- Current Gemma4 inference QKV view changes already exist in the working tree. Preserve them and include them consistently in the baseline.
- V100 is SM70. Port algorithms from vLLM into existing CUDA code; do not assume current vLLM kernels or its recent INT4 backend run on this GPU.

## Capacity arithmetic

For this model, Full Q8_KV costs 10560 bytes/token. SWA Q8_KV costs 105600 bytes/token across its 25 layers. Current SWA allocation is rounded from `window * slots + ubatch + 2 * block_size * slots`.

| Full pool | Full type | SWA slots/type | Full MiB | SWA MiB | Total KV MiB |
| --- | --- | --- | ---: | ---: | ---: |
| 131072 | Q8_KV | 12 / Q8_KV | 1320 | 1443.75 | 2763.75 |
| 262144 | Q8_KV | 12 / Q8_KV | 2640 | 1443.75 | 4083.75 |
| 262144 | Q4_0 or IQ4_NL | 12 / Q8_KV | 1440 | 1443.75 | 2883.75 |
| 262144 | Q4_0 or IQ4_NL | 8 / Q8_KV | 1440 | 1005.47 | 2445.47 |
| 262144 | Q4_0 or IQ4_NL | 4 / Q8_KV | 1440 | 567.19 | 2007.19 |

Weights occupy about 12498.5 MiB before workspace. Usable V100 memory was reported as 16145 MiB. Q8_KV at 262K/12 slots cannot fit even before workspace. Q4/IQ4 at 262K/12 slots adds 120 MiB to the existing KV allocation; it is a candidate, not a demonstrated fit. Measure actual peaks and maintain at least 256 MiB free under the tested workload before recommending a configuration. Reduced slot counts are measured alternatives and do not silently replace the 12-slot target.

## Implementation contract

1. Extend the existing paged cache/op to Q4_0 and IQ4_NL on CPU and CUDA. K and V remain matching within each domain; SWA and Full can differ. Reject mixed K/V explicitly. Preserve F16/Q8_KV behavior and existing Vulkan support; Vulkan must reject the new formats until it has kernels.
2. Reuse GGML row quantizers/dequantizers and CUDA `cpy-utils.cuh` codecs. IQ4_NL uses its real nonlinear codebook and scale fitting. Do not substitute uniform Q4 under the IQ4 name. Preserve packed tensor strides, page masks, multi-sequence write deduplication and COW behavior.
3. Apply normalized Hadamard after RoPE: rotate Q/K together, rotate V before storage and invert V rotation on attention output before Wo. Reuse `llama_mul_mat_hadamard` and existing FWHT dispatch. Preserve existing flag semantics: K uses the largest supported power-of-two divisor of head geometry; V uses blocks of 64. For Pulsar Full layers, this means K512/V64. A full-head V512 experiment requires an explicit distinct policy; do not silently change flag meaning.
4. Carry separate Full/SWA rotation inputs through paged graph input construction, reuse checks and input population. Changing rotation requires a new compatible graph and an empty cache; cached vectors from different bases cannot mix.
5. Version paged state metadata to encode rotation policies and codec geometry. Validate all metadata before changing allocator state or uploading restored payloads. Accept old PKV2/PKV3 only for their original unrotated compatible formats. Reject policy mismatches without mutating the current sequence. Preserve alias restoration and failed-load rollback.
6. Keep backend support checks before executing new formats. Do not turn unsupported paged kernels into CPU fallback over a GPU-owned cache.

Main files: `src/llama-kv-cache-paged.{h,cpp}`, `src/llama-graph.{h,cpp}`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cuda/pagedattn.cu`, and the CUDA op support guard. Extract/share existing Hadamard input generation only as needed; avoid unrelated cache refactoring.

## Validation and selection

- Extend `tests/test-backend-ops.cpp` for real Q4/IQ4 packed writes and attention against an independent dequantized dense oracle. Cover fragmentation, masks, aliases, block sizes and D256/D512; preserve the existing Q8 exact-byte boundary tests. CPU/CUDA IQ4 writer implementations must be checked for algorithm agreement before requiring identical bytes; decoded-value/output tolerances are separate from format correctness.
- Reuse `tests/test-save-load-state.cpp` for round trips, policy mismatch, malformed state and rollback. Report unsupported device-state cases as skips, never passes.
- Fix the existing `llama-kv-quality` F16 reference construction to clear all local/global rotation and native codec policies. It is a short smoke test, not a long-context quality certificate. Ordinary FA IQ4 reference runs require the IQ4_NL kernel pair to be compiled; the current root build does not include that pair.
- Use the same 3.5 GGUF, engine revision, token IDs, batch schedule and teacher-forced continuation for F16, paged Q8, Full Q4, Full Q4+Hadamard, Full IQ4 and Full IQ4+Hadamard. Keep SWA Q8 initially. Add local 4-bit only as a separately measured experiment.
- Evaluate short text, Korean dialogue, code, exact number/name retrieval and dispersed multi-needle tasks at 8K/32K/128K/262K where the configuration fits. Use sampled logits at fixed positions rather than saving all 262K vocabulary logits. Record KL, top-1 agreement, perplexity and task accuracy independently.
- For long-context quality, run controls and candidates sequentially and replay the same teacher-forced tokens; the existing two-context smoke tool cannot hold both long caches. Use one slot and a smaller matched microbatch to attempt the exact 262K Q8 control. F16 at 262K does not fit with GPU weights and is not an available all-GPU oracle. Report any unavailable length/control explicitly.
- Proposed adoption gate, subject to this design review: at short lengths with an F16 control, mean KL versus F16 increases by at most 0.01 relative to paged Q8 and top-1 agreement falls by at most 1 percentage point. At long lengths with a Q8 control, direct candidate-versus-Q8 mean KL is at most 0.01 and top-1 agreement is at least 99%. Task accuracy falls by at most 1 percentage point, with uncertainty reported. Critical deterministic retrieval cases must remain correct. Insufficient sample size is inconclusive. A failing candidate stays experimental.
- Measure PP/TTFT, pure TG/inter-token latency, concurrent throughput and peak/free VRAM at 1/4/12 slots. Use warmups and at least three matched repetitions without a profiler. Include long-context decode, not just 1K/8K. A lower byte count does not establish a speedup.

Candidate launcher arguments replace uniform `-ctk/-ctv`, which conflict with role options:

```text
-ctlk q8_kv -ctlv q8_kv -ctgk iq4_nl -ctgv iq4_nl
--k-cache-hadamard-global --v-cache-hadamard-global
--kv-paged --kv-blocks 8192 --kv-block-size 32
```

Q4_0 is the parallel candidate. No format becomes the default solely because it fits.

## Remaining serving optimization roadmap

| Unit | Existing bottleneck | Intended change | Evidence required |
| --- | --- | --- | --- |
| Metadata | `prepare`/`apply` copy complete allocator maps; `set_input` rebuilds/uploads both complete tables | Reuse table storage, update changed records and avoid redundant committed-state copies while preserving transactional prepare | COW/rollback lifecycle tests, host time and transferred bytes |
| Decode | Split count uses a batch-wide context bound | Bound splits by actual per-sequence useful span; preserve grouped D512 kernels already present | Mixed-length latency, empty split counts, register/local-memory profile |
| Prefill | Current kernel reuses KV across grouped heads but processes each query token separately | Prototype query/KV tiling with online softmax in existing CUDA backend; retain scalar fallback | D256/D512 numerics and end-to-end PP on SM70 |
| Pool utilization | Separate static Full/SWA pools cannot lend capacity | Assess compatible page-byte grouping and allocation policy from vLLM after codec validation | Budget/fragmentation tests and actual bytes saved |
| Graphs | Production disables graphs; caches are bounded by age rather than bytes | Assess bounded decode capture after memory headroom exists | Peak memory and stable latency benefit |

These are separately reviewable changes after the codec foundation. Prior attention Tensor Core prototypes were slower; instruction choice alone is insufficient. Do not claim a large serving speedup until matched measurements show it.

## Sources

- vLLM sources inspected at commit `f8471164bf9f5190752fd7b903d415a34fc4ba1c`: [block pool](https://github.com/vllm-project/vllm/blob/f8471164bf9f5190752fd7b903d415a34fc4ba1c/vllm/v1/core/block_pool.py), [KV manager](https://github.com/vllm-project/vllm/blob/f8471164bf9f5190752fd7b903d415a34fc4ba1c/vllm/v1/core/kv_cache_manager.py). The actual local downloads are retained under `/tmp/paged-vllm-*`.
- Local deployment and measurement context: `docs/pulsar-fused-qkv-20261011.md` and the existing uncommitted Gemma4 changes.
- Attention Matching integration: `2026-10-11-paged-kv-attention-matching-design.md` in this directory.
