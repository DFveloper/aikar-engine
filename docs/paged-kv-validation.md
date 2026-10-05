# Paged KV validation

Status: completed for the required Pulsar/Pulsar S text-inference scope on 2026-10-04. Paged KV is optional and disabled by default.

Source: llama.cpp PR #22569, revision `0b0f7bd7e3c85bda81645edcd7c2c639c67efec0`, by matiaslin. The page manager, interleaved layout and attention work were adapted to this fork's native memory interface. The external scheduler API, example drivers and new test executables were omitted. Vulkan is a native implementation, with no VUDA dependency. No commit or upstream submission was made.

## Scope and placement

Required GGUFs in `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar`:

- `Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf`
- `Lumen-3.1-Pulsar-LD-Q4_0_M.gguf`

These Gemma4 models have 30 layers and 16 query heads. SWA layers use dimension 256, eight KV heads and window 1024; full-attention layers use dimension 512 and two KV heads. Both F16 and Q8_KV are supported, including distinct full/SWA format domains. K and V must match within each domain. Q8_KV stores 64 signed values and an FP16 scale in 66 bytes, with no persistent F16 backing cache.

Test host: CUDA0/Vulkan1 are V100-SXM2-16GB; CUDA1/Vulkan0 are P620. CPU placement and CPU plus one accelerator are supported. This record covers the initial text delivery. Image support was added afterward; see [multimodal validation](paged-kv-multimodal-validation.md). General multi-GPU placement, other architectures, shared-KV Gemma4 variants, speculative decoding, training, Hadamard rotation, position shifts/division and device-only/partial snapshots are excluded. Unsupported configurations are rejected; server partial checkpoints and context/cache shifting are explicitly disabled. Full host state and prompt reuse are supported.

Evidence is in `/tmp/aikar-paged-artifacts` for this session. Separate CPU/CUDA/Vulkan builds were used in `/tmp/aikar-paged-kv`; the main workspace `build` now contains the combined CUDA/Vulkan executables. CUDA was built for architectures 61 and 70. The initial user changes are preserved in `user-before.patch`; the task was transferred without changing the real index or creating commits.

## Correctness and lifecycle evidence

| Check | Result | Evidence |
| --- | --- | --- |
| CPU paged operator | 24/24 | `final-cpu-24.log` |
| CUDA V100 paged operator | 24/24 | `final-op-CUDA0.log` |
| Vulkan V100 paged operator | 24/24 | `final-op-Vulkan1.log` |
| Vulkan P620 paged operator | 24/24 | `final-op-Vulkan0.log` |
| Two models x two types x CUDA/Vulkan, short lifecycle | 8/8 | `final-matrix-result.log`, `final-state-*.log` |
| Same model matrix, 1106-token prompts x three sequences, ubatch 128 | 8/8 | `long-matrix-result.log`, `long-state-*.log` |
| Final CUDA kernels, long model lifecycle | 4/4 | `long-cuda-final-result.log`, `final-long-state-*.log` |
| Single ubatch > SWA window: S, both types, both backends, ubatch 1536 | 4/4 | `large-ubatch-result.log` |
| P620 with CPU plus two Vulkan layers, both formats | 2/2 | `matrix-p620-cpu*.log` |
| Existing host save/load runner, CPU, both models/types | 7 supported cases per model passed | `save-load-matrix-result.log`, `save-load-cpu-*.log` |
| Vulkan Q8_KV repeated exact packed writes | 30/30 on each GPU | `final-extra-result.log`, `final-quant-repeat-*.log` |
| Ordinary CUDA Q8_KV attention | 1/1 | `ordinary-cuda-attention.log` |
| Ordinary CUDA state fixture | passed | `ordinary-cuda-state.log` |
| Existing CUDA F16 SET_ROWS | 44/44 | `final-cuda-set-rows-f16.log` |
| Affected CPU optimizers | 3/3 groups passed | `final-opt-cpu.log` |

The existing save/load runner skips tests 5 and 7 only in paged mode: these require device-only state. They are reported as skipped, not passed. Separate lifecycle assertions verify device-only/partial snapshot requests are rejected.

Operator cases use an independent double-precision dense attention reference with existing backend tolerances. They cover dimensions 256/512, GQA, reordered physical pages, sparse masks, two sequences at different positions, full/SWA visibility, position 2046 and a 2051-query batch. Packed writes are compared byte-for-byte with CPU reference quantization, including zero blocks, positive/negative ties and token/head/page strides. The GPU operators actually execute; CPU fallback is not used to satisfy GPU acceptance. Use `test-backend-ops -b CUDA0 -o 'PAGED_ATTN.*'` (or Vulkan1/Vulkan0) to include packed-write cases as well as attention.

Model lifecycle checks exercise fragmented allocation, partial removal, shared-prefix divergence, partial-page copy-on-write, asynchronous copy after decode, clear/reuse, pool exhaustion rollback and deduplicated shared full-state restoration. Corrupt, truncated, incompatible and insufficient-capacity restores leave the retained cache unchanged. Clean paged replay and restored replay enforce `max_abs < 1e-4`; the final required model runs observed maximum error zero. Long cases confirm SWA eviction and minimum-position reporting. Mixed Q8_KV full/F16 SWA runs passed on CUDA and Vulkan.

Numerical limitation: ordinary attention and paged FP32 attention have different arithmetic. Ordinary logits and greedy-token agreement were measured but are not identical in all model/type/backend cells. Diagnostic identical first-layer Q/K/V inputs gave max difference 3.0e-5 (RMS 7.8e-7) against dense FP32 reads of the same packed cache; first-token outputs were identical across heads. Later quantized model operations amplify differences. The diagnostic paths were removed. No identical-generation guarantee is made, and exact packed quantization on other drivers or extreme input floats is not established by these tests.

CUDA and Vulkan Q8_KV servers passed short prefix reuse, long-prefix reuse after SWA eviction, three parallel unequal requests, streaming cancellation followed by a usable request, and refusal with an eight-page budget followed by recovery. The shortened long-prefix request recomputes expired SWA history. See `server-*-q8-long.json/log`, `server-*-q8-budget8.json/log`, and `final-extra-result.log`.

Negative startup checks cover block sizes, negative/overflowing budgets, logical table index overflow, mixed K/V types, unsupported Qwen3.5 GGUF, Hadamard, multimodal and speculative configuration in the initial text delivery. The multimodal startup rejection was subsequently removed with image-aware paged attention. Training/default-context guards were reviewed; no training run with paged memory is claimed. Paged Q8_KV does not require Flash Attention; existing host save/load test 9 exercises FA disabled. Ordinary Q8_KV with FA disabled still rejects initialization (`ordinary-q8-fa-off-reject-completion.log`).

Baseline evidence includes `baseline-cuda-q8.log`, `option-red.log`, `paged-api-red.log` and Vulkan missing-support logs. Red/green regressions include shared-state deduplication, SWA minimum position, boundary-page capacity, Vulkan quantization scratch ordering/rounding and paged Q8_KV with FA disabled. An old, unrebuilt root llama-cli crashed against updated shared libraries during final auditing; the target was rebuilt before delivery. It is not counted as a paged model test.

## Matched serving measurements

All 14 runs completed. V100 placement is all GPU with one device, context 6144, three slots, batch 1024, ubatch 128, 18 CPU threads, unified KV, no checkpoints/context shifts/idle caching. Three simultaneous unique-prefix prompts have 90, 310 and 1110 tokens, with eight greedy output tokens each. Throughput below is the longest slot's server timing; wall time covers all three requests. Short slots wait while other prompts are processed, so their timing is not standalone decode speed. Peak VRAM is sampled every 0.3 seconds with no other GPU workload during measurement. These are workload measurements, not universal performance guarantees. Ordinary Vulkan Q8_KV was unavailable and has no invented baseline.

| Model | Backend | Type | Cache | Wall s | Long prompt tok/s | Long decode tok/s | Peak MiB |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Pulsar S | CUDA | F16 | ordinary | 2.077 | 629.12 | 60.13 | 13793 |
| Pulsar S | CUDA | F16 | paged | 2.682 | 490.10 | 33.73 | 13721 |
| Pulsar S | CUDA | Q8_KV | ordinary | 2.232 | 582.66 | 55.07 | 13423 |
| Pulsar S | CUDA | Q8_KV | paged | 2.938 | 438.73 | 34.14 | 13371 |
| Pulsar S | Vulkan | F16 | ordinary | 3.712 | 355.40 | 38.19 | 13812 |
| Pulsar S | Vulkan | F16 | paged | 5.646 | 247.20 | 9.48 | 13629 |
| Pulsar S | Vulkan | Q8_KV | paged | 6.035 | 235.49 | 7.77 | 13260 |
| Pulsar | CUDA | F16 | ordinary | 2.156 | 596.45 | 62.11 | 14877 |
| Pulsar | CUDA | F16 | paged | 2.767 | 468.38 | 35.26 | 14795 |
| Pulsar | CUDA | Q8_KV | ordinary | 2.303 | 556.80 | 52.41 | 14505 |
| Pulsar | CUDA | Q8_KV | paged | 3.032 | 420.83 | 33.33 | 14445 |
| Pulsar | Vulkan | F16 | ordinary | 3.793 | 346.29 | 38.45 | 14887 |
| Pulsar | Vulkan | F16 | paged | 5.721 | 243.07 | 9.49 | 14705 |
| Pulsar | Vulkan | Q8_KV | paged | 6.087 | 232.17 | 7.89 | 14336 |

Paged CUDA remains slower than ordinary CUDA in this workload; paged Vulkan has a larger decode slowdown. CUDA prefill uses warp reductions, while small decode batches retain block reductions and long-span partial reductions. The slower Vulkan subgroup experiment was removed in favor of the portable shared-reduction shader. Paged mode stays opt-in.

At these settings, each full layer reserves 384 pages and each SWA layer 206 pages, with block size 16. KV tensor bytes are 800849920 (763.75 MiB) for F16 and 412938240 (393.81 MiB) for Q8_KV, computed from actual layer dimensions, row sizes and page capacities; backend pool totals include small metadata tensors. Peak live counts were 97 full/95 SWA pages, leaving at least 287/111 free. Pools reserve their physical budget at initialization; live pages do not describe committed VRAM reduction by themselves.

Each paged run rebuilt 21 graphs. CUDA graph construction averaged 3.94-4.37 ms and Vulkan 1.15-1.29 ms; the timer excludes scheduler reset/allocation. Graph reuse is disabled for mutable page metadata. Raw commands, per-slot timings, page counts and graph costs are in `bench-results.json`, `bench-summary.json` and `bench-*.log`.

## Use and operational state

See [server instructions](../tools/server/README.md#paged-kv-in-this-fork) for Q8_KV serving, backend selection, block budget and separate domain types. For F16 set both `-ctk f16 -ctv f16`. Select one device with `-sm none`; choose CUDA0 or Vulkan1 on this host.

The original server was restored with its original nonpaged argv and verified healthy at port 11435. Its original nonpaged configuration was not silently changed. MuSR remains stopped as authorized. Changes are in the main workspace and isolated worktree, uncommitted. No push or external PR/comment was made.
