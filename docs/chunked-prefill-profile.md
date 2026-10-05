# Chunked prefill and 16-request memory investigation

Status: implemented and validated. Production is running with 16 slots and a 64-token mixed text prefill budget.

Target: 16 simultaneous text requests with about 8K input plus output tokens each; preserve single-image multimodal support. Measurements use Pulsar S Q4_0_XL, packed Q8_KV for K/V, V100 CUDA0, physical KV budget 8192 x 16 = 131072 tokens, CUDA Graphs disabled. Capacity and latency measurements cover this Q8_KV/CUDA configuration. Final functional checks cover F16/Q8_KV on CUDA and Vulkan. Other models and audio were not measured in this optimization pass.

## Streaming latency

One raw completion streams 128 tokens. After 16 non-empty content events, two unique 4430-token prompts arrive. Inter-event gaps measure client-visible streaming pauses; they can combine multiple tokens at UTF-8 boundaries. Each probe uses the same workload. Slot capacity differs as shown; only three slots are active in these latency probes. Nsight instrumentation is enabled in the first three cases, so these are diagnostic measurements, not unprofiled production throughput claims.

| Configuration | Largest gap during prefill | p95 gap during prefill | Peak GPU MiB | Minimum free MiB |
| --- | ---: | ---: | ---: | ---: |
| Existing 8 slots, batch/ubatch 1536 | 4457 | 3863 | 15751 | 394 |
| 16 slots, batch/ubatch 256 | 1047 | 1016 | 16089 | 56 |
| 16 slots, batch/ubatch 64 | 508 | 332 | 16035 | 110 |
| 16 slots, CPU tied embedding/output, batch/ubatch 1280 | 3819 | 3644 | 16077 | 68 |

Small global batches reduce pauses but reject image chunks larger than either batch limit. They are not a complete multimodal solution. The proposed text prefill budget must be separate from physical image capacity.

The current scheduler already appends generation tokens before prompts. It then fills the remainder of the batch with prompt tokens and executes the whole batch before returning generated results. Its prompt scan uses fixed slot order. Priority alone therefore does not bound the duration of each prefill step.

## GPU profile

`baseline8.nsys-rep` contains a valid Nsight Systems CUDA timeline. Direct SQLite aggregation of recorded kernel durations gives 21.713 seconds total kernel time; summed durations are not wall time when streams overlap.

- Warp paged attention prefill: 17.461 seconds, 80.42% of total kernel time.
- Paged block attention: 0.626 seconds, 2.88%.
- The `mul_mat_q` kernel family: 1.641 seconds, 7.56%.
- Paged KV writes: 0.055 seconds, 0.25%.

CUDA resource inspection shows a 256-byte thread stack for the runtime-dimension warp prefill kernel. SASS contains local-memory `LDL` and `STL` operations. Its `query[32]` and `acc[32]` arrays are indexed by loops whose limit depends on runtime `d`. Specializing the supported dimensions 256/512 and unrolling these loops is the first kernel optimization candidate. Removing local accesses and improving speed are hypotheses to verify, not measured improvements. Shared K/V tiles and tensor-core prefill are later candidates if needed.

The installed Nsight stats command could not import `nsysstats`; valid exported SQLite data was analyzed directly. The 64-token trace importer rejected an event-order error and retained its `.qdstrm`. Its client latency and NVML samples remain valid, but no converted kernel profile is claimed for that case.

## 16-request capacity and multimodal probes

All-GPU 16 slots with batch/ubatch 1536 fail during compute-buffer allocation. The model uses a tied `token_embd.weight` output tensor; an `output.weight`-only override does not move that tied output. Lowering `-ngl` also moves a Transformer layer in this fork, so it is not used as a substitute for output-only placement.

The working capacity probe keeps all Transformer layers and KV on CUDA and uses the existing tensor override:

```sh
--override-tensor '^(output|token_embd)\.weight$=CPU' -np 16 -b 1280 -ub 1280
```

This removes the approximately 396 MiB GPU copy of the tied embedding/output weight. Output computation then runs on CPU and has a latency/throughput tradeoff; the validated 16-slot serving configuration uses this placement.

- 16 simultaneous requests, each 7929-7930 input tokens plus 32 generated tokens: all completed. Peak active slots 16; completion wall time 415.865 seconds. Fixed-order prefill scheduling remains in use.
- 16 short prompts with simultaneous generation: all completed.
- One image budgeted at 1120 tokens, producing an actual 1170-token chunk, plus 15 simultaneous text requests: all completed. The image correctly returned `MEN WALK ON MOON`; its total prompt length was 3007 tokens including the installed chat template.
- GPU peak: 16077 MiB; minimum free: 68 MiB. This is a demonstrated capacity candidate with little spare memory, not a general OOM-free guarantee.

Image chunks remain atomic because Gemma4 image SWA is bidirectional. Text chunking alone cannot remove the latency of CPU vision encoding or the atomic image decode. The CPU output-placement candidate also shares CPU resources with vision encoding.

## Implemented scheduler and CUDA optimization

Added an opt-in `--prefill-chunk-size` server option, default 0. With generating requests, their decode tokens are included first and the option caps additional text tokens across all pending prompts. Enabled scheduling rotates pending slots after actual progress. Without generators, text uses the existing physical batch limit. Non-splittable prompts and image decode retain their physical limits. The deployment candidate uses 64 text tokens and physical batch/ubatch 1280.

CUDA warp prefill now specializes dimensions 256 and 512 and unrolls query/accumulator loops. Other supported dimensions retain the generic path. `cuobjdump` shows STACK 0 and LOCAL 0 for F16/Q8_KV specializations on sm70 and sm61, versus the original 256-byte stack. Existing independent operator tests on CUDA0 passed 34/34.

Round-robin prefill exposed a pre-existing SWA allocator assumption: inactive slots kept all of their preceding prefill chunk until their next decode. Sixteen 1280-token prefixes retained 20480 tokens in a pool sized for 18176. On SWA allocation pressure, the transactional candidate now releases inactive whole pages older than their last query window and retries both new-page and COW allocation. Current-ubatch sequences keep the existing earliest-query cutoff. Ordered native batch positions, rejected paged speculative decode and existing server prefix replay checks preserve supported behavior. Full-domain capacity stays unchanged.

A temporary metadata fixture proves new-page and shared partial-page allocation under pressure, preservation of valid shared pages, and unchanged committed state during prepare. Omitting the COW retry reproduces a failing COW allocation; the final allocator passes both checks. No new repository test executable was added.

### Final CUDA measurements

Identical 16-slot, tied CPU output, batch/ubatch 1280 settings and streaming workload, without Nsight:

| Code | Maximum TG event gap | p95 gap | Peak GPU MiB | Minimum free MiB |
| --- | ---: | ---: | ---: | ---: |
| Original scheduler and kernel | 3833 ms | 3630 ms | 16079 | 66 |
| Text budget 64, original kernel | 346 ms | 334 ms | 16079 | 66 |
| Text budget 64, specialized kernel and SWA reclaim | 238 ms | 228 ms | 16077 | 68 |

The final maximum gap is 93.8% lower than the original comparison. This trades prompt completion time for predictable generation progress; it is not a general throughput guarantee. Atomic image decode and CPU vision encoding can still pause TG.

Sixteen requests with 7929-7930 input tokens plus 32 generated tokens all completed, with 16 active slots observed. The run took 147.434 seconds, compared with 415.865 seconds in the earlier same-capacity prototype. A small CPU compilation occurred during the capacity phase of the final run, so this is a workload completion observation, not an isolated kernel speedup measurement. GPU peak was 16077 MiB, minimum free 68 MiB.

Final-code mixed workload passed: one 1170-token newspaper image with 15 chat text requests, plus 16 short simultaneous raw decodes. Headline result was `MEN WALK ON MOON`. The repeated prompt initially replayed with identical output; cancellation released the slot and a subsequent request completed. Peak GPU 16077 MiB, minimum free 68 MiB.

A separate repetition with 15 virtual-memory chat prompts produced one malformed Gemma4 channel sequence and the existing PEG parser returned HTTP 500. The image completed correctly. This output-format failure is retained in `scheduler-mm16-chat-parser-failure.json` and is not claimed fixed by chunking. A simpler exact-answer mixed workload passed. The first mixed run also passed all 16 responses before a prefix reuse assertion exposed unnecessary replay of short prefixes.

State restore passed on CUDA0 and Vulkan1 in both F16 and Q8_KV, including fragmented-vs-clean replay, copy-on-write, partial-page copy, failed-allocation rollback and asynchronous copy. Ordinary-vs-paged model logits are diagnostic only in that test; exact full-model parity is not asserted.

Short prefix reuse exposed a second issue: with partial checkpoints disabled, the server treated a position-zero SWA minimum as missing history and replayed the whole prefix. The paged-only guard now accepts prefixes whose SWA history still starts at zero. Positive eviction bounds retain the existing conservative replay check. `--no-cache-idle-slots` does not itself disable in-slot prefix reuse; the earlier investigation of this failure incorrectly attributed replay to that flag.

Final Vulkan scheduler checks passed in F16 and Q8_KV: repeated short prompt reused 5 tokens, unequal parallel requests completed, and cancellation released the slot followed by a successful new request. The deployed CUDA server also reused 5 prefix tokens and correctly read `MEN WALK ON MOON` with the default image minimum. Its health is OK and /slots reports 16 slots.

Memory candidates after profiling: reduce or recycle compute scratch according to actual batch shapes instead of retaining worst-case allocations, avoid redundant metadata/graph rebuild work, and investigate allocator high-water retention. Lazy page allocation helps idle or partially filled pools; it cannot remove KV bytes genuinely occupied by sixteen independent full contexts. Shrinking physical KV capacity is not counted as an optimization for this target.

## Serving configuration

`/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/start.sh` now defaults to 16 slots, text prefill budget 64, physical batch/ubatch 1280 and image maximum 1120. `PULSAR_PARALLEL` and `PULSAR_PREFILL_CHUNK_SIZE` control slot count and text budget. The server option itself defaults to 0, preserving existing scheduling when omitted.

Packed Q8_KV, 8192 physical blocks of 16 tokens and logical context 262144 remain selected. All Transformer layers and KV stay on CUDA0; tied embedding/output is on CPU. This saves approximately 396 MiB of GPU weights but slows isolated TG relative to the previous 8-slot all-GPU configuration. CUDA Graphs remain disabled. Measured free GPU memory is only 66-68 MiB at peak; capacity was demonstrated for the tested workloads and is not a guarantee against every allocation pattern or background GPU workload.

Atomic image encoding/decode can still pause TG. The recorded malformed-channel parser error is also unresolved. Original 8-slot startup settings are backed up at `/tmp/aikar-chunk16/start8.predeployment.sh`. MuSR remains stopped as authorized.

Evidence: `/tmp/aikar-chunk16`. Logs of interrupted/misconfigured placement probes and a temporary port collision are retained and are not counted as successful configurations. No commits, pushes or external submissions were made.
