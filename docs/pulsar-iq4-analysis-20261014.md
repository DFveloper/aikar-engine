# Pulsar IQ4_NL paged decode and prompt-cache analysis

Date: 2026-10-14. Source: `7ddea3e714c84c10464d5700b932482a4f7c75a0`.
Status: Measured baseline and implementation planning complete. Optimization implementation and adoption validation remain future work.

## Result

The actual launcher uses Q8_KV for local/SWA attention and IQ4_NL for global/Full attention. On the idle V100, one active request with 12 configured slots measured 76.43 tok/s at 1024 input tokens and 78.73 tok/s at 8192 input tokens. The first optimization target is the global IQ4 attention launch geometry. Its writer is a smaller contributor. Prompt caching has independently reproducible reuse failures and does not yet have persistent GPU prefix ownership.

The user confirmed that the goal is single-request decode >= 100 tok/s, not concurrent aggregate throughput. No current configuration measured here reaches that goal. All-F16 KV failed initialization at the current 12-slot/ubatch-1280 settings even with only 512 Full pages. Q8-local/F16-global works, but measures about 85 tok/s. The historical all-F16 ~100 tok/s configuration needs separate reproduction; it must not be substituted with a different slot count or weight offload without labeling the comparison.

Read the [design](superpowers/specs/2026-10-14-pulsar-iq4-radix-design.md), [decode plan](superpowers/plans/2026-10-14-pulsar-iq4-decode.md), and [prefix-cache plan](superpowers/plans/2026-10-14-pulsar-radix-cache.md) together. The design contains proposed API, lifetime, SWA coverage and budget contracts; the plans specify files, task order and validation gates.

## Measurement provenance

- GPU: Tesla V100-SXM2-16GB, SM70, 80 SMs, driver 580.178.04, 150 W limit. `nvidia-smi` reported no active GPU workloads before testing. CUDA reports 16144 MiB usable memory. The P620 was excluded with CUDA_VISIBLE_DEVICES=0.
- Binary: `build-pulsar-paged-decode-20261014/bin/llama-server`; Release build, CUDA architectures 61;70, CUDA Graphs compiled but disabled at runtime. The current paged CUDA object was newer than its source file; binary/library hashes identify the measured artifacts, but this analysis did not rebuild them.
- Target: `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/start.sh`; model `Lumen-3.5-Pulsar_S-LD-Q4_0_XL.gguf`. Retained launcher model, template, mmproj on CPU, 12 slots, logical context 262144, batch/ubatch 1280, prefill chunk 1024, threads 36, all GPU layers/experts, backend sampling, top-k 64, top-p 0.95, temperature 1.0, reasoning budget -1 and token-101 bias +1.
- Changed only listen address/port for isolated measurement, request seed 123 and ignore_eos=true to obtain a fixed output length. Servers were started and stopped sequentially; the production tmux launcher was not invoked. No product source was changed.
- HTTP `/completion`, exact saved token arrays of length 1024 or 8192 from one repetitive inventory-text fixture. Output: 256 tokens; two warmups plus three measured requests per length. This is a narrow baseline, not the five-repeat, alternating-order, mixed-workload adoption gate in the spec.
- Server decode rate is `(predicted_n - 1) * 1000 / predicted_ms`: this revision counts the first generated token with prompt evaluation. Client TTFT is time to first nonempty content event. Special-token events can delay this observable boundary.
- Repeated requests use the production global --cache-prompt setting. That setting forces request cache_prompt=true in `tools/server/server-schema.cpp:553`, even when the client sends false. Actual cache_n values are recorded; neither every repetition nor the 8K warmup is a cold-prefill sample.

Raw evidence and reusable measurement programs: `/tmp/pulsar-analysis-20261014/`. Each configuration has `*-manifest.json`, `*.jsonl` and `*.log`; exact token fixtures are `tokens-1024.json` and `tokens-8192.json`. `measure.py` launches isolated servers, `summarize.py` reproduces profile aggregation, and `summary.json` stores aggregate results. Keep these artifacts before cleaning /tmp. The report below retains the essential results even if temporary traces are removed.

SHA256 identifiers:

| Artifact | SHA256 |
| --- | --- |
| Model GGUF | f8428df24ae3961c98ac4137c7cb47c1b6326a45690aa3677601fe2d70b1c69c |
| chat_template.jinja | 91b64a1b0842f5865b9381fc9a90ef1a140c330562384ae68f95575f04533cdd |
| llama-server | 4de5c96a96b64c256d9439106a2b88c9651ee053db6462f4fb6eb1622c60b4a1 |
| libggml-cuda.so.0.25.3 | beed41c604aefaaa6f937fc947ac8bcf062cf3e7b2648bbcdb4d9e9b2e0463fe |
| libllama.so.0.5.0 | da63f245c2589d4540334528ff2b4499dd7737efd45d43c37ac650cb2e2f005b |

## Unprofiled throughput

| Local / global KV | Full pool | 1K median tok/s (range) | 8K median tok/s (range) |
| --- | ---: | ---: | ---: |
| Q8 / IQ4, production | 8000 pages | 76.432 (76.392-76.470) | 78.732 (78.615-78.761) |
| Q8 / IQ4, control | 512 pages | 76.659 (76.632-76.807) | 79.017 (78.880-79.172) |
| Q8 / F16, control | 512 pages | 85.848 (85.629-85.883) | 85.165 (85.006-85.256) |
| F16 / F16 | 512 pages | Initialization OOM | Not run |

The matched 512-page global-F16 control is 12.0% faster at 1K and 7.8% faster at 8K than IQ4. The 512-page IQ4 rate differs from the production pool by less than 0.4% at these two lengths, supporting use of that reduced pool for kernel attribution. It does not prove equivalent long-context, allocator-pressure or peak-memory behavior.

Production requires about 13.084 ms/decode step at 1K and 12.702 ms at 8K. A 100 tok/s goal is <= 10 ms/step, requiring approximately 3.084 ms (23.6%) and 2.702 ms (21.3%) reductions, respectively.

All-F16 failed allocating a 277.72 MiB compute buffer (`cudaMalloc: out of memory`, context initialization failure) with 512 pages and the same 12 slots/1280 ubatch. Nsight also made the 8000-page IQ4 configuration fail during initialization. These failed runs are excluded from throughput tables. Profiled successful runs use the 512-page IQ4 configuration.

## Actual decode kernel attribution

Nsight Systems 2023.4.4 captured successful 1K and 8K runs, one warmup plus one measured 64-token request at each length. Files: `iq4-profile.nsys-rep`, `iq4-profile.sqlite`, `iq4-profile.jsonl`. Profiling reduces measured TG; none of its rates are used as adoption evidence.

To locate decode, select IQ4 writer launches with gridX=2 (two KV heads, one token), group bursts separated by >50 ms, and use the final 300 launches of each 315-launch decode burst. Five Full layers per token identify 59 complete cyclic steps between the selected first and last model-token boundaries. The two repetitions agree closely. Prefill, RAM restore and startup kernels are excluded from this interval.

| Component | 1K ms/step, second burst | 8K ms/step, second burst |
| --- | ---: | ---: |
| Full IQ4 attention, five layers | 2.721 | 2.274 |
| Full IQ4 writer, five layers | 0.218 | 0.219 |
| SWA Q8 attention, 25 layers | 1.111 | 1.113 |
| SWA Q8 writer, 25 layers | 0.134 | 0.135 |
| Paged combine, all layers | 0.209 | 0.234 |
| Other model kernels | 7.553 | 7.570 |
| Sum of kernel durations | 11.947 | 11.545 |
| Interval wall time | 13.680 | 13.328 |

Wall-minus-kernel time is approximately 1.73-1.78 ms in these bursts, but includes copies, launch gaps, synchronization and profiler overhead. It is not a CPU-only measurement and must not be counted twice or assumed fully removable. Kernel sums also are not a critical-path proof when streams overlap.

The largest individual remaining kernels include Q4_0 matvec (~2.94 ms/step), another fused Q4_0 matvec variant (~0.79), RMS norm variants (~0.81 and ~0.58), MoE reduction (~0.58), and q8_1 activation quantization (~0.57). These establish a substantial model-compute floor beyond the KV format.

### Confirmed launch limitation

| Full IQ4 shape | Observed grid | Threads/block | Registers/thread | Mean kernel time/layer |
| --- | --- | ---: | ---: | ---: |
| 1K, split, D512/group1 | 4 x 5 = 20 blocks | 128 | 105 | 544.43 us |
| 8K, split, D512/group4 | 1 x 33 = 33 blocks | 128 | 225 | 454.31 us |

Only 20 or 33 blocks are available across 80 SMs for each layer launch. This is a structural parallelism limit. No Nsight Compute occupancy/stall counters were collected, so achieved occupancy and the exact instruction/memory limitation remain unmeasured. Registers were confirmed both in the trace and with cuobjdump for SM70. Fixed-dimension kernels had no reported local stack; the unrelated generic dimension-zero fallback has a 512-byte stack.

Source reason: `pagedattn.cu:282` enables short split only for Q8_KV. IQ4 uses split_size=256; n_splits is capped by span/split_size even though target_blocks=4*nsm. `GGML_CUDA_PAGED_SHORT_SPLIT=1` in start.sh cannot change IQ4 dispatch today.

Recommended first candidate: evaluate IQ4 spans 32/64/128/256 with the current target-block bound, nt <= 32, causal SM70, D512. Compare group1/group4 and a separately implemented group2 if needed. D512/group2 is not an existing dispatch specialization; adding it must update launch dispatch and tests explicitly. Preserve other shapes and backends. Include combine/scratch cost and numerical tests. Do not assume the shortest split wins.

Writer optimization alone has a ceiling around 0.22 ms/step in this fixture, far below the 2.70-3.08 ms needed. After split tuning, measure whether read instructions still dominate. If attention cannot supply the full gain, reduce measured redundant copies/launch work while preserving the exact model.

### Transfers and graph construction

Each 59-step interval includes 178 H2D transfers of 8 bytes, 118 transfers of 48 bytes, and 118 transfers of 16 bytes, among other small inputs. This confirms the old two-megabyte-table-upload decode description is not the current steady path; persistent dirty ranges are active. The exact owner of every 8-byte transfer was not instrumented.

Device-to-device copies include 50 x 8192 bytes, 25 x 16384 bytes, 10 x 4096 bytes and 5 x 32768 bytes per step, plus one 1048576-byte copy. They are not COW host round trips. `src/llama-graph.cpp:4157` unconditionally constructs cont operations for Q/K/V before paged attention; correlate copy launches with graph nodes before removing them. A contiguous fast path may avoid copies; strided inputs require correct strides through CPU/CUDA op implementations and support checks. Such a change must not merely delete cont nodes while retaining flat-pointer kernels.

## Existing operator cross-check

Command: `test-backend-ops perf -b CUDA0 -o PAGED_ATTN -p 'tokens=1,'` under the same paged fused/short-split/graphs-disabled environment.

| Full D512, two KV heads, one token | F16 us/run | Q8_KV us/run | IQ4_NL us/run |
| --- | ---: | ---: | ---: |
| query_start 8190 | 322.74 | 227.27 | 507.76 |
| query_start 32766 | 407.66 | 490.82 | 601.76 |

This fixture combines packed writes, paged attention and combine, uses its own fragmented/two-sequence page layout, and repeats the graph. Its values are not whole-model TG or standalone writer timing. The perf suite executed 16 CUDA cases; CPU was skipped. Its final '2/2 backends passed' text does not mean CPU was tested.

A focused correctness command used `-b CUDA0 -o 'PAGED_ATTN.*' -p 'cache_type=iq4_nl,dim=512,heads_kv=2,tokens=1,'`. All eight existing CUDA attention/write cases passed, covering block sizes 16/32/64/128. This validates the measured baseline subset; it is not coverage of every proposed kernel or cache lifecycle.

## Prompt-cache failure and proposed repair

At 8K, measured repetitions 2/3/4 report cache_n=[0,8191,0] for production, reduced IQ4 and global-F16. The production repeated-prompt TTFT alternates about 6.29/0.58/6.29 seconds. Explicit cache-enabled requests subsequently measured slot0 TTFT 0.457 seconds with cache_n=8191, slot1 5.952 seconds with cache_n=0, and slot2 5.945 seconds with cache_n=0 (32-token outputs). These are serial pinned-slot probes, not concurrent throughput.

Source findings explain the failure mechanism:

1. The prompt snapshot is saved at DONE_PROMPT, before generated tokens advance the SWA tail. Paged mode disables partial context checkpoints, so later reuse of an earlier prompt boundary can require RAM restoration.
2. `server_prompt_cache::load` clears the loaded payload and erases the cache entry. A successful restore consumes the snapshot. Its allocation path also skips publishing a prompt already fully contained in an entry, so a remaining snapshot may not be refreshed during the preceding request.
3. Explicit id_slot picks the slot before normal global RAM-cache selection. Cross-slot prefix candidates are not automatically attached through that path; active leader/follower sharing covers in-flight requests, not persistent completed GPU prefixes.
4. The load ranking requires both keep/similarity ratios to strictly improve, rather than selecting the longest compatible reusable state. Correct SWA coverage cannot be inferred from token LCP alone.

The alternating trace and consuming code are consistent with this diagnosis; this analysis did not add per-entry tracing to prove every allocation/load decision. The plan requires a regression with repeated restores and per-entry accounting before fixing it.

First repair immutable reusable RAM snapshots and cross-slot lookup. Then add a compressed token trie with retained Full/SWA page handles, active locks, bounded eviction and suffix-only prefill. Handle masks, E=N-1 replay boundaries, failed attach rollback, prepare/apply refcount snapshots, clear/restore generation and divergent SWA states explicitly. Existing COW stages page payload through CPU; change that path only if branching measurements justify device-copy work.

SGLang-level behavior is defined as persistent longest reusable prefix matching, shared page ownership, lock-safe eviction and suffix-only prefill. This model needs retained SWA boundary state in addition to Full pages. Same-hardware/model SGLang speed parity has not been measured. Reference revision resolved during this analysis: SGLang `72730d6360a5894b167500d5d1ad82c34c6bb116`; use its radix-cache and hybrid-component sources for the implementation comparison.

## Execution order and completion gates

1. Reproduce these baselines with the adoption fixture set: five measured repeats, alternating candidate order, Korean/code/retrieval prompts, explicit cache state and token/output controls.
2. Tune IQ4 short splits/grouping; validate numerics and end-to-end token latency before changing codecs or writer algorithms.
3. Reprofile; retain read improvements only if needed, defer writer changes unless justified, and assess contiguous fast paths/launch overhead for the residual gain.
4. Repair non-consuming RAM snapshots and explicit-slot lookup, then implement retained GPU prefix ownership and trie/eviction with complete SWA boundaries.
5. Run combined quality, 1/4/12-request workload, near-capacity, cancellation, restore and cold/warm tests. Require >=100 single-request tok/s at primary points, >=matched F16 when available, no quality gate failure, no leaked or evicted live pages and bounded peak memory.

The measurement servers have exited; GPU memory returned to its idle state. No production restart, source implementation, commit or submission was performed. The requested deliverable is the completed analysis and implementation documents; reaching the optimization goals requires executing and validating the plans.
