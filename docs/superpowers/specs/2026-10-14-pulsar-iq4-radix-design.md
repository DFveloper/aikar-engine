# Pulsar IQ4_NL decode and resident prefix cache design

Date: 2026-10-14
Status: Implementation authorized and candidate tested. The 100 tok/s adoption threshold remains unmet. Implementation rulings and evidence are in ../../pulsar-resident-prefix-validation-20261014.md.
Source revision: 7ddea3e71 (Paged KV: Volta optimization). Initial working tree was clean.

## Intent and scope

Optimize `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/start.sh`. The user explicitly confirmed that 100 tok/s means one active request's decode speed. Keep the configured 12-slot capacity; report concurrent throughput separately. Match or exceed the comparable F16 paged baseline, and provide SGLang-style reusable prefix caching. This is a planning deliverable, not a claim that either target has been reached.

Two independently reviewable workstreams: decode performance and resident prefix reuse. Use the companion plans in `../plans/2026-10-14-pulsar-iq4-decode.md` and `../plans/2026-10-14-pulsar-radix-cache.md`. Implement only after the contributor reviews the architecture and can explain its ownership, synchronization and correctness rules.

## Deployment snapshot

| Item | Observed launcher value |
| --- | --- |
| Executable | `/home/user/aikar-engine/build-pulsar-paged-decode-20261014/bin/llama-server` |
| Model | `Lumen-3.5-Pulsar_S-LD-Q4_0_XL.gguf` relative to launcher directory |
| Device | CUDA0, all layers and experts on GPU, 36 CPU threads |
| KV | Local K/V Q8_KV; global K/V IQ4_NL; no Hadamard flags |
| Capacity | Logical 262144; 8000 Full pages x 32 = 256000 physical token positions shared by slots |
| Scheduling | 12 slots; batch/ubatch 1280; prefill chunk 1024 |
| CUDA | Graphs disabled; paged fused enabled; short split enabled |
| Prompt reuse | cache-prompt; similarity 0.01; RAM budget 32768 MiB; no-cache-idle-slots |
| Other | Backend sampling default on; mmproj on CPU; image maximum 1120 tokens |

Directory name says 3.1; actual model filename says 3.5. Do not benchmark a model selected from the directory name. Environment overrides and a pre-existing tmux session can change effective behavior: collect live argv, startup log and executable/model/template hashes at measurement time. The launcher exits when its session is already alive.

The earlier model inspection records 30 layers: 25 SWA layers, D256, 8 KV heads, window 1024; 5 Full layers, D512, 2 KV heads; 16 query heads. Current trace writer counts and D256/D512 kernel shapes agree with these layer domains. The initial sandboxed GPU query failed; an approved unsandboxed query on 2026-10-14 succeeded: Tesla V100-SXM2-16GB, driver 580.178.04, CUDA driver API 13.0, 150 W limit, initially idle. New isolated measurements are recorded in `../../pulsar-iq4-analysis-20261014.md`; historical measurements remain separately labeled.

## Evidence and bottleneck candidates

| Finding | Source | Meaning and next measurement |
| --- | --- | --- |
| Short split is restricted to Q8_KV, causal, nt <= 32, SM70 | `ggml/src/ggml-cuda/pagedattn.cu:282` | IQ4 Full still uses minimum split span 256. Measure occupancy and combine overhead before extending the dispatch. |
| IQ4 writer uses one thread per 32-value block; D512 uses only 16 active threads in a 32-thread block | `pagedattn.cu:67`, `cpy-utils.cuh:160` | Serial max search, nearest-code search and weighted scale fitting may dominate small launches. Separate writer from attention duration. |
| IQ4 reads unpack nibbles, fetch nonlinear codebook values and multiply scales | `pagedattn.cu:91` | Compression saves bytes but adds instructions. Grouped kernels use indexed codebook loads; ungrouped kernels use warp lookup. Profile both, not a blanket replacement. |
| Fused D512 group4 begins at span >= 4096 for small batches | `pagedattn.cu:273` | Benchmark around 4095/4096/4097 and group1/group2/group4 candidates. Register pressure and reuse compete. |
| Split span remains batch-wide | `pagedattn.cu:271`, `paged_decode` | Mixed lengths may create empty splits; quantify wasted CTAs separately from single-request target. |
| Persistent tables and dirty-range upload already exist | `src/llama-kv-cache-paged.cpp:268-354` | Old full-table rebuild/upload recommendations are stale. `set_table_input` still synchronizes before uploads; `apply` still copies `plan.after`. Measure present cost. |
| COW page copy stages payload through CPU | `llama-kv-cache-paged.cpp:259`, `seq_cp` | Relevant to branching/cache reuse; not proof of a steady single-stream decode bottleneck. Count copied bytes and synchronization. |
| Graphs disabled in launcher | launcher; `ggml/src/ggml-cuda/ggml-cuda.cu` | Potential launch overhead; graph experiments require explicit VRAM budget and table lifetime proof. |
| Prompt cache is a list of serialized states | `tools/server/server-task.h:612`, `server-task.cpp:1711-1898` | Linear prefix scans and whole-state restoration; consumed entries are erased. No persistent global GPU radix ownership. |
| Prompt state saved at DONE_PROMPT | `server-context.cpp:4148` | `--no-cache-idle-slots` does not disable this save path. Measure its TTFT/readback cost. |
| Partial checkpoints disabled in paged mode | `server-context.cpp:1144` | `-ctxcp 8` becomes 0. Arbitrary SWA branch reuse requires retained state at a valid boundary. |
| Active leader/follower prefix sharing already exists | `server-context.cpp:1981`, `3102`, `3838` | Extend existing scheduling; do not implement a second independent follower protocol. |

Historical evidence in `docs/pulsar-paged-kv-262k-20261011.md`: 1K/one-sequence Q8 53.29 tok/s versus IQ4+Hadamard 49.93 tok/s; Full 32K/64-token attention Q8 19.919 ms versus IQ4 24.544 ms. These used earlier code, microbatch 128 and different rotation policy. They establish a past compression cost, not current deployment speed or an F16 comparison. The old ~1 ms host interval and full-table transfer timings must also be remeasured after the Volta optimization commit. The requested F16 ~100 tok/s is user-reported, not independently reproduced here.

## Approaches considered

1. Recommended: profile the actual mixed-codec deployment, optimize existing CUDA dispatch/writer/reads, then add a resident prefix index over existing page ownership. Preserves formats and creates small acceptance gates.
2. Replace attention with a new tiled/Tensor Core implementation first. Larger numerical and SM70 engineering risk; earlier MMA experiments were rejected in `build-paged-decode/validation`. Revisit only if profiles show attention remains dominant after simpler changes.
3. Increase RAM cache and tune slot similarity alone. Low implementation cost but cannot remove state transfers or give persistent GPU prefix sharing. Useful control, insufficient for the requested cache architecture.

No full SGLang runtime port, expert offload, speculative decoding or new quantization format in this proposal. Prefix caching improves repeated prefill and TTFT; it does not eliminate reading cached context during each decode token.

## Measurement contract

Primary gate: one active request, 12 configured slots, production model/template/sampling, 256 output tokens, representative 1K and 8K input token fixtures. For both points require median decode >= 100 tok/s and >= matched F16 baseline; if F16 is faster than 100, use F16 as the target. Use the server-reported predicted_per_second; this revision excludes the first generated token from decode timing, so the equivalent formula is (predicted_n - 1) * 1000 / predicted_ms for these requests. Also record client post-first-token rate, TTFT and inter-token latency; transport events are not token counts. Report all reasoning and visible output tokens consistently. Cache-warm and cache-cold results are separate.

Use two warmups and at least five measured repetitions per point without profiling, alternating baseline/candidate order. Record distribution and raw samples; increase repeats when uncertainty prevents a decision. Profile separate repetitions with Nsight Systems and targeted Nsight Compute counters supported on SM70. Do not report profiled throughput as the final rate.

Controls: current Q8-local/IQ4-global, Q8-local/F16-global (isolate Full codec), all-F16 paged (user baseline), all-Q8 paged (historical bridge). Same weights, token IDs, output length, batch limits and graph setting. Use equal feasible physical budgets for paired comparisons if F16 cannot allocate the production 8000-page pool; separately measure production IQ4 capacity. A reduced-pool comparison must be labeled and cannot prove all-F16 fits at 256000 tokens. No silent weight offload or slot reduction.

Extended matrix: input 1K/8K/32K/128K and feasible near-capacity; concurrency 1/4/12, mixed short/long requests, text and image controls. Full-pool prompt plus output plus shared live branches must fit; 262144 occupied Full tokens cannot fit in 256000 physical positions without sharing. Record long-context curves rather than promising flat 100 tok/s across all lengths.

Record writer/read/combine microseconds, other model kernels, host gaps, table bytes, COW D2H/H2D/D2D bytes, graph memory, live/free/pinned/cache pages by domain, resident hits, RAM restores, recomputed tokens and queue latency. Use Amdahl bounds: if the unoptimized remainder already exceeds 10 ms/token, IQ4 attention optimization alone cannot reach 100 tok/s.

## Decode design

Keep the packed IQ4_NL representation and existing public op signature. First evaluate split spans 32/64/128/256 for IQ4 D512 on SM70. Select by measured shape; include combine and scratch allocation cost. Current traces show 20/33 blocks across 80 SMs and 2.27-2.72 ms/step Full attention, making this the first implementation unit. Keep empty-split softmax identities, masks and per-sequence causality correct. Do not expand Q8-specific dispatch blindly.

If writer cost is material, implement a paged-specific cooperative writer, preserving codebook, max tie rule, rounding, nibble order and weighted scale fitting. Avoid changing the shared copy quantizer used by unrelated ops. Reduction reordering needs decoded-error/logit validation even when packed bytes differ. If reads dominate, evaluate paired nibble/scale loads and separate grouped/ungrouped lookup policies. Retain only changes improving end-to-end latency outside measurement noise.

Only then address measured host synchronization/state copies and optional bounded decode graphs. Keep event ordering across prior table readers and next uploads. Do not remove synchronization merely because uploads are small. A kernel microbenchmark win is insufficient for adoption.

## Resident prefix design

The desired SGLang behavior is longest reusable prefix lookup, page ownership surviving request completion, active reference locking, eviction of unused cache state, and suffix-only prefill. SGLang's radix implementation exposes prefix matching and lock references; its hybrid cache explicitly tracks component state. These are architecture references, not an assertion of identical runtime speed. See the official sources below.

Place a compact compressed token trie and cache entries beside `server_prompt_cache` in existing server-task files. Keep page retention, attach and release inside the paged memory implementation. Cache entries must own retained page handles independent of the 12 execution sequence IDs; reserving a slot per prefix would defeat the objective. Use generation-tagged handles so clear/restore cannot resurrect stale page IDs.

Each reusable boundary records exact tokens, absolute position, Full page/mask handles, and all SWA pages needed for the next query. Index only committed work. Publish at available completed prefill chunk boundaries and prompt endpoints E=N-1/E=N, with masks for partial pages; preserve the final token reevaluation required to obtain logits. Retain at most one interior boundary per active prompt in addition to endpoints. A token match is reusable only if both Full and SWA components cover the boundary. On a missing SWA boundary, walk back to a complete retained ancestor or take the existing RAM/cold path. Never attach old Full KV with the newest SWA tail and call it a hit. Publishing every 32-token SWA snapshot is deliberately avoided until its memory cost is measured.

Namespace by model/context generation, tokenizer/template identity, adapter configuration, KV types/rotation policy, position semantics and multimodal content identity. First implementation supports exact text-only compatible requests. Multimodal, unsupported adapters and incompatible state safely use current paths; eventual multimodal sharing requires stable content identity, not placeholder token equality.

Match -> validate both domains -> pin entry -> reserve private/COW pages -> atomically attach -> prefill suffix -> publish committed boundaries -> unpin on finish/cancel. No half-attached sequence on failure. All mutation runs on the existing server scheduling thread, ordered with GPU completion. Cache references and live sequence references are separate owners of existing physical blocks. Cache eviction releases only its references; live readers remain protected.

Use byte/page-accounted LRU among unlocked reclaimable entries. Reserve the next admitted batch's Full/SWA/COW requirements before adding cache entries. Evict cached boundaries until reservations fit; if every candidate is pinned, queue/defer admission or use an existing safe failure path. Do not evict active readers. Account for physical pages once even when multiple trie entries reference them; report logical retained references separately. Recompute reclaimable bytes after each eviction.

Keep RAM snapshots as a fallback tier using existing state serialization. A successful resident publication should avoid redundant automatic full-state serialization on the hot path; explicit slot save/restore remains available. Cache pressure and fallback must have observable counters. Do not reinterpret the existing RAM MiB option as a GPU cache limit.

SWA memory is a limiting resource: a 1024-token tail across 25 D256/8-head Q8 layers is approximately 103.125 MiB before alignment, with sharing reducing incremental cost. Twelve slots already reserve substantial SWA memory; many divergent cached tails can exhaust this pool even when Full pages are free. Bound retention by both domains and test real reclaimable capacity.

## Acceptance and safety gates

- Existing backend attention/packed-write and paged state tests pass without relaxing tolerances. CPU/CUDA support results must include actual executed case counts; unavailable backends are skips.
- Numerical tests cover IQ4 nibble halves with distinct values, codebook ties, masks, empty splits, D256/D512, block sizes 1/31/32/33/64/128, causal and noncausal paths.
- Quality: preserve current IQ4 behavior; compare teacher-forced logits with current IQ4 and F16/Q8 controls. Proposed adoption limits retain the prior design's mean KL increase <= 0.01 versus Q8-to-F16 and top-1 loss <= 1 percentage point, plus exact critical retrieval correctness. Historical IQ4 long-quality failures remain unresolved until rerun. Report uncertainty and unavailable controls.
- Resident cache: a warm aligned prefix reused in another slot causes zero KV payload D2H/H2D, no re-prefill of retained complete boundaries, and cold/warm continuation agreement under existing logit tolerance. Boundary rounding and final-token reevaluation are reported separately.
- Proposed cache performance gate: 8K shared prefix plus 128-token distinct suffix at 1/4/12 concurrency, resident warm median TTFT <= 50% of cold TTFT; unique-prefix p95 TTFT and decode latency regress <= 5%. These are proposed local goals, not measured SGLang parity.
- Compare with SGLang on identical hardware/model/tokenization only if that configuration is supported; otherwise report semantic parity and local measurements, with throughput parity unverified.
- Memory: no reference leak after cancellation/clear/restore; no eviction of pinned pages; deterministic rollback on allocation failure. Preserve >= 256 MiB free VRAM at tested peaks as a proposed operational margin.
- No product-code edits, production restarts, launcher changes, commits, pushes or PRs in this planning task. Reuse existing tests; no new file under tests without maintainer approval. Repository documents use ASCII.

## Sources and limits

- Local primary sources: launcher, `ggml/src/ggml-cuda/pagedattn.cu`, `cpy-utils.cuh`, `src/llama-kv-cache-paged.{h,cpp}`, `src/llama-block-manager.{h,cpp}`, `tools/server/server-{context,task}.{cpp,h}` as applicable.
- Historical local results: `docs/pulsar-paged-kv-262k-20261011.md`, `docs/pulsar-decode-overlap-expert-cache-20261011.md`; historical results are not current benchmarks.
- [SGLang radix cache](https://github.com/sgl-project/sglang/blob/72730d6360a5894b167500d5d1ad82c34c6bb116/python/sglang/srt/mem_cache/radix_cache.py), pinned source fetched and inspected 2026-10-14: match_prefix, evict, inc_lock_ref and dec_lock_ref.
- [SGLang hybrid component contract](https://github.com/sgl-project/sglang/blob/72730d6360a5894b167500d5d1ad82c34c6bb116/python/sglang/srt/mem_cache/unified_cache/components/README.md), pinned source fetched 2026-10-14: Full path-lock and SWA segment-lock preserve different component coverage.
- [Existing upstream paged KV proposal #22569](https://github.com/ggml-org/llama.cpp/pull/22569) and [prefix cache discussion #9677](https://github.com/ggml-org/llama.cpp/discussions/9677). Existing work was checked; this document does not request upstream submission.

Remaining measurement dependency: reproduce the user's historical all-F16 ~100 tok/s configuration; current all-F16 with 12 slots and ubatch 1280 failed allocation even at 512 Full pages. GPU access has been verified. Design review dependency: accept the proposed primary prompt lengths, quality thresholds, SWA boundary retention policy and cache memory/performance gates. These dependencies do not prevent delivering the requested plan; they prevent calling it a proven optimization or beginning implementation without review.


## Finalized prefix API and lifecycle contract

This section resolves the initial implementation alternatives. It is the proposed design to implement after review, not an existing API.

Use the normal exported memory API, not a private server cast. Add declarations beside existing memory functions in `include/llama.h`, wrappers in `src/llama-context.cpp:4990`, default unsupported virtual methods in `src/llama-memory.h`, and overrides in `src/llama-kv-cache-paged.{h,cpp}`. The server links through llama-common/llama shared libraries; an exported narrow bridge preserves this boundary.

```cpp
typedef uint64_t llama_memory_prefix_id;
LLAMA_API llama_memory_prefix_id llama_memory_prefix_retain(llama_memory_t mem, llama_seq_id src, llama_pos end);
LLAMA_API bool llama_memory_prefix_attach(llama_memory_t mem, llama_memory_prefix_id id, llama_seq_id dst);
LLAMA_API void llama_memory_prefix_release(llama_memory_t mem, llama_memory_prefix_id id);
```

Add a narrow read-only query used by eviction and budget enforcement:

```cpp
struct llama_memory_prefix_usage {
    uint64_t full_free_pages;
    uint64_t swa_free_pages;
    uint64_t full_retained_pages;
    uint64_t swa_retained_pages;
    uint64_t retained_bytes;
    uint64_t metadata_bytes;
};
LLAMA_API bool llama_memory_prefix_get_usage(llama_memory_t mem, struct llama_memory_prefix_usage * out);
```

The query returns false and zeroes out on unsupported memory; null out returns false. retained pages/bytes count each physical page once across all retained handles, including pages also referenced by active sequences. metadata_bytes includes retained registry maps. The server adds trie/token storage to enforce its 16 MiB metadata budget. Free pages come from the actual allocator after previous releases. Keep per-page retained-owner counts in paged memory so release updates usage without scanning every long token prefix. A cache eviction can free zero pages when active or other cache references remain; requery after eviction rather than equating retained_bytes with reclaimed_bytes.

IDs are nonzero, monotonically assigned within a memory instance and never reused after clear; zero means unsupported/invalid/unavailable. The owning memory object holds the retained registry, not a detached pointer. The server must pass the same memory instance and discard all cache objects before context destruction. Unknown/stale IDs fail attach and make release a no-op. Retain/attach validate committed positions and codec policy; exceptions do not escape the public C boundary. Allocation failure returns zero/false without mutation.

For exclusive end E, Full requires every position [0,E); SWA requires [max(0,E-window+1),E) for the next causal query at E. Retain validates each mask, not just min/max. Attach replaces the entire destination sequence transactionally, supports a nonempty idle destination, and retains both domains before releasing old ownership. Page table cache is invalidated after a successful attach. Existing write paths COW shared pages.

For replaying a complete prompt of length N without storing logits, publish E=N-1 as well as E=N when both are supported by committed state. A matching request selects E <= min(LCP,N-1) and re-evaluates its suffix. This avoids falsely declaring an N-only endpoint reusable when the final token needs logits. Never invent an intermediate SWA boundary from a split trie edge. Chunk snapshots are published at actual successful server decode boundaries only; sub-batches whose older SWA state is already gone are not reconstructed retrospectively.

Retained references live in the same block-manager registry copied by `init_batch`/`prepare`/`apply`. No retain, release, eviction or attach is allowed between preparing a batch and applying its final plan; otherwise an older `plan.after` could overwrite new refcounts. Serialize all such operations at scheduler batch boundaries, after GPU readers complete. On allocation failure, discard the failed prepared context, evict, and prepare again from current state. Do not mutate a live plan.

Full clear drops the prefix registry and reinitializes pools atomically; old IDs cannot release a recycled page. A single-sequence restore preserves unrelated cache ownership and allocates replacement pages using existing transactional state_read. Full state replacement first invalidates resident entries under the server's clear path. Metadata-parse failure preserves existing entries. Sleep/unload releases cache entries before destroying the context. The context-generation compatibility key changes on reload and adapter policy changes.

## Finalized cache policy and server integration

Add opt-in `--cache-prefix-gpu` (default false) and `--cache-prefix-gpu-mib` (default 256, positive) in existing `common/common.h` and `common/arg.cpp`, with descriptions in `tools/server/README.md`. This budget caps unique resident-cache GPU page bytes from existing pools; it does not allocate an additional 256 MiB pool. Keep an internal 16 MiB metadata cap, checked before publication. Existing `-cram` keeps its host-memory meaning. Defaults remain opt-in until validation gates pass.

Declare `server_resident_prefix_cache` beside server_prompt_cache in `server-task.h`; implement it in `server-task.cpp`. Use these concrete internal operations:

- `match(key, tokens, max_end)` returns `{entry_id, memory_prefix_id, end}` and increments an entry pin, or an empty result. key is a per-context configuration epoch; reject multimodal/active adapters before lookup in the first release.
- `publish(key, tokens, end, memory_prefix_id)` takes ownership on success and releases the supplied handle on failure; an identical key/token/end entry refreshes LRU without duplicating ownership.
- `unpin(entry_id)` removes exactly one request pin; wrap the receipt in a move-only scoped owner.
- `evict(full_pages_needed, swa_pages_needed)` releases unlocked least-recently-used endpoints until the allocator reports enough free pages, or returns failure when all remaining candidates are pinned.
- `clear()` drops entries and owned handles before context teardown. Expose counters through existing server metrics, not model response text.

Use the existing logical-page maps for retained handles initially. The one-interior-boundary policy and strict metadata budget bound their duplication; reject publication before allocation would exceed the cap. Do not introduce a persistent-map subsystem for the first implementation. Start with endpoints plus at most the most recent eligible chunk boundary per active prompt. Replace an unpinned interior boundary rather than retaining all 1024-token chunk snapshots. Keep common branch endpoints while the byte/page budgets permit. This policy can recompute an unmatched interior suffix; report that instead of promising arbitrary-token SWA reuse.

Concrete server contracts (internal C++, proposed for implementation):

```cpp
struct server_resident_prefix_match {
    uint64_t entry_id = 0;
    llama_memory_prefix_id memory_prefix_id = 0;
    llama_pos end = 0;
};
server_resident_prefix_match match(uint64_t key, const llama_tokens & tokens, llama_pos max_end);
bool publish(uint64_t key, const llama_tokens & tokens, llama_pos end, llama_memory_prefix_id id);
void unpin(uint64_t entry_id);
bool evict(uint32_t full_pages_needed, uint32_t swa_pages_needed);
void clear();
```

These are methods of server_resident_prefix_cache, which owns its memory instance and budgets. entry_id is monotonically assigned and zero denotes a miss. key is a monotonically assigned server compatibility epoch, not a lossy content hash: reset it on reload/template/adapter/position-policy changes. Exact token equality determines trie edges. match increments an entry pin; unpin consumes that receipt once. publish consumes the passed handle in every return path. It releases duplicates/failures, retains successful ownership, and never publishes end beyond tokens.size(). evict returns true only if the allocator query confirms both requested free-page counts. Cache clear invalidates outstanding receipts during teardown; RAII receipts are destroyed before the cache object.

The server owns admission reservations: demand is computed by the paged planner for the exact upcoming ubatch, including Full/SWA pages and COW. Use failed prepare as a safe backstop, not a fixed pages-per-request heuristic. If llama_decode returns allocation failure (ret=1), first evict unlocked entries and retry the original batch only after its failed memory context has been discarded; compare free-page counts to guarantee progress. If all entries are pinned or free counts do not improve, enter the existing idle-slot clearing/batch-halving path. Do not put an endless cache retry before a one-token context error. Cache entries must yield capacity to all 12 active slots. Since the present SWA pool has 448 pages at ubatch 1280, 12 retained independent 1024-token tails would consume 384 pages before new prefill/COW. Never reserve a fixed cache share that prevents configured concurrency.

Follow current effective cache_prompt semantics: `server-schema.cpp:553` forces true when the launcher enables --cache-prompt. Do not silently change that API precedence in the optimization. Test both global-force-on and global-off/request-false paths. A truly cold measurement must clear the slot or change tokens near the beginning and verify cache_n, not rely on the request flag alone.

Pinned slot selection currently bypasses normal global RAM cache selection unless a later SWA fallback triggers it. Resident lookup must run for both explicit id_slot and automatic selection before slot-specific LCP decisions. A compatible resident snapshot must survive successful loads and multiple consumers.

## Low-risk cache prerequisite

Before the resident trie, stabilize existing RAM snapshots as a separately reviewable change: `server_prompt_cache::load` currently clears payloads and erases a selected entry. Preserve snapshots as immutable data, clone prompt token/checkpoint metadata into the slot, and refresh list LRU by splice. Keep limit enforcement in alloc/update and measure extra host memory. Select the longest compatible usable prefix rather than the current pair of strict improvement ratios; SWA validity still needs validation, and a failed candidate must not destroy the prior slot state. This fixes consumable-cache behavior without claiming zero-copy GPU reuse. Repeated 8K requests must not alternate full prefill and RAM restore because the snapshot was consumed.
