# Pulsar implementation validation

Date: 2026-10-14
Base: 7ddea3e714c84c10464d5700b932482a4f7c75a0
Status: Candidate implementation; not approved for production adoption.

## Outcome

The requested >=100 tok/s single-request decode target is not achieved. The best verified production-shaped candidate preserves 12 configured slots, 8000 Full pages, 448 SWA pages, Q8_KV local/IQ4_NL global KV, all GPU weights, batch/ubatch 1280 and the original sampling settings. CUDA Graphs remain disabled.

| Prompt tokens | Original IQ4 median tok/s | Candidate median tok/s | Improvement |
| --- | ---: | ---: | ---: |
| 1024 | 76.432 | 96.761 | 26.6% |
| 8192 | 78.732 | 92.741 | 17.8% |

Candidate runs use two warmups and five measured requests, each producing 256 tokens. Original analysis used two warmups and three measured requests. Rates use `(predicted_n - 1)*1000/predicted_ms`. Repetitions reuse prefixes when global cache-prompt is forced on; these are decode measurements, not cold-TTFT comparisons. Do not claim an alternating five-repeat baseline matrix.

The matched 512-page Q8-local/F16-global baseline was 85.848/85.165 tok/s. All-F16 with 12 slots and ubatch1280 failed allocation. No measured 100 tok/s all-F16 baseline is available. The candidate exceeds the observed matched global-F16 values, but a fresh alternating candidate/F16 adoption comparison remains open.

## Changes

- SM70 IQ4 D512 causal decode uses short splits: 32 below 4K, 64 through 16K; the original long-context policy remains above 16K. IQ4 8K operator time improved from about 509us to 213us. D512/group2 measured 381us and was removed. Group1 and warp-codebook candidates were also rejected.
- SM70 Q8 D256 windowed decode can use existing grouped specialization with split16 when fused execution is enabled. Focused operator time was 39.6-43.5us versus roughly 45.5us. Combined server improvement is verified; a separate server ablation is not claimed.
- Paged attention avoids redundant contiguous copies only for tensors already contiguous. Strided inputs retain the existing conversion.
- Paged retained handles own Full/SWA page references independently of execution slots. Attach replaces destination metadata through a candidate state; partial-page writes use existing COW. Page copying uses existing backend tensor-copy infrastructure.
- A compressed exact-token trie retains committed endpoints and bounded interior boundaries. Entries use unique physical-byte accounting, a configurable GPU retention budget and a 16MiB metadata ceiling. Cache eviction releases cache ownership while active sequence ownership survives.
- Text RAM snapshots remain reusable across repeated restores. Media keeps its existing selection/consumption behavior. Explicit slot requests can search compatible text RAM snapshots.
- Resident caching is opt-in: `--cache-prefix-gpu --cache-prefix-gpu-mib 256`, requiring paged KV. Media, adapters and speculative draft requests use existing fallback paths.
- Metrics expose successful resident attachments, lookup misses, evictions, unique retained bytes and total cache metadata. Usage gauges are sampled between decode calls to avoid reading mutable allocator state from a metrics request during decode.

## Cache evidence

Production pool, RAM disabled, 8192-token prefix, 32 generated tokens:

- Cold prompt processing: 6007ms.
- First cross-slot replay: cache_n=8191, prompt_n=1, prompt processing 66ms, total request 425ms.
- Six repeated cross-slot replays all processed one prompt token.
- Concurrency 1/4/12 with an 8192-token prefix and 128-token suffix retained at least 8191 cached tokens for every request. At concurrency12, all 12 slots completed; per-request decode was approximately 24 tok/s, not the primary single-request target.
- Eighteen unrelated 2048-token prefixes completed under churn. A later 8192-token request safely recomputed after eviction.

This matrix proves selected reuse/capacity behavior. It uses 32/8 output tokens and one matrix pass, not the planned two-warmup/five-repeat 256-output acceptance matrix. Prompt processing time is not streaming first-token time. A separate streaming production probe measured warm first-token times around 60-90ms. No SGLang speed parity is claimed.

## Verification

- IQ4 CUDA dense-oracle numerical suite: 53/53 executed cases passed, including D512 nt1/4/12/32 and page/split boundary cases. Final grouping is the previously tested group4 policy.
- Q8 D256 CUDA suite: 20/20 passed after the grouped short-split change.
- CPU focused IQ4 operators: 19/19 passed before final server-only changes.
- CUDA paged state suite: all supported tests passed; device-only state tests are skipped by existing paged policy. The retained lifecycle now uses a 2048-token fixture when the supplied pool/context permit it, compares equivalent final-token batch shapes, invalidates handles on full restore/clear, and tests Full-only and SWA-only allocation failure followed by successful release/retry.
- CUDA fragmented state restore: passed with 3 sequences, 70 tokens each, batch256 and 64 physical pages.
- HTTP long-SWA regression: passed with 2048-token prompts, repeated cross-slot replay, source-slot replacement, branch suffixes, older divergence, cancellation followed by identical continuation and a nonzero successful-hit metric.
- RAM repeated cross-slot regression: passed.
- Earlier CPU slot suite: 10 passed, 1 incompatible paged-model skip, 2 image continuation failures. The same two image failures reproduced with the unmodified baseline forced to CPU; they are not fixed here.
- Independent whole-patch review identified an allocation-exception bug in trie splitting; allocation now completes before existing child ownership moves. Review also identified excessive pressure eviction and missing long-SWA coverage; both received targeted fixes/tests. The reviewer did not approve adoption while performance and broader quality gates remain open.

## Design rulings and limits

Ruling: Keep the existing transactional paged planner as the authority for exact batch allocation. Attach admission keeps one free page in each domain for initial private-tail replay, then prepare records the deficient domain and its next required free-page target. Server eviction retries only after that target is attainable, rather than requiring unrelated domains to gain pages. This prevents the reviewed whole-cache flushing pattern. It does not implement full advance reservation for every forthcoming suffix batch. A complete pre-admission planner interface remains an adoption gate.

Ruling: Retain synchronous backend copying and table synchronization. Aligned attach changes references without copying payload; branch COW can copy pages. A final Nsight proof of zero aligned-attach payload host transfers and measured COW cost remains open.

Prefix APIs operate on committed memory between prepared contexts. Retain/attach fail while a prepared context is alive; release must be called after its destruction. APIs require external serialization with decode and memory mutation. Handles are not serialized in exported state and become invalid on full clear/restore.

CUDA Graphs enabled with 8000 pages aborted with out-of-memory while allocating MMVQ activation caches. Attempts to exclude large paged/matmul prefill graph segments still failed at the same allocator; these experimental changes were removed. A reduced-pool graph experiment improved rates slightly but cannot substitute for the production capacity target. The launcher's Graph-disabled policy is retained. A >=256MiB free-VRAM margin is not established.

Remaining adoption gates: >=100 tok/s at both primary points; fresh alternating F16 comparison; broad teacher-forced Korean/code/number quality controls; long-context mixed workload regression; five-repeat cache/unique-prefix p95 acceptance; comparable SGLang run if supported; full advance reservation and broader incompatible-media/adapter cases.

## Reproduction and deployment

Isolated source: `/tmp/aikar-pulsar-implementation`. CUDA build: `/tmp/aikar-pulsar-build`. Build options and executable hashes are in measurement manifests. Unchanged CUDA objects were reused from the baseline after byte-identical compile-option comparison; modified CUDA and all modified C++ translation units were rebuilt. No original binary was replaced.

Artifacts are retained in the ignored repository directory `build-pulsar-validation-20261014/` when copied at final handoff. It contains raw JSONL, manifests, harnesses, test logs and the implementation ledger. GPU measurements were sequential; contaminated warmups and rejected candidates are excluded from final medians.

The authorized runtime deployment preserves Graph disable, 12 slots, 8000 pages and mixed KV types. Roll back by removing the two cache flags and selecting the previous binary.

## Runtime handoff

After explicit runtime authorization, `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/start.sh` now keeps the resident radix GPU cache disabled by default, retains the existing 32GiB RAM prompt cache, and enables `--metrics`. The runtime binary and shared libraries were updated from the verified build, and the `llama` tmux session was restarted successfully. Local health returned `{"status":"ok"}`.

Live local verification with a 6302-token prompt produced one cold request followed by five cross-slot requests: `cache_n=6301`, `prompt_n=1` on every warm request. Local metrics reported 5 successful resident hits, 1 miss, 3 evictions, 255.96 MB retained bytes and 99,288 bytes metadata.

With GPU resident caching disabled, the local 6302-token verification produced one cold request followed by five cross-slot RAM restores: `cache_n=6301`, `prompt_n=1` on every warm request. GPU prefix metrics remained zero. `https://chat.aeonthic.com/health` returned HTTP 200, but its public `/tokenize` request returned HTTP 403, so authenticated remote cache verification could not be completed from this environment. No remote cache-hit claim is made.
