# Pulsar Resident Prefix Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. The user authorized implementation. Continue inline execution; no commits or production deployment are authorized.

**Goal:** Reuse completed compatible prefixes across slots without KV payload host transfers, with correct Full/SWA state, bounded memory and observable warm TTFT improvement.

**Architecture:** A server-owned compressed token trie indexes committed reusable boundaries. Opaque retained handles in paged memory own Full and SWA page references independently of execution slots; attach is transactional. Existing RAM state cache and active leader/follower scheduling provide fallback and in-flight sharing.

**Tech Stack:** C++, existing llama_memory and block manager, CUDA page storage, current server scheduler, existing state/backend/HTTP tests.

**Spec:** `../specs/2026-10-14-pulsar-iq4-radix-design.md`

**Status:** Opt-in resident cache and reusable RAM snapshots implemented. Long-SWA repeated restore, source replacement, older divergence, full restore invalidation and independent Full/SWA pressure tests pass. Acceptance remains incomplete: see `../../pulsar-resident-prefix-validation-20261014.md` for measurements, design rulings and remaining gates.

## Global Constraints

- Primary goal is one active request, not aggregate throughput.
- Preserve 12 execution slots; resident prefixes do not consume execution sequence IDs.
- No new quantization formats, no weight offload and no implicit slot reduction.
- Reuse existing tests; no new file under tests without maintainer approval. Repository documents use ASCII.
- No commits, pushes, PR descriptions or production changes without separate applicable authorization.
- No claim of SGLang throughput parity without an identical supported comparison.

## Review Focus

1. A token match with missing historical SWA state must fall back safely (Tasks 1-3).
2. Trie splits/eviction must not free aliased pages still used by active sequences (Tasks 1, 2, 4).
3. Cancellation and failed attach must release locks exactly once without partial mutation (Tasks 1, 3, 4).
4. Clear, restore and model/adapter changes must invalidate stale handles (Tasks 1-3).
5. Multimodal placeholders must not cause cross-content cache hits (Tasks 2 and 3).

## Task 0: Make RAM prefix reuse repeatable

**Files:** `tools/server/server-task.{h,cpp}`, `tools/server/server-context.cpp`, existing `tools/server/tests/unit/test_slot_save.py` and `test_completion.py`.

**Interfaces:** Preserve server_prompt_cache::alloc/load/update signatures. load copies prompt metadata into the destination, preserves immutable payload storage, and refreshes LRU rather than consuming the entry. This prerequisite remains usable without the resident prefix API.

- [ ] Reproduce 8K repeat hits followed by misses and explicit cross-slot misses using the measurement report. Assert at least five consecutive repeated restores reuse the prefix after slot state has advanced beyond its SWA window.
- [x] Preserve immutable snapshot payloads in load; clone slot prompt metadata and move the retained entry to the LRU tail. Remove the payload clear/shrink/erase behavior only for successful retained snapshots.
- [ ] Select a compatible usable longest prefix and consider global cache candidates for explicit-slot requests. For the paged transactional restore path, validate metadata before committing; do not clear the previous slot state merely to search candidates. Keep serialized snapshot bytes and token limits enforced by existing alloc/update.
- [ ] Test independent consumers, eviction under RAM budget, failed state restore, and explicit id_slot selection. Report repeated RAM hit TTFT separately from future resident hits.

## Task 1: Retained paged-prefix ownership

**Files:** `src/llama-kv-cache-paged.{h,cpp}`, `src/llama-memory.h`, `src/llama-context.{h,cpp}`, `include/llama.h` for the narrow exported bridge; tests in `tests/test-save-load-state.cpp` and `tests/test-state-restore-fragmented.cpp`.

**Interfaces (finalized in the spec):** Use the exact uint64_t ID-based retain/attach/release and get_usage signatures in the spec. Declarations: include/llama.h. Wrappers: src/llama-context.cpp. Unsupported defaults: src/llama-memory.h. Overrides, owning registry and per-page retained counts: src/llama-kv-cache-paged.{h,cpp}. No server-side private casts.

- [x] Implement the reviewed exported bridge beside existing memory wrappers, with unsupported zero/false returns for non-paged backends and no exceptions crossing the C boundary.
- [ ] Extend existing state tests: retain a committed prefix, remove source sequence, attach to another sequence, and compare teacher-forced continuation to an uncached control. Assert page references survive source removal and return to baseline after final release.
- [ ] Add failure cases: wrong context generation, unavailable SWA window and failed COW allocation. Assert no destination mutation/refcount leak on failure. A valid attach to a nonempty idle destination replaces both domains atomically; test this as success, not an unsupported case.
- [ ] Add stale-ID, clear-then-release, sequence restore and prepare/apply interleaving tests. Prohibit prefix mutations while prepared plans are live; retry only after discarding a failed context.
- [ ] Implement retained handles using the current block-manager retain/release infrastructure. Capture only committed valid masks. Attach both domains atomically after reservation; generation mismatch is a miss. Do not serialize GPU payload to retain a prefix.
- [x] Run paged save/load tests and fragmented-state tests with compatible local fixtures; add explicit paged handle tests rather than relying on existing skipped device-only tests.

**Deliverable:** Retain/attach/release lifecycle independent of server trie, with no reserved execution slots and no KV payload transfer on a clean attach.

## Task 2: Compressed prefix index and bounded SWA boundaries

**Files:** `tools/server/server-task.{h,cpp}` beside existing prompt cache; existing `tools/server/tests/unit/test_slot_save.py` for end-to-end validation after Task 3.

**Proposed internal contracts:** `server_resident_prefix_cache` consumes a compatibility key, exact token vector and retained handle. `match` returns a pinned complete-boundary match; `publish` inserts a committed handle; `release` drops a request pin; `evict` releases unlocked handles. Use RAII pin ownership. Use the concrete argument/result contracts and ownership rules in the finalized spec section.

- [ ] Wire --cache-prefix-gpu and --cache-prefix-gpu-mib in common/common.h and common/arg.cpp, default false/256 MiB; enforce the 16 MiB metadata cap and document both options in tools/server/README.md. Reuse tests/test-arg-parser.cpp to reject nonpositive budgets and verify default-disabled behavior.
- [ ] Define compatibility keys: context/model generation, template/tokenizer identity, adapters, KV/rotation policy, absolute position semantics. Disable resident lookup for unsupported multimodal requests and adapters before comparing tokens.
- [ ] Define deterministic fixtures A+B, A+C, A, and unrelated D. Assert longest complete reusable ancestor, edge splitting, duplicate publication deduplication and explicit final-token replay allowance.
- [x] Implement compressed edges with token equality checks; a hash alone is insufficient. Tree edge splits do not imply a newly reusable SWA checkpoint.
- [ ] Publish only available committed boundaries, retaining endpoints E=N-1 and E=N plus at most one interior boundary per active prompt as specified. If a branch falls between retained SWA boundaries, select the nearest valid ancestor and recompute the gap. Record that gap as recomputed tokens, not a full cache hit.
- [ ] Use get_usage to account unique physical retained pages/bytes and free pages after eviction; add server trie/token bytes to registry metadata_bytes for the strict metadata cap. Retained bytes are not necessarily reclaimable bytes. Evict unused endpoints without double-freeing shared pages; prune empty trie edges after release.

**Deliverable:** A compact index over valid reusable state; no asynchronous disk/host tier or scheduler rewrite.

## Task 3: Server lookup, publish and fallback integration

**Files:** `tools/server/server-context.cpp`, `tools/server/server-task.{h,cpp}`, existing `tools/server/tests/unit/test_slot_save.py`, `test_completion.py` and `test_vision_api.py` only where necessary.

**Interfaces:** Consume retained handle APIs from Task 1 and trie contracts from Task 2. Preserve `/completion`, `/v1/chat/completions`, explicit slot save/restore and existing leader/follower state machine. Expose additive diagnostics using existing metrics/timing facilities.

- [ ] Extend cross-slot tests to force source-slot reuse after publication, then request an identical prefix plus suffix in another slot. Compare deterministic output/logits with cold evaluation; require suffix-only prompt work up to boundary rounding and final-token replay.
- [ ] Add SWA divergence earlier than the latest retained tail, changed image with identical placeholder positions, incompatible adapter/configuration, and state-clear cases. Assert safe miss/ancestor fallback instead of incorrect reuse.
- [ ] Integrate lookup before RAM restore/prefill. Pin, reserve, attach and only then update slot token bookkeeping. On failure release the pin and use existing RAM/cold fallback without partial slot state.
- [x] Publish only after a successful decode commits the relevant input prefix. Preserve existing follower sharing for in-flight work; no handle can expose scheduled-but-unwritten pages.
- [ ] Avoid redundant automatic DONE_PROMPT RAM serialization after successful resident publication. Keep RAM fallback for requests not retained and explicit state export unchanged. Preserve the existing global --cache-prompt override in server-schema.cpp and test both forced-on and global-off/request-false behavior.
- [ ] Release pins on finish, cancellation, slot reset and exceptions. Include generation checks to prevent a late completion attaching to a reused slot.
- [ ] Report resident hit tokens, RAM restored tokens, recomputed tokens, lookup/attach time and per-domain retained/reclaimable pages. Keep implementation diagnostics out of response text.
- [ ] Run `./tools/server/tests/tests.sh unit/test_slot_save.py unit/test_completion.py unit/test_vision_api.py` with candidate binary configured by `LLAMA_SERVER_BIN_PATH`; use dedicated paged/SWA fixtures because the default TinyLlama fixture cannot prove Pulsar SWA correctness.

## Task 4: Pressure, eviction and COW cost

**Files:** `src/llama-kv-cache-paged.{h,cpp}`, `src/llama-block-manager.{h,cpp}` only if existing reference accounting is insufficient, `tools/server/server-context.cpp`, `tools/server/server-task.cpp`; existing state/backend/server test files.

**Interfaces:** Admission supplies required Full/SWA/private-tail/COW pages; eviction releases only unlocked cache ownership. Existing allocator remains final authority; no separate allocator.

- [ ] Construct small-pool tests with multiple cached branches and active readers. Exhaust Full and SWA independently. Assert pinned entries survive, evicted references are released once and requests are deferred/rejected safely if no reclaimable pages exist.
- [ ] Reserve the next batch's allocation requirements before publication/admission. Evict LRU unlocked entries until both domains fit; requery actual free capacity after each release. Integrate ret=1 cache eviction before existing batch halving in server-context.cpp::decode, after the failed prepared context is discarded. Retry only if free counts improved; all-pinned/no-progress falls through to existing safe handling, including the one-token failure path.
- [ ] Test shared partial-page divergence, cancellation before/after decode completion, failed batch prepare, full clear and restore. Verify no leak after repeated cycles and no mutation visible to sibling branches.
- [ ] Profile `copy_page` and partial `seq_cp` merges. If host staging materially hurts branch reuse, implement device-to-device copying through existing backend copy infrastructure, preserving stream ordering and CPU fallback. Review a new backend operation separately if existing infrastructure cannot express it.
- [ ] Verify COW bytes/time separately from clean resident hits, and confirm no payload D2H/H2D on aligned clean attach. Keep numerical/state tests and exact reference accounting green.

## Task 5: Cache acceptance and combined deployment review

**Files:** Measurement changes in existing `scripts/server-bench.py` or existing cache benchmark after auditing its Flare-specific defaults; create `docs/pulsar-resident-prefix-validation-20261014.md` only when results exist.

- [ ] Run cold, same-slot warm, cross-slot warm, cross-completed-request branch, unique-prefix churn and RAM-only controls. Use 8K shared prefix + 128-token unique suffix, 256 generated tokens, concurrency 1/4/12, two warmups and five timed repetitions.
- [ ] Require warm resident median TTFT <= 50% of cold in the declared fixture, no > 5% unique-prefix p95 latency regression, zero payload host transfers on aligned resident attach and continuation correctness. Report per-configuration results; do not average away a failing concurrency point.
- [ ] Run near-capacity and mixed long/short workloads within the 256000-token Full budget; report retained SWA budget, active/cache references, queue latency and proposed >= 256 MiB free VRAM margin.
- [ ] Compare SGLang behavior against a pinned source revision. Only run speed parity claims when GPU, model, quantization, tokenizer and workload are comparable; unsupported parity remains unverified.
- [ ] Rerun the IQ4 decode plan's >= 100 tok/s and matched F16 gate with resident caching enabled. TTFT gains cannot substitute for the decode goal.
- [ ] Present measured default policy, memory limits, fallback behavior and a concrete rollback/deployment proposal. Await implementation/deployment authorization as applicable; no automatic production mutation or submission.

## Execution handoff

Review the spec and both plans together, with particular attention to the proposed ID/usage API, SWA checkpoint granularity and acceptance thresholds. Initial performance evidence is now available; extend baseline coverage in decode Task 1 and begin with split dispatch in Task 2. Cache Task 0 independently repairs repeatable RAM reuse. Each conditional optimization needs measured evidence; large API/ownership changes require contributor review before coding. The present deliverable ends with documents, not code or a service restart.
