# Pulsar Paged KV Q4/IQ4 and 262K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Native execution was already selected by the user. Steps use checkbox syntax for tracking.

**Goal:** Implement the approved paged Q4_0/IQ4_NL and Hadamard paths, measure 262144-token capacity and quality, and identify useful Decode Overlap and Expert Cache supplements on V100.

**Architecture:** Extend the existing combined per-layer K/V page tensor and existing local/global flags. Keep SWA Q8_KV initially; test Full Q4/IQ4 with and without existing K512/V64 rotation. Measure serving bottlenecks before introducing concurrency or dynamic expert residency.

**Tech Stack:** C++17, GGML CPU/CUDA, SM70 V100, existing backend/state fixtures, temporary benchmark drivers outside the repository.

**Spec:** [Approved codec/capacity design](../specs/2026-10-11-paged-kv-q4-iq4-262k-design.md).

## Global Constraints

- Preserve existing root changes, especially Gemma4 QKV views and backend tests. Reuse `/home/user/.codex/worktrees/9d53/aikar-engine`, with the root baseline patch already applied.
- No commits, pushes, PR text, new files under `tests/`, production launcher edits or production process interruption.
- Matching K/V type within each domain; accepted CPU/CUDA types are F16, Q8_KV, Q4_0 and IQ4_NL. Vulkan keeps F16/Q8_KV only.
- Use actual GGML packed formats; IQ4_NL is nonlinear. No permanent F16 KV or expert-weight replicas.
- Preserve existing Hadamard semantics: Full K512/V64, SWA K256/V64 for Pulsar when enabled; no silent V512 policy.
- State metadata must validate codec, geometry and rotation before mutation. Unsupported device-only/partial state cases are skips.
- Physical Full pool target is shared 8192 blocks x 32 tokens. Logical 262144 alone does not establish physical capacity or quality.
- A recommended configuration needs at least 256 MiB free at workload peak. Keep the 12-slot target visible; reduced slot counts are separately labeled alternatives.
- Timing uses three matched repetitions after warmup, without capture/profiling overhead. A byte reduction alone is not a serving speedup.
- AM is canceled. No fitting, compaction, beta tensors or AM state changes are in scope.

## Review Focus

- IQ4 CPU and CUDA writers can choose different scales: validate real packing and decoded error without inventing uniform-Q4 equivalence (Tasks 1-2).
- Cache bases cannot change while occupied: reject policy changes without corrupting old pages (Task 3).
- A graph reused after a policy/geometry change needs new rotation inputs rather than stale tensors (Task 3).
- Failed state imports and alias/COW divergence must leave live sequence outputs unchanged (Task 4).
- Long quality controls cannot hold two large caches at once; use sequential identical-token replay and label unavailable controls (Task 5).

## File and interface map

- `src/llama-kv-cache-paged.{h,cpp}`: accepted domain codecs, separate rotation policy, input matrices and PKV4 persistence.
- `src/llama-graph.{h,cpp}`: paged rotation inputs/reuse checks, Q/K/V transforms and inverse V transform.
- `src/llama-impl.{h,cpp}`, `src/llama-kv-cache.cpp`: share existing normalized Hadamard generator without changing the ordinary cache path.
- `ggml/src/ggml.c`: paged op format validation. Keep the seven existing input tensors and public signature.
- `ggml/src/ggml-cpu/ops.cpp`: packed row writes/reads using existing type traits.
- `ggml/src/ggml-cuda/pagedattn.cu`: format-specialized packed write/read in all grouped, scalar and split paths.
- `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`: explicit backend support guards.
- `tests/test-backend-ops.cpp`, `tests/test-save-load-state.cpp`: reuse existing numerical and lifecycle fixtures.
- `examples/kv-quality/kv-quality.cpp`: clear all reference rotation/native codec settings; reuse short F16 quality smoke.
- Temporary build/artifacts: reuse `/tmp/pulsar-am-experiment/build-am` with provenance explaining its historical directory name; new measurements go under `/tmp/pulsar-paged-262k/`.
- Final evidence: `docs/pulsar-paged-kv-262k-20261011.md`. Record binary/source/patch/model hashes and raw artifact paths.

## Task 1: CPU packed formats and contract

**Interfaces:** Keep `ggml_paged_attn(...)` unchanged. CPU writes consume contiguous F32 head rows and produce GGML-format rows at the existing cache byte strides. Constructor admits the four matching formats; unsupported device/geometry still rejects.

- [x] Extend the existing paged fixture with Q4_0 and IQ4_NL cases at D256/D512, block sizes 16/32/64/128, masked fragmented pages, multi-sequence writes and causal/noncausal SWA. Use fixed packed rows with known nibble values for nonlinear-codebook decode assertions and all-zero rows for scale-zero behavior.
- [x] Separate packed-write validation from attention output validation: assert Q4 nibble order, scale and unchanged non-written rows; for IQ4 compare decoded rows to the independently specified codebook and error bounds. Preserve the exact-byte Q8 rounding fixture.
- [x] Run the CPU paged cases before implementation; require an explicit unsupported-format/assertion failure, not a skip reported as success. Baseline is 86/86 CPU cases passed.
- [x] Extend GGML/cache validation and CPU row dispatch using existing `from_float_ref`/`to_float` traits or the existing format quantizers. Preserve barriers, write deduplication and strides. Reject new formats on Vulkan before pool execution.
- [x] Run `build-am/bin/test-backend-ops -b CPU -o PAGED_ATTN,PAGED_ATTN_WRITE`; require all supported cases to pass with the existing attention NMSE limit 1e-6. Do not relax the limit to hide a packing bug.

## Task 2: CUDA Q4/IQ4 paged write and attention

**Interfaces:** Replace the internal boolean format dispatch with `ggml_type` specialization across `paged_write`, `paged_read`, `paged_decode` and `paged_decode_block`. Preserve the existing head-group selection and split/combine ABI.

- [x] Run new CUDA0 format cases before changing CUDA kernels; require explicit unsupported output and record baseline F16/Q8 results.
- [x] Reuse `cpy_blck_f32_q4_0` and `cpy_blck_f32_iq4_nl` from `cpy-utils.cuh` for 32-component block writes. Keep one source head row and deduplicated destination slots; avoid an F16 staging cache.
- [x] Decode Q4_0 nibbles with signed scale and IQ4_NL nibbles with the existing device codebook. Implement all scalar, grouped D256/D512 and split paths, including zero scales. Keep Q8 exact rounding unchanged.
- [x] Update CUDA support checks. Run CPU and CUDA0 paged fixtures; require supported numerical cases, real packed-write assertions and fragmented-mask cases to pass. CPU/CUDA IQ4 scale differences need decoded-error evidence; do not require byte equality unless both algorithms actually agree.
- [x] Run existing paged performance mode for D256/D512, tokens 1/4/64 and context spans 8K/32K. Record register spills, write cost and split scratch bytes; keep the baseline for later end-to-end comparisons.

## Task 3: Existing Hadamard policies in paged graphs

**Interfaces:** Add `ggml_gen_hadamard(ggml_tensor * tensor)` to `llama-impl.h` by moving the existing generator. In paged cache add `ggml_tensor * build_input_k_rot(ggml_context *, bool swa) const`, `build_input_v_rot` with the same arguments, and `void set_input_rot(ggml_tensor *) const`. Rotation state is two K/V policy arrays indexed by domain and a matrix map keyed by rotation size.

- [x] Extend existing backend Hadamard fixtures with dense orthogonal-reference assertions at 64/256/512 and a paged rotation composition case. Assert normalized `H*H` is identity and attention with inverse V rotation agrees with the unrotated F16 result within the existing numerical tolerance.
- [x] Establish failing paged model smoke runs for global K/V flags and independent local flags before implementing the graph path. Use small caches to isolate policy acceptance from OOM.
- [x] Share the generator, store separate policy bits and create matrices only when required. `set_kv_hadamard_policy` rejects a changed effective policy if any live pages exist; the same policy is allowed. Preserve explicit/separate flag normalization in the context.
- [x] Add K/V rotation input arrays to `llm_graph_input_attn_kv_paged`; compare presence/shape/policy in `can_reuse`, populate through cache methods, rotate Q/K after RoPE and V before the paged op, invert V on output before Wo. Preserve packed cache backend ownership.
- [x] Run CPU/CUDA transform/attention fixtures and small Pulsar global/local policy smoke. Assert SWA bytes and outputs remain unchanged when only global rotation is enabled. Verify a live policy change rejects and leaves the original continuation reproducible.

## Task 4: Compatible state and rollback

**Interfaces:** PKV4 magic `0x504b5634`, followed by existing block-size/layer-count fields; each layer record has six uint32 fields: codec, D, 2*KV-heads, SWA flag, K rotation size (0 if disabled), V rotation size (0 if disabled). Page records keep the existing alias rules and four mask words for block sizes >32.

- [x] Extend existing save/load fixtures with Q4/IQ4 plus global/local rotation round trips, shared-page restore and divergent COW continuation. Add wrong codec/rotation size, truncated metadata/payload and invalid alias cases; assert failed imports preserve state bytes and continuation logits.
- [x] Demonstrate failures before PKV4 support. Implement versioned parsing that accepts PKV2/3 only for original unrotated F16/Q8 geometry; reject malformed or incompatible fields before allocator/payload mutation. Retain the existing candidate/pending-page transaction.
- [x] Run `test-save-load-state` with Pulsar, one/four slots, Q4 then IQ4 global K/V rotation and SWA Q8, enough pages for the fixture. Require all supported cases to pass and explicitly label partial/device-only skips.
- [x] Retest an old unrotated F16/Q8 fixture, block-size 32/64/128 and cross-policy mismatch. A legacy file cannot implicitly assign a rotation basis to existing packed rows.

## Task 5: Trustworthy quality and 262K capacity

**Interfaces:** Existing short tool stays `llama-kv-quality`. A temporary sequential driver consumes one saved token array and emits sampled teacher-forced logits per configuration; it never retains two long contexts. Results distinguish physical allocation, actual occupied token count and quality.

- [x] Fix F16 reference settings to clear K/V local/global rotation, separate/explicit flags and both TurboQuant bit policies. Verify reference settings independently before interpreting candidate KL.
- [ ] Run short F16/Q8/Full-Q4/Full-IQ4 controls with and without global K512/V64, matching batch/token schedule. Require candidate KL-versus-F16 increase <=0.01 relative to Q8 and top-1 agreement loss <=1 percentage point.
- [ ] Run sequential 8K/32K/128K/262K candidate-versus-Q8 replay where controls fit. Use one slot, a smaller matched ubatch and logits sampled at fixed positions. Require direct mean KL <=0.01 and top-1 agreement >=99%; label absent controls and small-sample accuracy as inconclusive.
- [ ] Include Korean instructions, code and exact/dispersed name-number retrieval. Critical deterministic retrieval must stay correct; task accuracy loss <=1 percentage point needs reported uncertainty, not only a small fixture pass.
- [ ] Measure 8192 Full blocks x32 for slots 1/4/8/12, initially SWA Q8 and ubatch 1280. Repeat smaller ubatches as separate configurations if needed. Require actual 262K occupation for the capacity claim and >=256 MiB free at workload peak for recommendation. At a completely occupied pool, reserve output pages or explain why decode cannot append another token.
- [x] Keep both Q4 and IQ4 experimental until quality and capacity pass. Report the 12-slot result even if only fewer-slot alternatives fit.

## Task 6: Serving speed and next optimization selection

**Interfaces:** Temporary benchmark/profile driver writes matched JSON for PP, TTFT, TG, common steady concurrent throughput, CPU/GPU idle gaps, H2D/D2H bytes and VRAM. No new server worker/cache contract is implemented in this task.

- [ ] Compare baseline Q8 with quality-passing codec candidates at 1K/8K/32K and longest fitting context, concurrency 1/4/12, same sampling, mixed-prefill policy and graph setting. Use warmup and three timed repetitions; profile separate runs.
- [ ] Measure whole-table transfers and allocator copies around paged `prepare/apply/set_input`; compute useful-span split counts for mixed-length sequences. Record which dominates before writing a separate metadata/split/query-tile implementation plan.
- [x] For Decode Overlap, trace exposed host gaps around `llama_decode`, sampled-token synchronization and `post_decode`, and identify shared/routed FFN branch lifetimes. vLLM DBO overlaps DP+EP all-to-all, which this single V100 deployment does not have. Do not remove synchronization without an input/output ownership design.
- [x] For Expert Cache, capture actual top-8 routing separately from timed runs: per-layer unique experts, cross-token reuse and cache misses under 116/112/108/96/80/64 slots. Replay the trace against byte-budgeted LRU and frequency-aware policies; include selected-expert pinning and zero prediction benefit as the conservative case.
- [x] Measure pinned H2D bandwidth at actual expert payload sizes. Compute transfer-time lower bounds and exposed time; do not assume overlap hides misses. Full packed expert weights are about11102.71 MiB; one layer has about370.09 MiB. Four fewer resident experts/layer would save about382.85 MiB before staging/maps, but all-GPU misses are currently zero.
- [x] If cache budgets or overlap look useful, write a separate concrete design for GGUF sliced expert ownership, logical-to-slot mapping, event/pin lifetimes, miss fallback, cancellation, graph invalidation and byte accounting. Reuse existing MMID and event infrastructure; no CPU loop launching one kernel per expert. Review that architecture before changing execution contracts, as required by AGENTS.md.
- [x] Write the final evidence report with accepted/rejected/inconclusive candidates and matched speedups. Require codec quality, numerical/state checks and meaningful end-to-end gain before recommending a default. Request one final independent code review after implementation; preserve the user's native implementation choice.

## Self-review and handoff

Execution disposition: Tasks1-4 are implemented and numerically/lifecycle verified. Short and8K/32K quality measurements ran, but broad acceptance failed; unchecked acceptance criteria do not imply a passing candidate. The user explicitly waived completing the long stress test after131072tokens; it stopped after180224completed tokens with8192pages allocated and12slots reserved. Full262144occupation, exact peak free memory,128K/262K quality and the full long/concurrent timing matrix remain uncertified. The follow-up selection is diagnostic, with no new dynamic-cache or overlap execution contract. Detailed deviations and evidence are in the execution ledger and final report.

The six tasks cover the approved codec foundation, persistence, quality/capacity and measured selection of serving follow-ups. Metadata delta uploads, transactional-state copy reduction, per-sequence split scheduling, tiled prefill, shared pools, bounded graphs and dynamic expert caching remain separate implementation units; this plan does not silently merge their execution contracts into the codec patch.

Native execution is preserved. Review this concrete plan once before product implementation; do not ask again for an execution-method selection. No commit steps are authorized.
