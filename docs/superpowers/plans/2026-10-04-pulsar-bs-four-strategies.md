# Pulsar backend sampling and four-strategy implementation plan

> Use superpowers:executing-plans inline. No commits, pushes, PRs, or new tests files.

**Goal:** Single decode >=90 tok/s and twelve-request aggregate >=600 tok/s, with100/1200 as targets and <=5% matched regression.

**Architecture:** Reuse the existing backend sampler, expert batching and paged memory infrastructure. Measure each change against one frozen runtime; keep unsupported policy/shape paths intact. Obtain concrete AGENTS.md confirmation before new concurrency or cross-layer execution contracts.

**Spec:** `2026-10-04-paged-production-performance.md`, plus the user-approved replacement of the8192 hard reasoning budget by token101 channel-end bias and existing-bs.

## Constraints and review focus

## Engine methods mapped to this implementation

| Method | Cost addressed | Existing implementation and next change |
| --- | --- | --- |
| GPU sampling | Full vocabulary D2H and CPU sampling | Reuse `-bs`; profile production top-k64/top-p0.95/temp1 and bias101+1. Return token IDs and policy-required probabilities; keep grammar and explicit budget fallbacks. |
| Expert batching | Repeated reads of the same expert weights | Existing MMQ groups expert IDs on GPU. Measure active-expert occupancy and choose existing GEMV/GEMM and tile sizes from actual inputs per expert, not total requests. |
| CUDA graphs | CPU kernel launch cost | Keep disabled with roughly98MiB historical headroom. Require fixed input/scratch buffers and a measured memory envelope at unchanged KV capacity before bounded decode capture. |
| CPU/GPU overlap | Exposed scheduling and output-processing gaps | Profile uncovered gaps first; preserve input lifetime, sequence ownership, RNG, cancellation and drain before moving next-batch preparation or previous-output processing across execution. |
| Decode-first chunked prefill | Long prompts blocking active decode | Decode tokens already precede text prefill and the mixed-workload chunk budget is64. Add allocation-aware admission before `llama_decode`, preserving atomic multimodal paths. |
| Prefix KV sharing | Repeated prefix compute and storage | Existing prefix leaders and page retain/release/COW already provide sharing. Verify divergent continuations and slot reuse; do not create a second prefix cache. |

PagedAttention reduces fragmentation and helps maintain batches; it does not accelerate weight multiplication by itself. Latest vLLM deployment compatibility on V100 must be checked against the exact release and backend; sm70 is not assumed compatible merely because an engine supports CUDA.

## Execution order and evidence

1. Profile existing GPU policy coverage and D2H. Earlier full-vocabulary twelve-output copies cost about3.9ms/step over PCIeGen3x4; remeasure after `-bs` rather than treating that old cost as still present. Preserve seeded distributions and policy semantics; GPU argmax alone is insufficient.
2. Measure routing for identical and distinct prompts. With116 experts and top8, twelve requests produce96 assignments, which can concentrate or scatter. Reject fixed tiles if padding wastes work; sweep existing MMQ tile choices, then consider occupancy-dependent dispatch.
3. Measure uncovered CPU/GPU gaps. The historical6-7ms launch API span overlaps GPU execution and cannot be added to the GPU critical path. Device token relay and two-batch lifetime changes require the concrete concurrency design gate.
4. Reduce8K attention and allocator retries. Historical D512+D256 attention was about9.8ms/step, almost the entire10ms step budget implied by1200 aggregate at twelve requests. Bound prefill by actual pages, including COW/SWA, and compare mixed PP/TG stream gaps.

### Allocation admission design awaiting API confirmation

- Add a non-mutating memory batch-prefix fit query with an ordinary-memory default that returns the requested length.
- In paged memory, simulate existing `prepare` on a state copy, using the same full/SWA page reclamation, COW and reference counts. Never alter device KV or live allocator state during admission.
- Server queries the rendered batch view before decoding and chooses a fitting prefix; zero fit preserves existing idle-slot clearing and exhaustion handling. Decode tokens remain first; atomic image and non-splittable batches retain existing validation.
- Reuse fragmented restore/COW/rollback fixtures and add assertions to an existing fixture for free-page exhaustion, shared partial pages and SWA reclamation. Compare errors, logits and slot state to the old retry path.
- Measure allocation failures, CPU admission cost, PP/TTFT, TG, stream gaps and peak VRAM. Keep only a net endpoint win; simulation can cost more than retries when the pool is unconstrained.

### Measurement corrections

- `no-pad-probe` omitted `--backend-sampling`; its92.85/375.48 result did not exercise `build_sampling` and is not padding-removal evidence.
- `GGML_CUDA_MOE_NCOLS_OPT` selects an existing MMQ tile optimization input; it does not yet implement occupancy-adaptive dispatch. The corresponding checkbox below records only diagnostic plumbing.
- The frozen matched512x3 GPU/CPU sampling medians are93.01/88.12 and406.66/356.06 at50tokens;85.24/81.28 and321.75/287.82 at8192tokens. No600/1200 achievement is established.

- Keep Q4_0 weights, Q8_KV,8192 physical blocks x16tokens, twelve slots, logical262144, GPU output/embedding and CPU mmproj.
- Budget-1 disables the reasoning sampler. Default `<channel|>` bias is101+1, a soft preference with no hard token bound. Keep grammar, request budgets, reasoning_control and probability-request fallbacks intact.
- CUDA graphs stay disabled unless measured memory accounting proves room without shrinking KV capacity. No permanent FP16 weight/KV copies.
- Verify actual backend execution, seeded stochastic output, penalties, channel closure, tools, cancellation, fragmented COW/rollback and VRAM. Explicitly test final-answer trailing close markers with static bias.
- Use50/4096/8192 prompts,512 outputs, concurrency1/2/4/8/12, three repeats. Report PP/TTFT, request and aggregate TG, common steady interval, stream gaps and memory separately.
- Never sum overlapping CPU/GPU spans. Operator gains do not establish endpoint gains; rejected MMA/tiled kernels remain rejected.

## Strategy1: GPU sampling and transfer reduction

**Files:** deployment `Lumen-3.1-Pulsar/start.sh`, existing `build-paged-decode/validation/decode_bench.py`, `src/llama-graph.cpp`, `src/llama-context.cpp`, `src/llama-sampler.cpp`, `common/sampling.cpp`.

- [x] Confirm token101 and Gemma4 reasoning end tag from the running model.
- [x] Configure default budget-1, existing-bs on, channel-end bias+1. Preserve rollback environment overrides and permit empty bias to disable it.
- [x] Fix benchmark request budget from0 to-1, remove inherited-bs flags for CPU ablation, record actual final candidate arguments.
- [x] Compare frozen selected runtime-bs on/off, first50/8192 with1/12 and512x3; broaden after a confirmed win.
- [ ] Profile actual stochastic production chain separately from greedy, including backend-prefix coverage and logits/probs/candidate transfers.
- [ ] Reuse existing state/sampling/completion fixtures for logit bias, seed, penalties, n_probs, grammar and budget fallback. Run thinking-on/off, tools and cancellation smoke tests.
- [ ] If unrequested sampler outputs transfer full arrays, propose opt-in output selection using existing tensors and fallback. Confirm the concrete cross-layer contract before implementation; compare same-input policies and actual D2H sizes.
- [x] Add opt-in `LLAMA_SAMPLING_NO_PAD=1` probe; keep default padded graph path until matched numerical and endpoint gates pass.

## Strategy2: Expert distribution and MoE dispatch

**Files:** `ggml/src/ggml-cuda/mmq.cu`, `mmvq.cu`, `mmid.cuh`, `ggml-cuda.cu`; existing `tests/test-backend-ops.cpp`.

- [ ] Collect per-layer expert occupancy for identical and distinct prompts, including empty experts and inputs per active expert. Disable instrumentation during timing.
- [ ] Benchmark existing MMVQ/MMQ at8/9/12/13 and actual gate-up/down shapes2816/1408 and704/2816, with measured routing.
- [x] Keep the existing Volta Q4_0 MMQ twelve-column reuse behind `GGML_CUDA_MOE_NCOLS_OPT`; endpoint adoption remains pending.
- [ ] Sweep existing tile parameters and GPU expert grouping. Keep GEMV for sparse expert occupancy when packing/padding outweighs reuse.
- [ ] Propose adaptive GPU dispatch only after identifying a measured crossover; no CPU expert download or per-expert host launch loop. Confirm any new execution pattern before coding.
- [ ] Require unchanged numerical limits, memory and server gains for both prompt distributions. The rejected fixed16-column FP16 loader is not adopted.

## Strategy3: CPU/GPU overlap and launch reduction

**Files:** `tools/server/server-context.cpp`, `src/llama-context.cpp`, `src/llama-graph.cpp`; existing scheduler/event interfaces.

- [ ] Trace exposed idle gaps, launch time, GPU work and D2H separately.
- [ ] Reuse graph/input metadata and remove synchronization only where stream ordering and buffer lifetimes prove safe.
- [ ] Specify immutable next-batch metadata, device token relay, prior-result processing, slot ownership, errors, cancellation drain and shutdown; request AGENTS.md confirmation before new concurrency.
- [ ] Keep stream order and sampler state intact. Test multi-slot cancellation, sleep/shutdown and mixed prefill.
- [ ] Consider bounded decode-only graph capture only after confirming VRAM fits with unchanged KV budget; otherwise keep graphs disabled.

## Strategy4: Long-context attention and page-aware prefill

**Files:** `ggml/src/ggml-cuda/pagedattn.cu`, `src/llama-kv-cache-paged.cpp`, `src/llama-block-manager.cpp`, `tools/server/server-context.cpp`.

- [ ] Sweep existing split/occupancy parameters at1/2/4/8/12, dimensions256/512 and actual SWA/8K spans. Retain only measured wins with existing tolerances.
- [x] Add opt-in `GGML_CUDA_PAGED_DECODE_BLOCKS=8` sweep for decode split occupancy; retain default16 after endpoint comparison.
- [ ] Inspect prefill failures and rollback. Bound chunks from existing free-page information including COW, SWA and live decode reservations, without dropping cache policy.
- [x] Add opt-in `LLAMA_PAGED_PREPARE_ADAPTIVE=1`: paged memory retries admission on a state copy with smaller ubatches before server-level decode retry; default remains unchanged pending endpoint gates.
- [ ] Confirm any new page-budget API/scheduler contract before coding. Preserve decode priority and prefix sharing.
- [ ] Run attention/write numerical matrix, fragmented restore/COW/rollback and mixed PP/TG/stream-gap comparison.

## Acceptance and reporting

- [ ] Record experiments, failures and matched medians in the existing progress ledger.
- [ ] Verify binary/library provenance and health after each exclusive diagnostic lease.
- [ ] Package only verified improvements. Script settings and runtime binaries are separate artifacts.
- [ ] If90/600 gates fail, report the remaining bottleneck and leave the objective incomplete; do not substitute a plan or-bs enablement for measured completion.
