# Paged production performance implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox syntax for tracking. Do not commit, push, submit a PR, or write external comments.

**Goal:** Reach at least 600 aggregate decode tok/s with a target of 1200, and at least 90 single-request decode tok/s with a target of 100, while preserving large-prefill gains, model quality, KV capacity, and prior performance.

**Architecture:** Recover the historical decode reference and profile single-request and batched decode before tuning mixed prefill. Optimize the measured decode bottleneck in attention, matrix-vector/matrix-matrix operations, or host/sampling work, then tune small mixed batches and scheduling. Apply metadata caching and query tiling only when profiles show a material bottleneck; keep reference paths available for comparison and rollback.

**Tech Stack:** C++17, CUDA sm_70/sm_61, ggml, existing llama-server scheduler, existing test-backend-ops and Python validation harness.

**Spec:** `docs/paged-prefill-optimization-20261004.md`, the production evidence below, and the user's request to explain the production slowdown and prepare an implementation plan. This document authorizes planning only; no runtime changes were made while preparing it.

**Updated performance requirement:** The user requires aggregate decoding at least 600 tok/s, targeting 1200 tok/s, and single-request decoding at least 90 tok/s, targeting 100 tok/s. Keeping the currently regressed approximately 40 tok/s decode rate is not an acceptable completion criterion.

## Global constraints

- Keep Q8_KV, physical pool 8192 blocks x 16 tokens, logical context 262144, 12 slots, GPU embedding/output, and CPU multimodal projector.
- Preserve FP32 online softmax and output accumulation, exact packed Q8_KV writes, causal/SWA bounds, valid masks, and sequence mappings.
- Keep reasoning budget and sampling settings unchanged in production. Measure thinking and visible content separately when the API exposes them.
- Do not replace the historical faster reference with the regressed deployed runtime when deciding whether a candidate passes. Keep both references and report any unavailable historical configuration explicitly.
- Do not enable CUDA graphs or allocate a persistent F16 KV copy with the current approximately 98 MiB free GPU memory.
- Reuse existing tests; do not add files under `tests/`.
- Production source is `/tmp/aikar-paged-kv`; the durable full snapshot is `build-paged-fused/sources.tar.gz`. The main workspace does not contain all paged integration changes. Do not build a candidate by assuming the root workspace is the complete production tree.
- Keep the existing runtime and start script rollback copy. Any runtime publication outside writable roots requires the applicable filesystem approval.
- Use ASCII in code and comments. Keep changes independently reviewable. A new tiled kernel is a larger design change and needs design review before implementation.

## Review focus

- Mixed sequence lengths and noncontiguous queries: grouping and tiling must not cross sequence, position, or KV-head boundaries.
- Partial pages, SWA expiry, and sparse masks: invalid positions must not enter softmax.
- Copy-on-write, restore, removal, and reused graph inputs: metadata caching must never retain an obsolete physical page or validity mask.
- Long prefills arriving during generation: improve PP without starving streams or pending slots; image processing has separate atomic constraints.
- Nearly full GPU memory and backend differences: candidates must run on V100 and retain a correct P620 fallback without increasing the physical KV budget.

## Production evidence and diagnosis

Evidence: current start script, production process/library mappings, `/tmp/paged-production-current.log`, the production source, and the matched reports in `build-paged-fused/validation/`.

| Measurement | Earlier benchmark | Observed production chat |
| --- | --- | --- |
| Endpoint | Bare `/completion` | Chat template, reasoning and production sampling |
| Load | 16 simultaneous requests, 12 slots | Two overlapping requests in the captured interval |
| Long output | 32 tokens per request | 1558 and 2044 generated tokens |
| Reported PP | 1234.41 prompt tokens / total workload wall second | Request timing: 729.58 and 462.59 prompt tokens/s |
| Reported TG | Short workload: 201.08 generated tokens / workload wall second | Request timing: 32.31 and 41.20 generated tokens/s |
| Mixed text prefill budget | 64 tokens when a request is generating | 64 tokens, confirmed by successive prompt progress increments |
| Grouped attention default | Enabled when operator query count is at least 128 | Usually disabled for 64 prefill tokens plus a few decode tokens |

The benchmark's 1234.41 tok/s is an aggregate wall-throughput metric, not per-request generation speed. Its long workload contains only 512 generated tokens in total; production's two requests contain 3602 generated tokens. A prefill speedup cannot remove those approximately 49 seconds of generation per request.

Production task 9 initially reports recent TG at 8.31 tok/s while task 10 is prefilling. After that prefill ends, its recent TG returns to about 40 tok/s. Its final average is 32.31 tok/s because time spent sharing the GPU with prefill remains in the denominator. Task 10 prefills 6781 tokens in 14.659 seconds and then generates 2044 tokens in 49.586 seconds. These are observed service times, not evidence that the kernel alone explains all latency.

The scheduler appends decode tokens first, then up to 64 text prefill tokens to the same batch while any request is generating. All work in that batch completes before subsequent stream progress. With at most 12 active streams, this mixed batch typically has no more than 76 queries, below the grouped path's 128-query threshold. Initial prefill with no generating slots can use the 1280-token limit, so some production PP does benefit. A prior production smoke logged 1342.50 PP tok/s, demonstrating that deployment does not universally lose the large-prefill path.

The grouped kernel shares K/V across query heads belonging to one token. It still scans cache positions with per-position reduction and online softmax, and does not reuse K/V across adjacent query tokens. Its split heuristic currently uses `(nh*nt + 3)/4`, but a grouped launch uses `(nh*nt/head_group + 3)/4`. This can leave fewer blocks than the heuristic intended. Performance consequences depend on shape and register pressure and require measurement.

Each `llm_graph_input_attn_kv_paged::set_input` reconstructs and uploads both entire page tables. Current dimensions imply 1.5 MiB per domain, 3 MiB per step, plus allocation and map traversal. That is a host/transfer optimization candidate, not a measured primary bottleneck. Log messages saying `graphs reused` refer to ggml compute graph reuse; CUDA graph replay remains disabled.

The old user-reported 100 tok/s single-request result has not been reproduced with a matched configuration. Neither 1234 PP tok/s nor 201 aggregate TG tok/s establishes recovery of 100 single-request TG tok/s. Reasoning budget 8192 can also delay visible content, but the captured timing does not establish how many generated tokens were hidden reasoning.

## Acceptance criteria

- Run each matched configuration three times with fixed prompts, token counts, seeds, endpoints, cache state, and reasoning settings; report medians and individual runs.
- Mandatory decode floor: single-request median at least 90 tok/s, targeting 100 tok/s; 12-active-request aggregate median at least 600 tok/s, targeting 1200 tok/s. Apply the decode matrix at approximately 50, 4096, and 8192 prompt tokens with 512 generated tokens per request. Report each context separately; do not hide a failing context in an overall average.
- Measure aggregate decode as total generated tokens across active requests divided by one common elapsed interval in which all 12 are generating. Retain endpoint completion throughput including PP/queue time as a separate metric. SSE event counts are not token counts. Check 1/2/4/8/12 concurrency for scaling and fairness.
- No more than 5% median regression in PP, single-request TG, aggregate TG, or mixed completion throughput against the corresponding matched pre-optimization reference. Absolute decode floors and relative regression gates both apply; a 100 tok/s matched reference makes the single-request gate at least 95 tok/s. Keep the current deployed baseline as an additional comparison, not the sole acceptance reference.
- Missing historical evidence or unmet absolute floors mean the performance objective remains incomplete. A local kernel gain or preserving approximately 40 tok/s cannot satisfy this requirement.
- Reproduce at least 1200 aggregate long-prompt wall tok/s on the earlier exact workload as a target, with no more than 5% regression against the same-session deployed baseline.
- Target at least 20% less mixed-workload prompt service time in addition to the mandatory decode and regression gates.
- For text mixed traffic, target p95 SSE output gap at most 250 ms and no more than 5% worse than the same-session baseline. Record maximum gaps as well. SSE events can bundle tokens; do not label event gaps as exact per-token latency.
- Keep GPU peak within the deployed baseline's measured envelope. Do not make room by reducing context, physical pages, reasoning quality, or output length.
- Retain all existing paged correctness cases, exact write bytes, state-restore checks, and multimodal behavior. Keep greedy output comparison alongside numerical operator comparisons.
- If a target fails, publish the failure and attribution; do not substitute operator speed for endpoint speed or aggregate speed for single-request speed.

### Task 1: Establish the production-shaped baseline and cost breakdown

**Files:** Extend the existing `build-paged-opt/validation/bench.py` validation harness or its copied counterpart in the candidate package. Read `tools/server/server-context.cpp`, `src/llama-graph.cpp`, and `ggml/src/ggml-cuda/pagedattn.cu` from the complete production source tree. Store reports under the candidate package's `validation/` directory.

**Interfaces:** Input is the deployed binary plus an explicit argument/environment snapshot. Output is JSON with workload kind, prompt/generated counts, wall time, request PP/TG, TTFT, first visible content, SSE gaps, memory peak, and raw timing evidence. Use null for unavailable reasoning counters; do not infer them from content.

- [ ] Preserve and rerun the earlier raw completion workload: 16 requests, 12 slots, approximately 7930 prompt tokens and 32 generated tokens each; short workload generates 128 each.
- [ ] Recover the historical approximately 100 tok/s binary, model, arguments, environment, KV configuration, context length, and reasoning/sampling settings from existing artifacts. Reproduce it where available, and benchmark historical, current paged, and candidate runtimes on matched inputs. Treat memory-incompatible or unavailable configurations as documented comparison limitations, not permission to reset the acceptance baseline to 40 tok/s.
- [ ] Add the mandatory 50/4096/8192-prompt-token decode matrix with 512 output tokens at concurrency 1/2/4/8/12. Record a common steady decode interval, exact generated counts, per-request rates, and endpoint wall rates. Verify the 90/600 minimum gates independently for each context.
- [ ] Add chat-template workloads: one short request generating 512 tokens; two approximately 7k-token requests generating 2048 each; one long prefill arriving after a stream has begun; and 12-stream mixed traffic. Pin reasoning disabled for the isolated decode comparison, then repeat with production reasoning 8192. Keep image and warm-cache follow-up checks separate.
- [ ] Record SSE timestamps and compute p50/p95/max output gaps per request. Verify report counts against the server's evaluated/generated counters and reject truncated runs.
- [ ] Capture a short CUDA/host timeline on an isolated candidate run: attention, projection/MLP, sampling, metadata construction/upload, and server scheduling. Read actual operator query counts and context spans from trace or bounded diagnostic logging.
- [ ] Compare deployment against `GGML_CUDA_PAGED_FUSED=0` under identical settings. Separate pure PP, pure decode, and mixed traffic in the report.
- [ ] Run three repetitions. Schedule model-bearing V100 benchmark runs exclusively; current free VRAM does not permit a second full model beside production. Do not stop production during planning.

### Task 2: Recover single-request and batched decode, then tune mixed attention

**Files:** Modify `ggml/src/ggml-cuda/pagedattn.cu` and the existing paged cases in `tests/test-backend-ops.cpp` in the complete production source tree. Inspect `ggml/src/ggml-cuda/mmvq.cu`, `ggml/src/ggml-cuda/mmq.cu`, `src/llama-context.cpp`, and existing common sampling code; change these only when the decode profile identifies a concrete bottleneck.

**Interfaces:** Keep `ggml_cuda_op_paged_attn` and packed cache format unchanged. In the existing dispatch, choose `head_group` before calculating the actual launch block count and `nsplits`. Preserve the environment disable/force comparison path.

- [ ] Build a decode-step cost budget from Task 1: single-request 90/100 tok/s corresponds to at most 11.11/10.00 ms per generated token; 12-active-request aggregate 600/1200 tok/s corresponds to at most 20/10 ms per step producing 12 tokens. Account for attention, model weight operations, logits/sampling, transfers, and host scheduling; attention-only gains do not establish these endpoint budgets.
- [ ] Compare ordinary/reference attention and paged attention at identical model, context, and output placement where supported. Identify the measured gap before choosing changes. Keep correctness comparisons against clean paged replay because ordinary and paged numerical outputs already differ in existing fixtures.
- [ ] Tune decode nt 1/2/4/8/12 independently from mixed-prefill dispatch. If projection/MLP dominates, evaluate existing MMVQ/MMQ dispatch and Volta configurations; if host/logits/sampling dominates, remove only demonstrated redundant work while preserving sampling and reasoning semantics. Require a bounded follow-up design with exact affected functions before any change outside paged attention.

- [ ] Extend existing operator correctness/perf cases with nt 1, 2, 12, 64, 65, 76, 128, 256, and 1280; D256/D512; Q8_KV/F16; start positions 0, 4096, and 8192; unequal sequences; causal/SWA; and partial groups. Reuse existing fixtures rather than new test files.
- [ ] Measure grouping 1/2 for D256 and 1/4 for D512 with split grid based on the selected group. Compare total attention plus combine time, registers/spills, temporary memory, and occupancy; do not minimize attention time while ignoring combine overhead.
- [ ] Calculate `nblocks` from the chosen launch grouping before the existing SM-based split heuristic. Retain the 256-token minimum partition span initially; test alternative split targets only after this isolated change is measured.
- [ ] Replace the blanket small-batch cutoff only for shapes where repeated measurements show a gain. Use query count, dimension, GQA ratio, context span, dtype, and device capability as needed; retain existing decode/fallback paths for losing shapes.
- [ ] Build with `cmake --build build-paged-cuda -j36 --target llama-server test-backend-ops test-arg-parser` from the complete source tree.
- [ ] Run `build-paged-cuda/bin/test-backend-ops -b CUDA0 -o PAGED_ATTN,PAGED_ATTN_WRITE` with grouping forced and disabled, then `build-paged-cuda/bin/test-arg-parser`. Repeat paged correctness on P620 with its CUDA device mapping. Require no packed-byte mismatches and existing numerical tolerances.
- [ ] Rerun Task 1 mixed and pure-decode workloads. Accept small-batch dispatch only if endpoint PP improves and TG/gap/memory criteria hold. If mandatory 90/600 decode floors remain unmet, continue measured decode work; do not declare the optimization complete after a mixed-PP improvement.

### Task 3: Remove metadata overhead if the profile justifies it

**Files:** `src/llama-kv-cache-paged.h`, `src/llama-kv-cache-paged.cpp`, `src/llama-graph.cpp`; reuse the state-restore validation referenced by `docs/paged-kv-validation.md`.

**Interfaces:** Change `table(bool swa) const` to return a cached `const std::vector<int32_t> &`. Add `uint64_t table_seq_revision(bool swa, llama_seq_id seq) const`. Each graph input tracks the uploaded revisions for both domains and every sequence; first use uploads the full table.

- [ ] Confirm the measured combined metadata CPU/transfer cost is material before implementing this task. Preserve the report if the task is skipped.
- [ ] Add regression assertions to existing validation for page-mask changes within a page, new allocation, SWA expiry, removal, clear, copy-on-write, sequence copy/keep, failed allocation, restore, and graph rebuild/reuse. Compare cached tables against a freshly reconstructed reference and restored logits against a clean paged run.
- [ ] Maintain persistent CPU table storage with per-sequence invalidation at every committed state mutation. Failed prepare operations must not publish candidate mappings. Keep table layout and physical IDs unchanged.
- [ ] Upload changed sequence rows using existing `ggml_backend_tensor_set` offsets; fully upload new graph inputs. Bind the cached table with `const auto &` in `set_input` to avoid copying it. Keep revisions local to each graph input so one graph's update cannot hide changes from another graph. Continue refreshing query and write-slot inputs every batch.
- [ ] Re-run state validation and Task 1 decode/mixed traces. Accept only if measured host/transfer time and endpoint latency improve without new stale mappings or GPU-memory growth.

### Task 4: Increase mixed prefill work under an output-latency budget

**Files:** `tools/server/server-context.cpp`; reuse the validation harness from Task 1. Preserve the existing `--prefill-chunk-size` interface initially.

**Interfaces:** The existing scheduler continues to prioritize decode and rotate `prefill_cursor`. The configured prefill chunk becomes the upper bound for adaptive text work in mixed batches; no-generating-slot prefill continues to use the normal batch limit. No new public option is required for the first experiment.

- [ ] Benchmark fixed chunk caps 64, 128, and 256 after Task 2, using delayed long-prompt arrival and 1/2/12 active generators. Keep the largest cap satisfying all latency gates; do not set chunk 1280 for mixed traffic by default.
- [ ] If a fixed cap fails across context lengths, implement a local adaptive budget using observed mixed decode-call wall time: start at 64, decrease on calls exceeding 150 ms, and increase by at most one power-of-two step after eight consecutive calls below 100 ms. Clamp to min(16, configured cap)..configured cap and available batch space; reset to 64 or the smaller cap when transitioning into mixed traffic. Preserve the existing disabled behavior for chunk size zero. This is a measured starting policy, not a promised latency bound.
- [ ] Verify batch time measurement includes actual completion, not only asynchronous enqueue time. Preserve round-robin pending-slot progress and reset control state after model/runtime reset.
- [ ] Validate long-context transitions, cancellation, slot reuse, multiple pending prompts, and image requests. Keep atomic image processing outside the text controller and report its gaps separately.
- [ ] Reject the scheduler candidate if p95 output gaps regress beyond the gate, PP does not improve, or any slot starves. Keep chunk 64 rollback available.

### Task 5: Tile adjacent prefill queries only if attention remains dominant

**Files:** `ggml/src/ggml-cuda/pagedattn.cu`, existing paged operator cases in `tests/test-backend-ops.cpp`.

**Interfaces:** Keep current operator inputs and output layout. Add a private templated tiled prefill kernel inside the same CUDA file; use the existing grouped/warp implementation for unsupported or irregular inputs.

- [ ] Use the post-Task-2 profile to decide whether repeated KV loads and per-position softmax dominate. Review the query-tile design before writing this larger kernel.
- [ ] Prototype tiles of 2 and 4 adjacent queries from the same sequence with contiguous positions; load/dequantize each K/V tile once and retain independent FP32 softmax/accumulators per query/head. Use bounded per-block scratch and no persistent expanded KV allocation.
- [ ] Guard each query's causal and SWA range independently. Route arbitrary ordering, cross-sequence tiles, unsupported dimensions, and incomplete tiles through the existing path until explicitly supported and tested.
- [ ] Compare Q8_KV/F16 results on sparse masks, page boundaries, uneven positions, and full/SWA attention. Measure spills and total attention time at nt 64/128/256/1280 on V100; verify fallback correctness on P620.
- [ ] Accept only with endpoint gains under Task 1 and all resource/latency gates. Do not repeat the rejected direct fused write/read experiment, which requantized new K/V per query and was slower.

### Task 6: Validate and package the measured winner

**Files:** Candidate `bin/`, `sources.tar.gz`, `optimization.patch`, and `validation/` artifacts; update `docs/paged-prefill-optimization-20261004.md` with final evidence. A later authorized deployment may update `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/start.sh`.

**Interfaces:** Package the complete tested source, binary, and matching libraries. Preserve the current deployed package as rollback and publish the exact environment/configuration with each report.

- [ ] Run the full paged operator matrix, exact write checks, argument parser, state-restore/COW validation, greedy output comparisons, and existing multimodal smoke check.
- [ ] Run the three-repeat baseline/candidate matrix and report aggregate PP, request PP, aggregate TG, single-request TG, mixed stream gaps, visible TTFT, and memory separately.
- [ ] Require the 90 single-request and 600 aggregate decode floors plus the historical-reference regression gates before publishing the candidate as meeting the performance objective. Report progress toward 100/1200 separately; do not describe either target as achieved without measured evidence.
- [ ] Attribute improvements by ablation: kernel only, metadata only if implemented, and scheduler plus kernel. Record skipped candidates and failed acceptance gates.
- [ ] Package only passing changes. Verify library resolution, snapshot rebuild inputs, and rollback commands before any deployment.
- [ ] After an authorized deployment, verify process/library mappings, health, slot count, representative mixed chat behavior, and memory. Do not claim 100 single-request tok/s without reproducing it.

## Execution order and stopping rules

Tasks 1 and 2 are the first implementation scope, with decode recovery taking priority over further PP gains. Task 3 is conditional on a host/transfer bottleneck; Task 4 depends on the new mixed-kernel measurements; Task 5 is a separate reviewed escalation if attention remains dominant. Task 6 closes only when the mandatory 90/600 decode floors and historical regression gates pass. If those goals cannot be reached within the measured hardware/resource limits, report the remaining bottleneck and keep the objective incomplete rather than weakening its criteria. No source, start script, or runtime changes were made in this planning turn.

### Task 2 bounded follow-up: Volta Q4 MoE at 9-12 queries

The deployed profile reports Q4 MMQ at about 13 ms per short 12-token step and 10.5 ms per long step. Original source selects MMQ once MoE output tokens exceed eight. Reuse the existing warp-per-token `mul_mat_vec_q_moe` kernel, which already supports runtime token count and independent expert IDs, for Q4_0 only on sm_70 at 9-12 tokens. Keep dense MMVQ and all other type/device limits unchanged. Update the MoE dispatch, synchronization predicate, per-device launch bound, and ID-path assertions consistently. Add existing-fixture cases at the 8/9/12/13 boundaries, both plain and gate fusion, with actual model expert shapes. Compare old MMQ and candidate using the same test binary and frozen libraries before endpoint adoption. Reject if numerical tolerances, kernel time, total endpoint throughput, or GPU envelope regress. This is reuse of an existing kernel, with no cache-format or sampling change.

### Proposed follow-up requiring review: parallel CPU sampling

Evidence: diagnostics show about0.85 ms CPU sampling per output token at twelve active requests, approximately10 ms serialized per decode step. The existing backend sampler rejects reasoning budgets and grammar, so enabling it cannot preserve the required production behavior.

Bounded design: add a common sampler entry point accepting immutable full-vocabulary logits. Extract and reuse the existing reasoning-budget, sampler-chain, grammar-first and rejection/resampling logic; preserve the current context-based and backend-sampler paths. Grammar rejection resets candidates from the same immutable logits rather than calling context getters from workers.

For regular generating slots with CPU sampling and no speculative drafts, prepare logits pointers on the server thread after synchronizing the context. Dispatch independent per-slot samplers through four workers using the existing cpp-httplib ThreadPool class, wait for all work, and consume tokens/accept/stream/release on the server thread in the original slot order. Single-output calls, first prompt output, backend sampling and speculation keep their existing path. Handle worker exceptions per slot, wait for every job before any context/slot mutation, and shut down the pool before releasing model/samplers. No model context API or CUDA operation runs in workers; no sampling policy or GPU allocation changes.

Affected files: common/sampling.h and .cpp, tools/server/server-context.cpp and CMakeLists.txt, and the existing tests/test-state-restore-fragmented.cpp sampling comparison fixture. Estimate100-150 changed lines. Validation compares cloned legacy/raw-logits samplers on the same logits with greedy and stochastic seeds, penalties, grammar rejection and reasoning limits; server runs compare fixed-seed output and candidate probabilities, followed by1/2/4/8/12 scaling and mixed/cancellation cases. Keep the old path available for ablation and reject unless endpoint gates improve without semantic or memory regression.

This introduces server sampling concurrency and needs design confirmation under AGENTS.md before implementation. Attention/combine validation can continue independently.

### Proposed attention follow-up requiring review: two KV positions per warp

Evidence: forced grouped long D512 twelve-query operator takes1247.71us, compared with2224.64us ungrouped. Packed Q8 reads take1282.21us and single long357.49us versus319.51us, so reject packed reads. Current best exploratory endpoint is92.56 single short,79.77 single long,348.14 aggregate short and274.76 aggregate long; mandatory floors remain unmet.

Bounded design: specialize the existing paged_decode template for sixteen lanes per query reduction, with each warp processing two interleaved KV positions for one query/head. Preserve FP32 query, dot products, online softmax and outputs; do not quantize queries or alter Q8 writes. Each subgroup handles its own positions, valid-page mask and softmax. At the end of its partition, merge the two subgroup states with a stable maximum and weighted sum using warp shuffles, then let only the first subgroup write the partial output. Keep the existing partition combine and input interfaces. Read V after the dot product for this specialization to avoid keeping K/V arrays live alongside larger query/output arrays. Use subgroup-specific shuffle masks while positions diverge and a full-warp merge after reconvergence.

Dispatch the experiment only for specialized D256/D512 decode (at most12 queries) with a nontrivial KV span, with the current paths retained as the reference. Derive split/occupancy changes from matched operator results; no unconditional adoption. The expected change is60-100 lines in ggml/src/ggml-cuda/pagedattn.cu, with no new file or subsystem. Validate the existing86-case attention/write matrix in default and forced modes on V100, retain P620 fallback, then fully offloaded state replay and fixed-output endpoint comparisons. Reject any accuracy, memory or endpoint regression, and keep90/600 and historic regression gates unchanged. This is a new CUDA execution pattern and requires AGENTS.md design confirmation before coding.

### Conditional bounded probe: reuse Volta sparse MMA for paged D512 decode

The selected 16-lane change improves long-single decode by approximately1.6%, while twelve-query grouped D512 remains about1.25ms per operator. Probe the existing flash_attn_ext_f16_process_tile sparse Q8 loader instead of introducing a new matrix math implementation. This is a separate reviewed execution pattern; do not implement before user design approval.

Limit the first probe to sm70, Q8_KV, D512, causal decode with at most12 queries, GQA divisible by8, and a long KV span. Each CUDA block owns one query and one group of8 heads from the same KV head, so mixed sequence IDs and positions cannot cross a group. Build transient per-query sparse indices on the GPU from existing page IDs, valid masks and query bounds. Encode the physical page/head-gap stride into the row index; use-1 for holes, excluded causal/SWA positions and tile padding. Reuse the existing sparse Q8 dequantization and Volta MMA math. Allow a null sparse additive mask only by treating valid indices as zero bias and invalid indices as negative infinity. Existing non-null-mask behavior stays the same.

Keep FP32 softmax metadata and the existing stable split-combine policy. The reused MMA path uses FP16 matrix operands/accumulation behavior inherited from ordinary attention; it is not bitwise identical to the scalar FP32 path. Numerical gates must therefore compare same-input outputs and same-KV logits using existing tolerances, and reject any material quality regression rather than relax tolerances. Test empty splits, sparse holes, partial pages, window edges, mixed query positions and replay. Keep all unsupported cases on the current kernels, including P620 and F16.

No persistent KV copy or CUDA graph capture. Bound transient indices and partial outputs by the actual query count/span, check allocation size before dispatch, and measure the production memory peak. Expected change is approximately120-180 lines in pagedattn.cu plus the existing sparse mask loader. Start opt-in; compare operator and full-server medians against the current frozen runtime, and retain only a measured win without more than5% regression. This probe does not establish the90/600 floors; final acceptance remains unchanged.

### Approved bounded follow-up: fused Q4_0 weight tiles on Volta

User approved this design on2026-10-05. Extend the existing mul_mat_f shared-memory loader to dequantize Q4_0 into FP16 pairs while retaining Volta MMA FP32 accumulators and expert-ID mapping. Restrict to sm70,9-12columns,K divisible64,rows divisible32 and supported contiguous weights/output. Pad to16columns with explicit input/output bounds, no full/permanent FP16 weight copy. Start opt-in GGML_CUDA_Q4_MMA; unsupported shapes/types keep existing dispatch. Verify existing numerical tolerances at8/9/12/13 including broadcast/down expert inputs, same-input logits, memory, then matched server medians. Reject any material numerical or runtime regression.
