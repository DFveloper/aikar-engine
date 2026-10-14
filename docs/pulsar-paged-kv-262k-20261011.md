# Pulsar Paged KV implementation and measurements

Date: 2026-10-11. Status: experimental implementation; long capacity stress test stopped at user direction after crossing131K. Independent product code review is complete. Production settings are unchanged. AM is excluded by user direction.

## Outcome so far

Existing local/global flags now support real GGML Q4_0 and IQ4_NL packed paged rows on CPU and CUDA. Hadamard rotation is independent for Full and SWA; Pulsar uses Full K512/V64 and SWA K256/V64 when enabled. PKV4 records codec, head geometry and both rotation sizes, with transactional validation before state mutation. The short quality tool clears every local/global rotation and TurboQuant policy for its F16 reference.

Neither codec is approved as a production default. Q4/IQ4 reduce Full KV bytes by 45.45% relative to Q8_KV, but measured attention and single-request decode are slower. The strict long-context top-1 gate also fails on some tested prompts. A larger physical pool is feasible with smaller microbatches, but allocation alone does not establish working 262K serving or acceptable quality.

An8192-page/262144-token pool was allocated with12slot reservations, and one active sequence completed180224actual tokens before the user requested stopping the stress test. This verifies processing beyond the old131072limit. It does not claim all262144tokens were occupied or that a full-pool HTTP workload passed.

## Current code versus vLLM

The source comparison uses vLLM commit `f8471164bf9f5190752fd7b903d415a34fc4ba1c`, principally its block pool and KV cache manager. Later DBO documentation is a separate architectural reference, not the measured baseline.

| Area | This implementation | Relevant vLLM approach | Consequence and next unit |
| --- | --- | --- | --- |
| Page ownership | Separate static Full/SWA physical pools, one combined K/V tensor per layer | Cache groups organized by compatible page geometry/bytes and shared allocation infrastructure | Measure unused reservations before attempting compatible page-byte grouping; do not merge incompatible layouts |
| Prefix reuse | Explicit sequence copy and reference counts, with COW | Hash-addressed prefix caching integrated with block allocation and eviction | Current shared pages do not establish automatic cross-request prefix caching |
| Allocator metadata | Complete candidate state copied during prepare and committed by copy | Block-manager orchestration avoids materializing all layer payloads for routine scheduling | Measure and reduce map copies while preserving failure rollback |
| COW payload copy | `copy_page` synchronizes, allocates a host vector, reads each layer page and uploads it again | Device-side block-copy paths keep same-device payload transfers on the device | Existing `ggml_backend_tensor_copy` and CUDA D2D buffer copy provide a smaller follow-up; measure shared-tail workloads and retain CPU fallback |
| GPU input metadata | Both full logical tables rebuilt and uploaded per batch | Backend block tables and slot mappings are maintained with scheduling | At262K/12slots/block32, two tables total1572864 bytes per upload; delta updates need graph-safe storage ownership |
| Prefill | Scalar online softmax per query, with existing grouped-head KV reuse | Backend-specific query/KV tiling, often FlashAttention/Triton | Current4-bit dequantization reduces bytes but adds instructions; query tiling is the larger PP opportunity |
| Decode splits | Batch-wide rounded maximum context bound and fixed occupancy target | Backend-specific split/work scheduling | Mixed short/long batches can allocate empty work; assess per-sequence useful spans and scratch bytes |
| Device support | V100 SM70, Full headD512 | Modern vLLM backends have differing hardware and geometry requirements | Copy scheduling ideas, not a recent kernel or flag without SM70/D512 validation |
| Overlap | Sampling waits for logits; existing QKV fan-out stream optimizer | DBO uses two microbatches around DP/EP all-to-all | This single V100 has no EP all-to-all; use measured host gaps and independent FFN branches |
| Expert cache | All routed weights resident; REAM cache is calibration-only | Open expert-cache proposal, not an established stable feature | Optional slot residency needs new loader/ownership contracts and measured miss cost |

## Implemented contract

- Matching K/V codec per domain: F16, Q8_KV, Q4_0 or IQ4_NL. Vulkan keeps F16/Q8_KV and rejects4-bit formats before execution.
- CPU reuses existing row traits. CUDA retains the exact Q8 rounding writer and uses existing Q4/IQ4 block quantizers. IQ4 uses its nonlinear codebook.
- Rotation policy changes reject occupied pages; identical policy remains valid. Graph reuse compares basis presence and sizes. Matrix data is published only after successful construction.
- PKV4 has six per-layer fields. PKV2/3 imports are accepted only for matching original unrotated F16/Q8 layouts. Invalid header, geometry, codec, basis, alias and truncated metadata/payload reject without changing live serialized state.
- No persistent F16 KV copy, expert-weight replica, AM fitting, new CLI, production launcher change or commit is included.

## Numerical and lifecycle evidence

CPU paged numerical suite passed138/138. CUDA passed138/138, including unchanged exact Q8 rounding checks and4-bit packed-row checks. Hadamard identity/composition fixtures passed5/5 on CPU and CUDA. After a selective IQ4 lookup adjustment, affected CUDA cases passed25/25; unchanged format paths retain the full-suite evidence.

Actual Pulsar Q4/IQ4 global rotation and Q4 global+local rotation save/load suites passed all supported cases. Device-only state cases are explicitly skipped. A live rotation change rejected byte-identically; rebuilt versus fresh graph logits differed by0.0. Full-state shared-page restore and divergent COW continuation matched exactly. Equivalent PKV2/3 fixtures restored Q8 at block sizes32/64/128 and rejected Q4 or rotated Q8 while retaining occupied state.

The four-slot Q4 and IQ4 global-rotation state suites also passed all supported cases. A fresh independent review found no critical or important product issues. One minor fixture limitation is deferred: identical low/high16-value halves could let a nibble-half-swap mutant pass. Validation covers CPU and SM70 CUDA; Windows, other GPU architectures and hardware-failure upload atomicity are not certified.

The independent F16 reference probe failed on the original remaining local rotation fields and passed with all codec/rotation/native-bit policies cleared. Initial diagnostic fixture/probe mistakes are documented in the execution ledger; failed setup runs do not count as behavioral proof.

## Quality evidence

All comparisons use the same GGUF and token schedule. Long contexts are freed between configurations. Continuation tokens come from the reference model. Chat prompts use the model's stored template. Timing from these comparisons includes CPU KL evaluation and is not a TG benchmark.

| Test | Candidate | Reference | Mean KL | Top-1 agreement | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| Short JSON prompt,128 samples | Q8 | F16 | 0.003000 | 98.44% | Control |
| Short JSON prompt,128 samples | Q4+Had | F16 | 0.003192 | 97.66% | Relative short gate passes on this prompt |
| Short JSON prompt,128 samples | IQ4+Had | F16 | 0.002828 | 99.22% | Relative short gate passes on this prompt |
| 8K,64 continuation samples | Q4+Had | Q8 | 0.001740 | 100% | Sample gate passes; broad quality inconclusive |
| 8K,64 continuation samples | IQ4+Had | Q8 | 0.004865 | 98.44% | Fails99% long gate |
| 32K,64 continuation samples | Q4+Had | Q8 | 0.002297 | 98.44% | Fails99% long gate |
| 32K,64 continuation samples | IQ4+Had | Q8 | 0.003158 | 98.44% | Fails99% long gate |
| Korean exact-code retrieval,125 samples through EOG | Q4+Had | F16 | 0.001401 | 99.20% | Answer47219 retained |
| Korean exact-code retrieval,125 samples through EOG | IQ4+Had | F16 | 0.001096 | 100% | Answer47219 retained |
| 1K final8 prefill logits | Q4+Had | Q8 | 0.055005 | 100% | Fails KL gate; too few samples to certify quality |

The retrieval is one case, and both candidate generation histories were reference-driven. It does not establish a task-accuracy loss below1percentage point. The first retrieval driver continued past EOG; its384-token metrics are superseded by the125-token run. A128-output tail diagnostic produced substantially larger divergences and extra workspace; it is retained in artifacts and is not used to certify a candidate. Long capacity and quality are separate gates.

## 262K allocation

Full Q8_KV costs10560 bytes/token across five layers. Full Q4_0/IQ4_NL costs5760 bytes/token, or1440MiB for262144 tokens. SWA Q8_KV across25layers costs105600 bytes/token. The original12-slot/ubatch1280/131K pool uses2763.75MiB KV. Doubling Full capacity with4-bit rows and unchanged SWA reservations uses2883.75MiB,120MiB more than the original total.

Actual V100 allocation of8192Full blocks x32 with Full4-bit/global Had and SWA Q8:

| Slots | Ubatch | KV MiB | Initial free MiB | Result |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1280 | 1678.49 | 309.5 | Context initialized |
| 4/8/12 | 1280 | varies | unavailable | Compute-buffer OOM |
| 1 | 128 | 1562.47 | 1607.5 | Context initialized |
| 4 | 128 | 1891.18 | 1259.5 | Context initialized |
| 8 | 128 | 2329.46 | 857.5 | Context initialized |
| 12 | 128 | 2767.74 | 407.5 | Context initialized |

These are initial context measurements, not workload peaks. Quantizing SWA too reduced the12-slot/ubatch128 KV allocation to2164.22MiB with1009.5MiB initial free, but this configuration has no quality acceptance. Keep the12-slot target explicit; a one-slot result cannot certify12-slot serving. A fully occupied8192-page pool has no append space beyond262144; reserve generation tokens within the logical budget.

Actual long prefill completed the180224-token checkpoint in1307.14seconds with12slots reserved and one active sequence, Full Q4+Had, SWA Q8, batch/ubatch128 and graphs disabled. Sampled `nvidia-smi` free memory remained310MiB after compute allocation, compared with407.5MiB initial free. The user then instructed stopping once the old131K boundary was crossed. The owned probe was terminated by SIGINT; exit-2 is intentional cancellation, not an OOM. Its in-process minimum-memory counter and final page-count record were not emitted, so exact workload minimum and full262K occupation remain unverified. Artifact: `artifacts/capacity-partial.json` and the unchanged progress log.

The GGUF declares `gemma4.context_length=262144`. Paged mode already enables unified logical context in both the context and server; physical8192-page capacity is shared across slots, not262K per each of12simultaneous requests. The full-capacity experiment retains12slot reservations and fills one active sequence. The existing server also caps each slot by the model's training context.

Experimental argument set, with the patched binary and the same model:

```text
--kv-paged --kv-block-size 32 --kv-blocks 8192 -c 262144 -np 12 -b 128 -ub 128
-ctlk q8_kv -ctlv q8_kv -ctgk q4_0 -ctgv q4_0
--k-cache-hadamard-global --v-cache-hadamard-global
```

Use `iq4_nl` for both global type flags to select IQ4. These arguments expose the implemented path, not a recommended production launcher. The physical-capacity check disables CUDA graphs; graph memory and a server workload peak need their own headroom check. Output tokens must fit inside the shared logical and physical budget.

## Speed evidence

Matched synthetic model timing uses one warmup plus three timed repetitions, identical saved prompt tokens,64generated tokens, ubatch128 and the same graph setting. Independent sequences use separate physical writes rather than shared-prefix aliases. These are API throughput measurements; HTTP/TTFT is measured separately.

| Prompt/active sequences | Codec | PP aggregate tok/s | TG aggregate tok/s |
| --- | --- | ---: | ---: |
| 1K/1 | Q8 | 652.28 | 53.29 |
| 1K/1 | Q4 | 650.47 | 50.23 |
| 1K/1 | Q4+Had | 644.71 | 49.02 |
| 1K/1 | IQ4+Had | 645.16 | 49.93 |
| 1K/4 | Q8 | 798.30 | 174.12 |
| 1K/4 | Q4+Had | 786.91 | 161.66 |
| 1K/12 | Q8 | 950.39 | 339.22 |
| 1K/12 | Q4+Had | 936.89 | 330.45 |
| 8K/1 | Q8 | 596.69 | 53.41 |
| 8K/1 | Q4+Had | 579.35 | 50.95 |

Initial Full attention32K/64-token kernel times were Q8=19.919ms, Q4=22.656ms and IQ4=24.544ms. FixedD512/group4 kernels have high register usage but no local spills; the generic dynamicD fallback has512bytes of stack. An IQ4 register-codebook lookup improved ungrouped SWA decode from roughly372us to286us but regressed grouped SWA prefill from780us to895us. The change is retained only for ungrouped kernels. This is a kernel improvement relative to initial IQ4, not a serving improvement over Q8.

The final selective lookup measurement is289.48us for ungrouped SWA decode and786.99us for grouped64-token prefill. Both affected numerical fixtures and the backend perf command passed; the grouped path retains the original lookup.

HTTP completion measurements use an isolated loopback server,12slots, context32768,4096Fullpages, batch/ubatch128 and CUDA graphs disabled for both configurations. Each request has about931prompt tokens and64generated tokens. Run one warmup and three measured waves at each concurrency. The server reports a small four-token prefix reuse in some requests despite `cache_prompt=false`; both configurations use the same prompts and scheduler. No production service is interrupted.

| Concurrent requests | Q8 wave wall seconds | Q4+Had wave wall seconds | Q4 relative wall improvement |
| ---: | ---: | ---: | ---: |
| 1 | 2.972 | 3.040 | -2.29% |
| 4 | 8.776 | 8.813 | -0.43% |
| 12 | 24.770 | 24.493 | +1.12% |

These small differences do not establish a substantial serving speedup. Request wall time includes prefill and decode. Streaming TTFT was not measured by this non-streaming client, and per-request TG under mixed concurrent work is not the same as pure batch TG. Raw136response timings are in `artifacts/server-bench.json`.

A separate matched streaming run completed136responses in24waves using the same configuration, prompt family,64output tokens and one warmup plus three timed waves. TTFT is elapsed client time to the first nonempty content event, including request queueing, scheduler prefill waits and transport. Final SSE events and64predicted tokens are checked for every response.

| Concurrent requests | Q8 mean TTFT seconds | Q4+Had mean TTFT seconds | Q8 mean wave seconds | Q4+Had mean wave seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.670 | 1.684 | 2.895 | 3.005 |
| 4 | 4.710 | 4.676 | 8.577 | 8.660 |
| 12 | 13.021 | 13.070 | 24.246 | 24.446 |

Mean post-first-content time was Q8/Q4=1.225/1.321seconds at concurrency1,3.621/3.723at4, and9.405/9.494at12. These include interleaved prefill and scheduling under concurrency and are not pure TG rates. No meaningful TTFT or wall-time gain appears. Raw evidence: `artifacts/server-stream-bench.json`, summarized in `server-stream-summary.json`.

A separate successful Q8 server profile completed one warmup and one64-token request. Within two sustained decode bursts,124logit-readback-to-next-kernel intervals measured median0.993ms/mean1.020ms. This includes host work and intervening copies, with profiler overhead. The profile uploads two98304-byte tables each step at context32768/12slots; the262K equivalent is eight times as large. This interval is about5% of a20ms token step and cannot support a claim of major TG gain from host overlap alone. Raw Nsight capture and analysis are retained separately from unprofiled timing.

## Follow-up selection

A CPU-only microbenchmark calls the actual paged `prepare`, `apply` and `table` methods with zero payload layers,12slots,8192 occupied Full pages and32 retained SWA pages per active sequence. It overwrites the last existing token, uses10warmup plus256measured repetitions, and runs alongside the long GPU capacity check. This measures metadata methods, not complete decode latency or GPU upload time. At one active sequence, median candidate copy=211.78us, prepare including its snapshot=164.23us, apply=135.92us, and both table builds=829.85us. At12active sequences, corresponding medians are204.86/177.14/125.10/853.85us. Both tables still total1572864bytes regardless of useful occupancy. Table generation is the largest measured metadata component; a server trace is required before attributing end-to-end latency to it. Raw matrix and source: `artifacts/metadata-bench.jsonl`, `metadata-bench.cpp`.

A separate idle-GPU probe uses the actual `ggml_backend_tensor_set` CUDA buffer path with pageable vectors and two table tensors,10warmup plus256measured repetitions. The1572864-byte transfer pair measured median663.116us/p95672.089us; both restored tables matched exactly. This path synchronizes after each tensor upload. It measures the transfer method, not an overlapped server transfer or an end-to-end speedup. Raw evidence: `artifacts/metadata-transfer.log` and `metadata-transfer.cpp`.

A source-derived mixed-length example illustrates split waste, rather than measuring a speedup: on SM70/80SM, Full D512 with16query heads and12decode tokens uses grouped4-head launch blocks=12, target blocks=320 and27splits. For one131K sequence and eleven1K sequences, the chunk size is4855tokens;38of324split blocks cover useful tokens and286have no useful attention iterations. Partial scratch is10658304bytes for this operation. Empty blocks still load query data, write partial results and participate in combine; they do not perform all missing attention iterations. Per-query split scheduling therefore needs evaluation on mixed lengths, with graph/scratch ownership kept explicit.

Do not enable compression as a speed default based on these results. Prioritize query/KV tiling and removal of repeated table uploads/allocator copies, then measure safe overlap. Optional112-expert residency has a plausible headroom tradeoff on one trace, but all-GPU weights currently have zero misses and dynamic caching introduces transfer cost. The concrete follow-up ownership design and measured cache table are in `docs/pulsar-decode-overlap-expert-cache-20261011.md`.

## Reproducibility

Source base: `8fb7fa3773aed443b8102874e91802595dc4602b`, plus the user's original root patch preserved in the isolated checkout `/home/user/.codex/worktrees/9d53/aikar-engine`. Product-only patch is `/tmp/pulsar-paged-262k/product.patch`. No commit is created. Existing Gemma4 QKV and MoE prune edits remain separate.

Model: `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.5-Pulsar_S-LD-Q4_0_XL.gguf`. GPU: Tesla V100-SXM2-16GB,SM70, usable approximately16145MiB. CPU build: `/tmp/pulsar-paged-262k/build-cpu`. CUDA build: `/tmp/pulsar-am-experiment/build-am`; the directory name is historical and contains this non-AM implementation. CUDA architecture70, Release, Vulkan disabled, FlashAttention build disabled; custom paged attention does not require FlashAttention.

Model SHA256: `f8428df24ae3961c98ac4137c7cb47c1b6326a45690aa3677601fe2d70b1c69c`. Binary, runner and saved-token hashes are in `artifacts/hashes.txt`.

The original `hashes.txt` predates the sparse-logit driver correction. Current binary, driver, product patch and262Ktoken hashes are recorded in `artifacts/final-hashes.txt`; model bytes did not change.

Commands and exit codes are recorded in artifact JSON files. Raw logs, saved token arrays, temporary runner source, resource reports, Nsight captures and cache replay are retained under `/tmp/pulsar-paged-262k/`. The execution ledger is `.superpowers/sdd/2026-10-11-pulsar-paged-kv-262k/progress.md`.

## Execution decisions and deferred review finding

The ledger contains these decisions, in execution order. They are retained here for handoff without requiring access to temporary files.

| Decision | Reason and cost if wrong |
| --- | --- |
| Reuse the managed worktree/historical CUDA build and add a CPU build | Preserve baseline and shorten RED/GREEN; additional disk use |
| Keep patches/logs without commits or recovery cleanup | User prohibits unauthorized commits; recovery depends on retained files |
| Preserve the original F16/Q8 writer and add a4-bit writer | Retain exact Q8 rounding; one extra internal template |
| Record unchanged commit ranges in completion wrapper | Commits prohibited; patch/log provenance replaces commit history |
| Defer initial kernel timing until common codec/rotation build | Comparable binaries; performance defects found later |
| Populate constant rotations once per allocated graph, with shape reuse checks | Avoid repeated upload; wrong allocator assumptions could stale bases, lifecycle probe passed |
| Use whole-graph comparison for composition fixtures | Oracle describes inverse output; fewer intermediate assertions |
| Verify unchanged SWA policy and identical-input operations, not whole-model outputs | Full-layer quantization propagates to later SWA activations; needs explicit policy assertions, probe passed |
| Omit dense oracle only in perf mode and match engine causal bounds | Avoid benchmarking the oracle; separate correctness suite required and passed |
| Publish Hadamard cache entries only after successful construction | Preserve allocation-failure invariants; temporary vector allocation |
| Test IQ4 register lookup and retain only ungrouped path | Reduce repeated nonlinear table loads; grouped path regressed and was reverted |
| Read strided routing views by byte span and gather IDs | Original diagnostic buffer was wrong; capture validation required and passed |
| Initially plan final-batch128logits for exact-capacity replay | Avoid append past pool; output ordering needed short validation |
| Reduce diagnostic tail to8logits after workspace OOM | Fit memory; too few samples for99% certification, later longquality not attempted |
| Stop teacher-forced retrieval at reference EOG | Exclude meaningless post-EOG tokens; supersedes earlier384sample result |
| Size concurrent logical budget for sum of independent sequences | Unified context shares capacity; exclude failed undersized run |
| Exclude failed step from initial partial API profile | Only64decode steps fit; partial API profile cannot establish server behavior |
| Reduce full long/concurrent timing matrix after quality rejection | Failed candidates cannot be adopted; less performance coverage |
| Retain ownership of measurements declined by reviewer | Code review cannot certify pending workload results; evidence remains parent's responsibility |
| Limit certification to CPU/SM70, not Windows/other GPUs/device failure atomicity | Available hardware and malformed-input tests; remaining platforms/failures unverified |
| Preserve and exclude pre-existing user changes from product review | Scope product patch; unrelated baseline defects not reviewed |
| Add matched streaming TTFT separately | Complement nonstream timing; additional diagnostic time |
| Accept user-requested stop after180224actual tokens | User waived remaining stress; near-full failure could remain undiscovered |
| Keep metadata method timings and split estimates separate from serving gain | Different measurement contexts; optimization may target a nondominant cost |

Deferred minor: packed representable-value fixture repeats the same16values in both nibble halves. A nibble-half-swap mutant could escape that specific fixture; diversify halves in a follow-up. Existing dense attention, writer and lifecycle suites remain unchanged in their acceptance limits. No critical or important product review findings remain.
