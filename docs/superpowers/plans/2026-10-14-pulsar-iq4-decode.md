# Pulsar IQ4_NL Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. The user authorized implementation. Continue inline execution; no commits or production deployment are authorized.

**Goal:** Reach >= 100 tok/s for one active request with 12 configured slots, and match or exceed comparable F16 paged decode.

**Architecture:** Optimize the existing mixed Q8-local/IQ4-global path using measured dispatch, quantization and memory-access changes. Preserve packed formats and allocator semantics. Escalate to graph/host changes only when measurements justify them.

**Tech Stack:** C++/CUDA, GGML, verified SM70/V100, existing backend/state tests, Nsight, HTTP streaming measurements.

**Spec:** `../specs/2026-10-14-pulsar-iq4-radix-design.md`

**Status:** Candidate implemented and tested. Five measured repetitions give 96.76 tok/s at 1K and 92.74 tok/s at 8K with 12 configured slots and the production 8000-page pool. The >=100 tok/s goal remains open. See `../../pulsar-resident-prefix-validation-20261014.md`; unchecked adoption gates are not implied complete.

**Measured priority:** Task 2 first: Full IQ4 attention costs 2.27-2.72 ms/step and launches only 20/33 blocks across 80 SMs. Task 4 follows if reads still dominate; Task 5 targets the remaining copy/launch/host cost. Task 3 is deferred: its entire current writer cost is only about 0.22 ms/step. Run Task 6 before deployment.

## Global Constraints

- Preserve local Q8_KV/global IQ4_NL, 12-slot capacity, model/template and production sampling in primary candidate measurements.
- Primary goal is one active request, not aggregate throughput.
- No new quantization formats, no weight offload and no implicit slot reduction.
- No commits, pushes, PR descriptions or production changes without separate applicable authorization.
- Reuse existing tests; no new file under tests without maintainer approval. Repository documents use ASCII.
- Read the spec and relevant source before implementation; review material architecture changes with the contributor.

## Review Focus

1. Empty or masked splits must not inject NaN into softmax (Task 2).
2. IQ4 ties and distinct nibble halves must survive writer/read optimization (Tasks 3 and 4).
3. Reusing persistent tables must not race an earlier CUDA graph or stream (Task 5).
4. Long-context F16 OOM must not turn into an unfair offloaded baseline (Tasks 1 and 6).
5. A kernel win must not conceal TTFT, mixed-batch or model-quality regressions (Task 6).

## Task 1: Reproduce deployment and attribute token time

**Files:** Inspect launcher, `scripts/bench-lumen-kv.sh`, `scripts/server-bench.py`, `tests/test-backend-ops.cpp`; extend existing benchmark scripts only after reviewing their argument handling. Store measurement artifacts under an ignored build validation directory.

**Interfaces:** Existing CLI role flags `-ctlk/-ctlv/-ctgk/-ctgv`, `--kv-paged`, HTTP `/completion` streaming and response timings. Produce a manifest plus raw per-run JSONL and kernel/host attribution, not a new serving API.

- [ ] Verify `nvidia-smi`, live server argv, GPU clocks/memory, `git rev-parse HEAD`, executable/model/template hashes and build CUDA options. Establish that compiled executable corresponds to the revision being measured.
- [ ] Add named benchmark profiles for current IQ4, Q8-local/F16-global, all-F16 paged and all-Q8 paged. Existing bench-lumen-kv defaults target an older model and do not supply this paged matrix; do not run it unchanged.
- [ ] Validate measurement client on a short request: count predicted tokens, handle reasoning/content SSE chunks, require a final event and distinguish first-token time from total request time. Fix output length at 256 tokens using supported controls; flag early termination instead of treating it as a comparable sample.
- [ ] Run the spec's 1K/8K one-active-request matrix: two warmups, five unprofiled repeats, alternating candidates. If F16 cannot fit, reduce the pool equally for all controls and label the result; run the production IQ4 budget separately.
- [ ] Profile separate representative runs with Nsight Systems; use Nsight Compute only for supported counters and selected writer/read/combine kernels. Record launch counts, instructions, memory traffic, occupancy, registers, spills and exposed host gaps.
- [x] Produce a table of milliseconds/token summing writer, attention/combine, other model work and exposed host work. Apply the 10 ms/token Amdahl bound. Rank Tasks 2-5 by measured potential; defer any change whose whole-server effect cannot exceed noise.

**Pass:** Reproducible baseline, actual F16 value or explicit unavailable result, and attribution with no unexplained double counting. Stop performance execution if the GPU is unavailable; do not infer measurements from source.

**Analysis already available:** Production 1K/8K medians 76.432/78.732 tok/s; matched 512-page IQ4 76.659/79.017; global-F16 85.848/85.165. All-F16 at 12 slots/ubatch 1280 failed allocation. Reduced-pool IQ4 profiling succeeded after production-pool profiling OOM. Focused eight CUDA correctness cases and 16 perf cases executed; CPU skipped. Extend this evidence to the full matrix above rather than treating it as adoption validation.

## Task 2: Tune IQ4 split and grouping dispatch

**Files:** Modify `ggml/src/ggml-cuda/pagedattn.cu`; extend `tests/test-backend-ops.cpp` existing paged cases.

**Interfaces:** Preserve `ggml_cuda_op_paged_attn` and existing op parameters. Internal dispatch consumes cache type, device capability, dimension, nt and span; produces head_group/nsplits without public API changes.

- [ ] Add cases for IQ4 D512, nt 1/4/12/32, spans around 32/64/128/256 and 4095/4096/4097. Include masked/empty split, fragmented pages, SWA and noncausal controls. Assert finite output and existing dense-oracle tolerance.
- [ ] Run `build-pulsar-paged-decode-20261014/bin/test-backend-ops -b CUDA0 -o 'PAGED_ATTN.*'`; record initial executed cases and results. A performance defect need not fail numerical assertions; record its slow baseline rather than inventing a failing correctness test.
- [ ] Evaluate split spans 32/64/128/256 on IQ4 SM70 for causal nt <= 32 with D512, retaining the existing target-block cap. Compare group1/group4; implement and test a distinct D512/group2 specialization only if the first comparison justifies it. Include partial-output allocation and combine duration. Keep non-SM70 behavior unchanged unless independently measured.
- [ ] Select the smallest dispatch change winning the end-to-end matrix. Verify batch-wide span remains correct for every sequence; per-sequence work reduction is a later conditional change if empty-CTA waste is material.
- [ ] Run backend correctness and perf modes (`perf -b CUDA0 -o 'PAGED_ATTN.*'`), then repeat affected HTTP points unprofiled. Reject wins that trade single-request speed for > 5% mixed-workload latency regression.

## Task 3: Reduce IQ4 writer overhead if material

**Files:** Modify `ggml/src/ggml-cuda/pagedattn.cu`; inspect `cpy-utils.cuh`; extend existing packed-write cases in `tests/test-backend-ops.cpp`.

**Interfaces:** Preserve `paged_write_q4` arguments and IQ4_NL packed row layout. Keep the general copy/set-rows quantizer unchanged; any cooperative variant is local to paged attention.

- [ ] Extend existing IQ4 writer fixtures with zeros, signed max ties, codebook boundaries, disparate upper/lower nibbles and masked aliases. Assert valid decoded values and no writes outside destination masks.
- [x] Recheck the measured ~0.22 ms/step writer bound after dispatch changes. Skip implementation unless it becomes a material contributor in a required workload; do not prioritize it ahead of attention or claim it supplies the ~3 ms needed.
- [ ] Prototype cooperative loading/quantization of 32-value blocks while preserving signed max tie selection, nonlinear nearest-code choice, nibble order and weighted scale fit. Compare deterministic bytes when operation order is unchanged; otherwise use decoded error and end-to-end logits with original tolerances.
- [ ] Compare old/new writer for D512/hk2 single-token and multi-token inputs; run CPU-oracle/CUDA paged tests, then short quality and whole-server tests. Remove the candidate if whole-server speed or quality gates fail.

## Task 4: Reduce IQ4 read instructions if material

**Files:** Modify `paged_read` and `paged_decode` in `ggml/src/ggml-cuda/pagedattn.cu`; extend `tests/test-backend-ops.cpp` only for uncovered layout cases.

**Interfaces:** Preserve packed bytes and float online-softmax accumulators. Existing warp-codebook policy remains selectable by template specialization.

- [ ] Record current grouped and ungrouped lookup instructions, load transactions and register use. The historical ungrouped improvement did not generalize to grouped prefill.
- [ ] Evaluate paired nibble/scale extraction and reuse across head groups without allocating an unpacked full cache. Benchmark LUT versus warp lookup separately for each measured shape.
- [ ] Verify D256/D512, fragmented pages, distinct nibble halves, causal/SWA masks and block sizes 1/31/32/33/64/128 with existing dense oracle.
- [ ] Retain only a measured policy with no significant end-to-end regression. Record extra scratch/register use and reject spill-driven regressions even if a selected microcase wins.

## Task 5: Conditional host synchronization and decode graphs

**Files:** Inspect/modify `src/llama-kv-cache-paged.{h,cpp}`, `src/llama-graph.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu` only where attribution warrants; tests in `tests/test-save-load-state.cpp` and existing backend suite.

**Interfaces:** Preserve `prepare`, `apply`, `set_table_input`, transactional failure behavior and graph input lifetime. A new internal event or graph-cache interface requires a focused follow-up design review before coding.

- [ ] Measure state copies and synchronization with dirty-range tables enabled; compare `LLAMA_PAGED_KV_FULL_UPLOAD` control only in isolated runs. Do not reimplement dirty upload already present.
- [ ] Correlate per-step D2D copies with graph nodes, including unconditional Q/K/V cont construction in src/llama-graph.cpp:4157. First evaluate bypassing cont only when the tensor really satisfies current contiguous assumptions. Any strided-input variant must carry source strides through CPU/CUDA kernels and support validation; add existing backend cases for strided inputs, masks and alias writes before using it.
- [ ] Separate copy/launch time from CPU-only time; profile wall-minus-kernel (~1.7 ms) includes transfers and profiler overhead. If host cost is material, first avoid redundant work without changing ownership. Any asynchronous upload must order against all prior readers and keep staging storage alive until completion.
- [ ] Add interleaved sequence/clear/restore/COW regression cases proving table version and page masks remain valid across graph reuse and upload boundaries. Run compute-sanitizer where supported in addition to numerical checks.
- [ ] Evaluate bounded decode-only graph capture with fixed storage and explicit byte accounting. Test alternating 1/4/12 active sequences plus prefill, cancellation, and page reuse. Reject if peak free VRAM drops below proposed 256 MiB margin or latency regresses.
- [ ] Keep a tested graphs-disabled fallback. Record the selected setting as a deployment recommendation; do not edit the production launcher automatically.

## Task 6: Adoption report

**Files:** Create `docs/pulsar-iq4-decode-validation-20261014.md` only when results exist; artifacts in the build validation directory. Update plan checkboxes with actual evidence.

- [ ] Build affected targets in the isolated execution workspace: `cmake --build build-pulsar-paged-decode-20261014 --target llama-server test-backend-ops test-save-load-state -j 8`. If using a fresh build, configure it with the captured baseline toolchain/CUDA flags first.
- [ ] Run CPU and CUDA paged backend tests and paged save/load tests with a compatible local model. Capture case counts; test-save-load-state has explicit paged device-state skips and must not be described as complete device-state coverage.
- [ ] Run teacher-forced quality comparisons, Korean/code/number retrieval, and supported long-context controls using the spec's quality gates. Do not change thresholds after seeing results.
- [ ] Run final unprofiled 1K/8K single-active-request gate and 1/4/12 concurrency/long-context regressions. Record model peaks, client p95 ITL/TTFT, cache status and exact F16 comparability limitations.
- [ ] Mark successful only when >= 100 tok/s and >= matched F16 at both primary points, correctness/quality pass, and capacity is retained. Otherwise state the remaining measured bottleneck and best verified rate; do not label the objective complete.
- [ ] Present a concrete launcher diff, rollback binary/settings and measured evidence for human deployment review. No commits, pushes or service restart are part of executing this plan by default.
