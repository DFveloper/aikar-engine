# Pulsar decode throughput implementation plan

> Execute inline with superpowers:executing-plans. User already delegated CPU-first design decisions. No commits or external submissions.

**Goal:** Analyze the remaining decode bottleneck and maximize measured generation throughput toward 20 tokens/s, examining 50 tokens/s only where supported by real workload measurements.

**Architecture:** Preserve the previous CPU paged F16 implementation, exact expert selections and no SWA-full. Profile actual tensor operations with the existing evaluation callback and compare uninstrumented timings. Screen thread/poll/affinity settings before changing the hottest CPU kernel. Evaluate existing verified speculative decoding separately on natural Korean, coding and repetitive workloads; do not present favorable copy workloads as general generation performance.

**Alternatives:** CPU kernel/scheduling optimization preserves the current hardware and model and is the primary path. Model-free speculative decoding can amortize weight reads when continuations are predictable; adoption depends on acceptance and latency. GPU layer streaming already lost to CPU with P620, so revisit only if the measured hot tensors fit usefully. Reducing experts or changing weights/model is outside this task's accuracy-preserving scope.

**Files:** diagnostics and raw data under build-pulsar-cpu-les/results/phase3; CPU Q4 GEMV helper reuse in ggml/src/ggml-cpu/arch/x86/repack.cpp; exact large-vector TANH partition in unary-ops.cpp and task dispatch in ggml-cpu.c; exact IQ4_NL native row decoding in CPU quants.c/quants.h with PAGED_ATTN dispatch in ops.cpp; tests/test-backend-ops.cpp for relevant boundary fixtures; scripts/serve-pulsar-iq3-temporary.sh for measured defaults and optional mixed KV; dated result report in docs.

## Constraints and review focus

- Preserve weights, routing, scales, output slots and expert reduction order.
- No SWA-full, global host security changes, commits, pushes or PRs.
- Use a separate candidate build before rebuilding libraries loaded by the active service.
- Operator callback splits graph execution and changes overhead; it diagnoses relative costs, not uninstrumented speed.
- Static/chunk scheduling must cover every expert/row once, including duplicate IDs, uneven rows and excess threads.
- Speculative throughput includes draft verification costs and rejected tokens; compare completion contents and token counts on each fixture.
- Record failed/slower candidates and distinguish single-stream generation from aggregate concurrent throughput.

## Tasks

- [x] Measure uninstrumented fixed-token decode at several thread counts and operation-level time/weight bytes; measure practical host read bandwidth.
- [x] Select the demonstrated bottleneck, add a meaningful focused regression fixture if kernel behavior changes, and measure a minimal alternative implementation.
- [x] Compare current and winning candidates sequentially at short/long context and on natural output; evaluate existing speculative decoding with acceptance statistics.
- [x] Review candidate source, run applicable tests/logit comparisons, deploy only measured improvements, verify API and report measured limits against 20/50 targets.
- [x] Evaluate the user's added Gemma4 DFlash/DSpark request with compatible pinned GGUF drafts, acceptance and matched CPU measurements; preserve the paged-speculation guard and distinguish conventional bounded SWA.
- [x] Evaluate the user's added MTP request with the Gemma4 26B-A4B assistant Q5_K_M draft on the same target and fixtures, checking acceptance and output agreement.

## Measured design decisions

The instrumented 12-step decode attributed about 38.6% of operation time to dense Q4 matrix multiplication, 29.6% to expert matrix multiplication across IQ3/Q4, 6.7% to paged attention and 2.7% to one large scalar TANH. Callback graph splits change overhead, so percentages diagnose candidates rather than predict full-graph speedup. Matrix weight spans total about 2.107 GB per token, not measured DRAM traffic. The streaming 1 GiB read screen saturated near 30.8 GB/s at 24 threads. Under a model that reads each span once, this suggests about 14.6 tokens/s before other work; cache reuse, layout and workload can change the correspondence. Do not present 20 or 50 as achieved or guaranteed.

Test an exact Q4_0 unsigned-offset integer dot already used in GEMM instead of repeated general signed correction in GEMV; preserve every floating-point accumulation. Split only sufficiently large TANH tensors into contiguous element ranges; retain strided row handling and exact tanhf. User additionally requested IQ4_NL global/Q8_KV local policies: measure that existing configuration and IQ4_NL global/F16 local, then test exact AVX2 nibble lookup/decode via the existing IQ4 table. Mixed domains must use the explicit global/local CLI flags because generic -ctk/-ctv conflicts with them.

Existing server explicitly rejects speculative decoding with paged KV. Benchmark speculation as an explicitly separate conventional bounded-SWA configuration; do not remove the guard or force unsafe rollback. First test model-free ngram methods; only the requested IQ3 GGUF is locally available, so no unapproved large draft model download is part of the plan.
