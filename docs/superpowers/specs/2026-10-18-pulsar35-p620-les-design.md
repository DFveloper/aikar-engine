# Pulsar 3.5 P620 lookahead expert prefetch design

Status: proposed implementation specification; implementation has not started.

## Intent and scope

Optimize Lumen 3.5 Pulsar inference on the local Quadro P620 and Xeon E5-2696 v3. The user approved reusing the existing expert cache, predicting the next layer's experts, transferring predicted experts asynchronously, preserving every expert selected by the real router, and retaining only measured performance improvements. Existing CPU kernel edits belong to earlier work and must be preserved. Do not commit, push, change the production launcher, or replace model files.

The first implementation is LES-derived cross-layer prefetch for single-token decode. It does not claim the complete MiniCPM LES implementation or its published speedups. Ready-expert-first execution inside an MMID operation is outside this change: existing grouped MMID operations consume all selected slots together. Splitting those operations and restoring their original reduction order requires separate design and measurement. SSD weight streaming and additional quantization are also outside this change.

## Observed environment

- Model: `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.5-Pulsar_S-LD-Q4_0_XL.gguf`, 13,121,388,640 bytes.
- Architecture: Gemma4, 30 layers, embedding width 2816, 116 experts per layer, 8 selected experts per token.
- Expert tensors including scale metadata: 11102.71 MiB. Other tensors: 1395.73 MiB. These are file tensor sizes, not device allocation measurements.
- One expert's fused gate/up and down payload: 3,345,408 bytes. A complete selection across 30 layers reads approximately 765.70 MiB before metadata and padding.
- GPU: Quadro P620, compute capability 6.1, 2048 MiB physical VRAM. Initial free memory was approximately 1991 MiB before CUDA initialization.
- CPU: 18 physical cores, 36 logical threads, AVX2, one NUMA node, 121 GiB RAM.
- GPU access requires the available executor outside the filesystem sandbox; ordinary sandbox commands do not expose the NVIDIA device nodes.

## Baselines

Existing binaries were used without rebuilding or editing source. Synthetic PP128 and separate TG32, two repetitions, F16 KV, short context:

| Execution | Threads | PP128 tok/s | TG32 tok/s |
| --- | ---: | ---: | ---: |
| CPU | 12 | 53.86 | 11.40 |
| CPU | 18 | 57.05 | 11.31 |
| CPU | 36 | 55.21 | 8.20 |
| CUDA layers, all 30 expert layers on CPU | 18 | 56.42 | 11.32 |

The CPU 36-thread measurements have high variance. CPU and hybrid rows use different existing build directories and are screening results, not evidence of a candidate implementation's improvement. Final comparisons must use the same binary and configuration except the feature under test. Raw files: `/tmp/pulsar35-les-cpu-baseline.{json,log}` and `/tmp/pulsar35-les-hybrid-baseline.{json,log}`. Partial layer offload screening: `/tmp/pulsar35-les-offload-sweep.{json,log}`.

Additional screening with the existing CUDA build, 12 threads, PP128/TG32 and two repetitions gave TG rates of 9.03, 8.81 and 9.13 tok/s for 1, 2 and 4 GPU layers respectively. These configurations did not improve the initial CPU screening result.

An existing `llama-completion` run used the same model, prompt, 32 generated tokens, greedy sampling, context 512, batch/ubatch 128, 12 decode threads and 18 batch threads, with 30 CPU expert layers and the remaining tensors assigned to CUDA. No-cache generation measured 11.27 tok/s; 128 MiB expert cache measured 4.73 tok/s. The 256 MiB cache could not initialize because CUDA compute-buffer allocation exhausted memory. Each is one screening run, not a repeated benchmark. Outputs from the no-cache and 128 MiB modes differed; no numerical parity claim is established by these runs. Logs and outputs are `/tmp/pulsar35-les-cache-{0,128,256}.{log,out}`. This reinforces the requirement to validate cache correctness and transfer cost before claiming any LES improvement.

## Existing integration points

- `src/llama-moe-cache.cpp` already owns bounded GPU banks, shared LRU groups, logical-to-physical slot maps, batched miss uploads, statistics and conservative large-batch fallback.
- `src/llama-context.cpp::sched_copy_experts` dispatches cache callbacks and copies selected experts for uncached graphs.
- `src/llama-graph.cpp::build_moe_cache_slots` inserts a host-weight slot-map lookup to establish a scheduler split after real routing.
- `src/models/gemma4.cpp` computes the real router from normalized attention output, inverse square-root embedding scaling, per-layer router scale and router weights. Expert scales and fused gate/up tensors must remain unchanged.

## Chosen behavior

Add an opt-in lookahead setting next to `--moe-cache-mib`, disabled by default. Require a nonzero expert cache, CUDA, Gemma4, inference mode and a single decode token. Use the existing path for prefill, unsupported batch shapes, other backends, training and unsupported adapters or weight transforms. Reject an explicitly requested combination that cannot provide a usable CUDA cache with a clear error rather than silently reporting LES as active.

At layer L, apply layer L+1's router normalization, scale and router matrix to an available earlier hidden state of matching embedding width. This state is a prediction input, not the true L+1 attention output. Select at most eight predicted IDs. Do not assume expert IDs correspond across layers. The current layer's real router remains authoritative, including its softmax weights, expert scales and expert reduction order. The final layer has no lookahead.

Extend the existing cache callback boundary to consume this prediction after the current layer's required slots have been prepared and protected, before the current layer's expert compute is submitted. The graph must explicitly order the prediction, actual slot preparation, and lookahead trigger; graph construction order alone is insufficient. The trigger follows the existing host-weight callback pattern and must not force an extra synchronization after current expert compute starts. If this ordering cannot be established with the existing scheduler, stop this implementation and revise the specification before introducing a new scheduler subsystem.

Create a separate transfer backend on the same CUDA device, owned by the cache, and use its stream and backend events. Confirm event interoperability and actual copy/compute overlap on SM61 before adopting it. Ordinary `tensor_set_async` on the compute backend does not establish overlap. Do not add a new GGML operation or general backend API for this initial implementation.

## Cache ownership and event contract

Each slot has a logical owner, generation, state and outstanding-use protection. States are empty, loading and ready. Loading reserves the slot immediately, but it is not a hit until its completion event is satisfied. A stale completion must never update a reassigned generation.

Demand selections have priority over speculative selections. Protect all current selected slots across fused gate/up and down consumers. Prefetch may use only empty slots or completed, unprotected slots; skip predictions when no safe slots remain. Do not increase the requested cache allocation to make a prediction fit. A shared LRU must retain `(layer, expert)` identity.

At the next real lookup, reuse completed predictions, wait for required in-flight predictions, and normally load every remaining true miss. Predictions that were not selected never alter the router or output. Upload complete gate/up and down payloads with original expert scales and padding behavior. Slot maps become visible only with the corresponding transfer dependencies. Preserve the existing conservative fallback when a batch cannot fit.

Initially disable CUDA graph capture only in LES mode, because host routing readback and mutable cache dependencies need explicit ordering. Preserve ordinary graph behavior when LES is disabled. Serialize cache mutations on the existing inference thread. On teardown or cancellation, drain transfer and compute uses before freeing or reusing buffers. Weight-only cache contents are not serialized with KV state.

## Memory and performance policy

Account separately for model buffers, KV, compute scratch, cache banks, guard slots, transfer staging and CUDA overhead. A small global cache can thrash across 30 layers even when within one-layer working-set capacity. Measure demand hit rate, prediction precision and recall, useful prefetched bytes, wasted bytes, demand wait time and end-to-end token latency. A high prediction hit rate alone is not a performance result.

Keep experts on CPU as a required baseline. Compare the current expert cache without prefetch against the new prefetch mode, using identical byte budgets. The cache may remain disabled in the recommended P620 configuration if PCIe traffic makes it slower.

Paged KV is an independent memory policy. Compare conventional F16 or existing supported quantized KV against paged KV at matched logical context, physical token capacity and concurrency. Test short context and a practical longer context separately. Do not assume paging speeds up single-request short-context decode or copy the V100 262K configuration onto a 2 GiB GPU.

## Validation and adoption

1. Establish the best existing CPU/hybrid configuration and a same-binary baseline before modifying inference code.
2. Add deterministic cache planner coverage for wrong predictions, zero safe slots, same expert number in different layers, in-flight ownership, required miss fallback and cache capacity boundaries. Reuse existing test infrastructure.
3. Compare fixed-input logits with LES disabled, cache enabled without lookahead, and lookahead enabled. Since expert execution order is unchanged, require identical outputs between cache modes on the same backend; investigate any discrepancy before timing.
4. Exercise shutdown with outstanding work, repeated context creation and prefill/decode transitions. Verify no expert or slot is reused before its final consumer completes.
5. Run matched sequential measurements with warmup and at least three repetitions. Record PP, TG, TTFT, latency distribution, peak VRAM, cache traffic and prediction statistics on Korean, coding and arithmetic prompts. Run a longer-context KV comparison separately.
6. Adopt a candidate only when correctness passes and its end-to-end improvement exceeds run variance. Keep a reproducible launch command and raw measurements. Report failed experiments and lack of improvement explicitly.

## Alternatives considered

Pure CPU execution and partial layer offload require no expert streaming and remain competitive on this hardware. Synchronous expert caching is the direct comparison for evaluating lookahead. Full LES with ready-first expert execution would require MMID scheduling changes and is deferred until prefetch measurements establish that transfer latency is worth targeting.

Reference: https://github.com/tc-mb/llama.cpp-omni/blob/master/docs/les/lookahead-expert-scheduling.md . Its published measurements concern MiniCPM on Apple hardware and do not predict Pulsar performance on P620.
