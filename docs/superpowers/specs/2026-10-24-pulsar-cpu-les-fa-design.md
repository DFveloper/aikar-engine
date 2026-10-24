# Pulsar CPU LES and cache-aware attention

The user selected CPU-first execution and delegated detailed design and implementation. Remove SWA-full everywhere in the active configuration. Work on the IQ3 Quality model without altering its weights or reducing the router's expert selection. No commits, pushes, PRs, new quantization types, or new attention operators.

## Approach

Use bounded paged KV and improve its existing CPU attention operator. Retain its scalar implementation as the correctness reference. Add blocked online softmax, SIMD dot/accumulate helpers, and query tiling so dequantized K/V tiles are reused. Select small working tiles suitable for the Xeon's 256 KiB L2. Preserve page validity, causal/noncausal visibility, sliding windows, GQA, and supported cache encodings. Ordinary CPU FA tile sizing can be adjusted only when separately measured.

Implement CPU LES as cross-layer router prediction plus an asynchronous, bounded worker that prepares host expert cache lines. A graph callback publishes an immutable per-expert priority snapshot before real expert computation and submits next-layer prediction before optional current-layer demand prefetch, because mandatory current reads are already performed by CPU compute. CPU MUL_MAT_ID computes prepared actual experts first and scatters results into their original selected-expert slots. The model's real router, per-expert scales, and final summation order remain authoritative. A prediction miss cannot omit computation. Cache preparation is a software hint, not a guarantee that hardware retains lines in LLC.

A CPU-only context owns the worker and immutable layer descriptors. Every decode iteration has a generation; stale work cannot publish readiness for another iteration. Stop/join the worker before its model buffers can be released. Bound speculative bytes per job and cancel superseded work. Preserve existing paths for prefill, unsupported backends, training, and unsupported model/router layouts. Restrict the initial prediction integration to the Gemma4 router path. Expose opt-in --cpu-les and --cpu-les-prefetch-mib controls; default inference remains unchanged.

The P620 streaming alternative is deferred because the measured CPU baseline was faster and left little GPU cache capacity. Pure software prefetch without ready-first expert scheduling does not fulfill this implementation's LES contract.

## Validation and adoption

Add deterministic CPU LES tests for prediction, wrong predictions, generation changes, bounded preparation, zero/invalid configuration, demand fallback, and shutdown. Exercise priority-aware MMID with real outputs and original slot ordering. Extend existing paged attention cases for page boundaries, invalid slots, empty partitions, sliding windows, GQA, F16 and Q8_KV, multiple query tokens, and noncausal inputs. Compare the optimized path against the retained reference without wall-clock correctness thresholds.

Use matched native Release builds and sequential measurements. Record PP/TG, output/logit comparisons, prediction accuracy, bytes prepared, and request latency on the actual GGUF at short and longer contexts. Hardware perf counters are denied by the current host; do not change system security settings. Keep only measured improvements enabled in the launcher; retain experimental LES as opt-in if it is correct but not faster. Verify Korean, arithmetic, streaming, and safe bounded-KV prompt reuse without SWA-full.
