# CPU LES and paged attention

This private CPU implementation targets the Gemma4 MoE router. Enable it with `--cpu-les`; `--cpu-les-prefetch-mib N` limits cache-line preparation per worker job to 1 through 1024 MiB (default 16). It requires CPU-resident decoder layers and a single sequence. It is experimental and disabled by default. Measurement on the Xeon E5-2696 v3 found a slowdown, so the temporary serving launcher leaves it disabled.

The context owns one asynchronous worker. During single-token decode, the graph snapshots prepared-expert priorities, then submits a prediction of the next layer's experts using its router and the current residual activation. Prediction precedes optional current-layer demand preparation. Each job has a byte budget, serial and generation. Superseded work is cancelled and cannot publish readiness into a later generation. Context destruction drains and joins the worker before releasing dependent resources. Prefill bypasses LES, LoRA graphs bypass LES, and training rejects LES.

Preparation reads one byte per cache line from the selected expert spans, including CPU-repacked buffers. A prepared hit means these software reads completed; it does not establish hardware cache residency. The existing CPU expert matrix loops visit prepared experts first and write into the original selected slots. Actual router decisions, expert scales and final reduction order remain unchanged. Wrong predictions only waste preparation work.

The CPU `PAGED_ATTN` implementation uses blocked online softmax, SIMD dot products and accumulations, F16C row conversion where available, and K/V tile reuse across queries and GQA heads. A maximum of 32 query/head rows and small key tiles bound per-thread scratch to about 192 KiB for optimized head dimensions. Mixed sequences use single-query tiles. Heads larger than 1024 retain the scalar implementation. Scratch comes from the existing graph work buffer. Page validity masks, sliding windows and causal/noncausal visibility still apply.

`GGML_CPU_PAGED_ATTN_REFERENCE=1` or the backend's `use_ref` setting selects scalar attention. Floating-point reduction order differs between implementations, and full-model logits need not be identical. On this quantized MoE model, small attention rounding differences can change later routing. LES on/off logits were byte-identical in the fixed-token probe; that result does not apply to scalar versus optimized attention. See the dated serving report for the numerical and throughput measurements.

The optional `ggml_mul_mat_id_set_expert_priority` metadata is implemented by the generic and repacked CPU paths. Its contiguous F32 vector has one entry per expert, with positive values visited first. It affects execution order only. Other backends do not implement this hint; the model integration is restricted to CPU.

## Validation commands

```bash
ctest --test-dir build-pulsar-cpu-les --output-on-failure -R 'test-cpu-les|test-arg-parser|test-server-component|test-batch-alloc|test-opt|test-tiled|test-alloc|test-generate-models|test-llama-context|test-context-q8-kv'
build-pulsar-cpu-les/bin/test-backend-ops test -b CPU -o PAGED_ATTN -p 'tokens=(1|7|8|9|17|33),.*nblocks=(1|4),'
```

Tests cover prediction misses, bounded preparation, generation changes, shutdown, graph reuse, rapid supersession and actual priority traversal with duplicate selections, including CPU_REPACK. Supersession tests exercise scheduling without timing thresholds; they do not deterministically cover every thread interleaving. Paged attention tests compare against the scalar path across cache types, tiling boundaries, GQA, multiple sequences, windows and noncausal visibility. The tests do not replace a broad language quality evaluation.
