# Static MoE expert pruning

`aikar-prune` implements static expert pruning for Gemma 4 26B A4B. The initial hard-pruning implementation supports the Q4_0 QAT GGUF layout loaded by `src/models/gemma4.cpp`. Other architectures and Gemma 4 variants fail with an unsupported-architecture error.

> Soft pruning does not modify model weights. It statically disables selected experts when the model is loaded. The selected profile cannot be changed while the server is running.

## Soft and hard pruning

Soft pruning writes a JSON profile containing original expert IDs. At model initialization, the profile identity hashes and shape constraints are validated. Disabled router logits are set to negative infinity before routing selection. Surviving router probabilities retain the model's original softmax and selected-weight normalization behavior.

Hard pruning consumes exactly one soft profile. It creates a new GGUF with compacted routed expert tensors and router rows, updates `gemma4.expert_count`, and records each original-to-new expert mapping. The dense parallel MLP in every Gemma 4 MoE layer is the shared expert path and is always copied without changes.

Hard pruning never edits the source model. It writes a temporary sibling file, finishes and validates it, then renames it to the requested output.

## ChatML JSONL

Each line must contain one object with a non-empty `messages` array:

```json
{"messages":[{"role":"user","content":"What is MoE?"},{"role":"assistant","content":"A sparse expert architecture."}]}
```

Assistant messages may contain separate reasoning and final content:

```json
{"messages":[{"role":"user","content":"Solve it."},{"role":"assistant","reasoning":"Work...","content":"Answer."}]}
```

The model's Jinja chat template and tokenizer render every record. For the extended form, reasoning and final content are rendered in that order in one assistant message with a newline separator. Errors include the JSONL line number. Supported loss masks are `all`, `assistant`, `reasoning`, and `content`.

Field masks use a deterministic token-start rule. A rendered token belongs to the field containing the byte offset at which that token starts. Template and special tokens belong to no message field and are evaluated only by the `all` mask.

## Analyze

```sh
build/bin/aikar-prune analyze \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --dataset calibration.jsonl \
  --ratios 0.05,0.10,0.15,0.20,0.25 \
  --metric router-output \
  --ppl-mask assistant \
  --max-layer-ratio 0.25 \
  --seed 42 \
  --output-dir pruning-results
```

Calibration collects selection count and frequency, router probability sum and mean, routed expert output L2 norm, and `mean(router_probability * output_norm)`. Experts are ranked once per layer by the final metric with original expert ID as the deterministic tie-breaker. Every larger ratio takes a longer prefix of the same ranking, so pruning sets are nested.

### Importance metrics

`--metric` accepts `router-output` (the existing default), `reap`, and `frequency`. Each metric uses the same per-layer ranking, Top-K safety checks, soft profiles, and hard GGUF rewrite path. `frequency` ranks by selection count.

REAP measures the conditional average contribution when an expert is selected:

```text
reap_sum[e] += gate_weight * L2(expert_output)
reap_count[e] += 1
REAP_score[e] = reap_count[e] > 0 ? reap_sum[e] / reap_count[e] : 0
```

The denominator is the expert's selection count, not the total calibration token count or total routing count. A rarely selected expert can therefore retain a high score. Every selected Top-K slot contributes one sample, including zero gate weights or zero output norms. Experts with no selections score zero. Completed calibration requires output sample counts to match routing selection counts.

For Gemma 4, the gate weight is read directly from `ffn_moe_weights_norm`: the router softmax probability divided by the sum of selected Top-K probabilities, with the graph's existing denominator clamp to `6.103515625e-5`. REAP does not apply a second normalization. Gemma 4's routing weight scale is 1. The expert output is measured after expert-specific down-projection scaling and before gate multiplication and summation, excluding the post-aggregation RMS normalization. The parallel dense shared expert is not ranked or pruned.

The existing `router-output` formula is also a conditional average. Both metrics now observe the same effective output tensor and count each routing selection once, so their scores agree. The historical collector missed `ffn_moe_down_scaled` and counted reshaped routing views; those collection errors are fixed for both metrics. The report exposes both scores for inspection.

Calibration gathers both metrics in one forward pass. Added persistent storage is 32 bytes per routed expert (routing selection count, output sample count, output norm sum, and weighted output sum). Activation norms use double-precision accumulation and bounded reads with at most 16 KiB of scratch space; full activation tensors are not copied or retained. The existing Top-K routing buffers are reused. FP16, BF16, and FP32 output tensors are supported. Non-finite activations, invalid gates, and accumulator overflow fail calibration rather than producing a pruning plan. GPU calibration adds output reads and callback synchronization; no extra model forward pass is required.

GGML reshape views inherit their source name with a ` (reshaped)` suffix. Calibration accepts only the exact original routing/output tensor names. Each selection, gate and entropy observation is recorded once; Top-K ID views are copied using their row stride. Scaled down-projection outputs populate both aikar and REAP sums. `router-output` retains its conditional-average formula, which equals REAP when both use these same correct observations. It is not frequency-weighted scoring.

REAP currently has the same architecture limits as the existing pruning implementation: Gemma 4 26B A4B with homogeneous expert counts. Other architectures and different numbers of experts per layer remain unsupported. Hard rewriting retains its Q4_0 QAT layout restriction.

`analyze` and `hard` disable CUDA graphs before backend initialization. Profiling callbacks split graph execution into multiple segments; captured segments retain separate MMVQ activation caches in VRAM. On a nearly full GPU these extra buffers can cause OOM in `ggml_cuda_prepare_mmvq_activation_cache`, even after the model and compute buffers loaded successfully. Direct execution reuses temporary CUDA pool buffers and preserves routing and score semantics. This setting affects only the pruning process, not `llama-server`. For older binaries, prefix the command with `GGML_CUDA_DISABLE_GRAPHS=1`. Model weights, KV cache, and the active compute batch still need available memory; a GPU already occupied by a serving process requires CPU offload or stopping that service before GPU calibration.

The first calibration writes `pruning-results/importance-cache.json`. The cache contains model and dataset identities, context size, baseline token/NLL aggregates, and the raw per-layer expert statistics. It does not contain a ratio-specific ranking. A later `analyze` command with compatible inputs loads this cache instead of running baseline calibration again. Use `--importance-cache FILE` to share one cache across output directories.

Ratio evaluation is enabled by default. Use `--no-evaluate` to generate profiles without loading and evaluating each soft-pruned model:

```sh
build/bin/aikar-prune analyze \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --dataset calibration.jsonl \
  --importance-cache calibration-importance.json \
  --ratios 0.05,0.10,0.15,0.20,0.25 \
  --no-evaluate \
  --output-dir pruning-profiles
```

To reduce GPU memory use during calibration or ratio evaluation, pruning accepts the same MoE CPU offload options as the other llama.cpp tools:

```sh
build/bin/aikar-prune analyze ... -ngl 999 -ncmoe 8
build/bin/aikar-prune analyze ... -ngl 999 -cmoe
```

`-ncmoe N` keeps routed expert tensors in the first N model layers on the CPU. `-cmoe` keeps all routed expert tensors on the CPU; the two options cannot be combined. These settings apply to baseline calibration, pruned evaluation and hard-output validation. They are part of the calibration execution fingerprint, so a cache made with different offload settings is rejected. `profiles` does not load the full model. CPU offload reduces VRAM use; speed depends on hardware and the number of GPU layers it allows you to retain.

`--evaluate` explicitly selects the default evaluated mode. Skipping ratio evaluation does not skip a cache miss's initial calibration. It skips only the full dataset passes for the generated soft-pruned profiles. `analysis.json` records `evaluation_enabled`, and unevaluated ratio entries record `evaluated: false`.

Once a cache exists, the `profiles` subcommand can create arbitrary compatible ratio profiles without opening or hashing the model or reading the dataset:

```sh
build/bin/aikar-prune profiles \
  --importance-cache calibration-importance.json \
  --ratios 0.06,0.12,0.18,0.24 \
  --output-dir another-profile-set
```

The selected `--ppl-mask` must have at least one evaluated token in the cached calibration. `--max-layer-ratio` and router Top-K safety checks still apply when profiles are generated.

New importance caches contain scalar statistics for all three metrics and require `routing_stats_version: 3`. Older caches, including version 2 caches that still lack correct aikar measurements, must be recalibrated for every metric. Explicit incompatible caches fail; automatic incompatible or malformed caches are replaced after successful fresh calibration. Legacy baseline checkpoints are not migrated. `profiles --metric reap` selects REAP explicitly; without `--metric`, `profiles` keeps the cache's recorded metric. `analyze` continues to default to `router-output`.

Cache validation checks model/dataset identities, tokenized record order and loss-mask digest, context, seed, batches, execution/build/backend/environment settings, and expert/count/loss consistency. Cache hits load vocabulary and tokenize for verification without an inference pass. Profiles carry collector version and calibration/token/execution fingerprints; obsolete generated profiles are rejected by soft/hard validation. Regenerate existing profiles from a new cache.

### Compare metrics with one calibration

Calibrate once, then compare the same ratios and evaluation data through the existing perplexity evaluator:

```sh
build/bin/aikar-prune analyze \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --dataset calibration.jsonl \
  --importance-cache comparison-importance.json \
  --ratios 0.10,0.20 --no-evaluate \
  --output-dir comparison-calibration

for metric in router-output reap frequency; do
  build/bin/aikar-prune analyze \
    --model gemma-4-26b-a4b-q4_0.gguf \
    --dataset calibration.jsonl \
    --importance-cache comparison-importance.json \
    --metric "$metric" --ratios 0.10,0.20 \
    --output-dir "comparison-$metric"
done
```

Use identical context size, loss mask, seed, batch/ubatch sizes, and execution options in each command. New caches record execution settings and `analyze` rejects mismatches. For ranking-only comparisons, use `profiles` with the same cache, ratios, and `--metric`, which requires no forward passes. Each profile lists the disabled expert IDs; the remaining IDs are the surviving experts. `analysis.json` records the metric, perplexity delta when evaluated, and `expected_expert_bytes_removed`. This byte estimate applies to hard expert/router compaction; soft masking does not reduce model size. The existing `hard` command's report records actual source and output GGUF sizes. This path does not add KL divergence evaluation.

Per-expert `analysis.json` entries include `selection_count`, `average_gate_weight`, `average_output_norm`, `REAP_score`, and `existing_aikar_score`, while preserving all prior fields. Older caches must be regenerated before producing these reports.

Perplexity uses stable accumulated negative log-likelihood:

```text
ppl = exp(total_nll / evaluated_token_count)
```

Baseline and pruned evaluations use identical rendered tokens, context windows, batches, and field masks. `analysis.json` contains all four field perplexities, timing, throughput, routing entropy, load imbalance, invalid-route count, and per-expert calibration statistics. `analysis.csv` contains one row per ratio. `README.txt` is a short human-readable summary.

JSONL reading, message parsing, template rendering, and tokenization use ordered parallel work queues. `--dataset-threads N` sets their worker limit; zero selects the physical core count. Results are collected in input order, and the pruning field masks use the same rendered text and token-start semantics in serial and parallel modes.

## Run the server

```sh
build/bin/llama-server \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --moe-prune-profile pruning-results/profile-020.json
```

The profile is loaded once before context creation. There is no hot reload, HTTP update, per-request mask, or per-sequence mask.

Validation covers format, version, architecture, full model SHA-256, expert identity SHA-256, routed MoE layer set, expert count, Top-K, expert ranges, duplicate IDs, equal surviving counts, and remaining-expert safety.

## Inspect and hard prune

```sh
build/bin/aikar-prune inspect \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --profile pruning-results/profile-020.json

build/bin/aikar-prune hard \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --profile pruning-results/profile-020.json \
  --output gemma-4-26b-a4b-pruned.gguf \
  --dataset calibration.jsonl \
  --validate
```

The hard converter accepts routed expert weights only when they are Q4_0 tensors with the original expert axis at GGML dimension 2, dimension 0 divisible by the Q4_0 block size of 32, and a tensor size equal to `row_size(ne0) * ne1 * ne2`. Each surviving expert slice is copied byte-for-byte. Router rows and optional per-expert scales are compacted without dequantization.

After conversion, the command reopens the output with the normal model loader and runs a short inference smoke test. When `--dataset` is present, it evaluates the hard model and the source model with the static soft profile, then records both perplexities and their absolute difference. The mapping, byte counts, and optional validation results are written to `<output>.report.json`.

## Limitations

- Hard pruning supports only Gemma 4 26B A4B Q4_0 QAT GGUF.
- All routed layers must retain the same expert count.
- Grouped routing, heterogeneous expert counts, interleaved expert storage, packed expert axes, and non-Q4_0 routed expert weights are rejected.
- Shared-expert pruning is not supported.
- Automatic benchmark-suite execution is not included.

Hard-pruning reports include source model path/hash, architecture, metric, requested/actual ratio, and a calibration fingerprint. The CLI adds final output path/hash and profile path/hash after validation. Generated profiles with obsolete provenance are rejected. Reports omit timestamps so identical inputs can be compared deterministically. Different ratio values that round to the same profile filename are rejected; use separate output directories.

Hard output publication backs up existing model/report with hard links, removes the old report before replacing the model, and restores the previous pair if publication fails. If rollback fails or the process stops between writes, the final report stays absent and `.validation.tmp.previous-model` / `.validation.tmp.previous-report` preserve recovery data. A subsequent publication refuses these paths until recovery is handled. This does not make two files a single atomic filesystem transaction; it prevents an old report from describing a new model.

`selection_frequency` is the fraction of calibration tokens that selected an expert. `routing_slot_fraction` records the old fraction of all selected slots. Shared experts are unchanged.
