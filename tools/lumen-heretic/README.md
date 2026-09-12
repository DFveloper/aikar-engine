# lumen-heretic

`lumen-heretic` is an experimental GGUF-native behavior-direction and Q4_0 projection-surgery tool for Gemma 4 E2B and Gemma 4 26B A4B. It does not support other architectures.

## Dataset

The preferred JSONL schema is:

```json
{"type":"target","prompt":"...","desired":"non_refusal"}
{"type":"control","prompt":"...","desired":"preserve"}
```

`desired` is optional and defaults to `non_refusal` for target records and `preserve` for control records. The legacy `group`, `target_token`, `target_tokens`, and `desired_score` fields remain supported. Prompts can be raw user messages with `--apply-chat-template`, or complete model prompts with `--raw-prompts`.

The loader rejects empty prompts and target/control overlap. It reports duplicates, extreme lengths, and statistically weak groups. The default recommended minimum is 32 target and 32 control records; 100-500 records per group are preferred for direction analysis.

One dataset can be split deterministically and by group:

```sh
lumen-heretic analyze \
  --model model.gguf \
  --dataset behavior.jsonl \
  --eval-split 0.2 \
  --seed 42 \
  --apply-chat-template \
  --output analysis
```

When an evaluation split is requested, direction extraction uses only the training portion. Separate files can instead be supplied with `--direction-dataset`, `--train-dataset`, and `--eval-dataset`.

## Scoring and preservation

The non-refusal objective generates a greedy response prefix and checks it with a configurable, case-insensitive refusal-pattern file. A pattern line can be `weight<TAB>pattern`; a line without a weight has weight 1. This is a lexical heuristic, not a semantic classifier. `--behavior-tokens` controls the prefix length.

Control preservation uses teacher-forced KL over the first `--kl-tokens` original-model positions. Reference probability distributions are cached in CPU memory. This favors correctness and avoids a second model on the GPU, but a large dataset, vocabulary, or token count can consume substantial host RAM.

Baseline evaluation without writing a GGUF is available with:

```sh
lumen-heretic evaluate \
  --model model.gguf \
  --dataset behavior.jsonl \
  --analysis analysis/analysis.json \
  --apply-chat-template \
  --kl-tokens 8 \
  --behavior-tokens 32 \
  --output baseline.json
```

## Optimization

```sh
lumen-heretic optimize \
  --model model.gguf \
  --dataset behavior.jsonl \
  --analysis analysis/analysis.json \
  --eval-split 0.2 \
  --apply-chat-template \
  --quant q4_0 \
  --train-alpha \
  --direction-layer-mode topk \
  --direction-topk 12 \
  --opt-both \
  --max-alpha 1.5 \
  --lambda-sparse 0.001 \
  --max-kl 0.10 \
  --steps 20 \
  --output edited.gguf
```

Use `--opt-attn`, `--opt-mlp`, or `--opt-both` for projection ablations. The 26B A4B `--experts` mode applies one layer direction and MLP alpha to all routed expert output projections. It does not edit router weights.

The optimizer uses SPSA-Adam. Every candidate starts from source GGUF bytes, applies the rank-1 update in a tensor-sized FP32 workspace, requantizes with the native Q4_0 codec, uploads the effective Q4_0 bytes, and evaluates that quantized model. Normal weights remain frozen and there is no full FP model copy. Candidates with non-finite metrics, broken or repetitive generation, or KL above `--max-kl` cannot become the best candidate.

The checkpoint at `OUTPUT.checkpoint.json` stores directions, bounded alpha state, Adam moments, candidate history, fingerprints, configuration, and PRNG state. `--resume` continues from it. The best evaluated candidate is restored before the final streaming GGUF write. Reports are written to `OUTPUT.report.json`, `OUTPUT.pareto.json`, and `OUTPUT.pareto.csv`.

The final writer copies unrelated and zero-alpha tensors byte-for-byte and preserves GGUF metadata, tokenizer data, and chat template. Only Q4_0 residual-writing projections are currently supported. Recovery LoRA, STE-QAT, semantic refusal classification, and expert-specific alpha parameters are intentionally outside this phase.
