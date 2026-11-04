#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
set -a
source "$SCRIPT_DIR/production.env"
source "$SCRIPT_DIR/data.sh"
set +a
BENCH_DIR=${BENCH_DIR:-$MODEL_DIR/logit-distillation/benchmark-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir "$BENCH_DIR"
BENCH_TEMP=$(mktemp -d "${TMPDIR:-/tmp}/qlion-throughput.XXXXXXXX")
trap 'rm -f "$BENCH_TEMP/final.gguf" "$BENCH_TEMP/final.gguf.qat-state.gguf"; rmdir "$BENCH_TEMP"' EXIT
common=(--train-file "$DATA" --distill-cache "$BENCH_DIR/teacher.dat"
    --distill-max-windows "${BENCH_WINDOWS:-2}" --val-split "$VAL_SPLIT"
    -c "$CTX" -b "$CTX" -np 1 -ngl "$GPU_LAYERS" -t "$THREADS" -tb "$THREADS" -dthr "$THREADS")
/usr/bin/time -o "$BENCH_DIR/extract.time" -f 'wall_s=%e max_rss_kib=%M' \
    "$BIN_DIR/llama-extract-logits" -m "$MODEL_DIR/Test.gguf" \
    --distill-student "$MODEL_DIR/Lumen-3.8-Pulsar-base.gguf" \
    --distill-top-k "$TOP_K" --distill-temperature "$TEMPERATURE" \
    -ub "$EXTRACT_UBATCH" "${common[@]}" > "$BENCH_DIR/extract.log" 2>&1
/usr/bin/time -o "$BENCH_DIR/train.time" -f 'wall_s=%e max_rss_kib=%M' \
    "$BIN_DIR/llama-finetune-qlion" -m "$MODEL_DIR/Lumen-3.8-Pulsar-base.gguf" \
    "${common[@]}" -ub "$TRAIN_UBATCH" --optimizer qlion2 --quant-type q4_0,q8_0 \
    --qat-targets "$TARGETS" --freeze-layers 0 --qat-static-graph "$STATIC_GRAPH" \
    --epochs 1 -lr "$LR" --save-every 0 --eval-every 0 --shuffle-dataset \
    --verbose-loss --qat-out "$BENCH_TEMP/final.gguf" > "$BENCH_DIR/train.log" 2>&1
rg 'distill_throughput' "$BENCH_DIR/extract.log"
rg 'qat_throughput|validation:|qat_memory:' "$BENCH_DIR/train.log"
cat "$BENCH_DIR/extract.time" "$BENCH_DIR/train.time"
echo "Logs: $BENCH_DIR"
