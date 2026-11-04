#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-/home/user/aikar-engine}
BIN_DIR=${BIN_DIR:-$REPO/build-distill-cpu/bin}
MODEL_DIR=${MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar}
source "$RUN_DIR/data.sh"
source "$RUN_DIR/targets.sh"
CTX=${CTX:-512}
UBATCH=${UBATCH:-128}
THREADS=${THREADS:-8}
MAX_WINDOWS=${MAX_WINDOWS:-0}
VAL_SPLIT=${VAL_SPLIT:-0.05}
CACHE=${CACHE:-$RUN_DIR/teacher-top512.dat}
OUTPUT=${OUTPUT:-$RUN_DIR/distilled-adapter.gguf}
exec "$BIN_DIR/llama-finetune-qlion" \
    -m "$MODEL_DIR/Lumen-3.8-Pulsar-base.gguf" \
    --train-file "$DATA" --distill-cache "$CACHE" \
    --distill-max-windows "$MAX_WINDOWS" --val-split "$VAL_SPLIT" \
    --optimizer qlion2 --quant-type q4_0,q8_0 \
    --qat-targets "$TARGETS" \
    --qat-out '' --qat-adapter-out "$OUTPUT" --save-every 0 \
    --epochs "${EPOCHS:-1}" -lr "${LR:-1e-5}" \
    --eval-initial --verbose-loss --shuffle-dataset \
    -c "$CTX" -b "$CTX" -ub "$UBATCH" -np 1 -ngl "${GPU_LAYERS:-0}" \
    -t "$THREADS" -tb "$THREADS" -dthr "$THREADS" "$@"
