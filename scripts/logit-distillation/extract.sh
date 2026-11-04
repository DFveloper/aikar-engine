#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-/home/user/aikar-engine}
BIN_DIR=${BIN_DIR:-$REPO/build-distill-cpu/bin}
MODEL_DIR=${MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar}
DATA=${DATA:-/mnt/openwebui/AIKAR/train_ko.jsonl}
CTX=${CTX:-512}
UBATCH=${UBATCH:-128}
THREADS=${THREADS:-8}
MAX_WINDOWS=${MAX_WINDOWS:-0}
VAL_SPLIT=${VAL_SPLIT:-0.05}
CACHE=${CACHE:-$RUN_DIR/teacher-top512.dat}
exec "$BIN_DIR/llama-extract-logits" \
    -m "$MODEL_DIR/Test.gguf" \
    --distill-student "$MODEL_DIR/Lumen-3.8-Pulsar-base.gguf" \
    --train-file "$DATA" --distill-cache "$CACHE" \
    --distill-top-k 512 --distill-temperature 1 \
    --distill-max-windows "$MAX_WINDOWS" --val-split "$VAL_SPLIT" \
    -c "$CTX" -b "$CTX" -ub "$UBATCH" -np 1 -ngl 0 \
    -t "$THREADS" -tb "$THREADS" -dthr "$THREADS" "$@"
