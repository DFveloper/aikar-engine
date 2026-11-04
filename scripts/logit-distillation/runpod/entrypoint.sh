#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    preflight|extract|train|benchmark|all) nvidia-smi -L ;;
esac
case "$mode" in
    help)
        echo 'Commands: preflight, plan, extract, train, benchmark, all, shell'
        echo 'Mount persistent /workspace with train_ko.jsonl, train.jsonl and Lumen-3.8-Pulsar/{Test.gguf,Lumen-3.8-Pulsar-base.gguf}.'
        ;;
    preflight)
        nvidia-smi
        "$BIN_DIR/test-logit-distillation"
        "$BIN_DIR/test-backend-ops" test -b CUDA0 -o DISTILLATION,MUL_MAT_ID_BACK "$@"
        "$BIN_DIR/test-backend-ops" test -b CUDA0 -o CPY -p fp4_kv
        "$BIN_DIR/test-backend-ops" grad -b CUDA0 -o DISTILLATION
        ;;
    plan)
        bash "$SCRIPT_DIR/production-extract.sh" --dry-run "$@"
        bash "$SCRIPT_DIR/production-train.sh" --dry-run "$@"
        ;;
    extract) exec bash "$SCRIPT_DIR/production-extract.sh" "$@" ;;
    train) exec bash "$SCRIPT_DIR/production-train.sh" "$@" ;;
    benchmark) exec bash "$SCRIPT_DIR/benchmark.sh" "$@" ;;
    all)
        bash "$SCRIPT_DIR/production-extract.sh" "$@"
        exec bash "$SCRIPT_DIR/production-train.sh" "$@"
        ;;
    shell) exec bash "$@" ;;
    *) echo "Unknown command: $mode" >&2; exit 2 ;;
esac
