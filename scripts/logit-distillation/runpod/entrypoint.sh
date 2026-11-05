#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
mode=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    preflight|extract|train|benchmark|all) nvidia-smi -L ;;
esac
case "$mode" in
    help)
        echo 'Commands: preflight, fetch-models, prepare-ko, budget, plan, extract, train, benchmark, all, upload-dataset, upload-checkpoints, jupyter-start, jupyter-status, shell'
        echo 'Mount /workspace with Lumen-3.8-Pulsar/{Test.gguf,Lumen-3.8-Pulsar-base.gguf}; prepare-ko downloads the default Korean data.'
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
    prepare-ko) exec python3 "$SCRIPT_DIR/prepare_ko.py" "$@" ;;
    fetch-models) exec python3 "$SCRIPT_DIR/fetch_models.py" "$@" ;;
    budget) exec python3 "$SCRIPT_DIR/budget.py" "$@" ;;
    jupyter-start)
        source "$SCRIPT_DIR/runpod/jupyter-env.sh"
        exec python -m jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=8888 \
            --ServerApp.port_retries=0 --ServerApp.root_dir=/workspace \
            --IdentityProvider.token="${JUPYTER_PASSWORD:-}" "$@"
        ;;
    jupyter-status)
        python -m jupyter server list
        if [[ -f /jupyter.log ]]; then tail -n 40 /jupyter.log; fi
        if [[ -f /workspace/logs/jupyter.log ]]; then tail -n 40 /workspace/logs/jupyter.log; fi
        ;;
    upload-dataset|upload-checkpoints)
        set -a
        source "$SCRIPT_DIR/production.env"
        set +a
        exec python3 "$SCRIPT_DIR/hf_upload.py" "${mode#upload-}" "$@"
        ;;
    benchmark) exec bash "$SCRIPT_DIR/benchmark.sh" "$@" ;;
    all)
        bash "$SCRIPT_DIR/production-extract.sh" "$@"
        exec bash "$SCRIPT_DIR/production-train.sh" "$@"
        ;;
    shell) exec bash "$@" ;;
    *) echo "Unknown command: $mode" >&2; exit 2 ;;
esac
