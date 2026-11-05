if [[ -z "${DATA:-}" ]]; then
    DATA_MODE=${DATA_MODE:-ko-knowledge-reasoning}
    if [[ "$DATA_MODE" == ko-knowledge-reasoning ]]; then
        DATA=${KO_DATA:-$(dirname -- "${MODEL_DIR:-/workspace/Lumen-3.8-Pulsar}")/ko-knowledge-reasoning.jsonl}
        python3 "$(dirname -- "${BASH_SOURCE[0]}")/prepare_ko.py" --output "$DATA" --seed "${KO_SEED:-42}"
    elif [[ "$DATA_MODE" == mixed ]]; then
    DATA=${MIXED_DATA:-${MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar}/logit-distillation/mixed-train-ko-first-1to1.jsonl}
    python3 "$(dirname -- "${BASH_SOURCE[0]}")/mix_data.py" \
        --ko "${DATA_KO:-$(dirname -- "${MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar}")/train_ko.jsonl}" \
        --other "${DATA_OTHER:-$(dirname -- "${MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar}")/train.jsonl}" \
        --output "$DATA" --seed "${MIX_SEED:-42}"
    else
        echo "Unknown DATA_MODE: $DATA_MODE" >&2
        return 2
    fi
fi
