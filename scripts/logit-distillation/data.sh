if [[ -z "${DATA:-}" ]]; then
    DATA=${MIXED_DATA:-${MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar}/logit-distillation/mixed-train-ko-first-1to1.jsonl}
    python3 "$(dirname -- "${BASH_SOURCE[0]}")/mix_data.py" \
        --ko "${DATA_KO:-$(dirname -- "${MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar}")/train_ko.jsonl}" \
        --other "${DATA_OTHER:-$(dirname -- "${MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar}")/train.jsonl}" \
        --output "$DATA" --seed "${MIX_SEED:-42}"
fi
