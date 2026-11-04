if [[ -z "${DATA:-}" ]]; then
    DATA=${MIXED_DATA:-${MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar}/logit-distillation/mixed-train-1to1.jsonl}
    python3 "$(dirname -- "${BASH_SOURCE[0]}")/mix_data.py" \
        --ko "${DATA_KO:-/mnt/openwebui/AIKAR/train_ko.jsonl}" \
        --other "${DATA_OTHER:-/mnt/openwebui/AIKAR/train.jsonl}" \
        --output "$DATA" --seed "${MIX_SEED:-42}"
fi
