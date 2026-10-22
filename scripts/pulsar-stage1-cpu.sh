#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar}
trainer=${QLION_BIN:-$repo_dir/build-pulsar-cpu-candidate-20261021/bin/llama-finetune-qlora}
model=${MODEL_PATH:-$model_dir/Lumen-3.5-Pulsar_S-LD-Q4_0_XL.gguf}
train_file=${TRAIN_FILE:-$model_dir/Benchmaxx.jsonl}
types=${TENSOR_TYPE_FILE:-$model_dir/Q4_0_XL.txt}
output=${OUTPUT_PATH:-/mnt/data/AIKAR/Lumen-3.1-Pulsar/Lumen-3.6-Pulsar_S-stage1-cpu-lora.gguf}
ubatch=${UBATCH:-512}
recompute=${RECOMPUTE:-off}
threads=${THREADS:-18}

[[ -x "$trainer" ]] || { echo "Trainer is not executable: $trainer" >&2; exit 1; }
for input in "$model" "$train_file" "$types"; do
    [[ -r "$input" ]] || { echo "Missing input: $input" >&2; exit 1; }
done
if [[ ! "$ubatch" =~ ^[1-9][0-9]*$ ]] || (( ubatch > 8192 || 8192 % ubatch != 0 )); then
    echo "UBATCH must be a positive divisor of 8192." >&2
    exit 1
fi
if [[ "$recompute" != on && "$recompute" != off ]]; then
    echo "RECOMPUTE must be on or off." >&2
    exit 1
fi
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || { echo "THREADS must be positive." >&2; exit 1; }
shopt -s nullglob
for existing in "$output" "$output.init.gguf" "$output.state" "$output".epoch*.ckpt*.gguf "$output".ckpt*.gguf; do
    if [[ -e "$existing" ]]; then
        echo "Output already exists; choose a new OUTPUT_PATH: $existing" >&2
        exit 1
    fi
done
if pgrep -f '(^|/)[l]lama-finetune-ql(ora|ion)( |$)' >/dev/null; then
    echo "Training is already running." >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$output")"
ulimit -c 0
export GGML_NO_BACKTRACE=1

exec "$trainer" \
    --model "$model" --train-file "$train_file" --lora-out "$output" \
    --val-split 0.0005 --eval-every 50 --eval-initial \
    --lora-alpha 32 --lora-rank 64 --seed 3407 \
    --lora-qat q4_0 --tensor-type-file "$types" \
    --activation-recompute "$recompute" --kv-cache-training -ctk f16 -ctv f16 \
    -fa on -c 8192 -b 8192 -ub "$ubatch" \
    --lr-scheduler cosine -lr-min 8e-17 \
    --lora-targets attn_q.weight,attn_k.weight,attn_v.weight,attn_output.weight \
    --epochs 1 --save-every 100 -lr 8e-6 \
    --warmup-steps 15 --warmup-init-ratio 1e-15 -wd 0 \
    --preserve-thinking -ngl 0 -dthr "$threads" -t "$threads" -tb "$threads"
