#!/usr/bin/env bash
set -euo pipefail

model_dir="${PULSAR_MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar}"
prune_bin="${PULSAR_PRUNE_BIN:-/home/user/aikar-engine/build/bin/aikar-prune}"
model="${REAM_MODEL:-$model_dir/Lumen-3.1-Pulsar-Q4_0_XL.gguf}"
dataset="${REAM_CALIBRATION:-$model_dir/../train_ko.jsonl}"
output_dir="${REAM_OUTPUT_DIR:-/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar}"
output="${REAM_OUTPUT:-$output_dir/Lumen-3.5-Pulsar_S-exp1-LD-Q4_0_XL.gguf}"
log="${REAM_LOG:-$model_dir/ream.log}"
for arg in "$@"; do
  if [[ "$arg" == --dry-run && -z "${REAM_LOG:-}" ]]; then log="$model_dir/ream-dry-run.log"; fi
done
compute_device="${REAM_COMPUTE_DEVICE:-gpu}"
feature_precision=f32
if [[ "$compute_device" == gpu ]]; then feature_precision=f16; fi
feature_precision="${REAM_FEATURE_PRECISION:-$feature_precision}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GGML_VK_VISIBLE_DEVICES=
export GGML_CUDA_DISABLE_GRAPHS=1
export LD_LIBRARY_PATH="$(dirname "$prune_bin")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

args=(hard --method ream --model "$model" --dataset "$dataset" --output "$output"
  --target-experts "${REAM_TARGET_EXPERTS:-64}" --ream-group-size "${REAM_GROUP_SIZE:-16}"
  --ream-merging logits+weights --ream-sequential --ream-compute-device "$compute_device"
  --ctx-size "${REAM_CTX_SIZE:-4096}" --batch-size "${REAM_BATCH_SIZE:-512}"
  --ubatch-size "${REAM_UBATCH_SIZE:-512}" --threads "${REAM_THREADS:-36}"
  --dataset-threads "${REAM_DATASET_THREADS:-36}" --seed "${REAM_SEED:-42}"
  --n-gpu-layers "${REAM_GPU_LAYERS:-99}" --cpu-moe
  --ream-activation-samples "${REAM_ACTIVATION_SAMPLES:-32768}"
  --ream-chunk-size "${REAM_CHUNK_SIZE:-4096}"
  --ream-feature-precision "$feature_precision"
  --ream-input-cache-mib "${REAM_INPUT_CACHE_MIB:-6144}"
  --ream-expert-cache-mib "${REAM_EXPERT_CACHE_MIB:-12288}"
  --ream-max-memory-mib "${REAM_MAX_MEMORY_MIB:-4096}")
mkdir -p "$(dirname "$output")"
args+=(--ream-work-dir "${REAM_WORK_DIR:-$model_dir/pulsar-ream-work-$$}")
args+=(--ream-activation-dir "${REAM_ACTIVATION_DIR:-$model_dir/pulsar-ream-activ-$$}")
"$prune_bin" "${args[@]}" "$@" 2>&1 | tee "$log"
