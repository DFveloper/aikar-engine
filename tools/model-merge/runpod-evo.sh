#!/usr/bin/env bash
set -euo pipefail

if (( $# < 6 )); then
    echo "usage: $0 BUILD_DIR BASE.gguf OUT.gguf CALIBRATION TARGET_TYPE MODEL.gguf [MODEL.gguf ...]" >&2
    exit 2
fi

build_dir=$1
base=$2
output=$3
calibration=$4
target_type=$5
shift 5

binary="$build_dir/bin/llama-merge"
if [[ ! -x "$binary" ]]; then
    echo "llama-merge executable not found: $binary" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for the RunPod CUDA preset" >&2
    exit 2
fi

mapfile -t gpu_indices < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
if (( ${#gpu_indices[@]} == 0 )); then
    echo "no visible NVIDIA GPUs" >&2
    exit 2
fi

threads=${THREADS:-$(nproc)}
available_kib=$(awk '/MemAvailable:/ { print $2; exit }' /proc/meminfo)
memory_budget=${MEMORY_BUDGET:-$((available_kib * 1024 * 7 / 10))}
population=${POPULATION:-$((2 * ${#gpu_indices[@]}))}
if (( population < 4 )); then
    population=4
fi
generations=${GENERATIONS:-10}
elite_count=${ELITE_COUNT:-2}
context_size=${CTX_SIZE:-512}
gpu_layers=${GPU_LAYERS:--1}
seed=${SEED:-42}
device_mode=${EVO_DEVICE_MODE:-replica}

candidate_dir=$(mktemp -d "${TMPDIR:-/tmp}/llama-merge-evo.XXXXXX")
trap 'rmdir "$candidate_dir" 2>/dev/null || true' EXIT

device_args=()
case "$device_mode" in
    replica)
        device_names=()
        for index in "${!gpu_indices[@]}"; do
            device_names+=("CUDA$index")
        done
        device_list=$(IFS=,; echo "${device_names[*]}")
        if (( ${#device_names[@]} == 1 )); then
            device_args=(--device "$device_list")
        else
            device_args=(--devices "$device_list")
        fi
        ;;
    split)
        ;;
    *)
        echo "EVO_DEVICE_MODE must be replica or split" >&2
        exit 2
        ;;
esac

model_args=()
for model in "$@"; do
    model_args+=(--model "$model")
done

echo "RunPod Evo: GPUs=${#gpu_indices[@]} mode=$device_mode population=$population threads=$threads"
echo "RunPod Evo: temp=$candidate_dir memory_budget=$memory_budget"

"$binary" \
    --base "$base" \
    "${model_args[@]}" \
    --output "$output" \
    --method evo \
    --calibration "$calibration" \
    --target-type "$target_type" \
    --population "$population" \
    --generations "$generations" \
    --elite-count "$elite_count" \
    --seed "$seed" \
    --ctx-size "$context_size" \
    --gpu-layers "$gpu_layers" \
    --threads "$threads" \
    --memory-budget "$memory_budget" \
    --temp-dir "$candidate_dir" \
    --merge-gpu \
    "${device_args[@]}"
