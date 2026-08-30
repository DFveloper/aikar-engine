#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

model="${LUMEN_MODEL:-/mnt/openwebui/AIKAR/Lumen-3-Pulsar/Lumen-3-Pulsar-it-agentic-q4_0.gguf}"
template="${LUMEN_TEMPLATE:-/mnt/openwebui/AIKAR/Lumen-3-Pulsar/chat_template.jinja}"
bench_bin="${LLAMA_BENCH:-$repo_dir/build/bin/llama-bench}"
cli_bin="${LLAMA_CLI:-$repo_dir/build/bin/llama-cli}"
quality_bin="${LLAMA_KV_QUALITY:-$repo_dir/build/bin/llama-kv-quality}"
mode="${1:-quick}"
configs="${LUMEN_KV_CONFIGS:-f16 q8_hadamard q4_hadamard turbo3 turbo4 local_f16_global_turbo3 local_f16_global_turbo4 local_f16_global_turbo3_turbo4 local_f16_global_turbo4_turbo3}"
contexts="${LUMEN_CONTEXTS:-4096,16384,32768,65536,131072,262144}"
repetitions="${LUMEN_REPETITIONS:-3}"
predict="${LUMEN_PREDICT:-64}"
n_cpu_moe="${LUMEN_N_CPU_MOE:-0}"
dataset="${LUMEN_DATASET:-/mnt/openwebui/AIKAR/agentic-sample.jsonl}"
dataset_samples="${LUMEN_DATASET_SAMPLES:-1}"
dataset_context="${LUMEN_DATASET_CONTEXT:-16384}"
dataset_batch="${LUMEN_DATASET_BATCH:-4096}"
dataset_ubatch="${LUMEN_DATASET_UBATCH:-256}"
dataset_chars="${LUMEN_DATASET_CHARS:-2500}"

if [[ ! -r "$model" ]]; then
    printf 'model not found: %s\n' "$model" >&2
    exit 1
fi
if [[ ! -r "$template" ]]; then
    printf 'chat template not found: %s\n' "$template" >&2
    exit 1
fi

export GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cache_args() {
    case "$1" in
        f16)          printf '%s\n' -ctk f16 -ctv f16 ;;
        q8_hadamard)  printf '%s\n' -ctk q8_0 -ctv q8_0 --k-cache-hadamard --v-cache-hadamard ;;
        q4_hadamard)  printf '%s\n' -ctk q4_0 -ctv q4_0 --k-cache-hadamard --v-cache-hadamard ;;
        turbo3)       printf '%s\n' -ctk turbo3 -ctv turbo3 ;;
        turbo4)       printf '%s\n' -ctk turbo4 -ctv turbo4 ;;
        local_f16_global_turbo3)        printf '%s\n' -ctlk f16 -ctlv f16 -ctgk turbo3 -ctgv turbo3 ;;
        local_f16_global_turbo4)        printf '%s\n' -ctlk f16 -ctlv f16 -ctgk turbo4 -ctgv turbo4 ;;
        local_f16_global_turbo3_turbo4) printf '%s\n' -ctlk f16 -ctlv f16 -ctgk turbo3 -ctgv turbo4 ;;
        local_f16_global_turbo4_turbo3) printf '%s\n' -ctlk f16 -ctlv f16 -ctgk turbo4 -ctgv turbo3 ;;
        *)
            printf 'unknown KV configuration: %s\n' "$1" >&2
            return 1
            ;;
    esac
}

run_quick() {
    [[ -x "$bench_bin" ]] || { printf 'benchmark executable not found: %s\n' "$bench_bin" >&2; exit 1; }
    for config in $configs; do
        mapfile -t kv_args < <(cache_args "$config")
        printf 'Lumen KV benchmark: config=%s mode=quick\n' "$config" >&2
        "$bench_bin" -m "$model" -ngl 999 -ncmoe "$n_cpu_moe" -fa on -b 512 -ub 512 -r "$repetitions" -o jsonl -p 512 -n "$predict" "${kv_args[@]}"
    done
}

run_context() {
    [[ -x "$bench_bin" ]] || { printf 'benchmark executable not found: %s\n' "$bench_bin" >&2; exit 1; }
    for config in $configs; do
        mapfile -t kv_args < <(cache_args "$config")
        printf 'Lumen KV benchmark: config=%s mode=context contexts=%s\n' "$config" "$contexts" >&2
        "$bench_bin" -m "$model" -ngl 999 -ncmoe "$n_cpu_moe" -fa on -b 512 -ub 512 -r "$repetitions" -o jsonl -p 0 -n "$predict" -d "$contexts" "${kv_args[@]}"
    done
}

run_quality() {
    [[ -x "$quality_bin" ]] || { printf 'quality executable not found: %s\n' "$quality_bin" >&2; exit 1; }
    local system_prompt='You are an agentic coding assistant. Follow instructions exactly and keep structured outputs valid.'
    local prompts=(
        'Give a two-sentence diagnosis for a unit test that fails only after prompt-cache reuse.'
        'Return only JSON for a tool call named inspect_file with path src/llama-kv-cache.cpp and line 220.'
        'Write a compact C++ function that checks whether a positive integer is a power of two, then list two edge cases.'
        'Reason step by step about why online softmax is numerically stable across tiled attention, then give the invariant.'
        'A long cached conversation already established that CUDA SM70 is the target. Give the next three profiling actions as a JSON array.'
    )

    for config in $configs; do
        [[ "$config" == f16 ]] && continue
        mapfile -t kv_args < <(cache_args "$config")
        for prompt in "${prompts[@]}"; do
            printf 'Lumen KV quality: config=%s prompt=%s\n' "$config" "$prompt" >&2
            "$quality_bin" --jinja -m "$model" -ngl 999 -ncmoe "$n_cpu_moe" -fa on -c 4096 -b 4096 -ub 512 -n "$predict" --chat-template-file "$template" -sys "$system_prompt" -p "$prompt" "${kv_args[@]}"
        done
    done
}

run_agentic() {
    [[ -x "$cli_bin" ]] || { printf 'CLI executable not found: %s\n' "$cli_bin" >&2; exit 1; }
    local system_prompt='You are an agentic coding assistant. Follow instructions exactly and keep structured outputs valid.'
    local prompts=(
        'Diagnose a unit test that fails only after prompt-cache reuse. Give the smallest safe investigation plan.'
        'Return only a JSON tool call for inspect_file with path src/llama-kv-cache.cpp and line 220.'
        'Implement a compact C++ online-softmax update and explain its numerical invariant.'
        'Analyze the bandwidth bottleneck of batch-1 attention at a 64K cached context on a V100.'
        'Continue a long cached agent turn: list the next profiling actions and the measurements that decide dispatch.'
    )

    for config in $configs; do
        mapfile -t kv_args < <(cache_args "$config")
        for prompt in "${prompts[@]}"; do
            printf 'Lumen agentic run: config=%s prompt=%s\n' "$config" "$prompt" >&2
            "$cli_bin" --jinja -m "$model" -ngl 999 -ncmoe "$n_cpu_moe" -fa on -c 4096 -b 512 -ub 512 -n "$predict" --chat-template-file "$template" -sys "$system_prompt" -p "$prompt" -cnv --single-turn --perf "${kv_args[@]}"
        done
    done
}

run_dataset() {
    [[ -x "$quality_bin" ]] || { printf 'quality executable not found: %s\n' "$quality_bin" >&2; exit 1; }
    [[ -r "$dataset" ]] || { printf 'dataset not found: %s\n' "$dataset" >&2; exit 1; }

    local prompt_dir
    prompt_dir="$(mktemp -d)"
    trap 'rm -rf -- "$prompt_dir"' RETURN
    python3 - "$dataset" "$prompt_dir" "$dataset_samples" "$dataset_chars" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
limit = int(sys.argv[3])
max_chars = int(sys.argv[4])
written = 0
with source.open(encoding="utf-8") as stream:
    for line in stream:
        if written >= limit:
            break
        record = json.loads(line)
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue
        parts = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "unknown"))
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            parts.append(f"[{role}]\n{content}")
        if parts:
            selected = []
            selected_chars = 0
            for part in reversed(parts):
                if selected and selected_chars + len(part) > max_chars:
                    break
                selected.append(part)
                selected_chars += len(part)
            prompt = "\n\n".join(reversed(selected))
            (output / f"sample-{written:03d}.txt").write_text(prompt, encoding="utf-8")
            written += 1
if written == 0:
    raise SystemExit("no messages records found in dataset")
PY

    for config in $configs; do
        [[ "$config" == f16 ]] && continue
        mapfile -t kv_args < <(cache_args "$config")
        for prompt_file in "$prompt_dir"/*.txt; do
            printf 'Lumen dataset quality: config=%s dataset=%s sample=%s\n' "$config" "$dataset" "$prompt_file" >&2
            "$quality_bin" --jinja -m "$model" -ngl 999 -ncmoe "$n_cpu_moe" -fa on -c "$dataset_context" -b "$dataset_batch" -ub "$dataset_ubatch" -n "$predict" --chat-template-file "$template" -f "$prompt_file" "${kv_args[@]}"
        done
    done
}

case "$mode" in
    quick)   run_quick ;;
    context) run_context ;;
    quality) run_quality ;;
    agentic) run_agentic ;;
    dataset) run_dataset ;;
    all)
        run_quick
        run_context
        run_quality
        run_agentic
        run_dataset
        ;;
    *)
        printf 'usage: %s [quick|context|quality|agentic|dataset|all]\n' "$0" >&2
        exit 1
        ;;
esac
