#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
model_dir="${PULSAR_MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar}"
server_bin="${LUMEN_SERVER_BIN:-$repo_dir/build-paged-decode/staging-bin/llama-server}"
server_lib_dir="$(dirname "$server_bin")"

exec env CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$server_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$server_bin" -m "$model_dir/Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf" \
    --chat-template-file "$model_dir/chat_template.jinja" \
    --host "${PULSAR_HOST:-127.0.0.1}" --port "${PULSAR_PORT:-18136}" \
    -ngl 99 -ncmoe 0 -dev CUDA0 -c 8192 -np 1 -b 1280 -ub 1280 \
    -ctk q8_kv -ctv q8_kv --kv-paged --kv-block-size 16 --kv-blocks 512 \
    -kvu -fa on -t 36 -tb 36 --backend-sampling --reasoning-budget -1 \
    --no-cache-idle-slots "$@"
