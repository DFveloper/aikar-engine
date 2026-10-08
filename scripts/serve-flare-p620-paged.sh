#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
model_dir="${FLARE_MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.2-Flare}"
server_bin="${FLARE_SERVER_BIN:-$repo_dir/build-flare-p620/bin/llama-server}"
server_lib_dir="$(dirname "$server_bin")"
parallel_slots="${FLARE_PARALLEL:-4}"
kv_blocks="${FLARE_KV_BLOCKS:-$((576 * parallel_slots))}"
ram_cache_mib="${FLARE_CACHE_RAM_MIB:-32768}"
cache_type="${FLARE_CACHE_TYPE:-f16}"
prefill_chunk_size="${FLARE_PREFILL_CHUNK_SIZE:-64}"
template_file="${FLARE_CHAT_TEMPLATE:-$repo_dir/scripts/flare-p620-chat-template.jinja}"
system_prompt="$(python3 - "$template_file" <<'PYTHON'
import re
import sys
from pathlib import Path

match = re.search(r"{%\s*set\s+system\s*%}(.*?){%\s*endset\s*%}", Path(sys.argv[1]).read_text(), re.DOTALL)
if match is None or not match.group(1).strip():
    raise SystemExit("chat template has no static system prompt")
sys.stdout.write(match.group(1))
PYTHON
)"

exec env CUDA_VISIBLE_DEVICES="${FLARE_GPU:-1}" LD_LIBRARY_PATH="$server_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    LLAMA_SERVER_SYSTEM_PROMPT="$system_prompt" \
    "$server_bin" -m "$model_dir/Lumen-3.2-Flare-stage4-LD-Q4_0_L.gguf" \
    --alias Lumen-3.2-Flare --jinja --reasoning off --chat-template-file "$template_file" \
    --host "${FLARE_HOST:-127.0.0.1}" --port "${FLARE_PORT:-11437}" \
    -fit off -ngl 99 -dev CUDA0 -c "$((9216 * parallel_slots))" --kv-unified-per-slot 9216 -np "$parallel_slots" \
    -b "$((9216 * parallel_slots))" -ub 256 -ctk "$cache_type" -ctv "$cache_type" -kvu -fa on -t 4 -tb 4 \
    --kv-paged --kv-block-size 16 --kv-blocks "$kv_blocks" --backend-sampling \
    --prefill-chunk-size "$prefill_chunk_size" --cache-prompt --no-cache-idle-slots --cache-ram "$ram_cache_mib" --ctx-checkpoints 0 --no-context-shift \
    -n 512 --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0 \
    --samplers 'top_k;top_p;temperature' --slots "$@"
