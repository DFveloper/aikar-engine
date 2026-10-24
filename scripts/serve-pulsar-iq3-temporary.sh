#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
server_bin="${PULSAR_IQ3_SERVER_BIN:-$repo_dir/build-pulsar-throughput/bin/llama-server}"
model="${PULSAR_IQ3_MODEL:-/mnt/openwebui/AIKAR/Adaptive-bit/models/Lumen-3.1-Pulsar-IQ3-Quality.gguf}"
server_lib_dir="$(dirname "$server_bin")"
kv_args=(-ctk f16 -ctv f16)
les_args=()
spec_args=()
paged_default=1

case "${PULSAR_IQ3_SPEC:-none}" in
    none) ;;
    dspark|dflash|mtp)
        spec_type="$PULSAR_IQ3_SPEC"
        case "$spec_type" in
            dspark) draft_default="$repo_dir/build-pulsar-throughput/drafts/dspark-q4.gguf" ;;
            dflash) draft_default="$repo_dir/build-pulsar-throughput/drafts/dflash-q8.gguf" ;;
            mtp) draft_default="$repo_dir/build-pulsar-throughput/drafts/mtp-q5.gguf" ;;
        esac
        draft_model="${PULSAR_IQ3_DRAFT_MODEL:-$draft_default}"
        [[ -r "$draft_model" ]] || { printf 'draft model not found: %s\n' "$draft_model" >&2; exit 1; }
        [[ "${PULSAR_IQ3_PAGED:-0}" == 0 ]] || { printf 'Speculative decoding requires PULSAR_IQ3_PAGED=0\n' >&2; exit 2; }
        paged_default=0
        spec_args=(-md "$draft_model" --spec-type "draft-$spec_type"
            --spec-draft-n-max "${PULSAR_IQ3_DRAFT_MAX:-3}" --spec-draft-n-min 1 --spec-draft-p-min 0
            -devd none -ngld 0 -td "${PULSAR_IQ3_THREADS:-18}" -tbd "${PULSAR_IQ3_THREADS_BATCH:-18}")
        ;;
    *) printf 'PULSAR_IQ3_SPEC must be none, dspark, dflash or mtp\n' >&2; exit 2 ;;
esac

case "${PULSAR_IQ3_PAGED:-$paged_default}" in
    0) ;;
    1)
        kv_type="${PULSAR_IQ3_KV_TYPE:-f16}"
        kv_global="${PULSAR_IQ3_KV_GLOBAL:-$kv_type}"
        kv_local="${PULSAR_IQ3_KV_LOCAL:-$kv_type}"
        for cache_type in "$kv_type" "$kv_global" "$kv_local"; do
            case "$cache_type" in
                f16|q8_kv|iq4_nl|q4_0) ;;
                *) printf 'KV types must be f16, q8_kv, iq4_nl or q4_0\n' >&2; exit 2 ;;
            esac
        done
        kv_args=(--kv-paged --kv-block-size 16 -ctgk "$kv_global" -ctgv "$kv_global" -ctlk "$kv_local" -ctlv "$kv_local")
        ;;
    *) printf 'PULSAR_IQ3_PAGED must be 0 or 1\n' >&2; exit 2 ;;
esac

case "${PULSAR_IQ3_LES:-0}" in
    0) ;;
    1) les_args=(--cpu-les --cpu-les-prefetch-mib "${PULSAR_IQ3_LES_PREFETCH_MIB:-4}") ;;
    *) printf 'PULSAR_IQ3_LES must be 0 or 1\n' >&2; exit 2 ;;
esac

for argument in "$@"; do
    case "$argument" in
        --swa-full|--swa-full=*) printf 'SWA-full is disabled for this launcher\n' >&2; exit 2 ;;
    esac
done

[[ -x "$server_bin" ]] || { printf 'server not found: %s\n' "$server_bin" >&2; exit 1; }
[[ -r "$model" ]] || { printf 'model not found: %s\n' "$model" >&2; exit 1; }

exec env -u LLAMA_ARG_CACHE_TYPE_K -u LLAMA_ARG_CACHE_TYPE_V \
    -u LLAMA_ARG_CACHE_TYPE_K_LOCAL -u LLAMA_ARG_CACHE_TYPE_V_LOCAL \
    -u LLAMA_ARG_CACHE_TYPE_K_GLOBAL -u LLAMA_ARG_CACHE_TYPE_V_GLOBAL \
    LLAMA_ARG_SWA_FULL=0 LLAMA_ARG_CPU_LES=0 \
    LD_LIBRARY_PATH="$server_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$server_bin" -m "$model" --alias Lumen-3.1-Pulsar-IQ3-Quality \
    --host "${PULSAR_IQ3_HOST:-127.0.0.1}" --port "${PULSAR_IQ3_PORT:-11438}" \
    -fit off -dev none -ngl 0 -c "${PULSAR_IQ3_CTX:-8192}" -np 1 \
    -t "${PULSAR_IQ3_THREADS:-18}" -tb "${PULSAR_IQ3_THREADS_BATCH:-18}" \
    -b 1024 -ub 512 -fa on "${kv_args[@]}" "${les_args[@]}" "${spec_args[@]}" \
    --cache-prompt --no-cache-idle-slots --cache-ram 0 --ctx-checkpoints 0 --no-context-shift \
    --reasoning "${PULSAR_IQ3_REASONING:-off}" --metrics -n 1024 "$@"
