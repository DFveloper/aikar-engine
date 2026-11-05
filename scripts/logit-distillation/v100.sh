#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR=${BUILD_DIR:-$REPO/build-v100}
BIN_DIR=${BIN_DIR:-$BUILD_DIR/bin}
MODEL_DIR=${MODEL_DIR:-/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar}
TEACHER=${TEACHER:-$MODEL_DIR/Test.gguf}
STUDENT=${STUDENT:-$MODEL_DIR/Lumen-3.8-Pulsar-base.gguf}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-/tmp/lumen-teacher-logits}
CACHE_DIR=${CACHE_DIR:-$DOWNLOAD_DIR/caches/a04b543e8c17a4208216d0b0f881735fff378e8d465409e2a7430598f35459fa}
DATA=${DATA:-$CACHE_DIR/train.jsonl}
CACHE=${CACHE:-$CACHE_DIR/teacher.dat}
OUTPUT=${OUTPUT:-$REPO/lumen-v100-recovery.adapter.gguf}
CTX=${CTX:-1024}
UBATCH=${UBATCH:-128}
MAX_WINDOWS=${MAX_WINDOWS:-7607}
VAL_WINDOWS=${VAL_WINDOWS:-32}
VAL_SPLIT=${VAL_SPLIT:-0.05}
THREADS=${THREADS:-8}
RECOMPUTE=${RECOMPUTE:-off}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export GGML_CUDA_P2P=${GGML_CUDA_P2P:-1}
if [[ $GGML_CUDA_P2P == 0 ]]; then unset GGML_CUDA_P2P; fi
export LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$SCRIPT_DIR/targets.sh"
mode=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
common=(-dev CUDA0,CUDA1 -sm tensor -ts 1,1 -ngl all --fit off -fa on -np 1
    -c "$CTX" -b "$CTX" -ub "$UBATCH" -t "$THREADS" -tb "$THREADS")
dataset=(--train-file "$DATA" --distill-cache "$CACHE" --distill-max-windows "$MAX_WINDOWS"
    --distill-validation-windows "$VAL_WINDOWS" --val-split "$VAL_SPLIT" -dthr "$THREADS")
case "$mode" in
    download) command=(hf download DFveloper/lumen-teacher-logits --repo-type dataset --local-dir "$DOWNLOAD_DIR") ;;
    build)
        cmake -S "$REPO" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
            -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_COMPILER:-/usr/bin/g++-13}" \
            -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_QUANTS=f16-f16 \
            -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_MTMD=OFF
        exec cmake --build "$BUILD_DIR" --target llama-completion llama-server llama-extract-logits llama-finetune-qlion test-logit-distillation test-backend-ops -j "${JOBS:-8}"
        ;;
    infer) command=("$BIN_DIR/llama-completion" -m "$STUDENT" "${common[@]}" -p "${PROMPT:-Hello}" -n "${TOKENS:-32}" -no-cnv) ;;
    teacher) command=("$BIN_DIR/llama-completion" -m "$TEACHER" "${common[@]}" -p "${PROMPT:-Hello}" -n "${TOKENS:-32}" -no-cnv) ;;
    extract) command=("$BIN_DIR/llama-extract-logits" -m "$TEACHER" --distill-student "$STUDENT" "${common[@]}" "${dataset[@]}" --distill-top-k 512 --distill-temperature 1) ;;
    train)
        export GGML_CUDA_ALLREDUCE=${GGML_CUDA_ALLREDUCE:-none}
        export GGML_CUDA_DISABLE_GRAPHS=${GGML_CUDA_DISABLE_GRAPHS:-1}
        if [[ $GGML_CUDA_DISABLE_GRAPHS == 0 ]]; then unset GGML_CUDA_DISABLE_GRAPHS; fi
        command=("$BIN_DIR/llama-finetune-qlion" -m "$STUDENT" "${common[@]}" "${dataset[@]}"
            --optimizer qlion2 --quant-type q4_0,q8_0 --qat-targets "$TARGETS"
            --override-tensor '^output\.weight$=CUDA0' --kv-cache-training -ctk f32 -ctv f32
            --qat-static-graph off --activation-recompute "$RECOMPUTE" --epochs "${EPOCHS:-3}" -lr "${LR:-1e-5}"
            --qat-out '' --qat-adapter-out "$OUTPUT" --save-every 0 --verbose-loss)
        ;;
    *) echo 'Usage: bash scripts/logit-distillation/v100.sh {download,build,infer,teacher,extract,train} [--dry-run | extra arguments]'; exit 0 ;;
esac
if [[ ${1:-} == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}" "$@"
fi
