#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=run
resume=latest
fresh_optimizer=()

while (( $# )); do
    case "$1" in
        --from-best) resume="$root/logit-distillation/interleaved/stagnation-20261113/shared-kv-phase-512/shared-kv.adapter.gguf.step136.gguf"; fresh_optimizer=(--fresh-optimizer); shift ;;
        --dry-run) mode=dry-run; shift ;;
        --check) mode=check; shift ;;
        --resume) resume=${2:?Usage: ./train.sh --resume CHECKPOINT}; fresh_optimizer=(); shift 2 ;;
        --fresh) resume=; fresh_optimizer=(); shift ;;
        -h|--help)
            cat <<'HELP'
Usage: ./train.sh [--from-best | --resume CHECKPOINT | --fresh] [--dry-run | --check]

Default: resume the latest complete checkpoint from the current run.
--from-best starts Q/O training from Best weights with a new optimizer and data position zero.
--fresh starts from the base model. Edit the arguments below to change training.
--dry-run prints the command. --check also checks input files without using GPUs.
Validation measures loss; it does not stop training or change the learning rate.
HELP
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

exec /usr/bin/python3 "$root/scripts/train.py" \
    --root "$root" \
    --mode "$mode" \
    --resume "$resume" "${fresh_optimizer[@]}" \
    --auto-microbatch --microbatch-candidates 4096,2048,1024,768,512,256,128 \
    --library-path "$root/.cache/ga16-runtime":/home/user/aikar-engine/build-validation-runtime:/home/user/aikar-engine/build-joint/bin:/home/user/aikar-engine/build/bin \
    --fp32-residual \
    --source-shuffle-seed 42 --source-shuffle-from 0 \
    --async-save \
    --joint-validation --joint-fusion output --joint-val-microbatch 768 \
    --joint-start-initial \
    -- \
    "/home/user/aikar-engine/build-validation-runtime/llama-finetune-qo" \
    -m "$root/Lumen-3.8-Pulsar-base.gguf" \
    --train-file "$root/logit-distillation/interleaved/runs/20261112T001619Z-476684/mixed.json" \
    --distill-cache "$root/logit-distillation/interleaved/runs/20261112T001619Z-476684/mixed.json" \
    -dev CUDA0,CUDA1 -sm tensor -ts 1,1 -mg 0 -ngl all \
    --n-cpu-moe 0 --fit off -fa on -np 1 \
    -c 16384 -b 16384 -ub 4096 --grad-accum 6 \
    -t 8 -tb 8 -dthr 8 \
    --optimizer qlion2 --quant-type q4_0,q8_0 \
    --qat-targets blk.11.attn_q.weight,blk.11.attn_output.weight,blk.17.attn_q.weight,blk.17.attn_output.weight,blk.23.attn_q.weight,blk.23.attn_output.weight,blk.29.attn_q.weight,blk.29.attn_output.weight \
    --override-tensor '^output\.weight$=CUDA0' \
    --qat-static-graph off --activation-recompute on \
    --epochs 1 \
    -lr 1e-6 -lr-min 1e-7 --lr-scheduler cosine --warmup-steps 0 \
    --eval-every 8 \
    --qat-adapter-only --qat-adapter-out interleaved.adapter.gguf \
    --save-first-at 0 --save-every 4 \
    --verbose-loss
