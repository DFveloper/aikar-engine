# Pulsar S QLoRA on V100

Validated on 2026-10-01 with Tesla V100 SXM2 16GB and Xeon E5-2696 v3. The model is `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf`.

## Changes

- Gemma4 training registers both shared and routed MoE FFNs for activation recomputation. Replay keeps the original expert indices; attention and KV writes stay outside replay.
- CPU router backward handles batched, strided top-k indices. CUDA handles the same operation directly, avoiding CPU graph splits. GET_ROWS_BACK preserves all four source dimensions.
- CUDA MoE input gradients size the gather buffer to the microbatch instead of always using 128 entries. The existing overflow path handles repeated expert routes.

## Measured settings

The rank 48 run trains attention Q/O and dense FFN gate/up/down adapters, with frozen experts, full GPU placement, recompute, F16 KV, context/batch 8192, and the original GGUF tensor type rules.

| Microbatch | Peak GPU memory | Short-prefix throughput |
| --- | --- | --- |
| 64 | 15,287 MiB | 44.0 tokens/s |
| 128 | 15,787 MiB | 42.0 tokens/s |

These measurements use the first six rows of the real stage1 dataset and stop after 12 microbatches. Throughput uses microbatches 4-12. They do not cover a complete 8192-token optimizer window or establish sustained epoch throughput. The two settings process different prefix lengths. Microbatch 64 is the conservative starting point because it leaves more memory headroom. Check later windows and evaluation before increasing it.

The stage runner at `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/stage_training.py` now defaults to microbatch 64, CPU MoE layers 0, recompute on, and F16 KV. Stage rank, alpha, learning rates, tensor type preservation, and checkpoint retention are preserved.

```bash
bash /mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/train-stages.sh \
  --run-dir /home/user/pulsar-s-training/pulsar-s-v100-recompute
```

The old training process was stopped. Validation runs finished; a new full curriculum was not started.

## Verification

- Real rank 48 training completed with recompute on and off for a 256-token window. All 300 saved adapter tensors matched exactly, with loss 1.744860. CUDA fusion was disabled in both runs for this comparison; fusion can otherwise change rounding when replay changes tensor lifetimes.
- `test-opt`: 9/9 backend/optimizer combinations passed, including MoE replay gradient and update comparisons.
- `test-qat`: CPU and CUDA0 reference tests passed, including batched router gradients and duplicate-route overflow.
- CUDA GET_ROWS_BACK backend tests passed, including strided indices and batched inputs.
- Stage runner: four unit tests and the real four-stage dry run passed.
- Full CTest has two remaining failures: `test-chat` throws while parsing tool responses, and `test-generate-models` segfaults. Five dependent recurrent/save-load tests consequently do not run. `test-state-restore-fragmented` passed when rerun with GPU access. The full suite is not green.

Raw validation logs and benchmark JSON files are in `/tmp/pulsar-recompute/`; `all48-64-real-final.log` contains the exact benchmark command. Existing user changes were retained; no commit or submission was made.
