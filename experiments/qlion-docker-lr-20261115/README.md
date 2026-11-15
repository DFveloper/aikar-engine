# QLion with Docker learning-rate settings

User-requested experiment started on 2026-11-15. The previous QLion2 controller was stopped. The new run restores the latest complete checkpoint, step104 / dataset window624, and continues the original data order and schedule position.

| Setting | Applied value |
| --- | --- |
| Optimizer | QLion (`qlion_qat` in native logs/state) |
| Adaptive LR scaling | Off, `lr_scale_unit=0` |
| Base LR | 1e-5 |
| Minimum LR | 1e-6 |
| Scheduler | Cosine |
| Warmup | 50 steps; resume step104 is past warmup |
| Residual | F32 |
| Optimizer transition | Reset momentum, residual values and adaptive LR state; preserve weights and progress |
| Context / microbatch / accumulation | 16384 / 768 / 6 |
| Training targets | Existing eight Q/output tensors |
| Save / validation interval | 4 / 8 optimizer steps |

LR, minimum LR and warmup match `scripts/logit-distillation/production.env` inside local image `aikar-distill:recovery-jupyter`, digest `sha256:56ded6caedc568166ad8eec29096093bfbb3cbf94713a202e98d5d65bef29aed`. The image production launcher uses cosine scheduling. Other training settings retain the existing local values.

The persistent model-directory `train.sh` now selects QLion, these LR settings, F32 residuals and the existing optimizer-switch reset option. Its prior contents and the exact diff are saved as `train-before.sh` and `train.patch`.

Verified in the actual native run:

- `optimizer=qlion_qat lr_scale_unit=0`.
- `residual_type: f32`.
- `optimizer switch qlion2 -> qlion_qat; reset momentum, residual and adaptive LR state; preserve model and progress`.
- Scheduler starts at step104 with warmup50, LR1e-5 and minimum1e-6.
- The new step104 adapter/state pair was saved successfully; its metadata confirms QLion, F32 residual, optimizer step104 and dataset window624.
- Real training forward/backward microbatches are progressing.
- Launcher input checks, shell syntax, existing ten launcher tests and `git diff --check` pass.

`active-run.json` identifies the run. `command.json` contains the exact invocation. `checkpoint-verification.json` records the saved metadata checks. Previous run metadata remains in `previous-run.json`.

The initial joint validation snapshot is step104, before any QLion updates. That result is a starting reference. Learning quality must be assessed using later snapshots; no improvement is claimed from startup or changing the optimizer alone.
