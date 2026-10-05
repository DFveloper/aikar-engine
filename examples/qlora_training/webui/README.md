# Lumen training monitor

This is the Flare training WebUI adapted for Pulsar S. It is a read-only monitor: closing the browser does not stop training. The server and trainer survive SSH disconnects when launched in tmux. Starting the monitor alone does not detach an existing foreground trainer from SSH.

Deployment copies this directory to `<model_dir>/webui/` and the scripts from `../pulsar-s/` into the Pulsar model directory. The installed model directory is `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar`.

## Run a stage

```bash
cd /mnt/openwebui/AIKAR/Lumen-3.1-Pulsar
bash stage1.sh
```

Like Flare, each stage creates the `train` tmux session with `webui` and `finetune` windows. Open `http://<server-ip>:8787`, or attach with `tmux attach -t train`. Detach with Ctrl+B then D. Scripts reject another running GPU trainer. Finish the current stage and close its old tmux session before starting another stage.

`stage1.sh`, `stage2.sh`, `benchmaxx.sh` (also `stage3.sh`), and `stage4.sh` contain direct trainer commands. They do not automatically merge adapters or run another stage. Stage2/3/4 expect the previous merged GGUF, like Flare. `stage_training.py` and `train-stages.sh` remain available for the earlier sequential training/merge workflow.

The defaults are full CUDA placement, context/batch 8192, microbatch 64, recompute on, F16 KV, and the original model's mixed quantization rules in `Q4_0_XL.txt`. Rank, alpha, LR, data, and warmup follow the Flare curriculum. Pulsar uses `attn_q.weight` instead of the Flare fused QKV projection, and stage1 uses its attention output projection.

Override inputs/output with Flare's environment variables:

```bash
MODEL_PATH=/path/to/previous-merged.gguf \
OUTPUT_PATH=/home/user/pulsar-s-training/stage2-adapter.gguf \
bash stage2.sh
```

`TRAIN_FILE`, `QLION_BIN`, `QUANT_TYPE`, and `UBATCH` are also supported. These direct scripts retain all checkpoints; choose storage with enough space. The sequential runner retains its existing checkpoint limit.

To merge an adapter while preserving tensor types:

```bash
/home/user/aikar-engine/build/bin/llama-export-lora \
  -m Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf \
  --lora Lumen-3.1-Pulsar_S-stage1-Q4_0_XL-lora.gguf \
  -o /home/user/pulsar-s-training/Lumen-3.1-Pulsar_S-stage1-LD-Q4_0_XL.gguf \
  --type q4_0 --tensor-type-file Q4_0_XL.txt -t 36
```

## Attach a monitor to an existing log

```bash
LOG_PATH=/home/user/pulsar-s-training/20261001-100418/stage1/train.log \
bash start-monitor.sh
```

This creates only the WebUI window. It does not restart the trainer or change its SSH attachment.

## Fixes and validation

- Parse only `train:` progress; validation microbatches cannot change training progress or epoch.
- Count epoch boundaries once and convert local training/validation/checkpoint counters to global optimizer windows. Preserve native QAT global validation steps and initial resumed QLoRA evaluation.
- Keep metadata and epoch context when the log exceeds the tail limit. Track physical microbatch activity so a slow optimizer window is not falsely marked stalled.
- Preserve the first microbatch and all subsequent microbatch samples before downsampling; do not discard the first optimizer window. A single validation sample renders as a point without a fabricated line or area. Missing critical-token statistics display placeholders.
- Both separate and combined charts use the same x-axis from the run start to the current partial optimizer window. Y-axis bounds use only observed values inside that range; future points cannot flatten the current plots. Sparse Val points retain their real spacing; empty/single series render without invalid coordinates. Ordinary SFT exposes its train loss as unweighted NLL so the combined NLL graph remains available without critical-token statistics.
- Elapsed time includes completed epochs. Checkpoint identities include epoch, so repeated local checkpoint numbers do not collapse.

```bash
python3 -m unittest discover -s examples/qlora_training/webui -p 'test_*.py' -v
node examples/qlora_training/webui/test_charts.cjs
```

Eleven Python tests and the JavaScript coordinate checks pass. Script integration runs all four commands through a fake trainer, including paths with spaces, dollar signs, backticks and quotes. Actual Flare epoch2 logs and the live Pulsar HTTP endpoint were checked. No new GPU training was started for these changes.

Browser validation uses the existing Playwright package under `tools/ui/node_modules` and an installed headless browser. Run `node examples/qlora_training/webui/test_browser.cjs` against the running monitor. It checks live and multi-epoch plots on desktop/mobile in both chart modes, including the first point, Val positions, single-point areas, missing statistics, and runtime errors. Set `WEBUI_EPOCH2_FIXTURE` to replay an actual status JSON instead of the synthetic epoch2 fixture.
