# Runpod A100 recovery container

Build from this working tree at the repository root:

```bash
docker build -f scripts/logit-distillation/runpod/Dockerfile -t aikar-distill:a100 .
```

The build targets A100 compute capability 8.0 explicitly and does not need a GPU during image construction. Flash attention is disabled in this initial training image to use the existing differentiable attention graph. CUDA performs KL forward/backward, QLion2 updates and frozen Q4_1 MoE activation gradients. Lumen FP4 KV fake quantization runs on CUDA; it uses software E2M1 quantization and does not require Blackwell FP4 hardware.

Use an A100 80 GB first, with at least 64 GB host RAM and a persistent volume with 300 GB or more free for the current checkpoint retention policy plus source models. A100 40 GB may require a smaller TRAIN_UBATCH and is not yet verified. A smaller microbatch detaches past-KV gradients in the existing trainer. Teacher extraction and student training run sequentially, so both weight sets need not reside in VRAM together. No cloud resources are provisioned by these scripts.

Persistent volume layout:

```text
/workspace/train_ko.jsonl
/workspace/train.jsonl
/workspace/Lumen-3.8-Pulsar/Test.gguf
/workspace/Lumen-3.8-Pulsar/Lumen-3.8-Pulsar-base.gguf
```

The Korean source is reduced to its first user/assistant pair per conversation, preserving the first assistant's reasoning. The original file stays intact. It is mixed with an equal number of sampled English conversations using seed 42. Targets remain attention in blocks 11, 17, 23 and 29, with no ground-truth CE. Shared KV means only Q/output weights have effective updates in these blocks.

Local NVIDIA Container Toolkit examples:

```bash
docker run --rm --gpus all --ipc=host -v /path/to/persistent-data:/workspace aikar-distill:a100 preflight
docker run --rm --gpus all --ipc=host -v /path/to/persistent-data:/workspace aikar-distill:a100 plan
docker run --rm --gpus all --ipc=host -v /path/to/persistent-data:/workspace aikar-distill:a100 benchmark
docker run --rm --gpus all --ipc=host -v /path/to/persistent-data:/workspace aikar-distill:a100 extract
docker run --rm --gpus all --ipc=host -v /path/to/persistent-data:/workspace -e RUN_ID=a100-recovery-01 aikar-distill:a100 train
```

For Runpod, publish the image to your own registry, select it as the Pod image, attach the persistent volume at /workspace, and choose the command `preflight`, `plan`, `benchmark`, `extract`, `train` or `all`. Run preflight and benchmark before starting a full recovery job. Model files and data are volume inputs and are not baked into the image. Set RUN_ID explicitly for a resumable run. Existing production.env settings can be overridden with environment variables, including PROD_DIR, EPOCHS, SAVE_EVERY, CTX and TRAIN_UBATCH. Do not change data/packing settings between extraction and training.

To resume, retain the original RUN_ID and set RESUME to a full checkpoint path under /workspace, including its sibling .qat-state.gguf file. All commands/logs/manifests remain on the volume. The input defaults use a new mixed-train-ko-first-1to1.jsonl name so old full-multiturn caches are rejected by the manifest checks. Existing Korean-multiturn throughput and dataset plans are historical and must be recalculated with `plan`.

No A100 throughput estimate is claimed from CPU or P620 measurements. An actual A100 run is required to establish speed, memory use and recovery quality.

See VALIDATION.md for the measured CPU/P620 results, supported CUDA cases and the broader QAT regression failure. When copying a whole local model directory to a new machine, set MIXED_DATA to a fresh output path if the copied mixture manifest refers to old source paths; source identity checks deliberately reject that stale manifest.
