# Runpod recovery with JupyterLab

The image uses official Runpod Ubuntu 24.04 with JupyterLab and SSH. CUDA 12.8 targets A100 (sm_80) and RTX PRO 6000 Blackwell (sm_120). Native QLion2 training does not require PyTorch. Full GPU training remains unverified locally; see VALIDATION.md.

```bash
docker build -f scripts/logit-distillation/runpod/Dockerfile -t aikar-distill:recovery-jupyter .
```

## Pod settings

- Public image: ghcr.io/dfveloper/aikar-distill:recovery-jupyter. No registry credential is required.
- Start command: /start.sh. Expose HTTP port 8888 and enable Start Jupyter Notebook.
- A100 80 GB or RTX PRO 6000 Blackwell 96 GB, at least 64 GB host RAM.
- Container disk: 30 GB. Persistent volume: at least 300 GB free, mounted at /workspace.
- Set HF_TOKEN as a runtime secret with repository write permission.

Input models:

```text
/workspace/Lumen-3.8-Pulsar/Test.gguf
/workspace/Lumen-3.8-Pulsar/Lumen-3.8-Pulsar-base.gguf
```

The local development server's model files are not present on a new Pod. Download the public input bundle before benchmarking:

```bash
distill fetch-models --repo dfveloper/aikar-distill --tag lumen-models-20261104 --output /workspace/Lumen-3.8-Pulsar
```

The input bundle is a GHCR artifact, not a runnable image. The downloader verifies SHA-256 for each chunk and the complete files, and resumes verified chunks after interruption. It requires no registry login. Keep about 28 GB free for the input models in addition to training storage.

Data, teacher cache, checkpoints, HF cache and benchmark temporary files are stored under /workspace. A Network Volume survives Pod termination; a regular Volume Disk does not.

## Jupyter startup and repair

The /start.sh wrapper keeps Runpod's original startup and supplies a token when JUPYTER_PASSWORD is unset. The random token is saved at /run/aikar-jupyter/token with mode 600 and reused after container restart. A recreated Pod gets a new token unless JUPYTER_PASSWORD is set. Jupyter config, runtime and logs use container-local /run/aikar-jupyter because Runpod network volumes may reject chmod. Models, training data and checkpoints stay in /workspace. Read this file from the Pod terminal to log in. Set JUPYTER_PASSWORD to provide your own token. No shared password is baked into the image. JUPYTER_DISABLE_AUTH=true is honored only when explicitly set.

```bash
distill jupyter-status
distill jupyter-start
```

The second command runs Jupyter in the foreground and fails on a port conflict instead of choosing 8889. Ensure the Pod exposes HTTP port 8888. The original startup log remains /jupyter.log. A custom command that bypasses /start.sh also bypasses automatic startup.

## Korean knowledge and reasoning

The default samples up to 6000 reasoning rows from [Korean reasoning 12k filtered v2](https://huggingface.co/datasets/Jongsim/claude-opus-4.6-reasoning-12k-ko-filtered-v2) (Apache 2.0) and 4000 knowledge/explanation QA rows from [KoCommercial-Dataset](https://huggingface.co/datasets/MarkrAI/KoCommercial-Dataset) (MIT). Only pinned train splits are used. Questions, final answers and reasoning must contain Korean. JSON-string messages are parsed once, initial system/first user/assistant are preserved, and reasoning is stored as reasoning_content. Empty/incomplete answers, summary/translation/story requests and duplicate normalized prompts are filtered. These filters do not independently verify factual correctness.

Preparation uses seed 42 and reservoir sampling. Revisions, licenses and policy are saved beside /workspace/ko-knowledge-reasoning.jsonl and uploaded with the cache. KO_REASONING_ROWS, KO_KNOWLEDGE_ROWS and KO_DATA override counts and output. Changed settings require a new output name. DATA selects an existing JSONL. DATA_MODE=mixed restores the historical Korean/English 1:1 mixture with Korean first-turn truncation.

## About ten hours

Start with context=microbatch=1024 and accumulation=1. Smaller microbatches detach past-KV gradients. Benchmark each GPU separately; VRAM size alone does not determine speed.

```bash
distill preflight
distill prepare-ko
BENCH_WINDOWS=8 distill benchmark
# Replace <timestamp> with the printed benchmark directory.
distill budget --bench-dir /workspace/Lumen-3.8-Pulsar/logit-distillation/benchmark-<timestamp> --hours 10 --output /workspace/recovery-budget.env
source /workspace/recovery-budget.env
distill plan
distill extract
RUN_ID=recovery-01 distill train
```

The budget tool discounts measured extraction/training rates by 20%, reserves two hours for setup, validation, saves and uploads, caps worst-case cache size at 64 GiB and retains at most nine checkpoint pairs by selecting SAVE_EVERY. The environment file includes matching DATA/context/microbatch settings, PROD_MAX_WINDOWS and a fresh PROD_DIR. Use it for both stages. Plan reports exact packed counts and storage; the corpus can contain fewer windows than the cap.

This is an estimate, not an automatic wall-time stop. Slow networking or preparation can exceed the reservation. Rebenchmark after switching GPU, data or context. Use tmux for long commands.

Teacher extraction and training run sequentially. The loss remains pure top-512 plus tail forward KL, with no ground-truth CE. Only attention in blocks 11, 17, 23 and 29 is selected. Shared KV means Q/output weights have effective updates; block 5 and FFN stay frozen.

## HF publication and resume

HF_UPLOAD=1 is the production default. HF_TOKEN is read at runtime; whoami resolves its account namespace. No token or account name is baked into the uploader. Bare names use the token owner's namespace; explicit namespace/repository values are accepted:

```text
HF_DATASET_REPO=lumen-teacher-logits
HF_MODEL_REPO=lumen-recovery-checkpoints
```

Repositories are created public; existing private targets are rejected. HF_UPLOAD=0 disables publishing. Plan and benchmark do not publish. After extraction, the dataset repository receives caches/<cache-sha256>/teacher.dat, train.jsonl, cache.json and optional provenance.json after checksum checks.

At each SAVE_EVERY checkpoint the trainer saves the local full model/state and a QLion adapter. The completed adapter is detected within about one second and queued for background upload while training continues. All adapter checkpoints use the single HF_MODEL_REPO under runs/<run-id>/<checkpoint-name>/ with adapter.gguf, model.gguf.qat-state.gguf and run.json, added in one Hub commit. The full model stays local. A final-adapter save event also triggers upload immediately, before waiting for process cleanup. Uploads are serialized; a slow earlier upload can delay the next one. Failed uploads retry, unchanged receipts are skipped, and local files are retained:

```bash
distill upload-dataset
distill upload-checkpoints --run-dir "$PROD_DIR/runs/recovery-01"
```

Logit publication is unchanged: the complete extraction cache is uploaded after extraction finishes. Extraction resumes complete windows from .partial files. Local training resume requires the original RUN_ID, unchanged settings and RESUME pointing to the full-model checkpoint with its sibling .qat-state.gguf. Each exported adapter embeds its exact base hash; after resume its base is the resume checkpoint, so retain earlier adapters/base checkpoints when rebuilding from HF. These QLion adapters are quantized tensor updates, not LoRA. Separate experiments use distinct RUN_ID values within the same model repository.
