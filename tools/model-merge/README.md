# llama-merge

`llama-merge` merges compatible GGUF models. TIES preserves a shared quant type,
while Evo evaluates quantized candidates on calibration data and evolves a
per-tensor distribution over all input models.

## TIES

```sh
llama-merge --base base.gguf \
    -m model-a.gguf -m model-b.gguf \
    -o merged.gguf --method ties --density 0.5 \
    -t 16 --memory-budget 16G
```

## Quant-aware evolutionary merge

```sh
llama-merge --base base.gguf \
    -m model-a.gguf -m model-b.gguf -m model-c.gguf \
    -o evolved-q4-k.gguf --method evo \
    --calibration calibration.jsonl --target-type q4_k \
    --population 8 --generations 10 --elite-count 2 \
    --sigma0 0.10 --seed 42 \
    --gpu-layers -1 --device CUDA0 --merge-gpu \
    -t 32 --memory-budget 128G
```

Use a device name reported by a llama.cpp executable. A Vulkan-only build can
use `--device Vulkan0`. `--merge-gpu` offloads Evo weighted accumulation to the
first selected accelerator. Candidate quantization and GGUF I/O remain on the
CPU. GPU merge currently applies to Evo only. TIES trimming and sign consensus
use the CPU implementation.

Evo creates, evaluates, and releases one complete candidate at a time. Quantized
output types use the same per-tensor policy as `llama-quantize`, including mixed
K-quant rules and shape fallbacks. Integer and other non-float tensors are copied
byte-for-byte after the inputs are checked for equality. Candidate generation
reports completed GB, average GB/min, elapsed time, and ETA.

`--evo-mode` selects where source and candidate weights live:

- `low-ram`: read sources from storage, write one candidate to `--temp-dir`,
  evaluate it, delete it, and regenerate the winner to the final output.
- `ssd-rich`: read sources from storage and write candidates one at a time to an
  SSD `--temp-dir`. Keep only the best evaluated candidate file and promote it
  to the final output without regenerating it.
- `ram-rich`: cache every input GGUF in host RAM, write one candidate at a time
  to a RAM-backed `--temp-dir`, evaluate and delete it, then regenerate the
  winner to persistent storage. The full-size preset expects about 180 GB RAM.
- `normal`: read input GGUFs from SSD, CPU-merge one candidate to a RAM-backed
  `--temp-dir`, load it for GPU fitness, delete it, and regenerate the winner to
  persistent storage.

`normal` and `ram-rich` require an explicit `--temp-dir` and use CPU merge.
`low-ram` and `ssd-rich` may use `--merge-gpu`. All four modes accept
`--temp-dir`; if it is omitted in a disk mode, the system temporary directory is
used. All modes evaluate one candidate at a time and accept at most one explicit
fitness device.

Fitness keeps the selected `--ctx-size` but processes it in smaller logical and
physical batches. `--eval-batch` defaults to 128 and `--eval-ubatch` defaults to
64, which leaves graph memory for a 13.8 GiB model on a 16 GiB V100. Lower these
values if another process uses the GPU; this changes peak memory and throughput,
not the selected evaluation context window.

The `/mnt/openwebui/AIKAR/evo/merge.sh` V100 recipe uses batch 256 and ubatch
128. On the 26B-A4B Q4_0 candidate this keeps all 31 layers on the V100 and uses
about 14.4 GiB at `--ctx-size 1024`. GPU merge memory is released before the
fitness model is loaded. Set `MERGE_GPU=0` only when CPU merge is preferred.

Fitness uses sparse cross-entropy in the llama compute graph. On a CUDA build,
the loss stays on the GPU and only one scalar is copied to the host per batch.
Prompt tokens in `prompt`/`response` and `messages` JSONL records are masked;
only response tokens affect fitness.

Use `--device CUDA0` when one candidate fits on one GPU. Omit `--device` and
`--devices` to let llama.cpp split one candidate across all visible GPUs. The
four candidate-at-a-time modes do not evaluate replicas concurrently.

`--device` and `--devices` are mutually exclusive. `--seed 0` is a valid,
deterministic seed. The temporary directory needs room for one candidate, plus
the retained winner in `ssd-rich` mode. Before each candidate, the merger checks
free space and keeps a 5% or 2 GB minimum reserve. It reports required and
available space if the directory is too small.
If compatible checkpoints differ only in `tokenizer.chat_template`, pass
`--ignore-chat-template`. Tokenizer vocabulary and all other architecture
metadata remain strictly checked. `messages` calibration records then bypass
Jinja and use neutral `role:\ncontent` serialization, with only the last
assistant payload contributing to loss.

Calibration files may be plain text or JSONL. JSONL accepts the same basic
schemas as the QLoRA trainer: `messages`, `prompt` plus `response`, or `text`.

The Evo target types are experimental: `q4_0`, `q3_k`, `q4_k`, and `mxfp4`.
The optimizer is a separable CMA-ES over per-tensor model weights. Fitness is
mean token negative log-likelihood measured after candidate quantization, so
quantization error participates directly in selection.

## RunPod preset

Build with CUDA and place final models/calibration data on persistent storage:

```sh
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-merge -j"$(nproc)"

tools/model-merge/runpod-evo.sh build \
    /workspace/base.gguf /workspace/evolved-q4-k.gguf \
    /workspace/calibration.jsonl q4_k \
    /workspace/model-a.gguf /workspace/model-b.gguf
```

The wrapper discovers visible NVIDIA GPUs, uses all CPU cores, sizes the merge
worker budget from available RAM, and stores each candidate under local `/tmp`.
Its default `EVO_MODE=low-ram` uses GPU merge and `EVO_DEVICE_MODE=single` uses
`CUDA0`. Set `EVO_DEVICE_MODE=split` when the model needs combined VRAM:

```sh
EVO_DEVICE_MODE=split POPULATION=8 GENERATIONS=20 \
    tools/model-merge/runpod-evo.sh build BASE OUT CALIB q4_k MODEL_A MODEL_B
```

Useful overrides are `EVO_MODE`, `EVO_TMPDIR`, `THREADS`, `MEMORY_BUDGET`,
`POPULATION`, `GENERATIONS`, `ELITE_COUNT`, `CTX_SIZE`, `EVAL_BATCH`,
`EVAL_UBATCH`, `GPU_LAYERS`, and `SEED`.

Keep source/final GGUFs on `/workspace`, but keep `--temp-dir` on local storage.
RunPod documents typical standard network-volume throughput as 200-400 MB/s,
so candidate churn on the network volume can dominate runtime. See the current
[RunPod network-volume documentation](https://docs.runpod.io/storage/network-volumes)
and [Pod GPU selection guide](https://docs.runpod.io/pods/choose-a-pod).

## INI configuration

```ini
base = base.gguf
models = model-a.gguf, model-b.gguf, model-c.gguf
output = evolved-q4-k.gguf
method = evo
calibration = calibration.jsonl
target_type = q4_k
population = 8
generations = 10
elite_count = 2
sigma0 = 0.10
seed = 42
gpu_layers = -1
device = CUDA0
# Or: devices = CUDA0, CUDA1, CUDA2, CUDA3
merge_gpu = true
ctx_size = 512
eval_batch = 128
eval_ubatch = 64
threads = 32
memory_budget = 128G
temp_dir = /tmp/llama-merge-evo
ignore_chat_template = true
evo_mode = low-ram
```
