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

Evo writes candidates tensor-first: each source tensor is read and decoded once
per generation, then all candidates are merged, quantized, and written in
parallel within the memory budget. Quantized output types use the same
per-tensor policy as `llama-quantize`, including mixed K-quant rules and shape
fallbacks. Integer and other non-float tensors are copied byte-for-byte after
the inputs are checked for equality. Candidate generation reports completed
GB, average GB/min, elapsed time, and ETA while it writes the output tensors.
The best evaluated candidate is promoted directly to the final output instead
of being merged and quantized a second time.

Fitness uses sparse cross-entropy in the llama compute graph. On a CUDA build,
the loss stays on the GPU and only one scalar is copied to the host per batch.
Prompt tokens in `prompt`/`response` and `messages` JSONL records are masked;
only response tokens affect fitness.

For multiple GPUs, choose one of these modes:

- `--devices CUDA0,CUDA1,...` loads one candidate on each GPU and evaluates
  candidates concurrently. Use this when one candidate fits on one GPU.
- Omit `--device` and `--devices` to let llama.cpp split one candidate across
  all visible GPUs. Use this for a candidate that does not fit on one GPU.

`--device` and `--devices` are mutually exclusive. `--seed 0` is a valid,
deterministic seed. Temporary candidates default to the system temporary
directory; set `--temp-dir` to fast local storage. The directory needs free
space for roughly `population * candidate GGUF size` during each generation.
Later generations can temporarily need one additional candidate file while the
best result from the previous generation is retained. Before each generation,
the merger checks free space and keeps a 5% or 2 GB minimum reserve. It reports
the required and available space before writing if the temp directory is too
small. Do not use a RAM-backed `/tmp` for large models.
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
worker budget from available RAM, enables GPU merge, and stores generation
candidates under local `/tmp`. Its default `EVO_DEVICE_MODE=replica` evaluates
one candidate per GPU. Set `EVO_DEVICE_MODE=split` when the model needs combined
VRAM:

```sh
EVO_DEVICE_MODE=split POPULATION=8 GENERATIONS=20 \
    tools/model-merge/runpod-evo.sh build BASE OUT CALIB q4_k MODEL_A MODEL_B
```

Useful overrides are `THREADS`, `MEMORY_BUDGET`, `POPULATION`, `GENERATIONS`,
`ELITE_COUNT`, `CTX_SIZE`, `GPU_LAYERS`, and `SEED`.

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
threads = 32
memory_budget = 128G
temp_dir = /tmp/llama-merge-evo
ignore_chat_template = true
```
