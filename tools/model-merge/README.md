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
use `--device Vulkan0`; omit `--device` to use the default backend selection.
`--merge-gpu` offloads Evo weighted accumulation to that device. Candidate
quantization and GGUF I/O remain on the CPU.
GPU merge currently applies to Evo only. TIES trimming and sign consensus use
the CPU implementation.

Calibration files may be plain text or JSONL. JSONL accepts the same basic
schemas as the QLoRA trainer: `messages`, `prompt` plus `response`, or `text`.

The Evo target types are experimental: `q4_0`, `q3_k`, `q4_k`, and `mxfp4`.
The optimizer is a separable CMA-ES over per-tensor model weights. Fitness is
mean token negative log-likelihood measured after candidate quantization, so
quantization error participates directly in selection.

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
merge_gpu = true
ctx_size = 512
threads = 32
memory_budget = 128G
```
