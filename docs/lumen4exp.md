# Lumen 3.8 Pulsar Next

The experimental architecture name is Lumen-4-Pulsar-exp. Its GGUF architecture is `lumen4exp`. It reuses the Gemma 4 graph, tensor names, tokenizer, RoPE and MoE computations.

## Convert a Gemma 4 GGUF

```sh
python3 scripts/convert_gemma4_to_lumen4exp.py gemma4.gguf lumen4exp.gguf
```

The converter preserves weight payloads and changes the architecture metadata. It does not train or distill the model. Conversion changes the model's attention behavior, so preserving weights does not preserve its logits or establish its quality.

Conversion also configures YaRN for a 1,048,576-token (1M) context by default. It records `lumen4exp.context_length`, `rope.scaling.type = yarn`, the original context length, and `factor = target / original`. A 131,072-token source gets a factor of 8. An already scaled source uses its `rope.scaling.original_context_length` instead of applying scaling twice. Previous global scaling settings are replaced; SWA window and frequency metadata are preserved. Set `--context-length N` to choose another target at least as large as the original context.

The model loads these YaRN settings without runtime RoPE flags. Use `-c 1048576` to allocate the full context at inference. The metadata marks the extension as not finetuned: conversion alone does not validate 1M-context quality. Distillation or long-context training and evaluation are still required.

The default `lumen4exp.attention.shared_kv_layers` is 19, interpreted as the trailing 19 transformer layers. Only global attention layers in this range reuse KV. For the 30-layer model, layer 5 is the only global KV owner; global layers 11, 17, 23 and 29 all reuse layer 5. All indices are zero-based. Every SWA layer retains its own KV projections and cache. `--shared-layers 0` disables sharing. The temporary independent-layer-17 override has been removed.

A source checkpoint must contain independent K projection and K normalization weights for all SWA layers. The converter rejects checkpoints whose original sharing removed these weights. Merge sharded GGUFs before conversion. Existing output files are never overwritten.

HF checkpoints can use `Lumen4ExpForCausalLM` or `Lumen4ExpForConditionalGeneration` in their `architectures` configuration and the existing HF converter.

## Default FP4 KV cache

Lumen4Exp stores global K and V in runtime-only `fp4_kv`: E2M1 values with one E4M3 scale for each group of 16 channels, after normalization and K RoPE. There is no FP32 row or tensor master scale. This uses the per-16 format described by [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). It does not import DeepSeek's attention architecture or cache sharing rules.

```text
block_scale = E4M3(max(abs(group_of_16)) / 6)
value       = E2M1_value * block_scale
```

The number of independent global cache layers comes from model metadata and sharing rules; it is not fixed at two. A 512-channel head row occupies `8 * 36 = 288` bytes. With two KV heads, each independent global layer uses about 1.073 GiB at 1,000,000 tokens. In the earlier two-layer test configuration, two independent global layers and two KV heads require `1000000 * 2 * 2 * 2 * 288 = 2304000000` bytes (2.146 GiB). At the aligned 1,000,192 cells, the global cache uses 2,197.69 MiB. SWA keeps its configured cache types and adds 900 MiB in the four-slot test configuration. Appending rows does not change existing groups or require full-cache requantization.

The default is selected automatically for `lumen4exp`; no cache-type flags or checkpoint conversion are needed. Runtime type 150 is separate from GGUF NVFP4 weight type 40. Existing NVFP4 weight codecs and conversion remain compatible. Old saved KV states are rejected by their different cache type.

CPU cache writes, dequantization and direct Flash Attention support this format. CUDA supports packed cache writes and GPU dequantization before F16 Flash Attention, including on V100; no native FP4 tensor-core instructions are required. FP4 cache rows span the KV heads so tensor parallelism can split the head channels while sharing token indices. Unsupported backends keep the packed cache on CPU. Small-model FP4 KV inference passed on two V100s. Real-model validation is pending after a repeated PCIe/NVLink device failure; see the V100 guide for kernel logs. E4M3 scales have finite range: extremely small values can underflow, and very large values can saturate; there is no master scale to extend that range.

The existing training path uses the same per-16 FP4 quantizer in forward and STE gradients in backward. It currently restores FP32 values for the differentiable attention graph; this is not a complete FP4 backward kernel or a measured reduction in training activation memory.

See [attention profiling](lumen4exp-nvfp4-cpu-attention.md) for the preceding optimization and its historical NVFP4 measurements.

## Validation

```sh
PYTHONPATH=gguf-py python3 gguf-py/tests/test_lumen4exp.py -v
build/bin/test-quantize-fns
build/bin/test-backend-ops test -b CPU -o CPY,SET_ROWS,FLASH_ATTN_EXT -p fp4_kv
PYTHONPATH=gguf-py python3 gguf-py/tests/test_quants.py --libggml build/bin/libggml-base.so --type nvfp4 --quick
```

The 2026-11-04 implementation passed 66 CPU backend cases for cache writes, copies and attention, the quantization suite, scalar fallback verification and six converter tests. A real server retained the 1M context and four slots, reported global `fp4_kv` K/V totaling 2,197.69 MiB plus 900 MiB SWA, and completed a 652-token prompt with eight generated tokens. Compute reservation was 1,085.79 MiB. This checks execution and storage, not trained model quality or 1M-token active-context throughput. No production server was restarted. Session validation artifacts are `build/fp4-kv-*.log` and `build/fp4-kv-smoke.json`.

An additional strict sequence-state replay check returned successfully from the save/restore APIs but did not produce bitwise-identical next-token logits. Numerical error and cache-placement effects were not measured before the original checkpoint was replaced; no bitwise replay guarantee is claimed. Further model checks were stopped to avoid interfering with the user's independent KV-layer experiments. The successful server and backend results above refer to the earlier Lumen4Exp checkpoint.

## Two V100 recovery workflow

See [the V100 launcher](../scripts/logit-distillation/V100.md) for tensor-parallel inference, downloaded teacher logits and attention-only recovery. Global KV remains FP4; the training launcher selects F32 for SWA caches to support differentiable cache writes.
