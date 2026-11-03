# Lumen 3.8 Pulsar Next

The experimental architecture name is Lumen-4-Pulsar-exp. Its GGUF architecture is `lumen4exp`. It reuses the Gemma 4 graph, tensor names, tokenizer, RoPE and MoE computations.

## Convert a Gemma 4 GGUF

```sh
python3 scripts/convert_gemma4_to_lumen4exp.py gemma4.gguf lumen4exp.gguf
```

The converter preserves weight payloads and changes the architecture metadata. It does not train or distill the model. Conversion changes the model's attention behavior, so preserving weights does not preserve its logits or establish its quality.

The default `lumen4exp.attention.shared_kv_layers` is 13, interpreted as the trailing 13 transformer layers. Only global attention layers in this range reuse KV. For a 30-layer model with global layers at zero-based indices 5, 11, 17, 23 and 29, the KV sources are 5, 11, 11, 11 and 11. Every SWA layer retains its own KV projections and cache. `--shared-layers 0` disables sharing.

A source checkpoint must contain independent K projection and K normalization weights for all SWA layers. The converter rejects checkpoints whose original sharing removed these weights. Merge sharded GGUFs before conversion. Existing output files are never overwritten.

HF checkpoints can use `Lumen4ExpForCausalLM` or `Lumen4ExpForConditionalGeneration` in their `architectures` configuration and the existing HF converter.

## NVFP4 with an FP32 master scale

The existing `NVFP4` name and type number 40 are retained. Each row contains an FP32 decode scale followed by the existing E2M1 values and E4M3 scales for groups of 16 elements:

```text
master_scale = max(abs(row)) / (448 * 6)
block_scale  = E4M3(max(abs(block / master_scale)) / 6)
value        = E2M1_value * block_scale * master_scale
```

Zero rows use a master scale of 1. Very small scales are clamped to a representable positive FP32 value. A row of 512 values occupies `4 + 8 * 36 = 292` bytes. The existing 64-element block payload is unchanged; the four-byte master scale belongs to the whole row.

Lumen applies NVFP4 only to global K and V, after normalization and K RoPE. Each token's KV head vector is a row with its own scale, so appending tokens does not change previously cached rows. SWA cache types retain their existing configuration. Attention reads dequantized values.

This follows the two-level scale formula in the [NVIDIA NVFP4 reference](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html). [DeepSeek V4.1 Flash section 2.4.4](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS4) uses the same 16-channel block format after RoPE but explicitly omits the FP32 master scale. Lumen retains it.

The changed layout uses quantization version 3. The converter upgrades older NVFP4 weight rows by prepending an FP32 scale of 1, preserving their original block bytes and decoded values. Old block-only NVFP4 weights must be converted before loading with this engine.

The scale-aware codec and global cache currently execute on CPU. CUDA, Vulkan, SYCL and WebGPU decline operations involving this NVFP4 storage layout, allowing the scheduler to use CPU. Their existing block-only kernels cannot interpret the new row header. Native GPU support and training are separate work.

## Validation

```sh
PYTHONPATH=gguf-py python3 gguf-py/tests/test_lumen4exp.py -v
build/bin/test-quantize-fns
build/bin/test-backend-ops -b CPU -o CPY,SET_ROWS -p nvfp4
PYTHONPATH=gguf-py python3 gguf-py/tests/test_quants.py --libggml build/bin/libggml-base.so --type nvfp4 --quick
```
