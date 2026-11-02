# llama-export-qlion

Merge quantized QLion tensor replacements without dequantization or requantization:

```sh
llama-export-qlion -m base.gguf --adapter first.gguf --adapter second.gguf -o merged.gguf
```

The tool supports v1 full tensors and v2 rows/experts. Every replacement must match the base architecture, name, type, shape, and original SHA-256. Payload hashes are checked before use and again during copying. Overlapping byte ranges are rejected, including identical replacements. Input paths, symlinks, and hard links cannot be used as the output. Split GGUF is unsupported. The tool copies the base file and changes only validated tensor ranges; metadata, padding, and untouched tensors are preserved. Failed writes leave an existing output intact.

`--allow-compatible-base` permits a different whole-file hash only after all original replacement hashes match. This allows metadata changes but does not prove equality of untouched tensors. The tool warns in this mode. Whole-file hashes are computed directly without a persistent cache.

Adapters are inference artifacts, not optimizer checkpoints. Resume training with the existing model checkpoint plus `.qat-state.gguf`.

Runtime loading through common tools is available:

```sh
llama-completion -m base.gguf --qlion-adapter first.gguf --qlion-adapter second.gguf -p hello -n 16
```

The core model loader validates all adapters and applies replacements before backend upload or repacking. mmap uses a private writable mapping, so replacement pages are copied while the base file remains unchanged. Lazy tensors retain their mapping and see the same overlay. Resident tensors are assembled before backend upload. Tied tensor copies use the same patched source bytes. The mapping owns its pages for the model's lifetime; adapter files are only needed during loading.

Applications calling the core API can set a null-terminated list before loading:

```cpp
const char * adapters[] = { "first.gguf", "second.gguf", nullptr };
auto params = llama_model_default_params();
params.qlion_adapters = adapters;
params.qlion_allow_compatible_base = false;
auto * model = llama_model_load_from_file("base.gguf", params);
```

The list and strings must live until the load call returns. Rebuild consumers when using the extended model parameter structure. `--qlion-allow-compatible-base` enables the same explicit compatibility path as the exporter. Split models, vocabulary-only loading, file-handle/embedded model loading, and the independent CPU LES disk cache are unsupported. Failed validation returns a model loading error and discards private mappings; it does not write to the base model.
