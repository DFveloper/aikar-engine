# export-lora

Apply LORA adapters to base model and export the resulting model.

```
usage: llama-export-lora [options]

options:
  -m,    --model FNAME                  model path from which to load base model
         --lora FNAME                   path to LoRA adapter (use comma-separated values to load multiple adapters)
         --lora-scaled FNAME:SCALE,...  path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)
         --tensor-type REGEX=TYPE       output type for matching merged tensors (first match wins)
         --tensor-type-file FILE        whitespace-separated REGEX=TYPE entries (first match wins)
  -o,    --output, --output-file FNAME  output file (default: 'ggml-lora-merged-f16.gguf')
```

For example:

```bash
./bin/llama-export-lora \
    -m open-llama-3b-v2.gguf \
    -o open-llama-3b-v2-english2tokipona-chat.gguf \
    --lora lora-open-llama-3b-v2-english2tokipona-chat-LATEST.gguf
```

Multiple LORA adapters can be applied by passing comma-separated values to `--lora FNAME` or `--lora-scaled FNAME:SCALE,...`:

```bash
./bin/llama-export-lora \
    -m your_base_model.gguf \
    -o your_merged_model.gguf \
    --lora-scaled lora_task_A.gguf:0.5,lora_task_B.gguf:0.5
```

Each adapter is applied only to base tensors for which it contains both the `lora_a` and `lora_b` tensors. If only one tensor in a pair exists, the export fails.

Use `--tensor-type` or `--tensor-type-file` to override the output type of tensors changed by a LoRA adapter. Tensors copied unchanged from the base model keep their original data and type.

```bash
./bin/llama-export-lora \
    -m base-model.gguf \
    -o merged-model.gguf \
    --lora lora-a.gguf,lora-b.gguf \
    --type q4_0 \
    --tensor-type 'token_embd.weight=Q8_0' \
    --tensor-type-file tensor-types.txt
```
