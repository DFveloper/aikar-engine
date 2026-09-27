# QLion Q8_0 and mixed-weight training on a 16 GB GPU

Date: 2026-09-27
Status: design approved in chat; implementation pending review of this document

## Goal

Train all supported quantized weights in Q8_0-only, Q4_0+Q8_0, and MXFP4+Q8_0 GGUF models with `llama-finetune-qlion`. Keep the entire Flare model on CUDA0, retain an 8192-token context, and complete an optimizer update on the machine's V100 16 GB GPU. Provide a training script based on the MoXXf 2B `stage1.sh` settings.

## Existing behavior and constraints

`finetune_qat.cpp` accepts one unmixed Q4_0 or MXFP4 type. Parameter registration and QLion step kernels exclude Q8_0. The Flare Q4_0_L file has 35 Q8_0 tensors, and Q4_0_XL has 70. They are attention output weights, plus FFN down weights in XL. Nonquantized tensors remain frozen.

The previous Flare run reported about 10.1 GB for quantized weights and QLion states, 1.2 GiB for the CUDA scheduler buffer at `-ub 256`, and an estimated 4.9 GB Q8_0 gradient accumulator for `-c 8192 -b 8192 -ub 256`. The last allocation would leave too little room on a 16 GB GPU. The machine has more than 100 GiB available system RAM.

## Model format selection

Extend `--quant-type` to accept the exact sets `q8_0`, `q4_0,q8_0`, and `mxfp4,q8_0`, while preserving strict `q4_0` and `mxfp4`. Input inspection rejects every quantized tensor outside the selected set and requires at least one tensor of each selected type. The parameter filter selects each allowed quantized tensor by its own type. State serialization records the canonical selected set and validates it on resume. Checkpoints and final outputs retain each tensor's original GGUF type.

## QLion update paths

Keep Q8_0 momentum and Q4_0 residual for every trainable tensor. Add Q8_0 weight decode, update, requantization, and residual calculation to the existing CPU, CUDA, and Vulkan QLion operations. Cover dense, tied embedding, sparse embedding rows, routed MoE, and accumulated-gradient variants where each backend supports the corresponding existing Q4_0 or MXFP4 path. Extend the quantized backward dispatch for Q8_0 wherever it is currently restricted to Q4_0/MXFP4. Use existing type traits and quantization helpers where possible. No floating-point master copy is stored.

## Host gradient accumulation

Add an explicit `--qat-grad-accumulator cpu` option; the default remains `device`. Thread the choice through the trainer, llama optimizer parameters, and ggml optimizer context. With `cpu`, allocate only persistent Q8_0 gradient accumulators and their sparse-row bookkeeping in a CPU backend buffer. Keep model weights, Q8_0 momentum, Q4_0 residual, and the forward/backward graph on CUDA0. Let the scheduler transfer the current microbatch gradient to the CPU accumulation operation, then transfer the accumulated Q8_0 gradient for the optimizer step. Preserve the existing accumulation and update order. Check graph placement and transfer sizes during the first full update; avoid any full-model duplicate accumulator on CUDA. This mode can be slower because gradients cross the host/device boundary each microbatch.

## 16 GB training script

Add `/mnt/openwebui/AIKAR/Lumen-3.2-Flare/train-qlion-v100-16gb.sh`, based on `/mnt/openwebui/AIKAR/Lumen-3.2-MoXXf-2B/stage1.sh`. Default to the Flare Q4_0_L GGUF and the existing MoXXf training JSONL, with environment overrides for model, dataset, and output. Use CUDA0, all GPU layers, `--quant-type q4_0,q8_0`, `--qat-grad-accumulator cpu`, `--activation-recompute on`, FlashAttention, `-c 8192 -b 8192 -ub 256`, and the prior optimizer settings. Keep output names distinct from existing checkpoints. The script checks that the model and dataset exist and does not overwrite an existing output. Document how to select Q4_0_XL and MXFP4+Q8_0 inputs.

## Verification

Extend `examples/qlora_training/test-qat.cpp` and existing backend tests; add no test file. Compare Q8_0 updates with the reference block update for CPU, CUDA, and Vulkan, including F32 and Q8_0 accumulated gradients. Exercise the existing tied, row, and routed paths with Q8_0 where supported. Test strict format rejection, mixed input acceptance, and save/resume type preservation. Run the existing QAT and argument parser suites.

On the V100, run a short 8192-token training sample through at least one complete optimizer update with all model layers on CUDA0. Confirm that a Q4_0 tensor and a Q8_0 tensor both change, inspect the output and resumed state, and measure peak GPU memory. Accept the script only if the measured peak is below the GPU's usable 16 GB and no allocation failure occurs. If `-ub 256` exceeds the limit, reduce `-ub` while retaining `-c 8192 -b 8192` and repeat the measurement.

## Boundaries

The change supports the three requested weight combinations. It does not train F32, F16, or BF16 tensors. It does not convert quantized weights to a different GGUF type. It does not submit a PR, commit, or start a full training run.
