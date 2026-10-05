# Pulsar S CUDA Training Optimization Design

## Goal

Increase Lumen 3.1 Pulsar S QLoRA training throughput on the Tesla V100 while reducing temporary GPU memory, without changing numerical results beyond the existing CUDA tolerance.

## Scope

The change targets the CUDA optimizer graph used by `llama-finetune-qlora`. It does not change dataset semantics, LoRA rank, training schedule, or window stride.

## Approach

1. Extend the existing CUDA `OUT_PROD` fusion so a gradient outer-product followed by an in-place F32 accumulation uses one kernel and does not materialize an intermediate result.
2. Reuse existing CUDA MoE routed-backward paths and add only measured temporary-buffer or duplicate-atomic reductions where the operation contract permits it.
3. Keep all fast paths guarded by exact shape, type, contiguity, and alias checks; unsupported graphs use the existing implementation.
4. Validate with backend numerical tests, QAT tests, a short CUDA training comparison, and V100 throughput/VRAM measurements.

## Success Criteria

- Existing CPU and CUDA tests remain green.
- A 256-token CUDA training comparison has matching loss and adapter tensors within the repository's established tolerance.
- The real V100 benchmark shows higher tokens/s or lower peak VRAM; changes with neither benefit are rejected.
