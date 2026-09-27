# Pulsar S F16 and Q8_KV Fusion Design

## Objective

Reduce Lumen 3.1 Pulsar S decode kernel count on CUDA, especially Volta, without changing model weights, routing semantics, KV formats, sampling, or multi-token expert execution.

## Scope

- Add fused top-k routing for exactly 116 experts and top-8 selection through the existing CUDA top-k MoE kernel.
- Fuse the Gemma4 router sequence `RMS_NORM -> SCALE -> MUL` into one CUDA kernel.
- Fuse the Gemma4 V-cache sequence `RMS_NORM -> VIEW -> SET_ROWS` for F16 and Q8_KV destinations.
- Reuse existing graph-fusion and backend-test infrastructure.

Q4_KV, Turbo KV changes, backend sampling, shared three-way RMSNorm, Q/K/V activation quantization reuse, managed-memory policy changes, and multi-token expert down/reduction are excluded.

## 116-Expert Top-K

The CUDA top-k kernel must allocate register slots with ceiling division because 116 is not divisible by the 32-thread warp size. Only 116 is added to the non-power-of-two eligibility list. Existing expert counts and routing operations keep their current behavior.

Tests cover 116 experts, top-8, one decode row, and the existing gating and normalization variants.

## Router Preprocessing

Match only a direct F32 `RMS_NORM -> SCALE -> MUL` chain. The scale bias must be zero, the multiply must consume the scale result, tensors must have compatible shapes and contiguous rows, and the final output must pass the existing fusion memory-range check.

The fused kernel preserves operation order as `(rms_scale * x) * scalar * vector_scale`. The existing RMSNorm and RMSNorm-plus-MUL paths continue to use a scalar of one.

Tests use the Pulsar hidden size 2816 for one and four rows and compare the whole graph against the CPU backend.

## V Norm and Cache Write

Match only a direct F32 `RMS_NORM -> VIEW -> SET_ROWS` chain with I64 row indices, `ne[3] == 1`, contiguous input rows, head dimensions 256 or 512, and F16 or Q8_KV destinations.

F16 layout flattens all KV heads into one cache row per token. Q8_KV layout keeps one cache row per KV head and token. The fused kernel normalizes each head independently, then writes the final cache representation without materializing normalized F32 values.

Q8_KV quantization keeps the existing 64-value block format, FP16 scale storage, nearest-even CUDA rounding, and clamp range. Its reduction order within each 64-value block remains the same as the existing Q8_KV cache-write kernel.

Tests cover SWA geometry `(D=256, H=8)` and global geometry `(D=512, H=2)` for both F16 and Q8_KV.

## Validation

- Build and run focused `test-backend-ops` cases on CUDA0.
- Run the broader backend operation suite supported by the local build.
- Run a Pulsar S decode smoke test with F16 KV and Q8_KV when the deployed model and executable are available.
- Compare output tokens and inspect CUDA graph reuse and kernel count when profiling tools are available.

