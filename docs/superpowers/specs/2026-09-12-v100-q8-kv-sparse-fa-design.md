# V100 Direct Q8_KV Sparse Flash Attention Design

## Goal

Add a fast sparse Flash Attention decode path for V100 when both KV tensors use `GGML_TYPE_Q8_KV`. The target workload is Gemma 4 sliding-window attention with one query token, head dimensions K=256 and V=256, GQA ratio 2, and a 1024-token window. Full-attention layers and unsupported shapes keep their existing dense kernel selection.

## Current State

The graph already records a maximum sparse KV count in Flash Attention op parameter 4, and CUDA already compacts a mask into row indices. The F16 MMA kernel can gather those rows, but its sparse shape list only covers the existing D512 and D576 paths. The Q8_KV selector returns vector or tile kernels before sparse MMA selection, and the MMA implementation interprets global K and V storage as F16.

The local tile launcher also has a positional argument mismatch after the upstream `launch_fattn` signature gained `use_sparse`. Its integer warp size is currently interpreted as `true`, which causes a false sparse launch with `n_kv_max == 0`. The existing six-call correction remains part of this change.

## Architecture

The graph passes `min(hparams.n_swa, k->ne[1])` as `n_kv_max` only for sliding-window layers. The existing mask compaction kernel produces at most that many row indices per query. Full-attention layers pass zero and remain dense.

CUDA selects sparse MMA before the Q8_KV vector/tile early return only when every direct-Q8 requirement is satisfied. The first supported V100 shape is:

- compute capability 7.0
- one query token
- K and V type `GGML_TYPE_Q8_KV`
- K head dimension 256
- V head dimension 256
- GQA ratio 2
- mask present, no ALiBi bias, and no logit softcap
- positive `n_kv_max`
- KV length at least 1024, where the direct path outperforms the existing vector kernel on V100

The MMA kernel keeps its current half-precision shared-memory layout. A Q8_KV-specific sparse loader gathers each selected `block_q8_kv` row from global memory, applies its F16 scale while converting packed int8 values to `half2`, and writes the result directly into the same swizzled shared tile consumed by the existing Volta MMA code. There is no full F16 KV staging buffer and no separate global-memory dequantization pass.

K and V use the same compacted row indices. Invalid padded slots write zero to the KV tile and negative infinity to the mask tile, matching the existing sparse F16 behavior. The first implementation uses one shared-memory stage because indexed gathers cannot use the current contiguous multi-stage copy path.

## Kernel Shape and Bounds

Sparse decode stays at `ncols1 == 1`, since each query owns a different compacted index list. V100 uses the existing padded `ncols2 == 32` MMA geometry to provide enough work per block. All Q loads, sink loads, output writes, and fixup writes must retain or gain guards against `zt_Q + local_head >= ne02`, so the physical 32-column tile is safe for Gemma's logical GQA group of 2.

The Q8 loader receives byte strides for global K and V. It must not reuse the F16 `half2` row stride calculation. Shared-memory strides and all downstream MMA math remain unchanged.

## Selection and Fallback

Sparse eligibility remains centralized in `ggml_cuda_flash_attn_ext_mma_f16_should_use_sparse`. It is extended for the D256/V256/GQA2 V100 case and distinguishes the direct Q8_KV variant from the F16 variant. Kernel selection checks sparse eligibility before the Q8_KV early return.

Any unsupported type combination, shape, batch size, mask layout, bias, softcap, GPU architecture, or short KV length falls back to the existing Q8_KV vector/tile selection. Full-attention Gemma layers do not receive `n_kv_max`, so they cannot select sparse kernels.

No user-facing flag is required. The optimization is automatic under `-fa on -ctk q8_kv -ctv q8_kv`.

## Failure Handling

Host-side selection prevents unsupported template combinations from launching. Device code does not assert on a zero sparse bound because a sparse kernel cannot be selected without a positive `n_kv_max`. If mask compaction sees more live rows than the bound, it stores only the first `n_kv_max` rows; the graph-provided SWA bound must therefore equal the semantic maximum number of unmasked rows.

For Gemma sliding-window masks, causal masking plus the window rule produces at most `min(hparams.n_swa, k->ne[1])` live rows. Full-attention masks are excluded by the zero bound.

## Files

- `src/llama-graph.cpp`: pass the SWA bound to `build_attn_mha`.
- `ggml/src/ggml-cuda/fattn.cu`: extend sparse eligibility and select sparse MMA before Q8_KV dense paths.
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`: add the direct Q8_KV sparse loader, template selection, byte-stride handling, and padded-head safety.
- `ggml/src/ggml-cuda/fattn-tile.cuh`: keep the corrected non-sparse `launch_fattn` argument.
- `tests/test-backend-ops.cpp`: add the supported D256/V256/GQA2/Q8_KV sparse case to the existing Flash Attention matrix.

No new test file is added.

## Verification

1. Preserve the existing failing runtime command as the regression case and verify the `n_kv_max > 0` assertion is gone.
2. Build the CUDA server for compute capability 7.0 and run the target GGUF on V100 only.
3. Verify startup, warmup, and a real completion request with `-fa on -ctk q8_kv -ctv q8_kv -c 262144 -kvu -np 4`.
4. Compare deterministic short-prompt logits or token output against Flash Attention disabled, using tolerances appropriate for Q8 KV dequantization and F16 MMA accumulation.
5. Confirm through temporary local instrumentation or debugger inspection that SWA layers select direct Q8 sparse MMA and full-attention layers select the existing dense path. Remove temporary instrumentation before handoff.
6. Measure decode throughput at representative populated KV lengths, including 1K, 4K, and a longer context. Keep the sparse threshold only where it outperforms the existing Q8 vector/tile path on V100.
7. Run the existing CUDA backend tests relevant to Flash Attention and a final clean incremental build.

## Non-Goals

- Prefill or multi-query-token sparse kernels
- Sparse Q8_KV support for architectures other than V100
- D512, D576, or other new direct-Q8 shapes
- A global F16 KV staging buffer
- Changes to the Q8_KV format
- Changes to full-attention layer behavior
