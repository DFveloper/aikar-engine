# CUDA Q4_0 Training Memory Design

Date: 2026-09-23
Status: Proposed for implementation

## 1. Objective

Reduce peak VRAM during QLoRA and QLion training on NVIDIA V100 by keeping frozen Q4_0 weights quantized throughout forward and backward matrix multiplication. Avoid full-weight F16 or F32 dequantization. Also reduce output and activation buffers without changing the default training result.

The first supported target is:

- CUDA on NVIDIA V100
- frozen Q4_0 base weights
- LoRA and QLion trainable state
- the Lumen-3.2-Flare-base-Q4_0.gguf model under `/mnt/openwebui/AIKAR/Lumen-3.2-MoXXf-2B`

Other devices, backends, and quantization types keep their current behavior unless they can use an existing quantized kernel without new code.

## 2. Current Memory Failure

The model file is about 2.44 GiB and its persistent CUDA model buffer is about 2.66 GiB. The unexpectedly high training peak is not caused by the frozen model storage itself.

For a quantized `MUL_MAT`, the V100 CUDA heuristic selects cuBLAS when the token batch is larger than 96 columns. The cuBLAS path converts the complete frozen source matrix to F16. The tied token embedding and output matrix contains 402,653,184 values, so this conversion needs about 768 MiB.

The gradient of the activation input to that multiplication is represented as `OUT_PROD`. Its current CUDA implementation converts the complete quantized source matrix to F32 before SGEMM. The same tied matrix therefore needs about 1.50 GiB. The CUDA virtual-memory pool retains these pages until backend destruction, so the temporary conversion raises the remaining process peak even after the operation completes.

The scheduler buffer is the other major allocation. It was measured at about 4.71 GiB with microbatch 512 and about 1.18 GiB with microbatch 128. Its main causes are retained forward activations, all-token LM-head logits and loss gradients, and the lack of real activation recomputation. The current `grad_checkpoint_interval` implementation only marks tensors persistent. It does not recompute anything and can lengthen tensor lifetimes.

## 3. Design Overview

The implementation has four independent parts:

1. Force eligible frozen Q4_0 forward multiplications to use the existing CUDA quantized MMQ kernel through the existing `GGML_PREC_Q4` source-precision hint.
2. Add a direct CUDA Q4_0 `OUT_PROD` kernel for the activation gradient, with bounded tiled F16 fallback for unsupported layouts.
3. Compact the LM-head, logits, and sparse loss to supervised token rows only.
4. Replace the fake checkpoint behavior with optional FFN activation recomputation. Recomputation is off by default and can be enabled explicitly.

The four parts must be independently testable and measurable. A failure in one optimization must not silently select an unbounded full-weight dequantization path in another.

## 4. Forward Quantized Matrix Multiplication

### 4.1 Graph policy

When an optimization graph is built for QLoRA or QLion, `build_lora_mm` marks the frozen base `MUL_MAT` with:

```cpp
ggml_prec_set_src(base_mm, GGML_PREC_Q4, 0);
```

The existing API currently accepts only source index 1 for `MUL_MAT`. It will be extended to accept source index 0 because source 0 is the frozen weight in ggml matrix multiplication. This is an API correction and avoids adding a training-only operation or global environment variable.

The hint means that the backend must not internally expand source 0 above its stored four-bit rank when a compatible quantized kernel exists. It does not change the tensor type, accumulator type, or numerical contract.

The hint is attached only when all of the following are true:

- the graph is an optimizer graph;
- the source weight is frozen;
- the source type is Q4_0;
- CUDA is allowed to execute the operation.

LoRA A and B multiplications remain F32 or their configured QAT types. Inference graphs and non-training graphs retain the existing performance heuristic.

### 4.2 CUDA dispatch

CUDA reads the source-0 precision hint before applying the V100 batch-size heuristic. For contiguous Q4_0 weights and supported F32 input/output, `GGML_PREC_Q4` selects `ggml_cuda_mul_mat_q` even when `ne11 > 96`.

Capability and correctness checks still take priority over the hint. Unsupported shapes use the bounded fallback described below instead of full-weight conversion. Builds that explicitly disable MMQ report the unsupported forced policy during optimizer initialization rather than silently consuming the larger buffer.

The initial implementation does not change MMQ heuristics for ordinary inference.

## 5. Direct Q4_0 Backward Kernel

### 5.1 Required operation

For a frozen base multiplication, backward needs only the gradient with respect to the activation input. The base weight has no gradient. Given the ggml `OUT_PROD` layout, the kernel computes the mathematically equivalent matrix product while reading Q4_0 blocks directly.

The kernel:

- loads each Q4_0 block and its F16 scale;
- unpacks nibbles in registers;
- converts only the values used by the active tile;
- accumulates products in F32;
- writes F32 destination values;
- supports the operation's existing beta/accumulation behavior;
- does not allocate a tensor proportional to the frozen weight size.

The first optimized layout is a contiguous two-dimensional Q4_0 source and contiguous F32 gradient source/destination. It must cover the tied output matrix and dense attention/FFN projections in the test model. Batch dimensions with standard ggml broadcasting are added only where required by the test model.

### 5.2 Kernel tiling

Each CUDA block owns a tile of destination rows and columns. Destination rows are grouped by the 32-value Q4_0 block. The kernel traverses reduction rows and reuses each loaded Q4_0 block across the destination-column tile. Weight dequantization stays in registers or shared memory for that tile. No dequantized weight tile survives the kernel launch.

Tile dimensions are selected for Volta limits and validated with occupancy and correctness tests. The design does not require tensor cores, which avoids depending on an F16 copy of the frozen matrix.

### 5.3 Bounded fallback

If a Q4_0 `OUT_PROD` shape cannot use the direct kernel, CUDA uses a fixed-size tiled F16 fallback:

1. Select a destination-row tile aligned to the Q4_0 block size and a bounded reduction tile.
2. Dequantize only the matching source-0 tile to F16.
3. Convert only the matching source-1 reduction tile to F16.
4. Run `cublasGemmEx` with F32 accumulation for the matching destination slice.
5. Accumulate reduction tiles into the F32 destination and reuse both F16 workspaces.

The default workspace cap is 64 MiB. The cap is independent of total weight size. Allocation failure reduces the tile size down to one aligned row tile. If even the minimum tile cannot be allocated, the operation fails through the existing CUDA error path; it must not fall through to full F32 dequantization.

Non-Q4_0 types retain the existing implementation in this first version.

## 6. Supervised-Output Compaction

Training currently sets every token as an output even when sparse labels mask most tokens. With a 262,144-token vocabulary, each 512-row logits or logits-gradient tensor is 512 MiB.

The training batch builder will set `batch.logits[i]` only for tokens with an active target. The existing output-ID graph input and Gemma4 `ggml_get_rows` path then compact the final hidden state before the output norm and LM head. Sparse targets and weights are packed in the same order.

Requirements:

- loss normalization remains based on the same logical active-label count;
- token order and target alignment remain exact;
- all tokens still execute transformer layers and update the KV cache;
- a microbatch with zero active labels uses one zero-weight sentinel output row, preserving optimizer accumulation cadence without allocating all-token logits;
- mixed supervised and unsupervised microbatches use one compact loss graph;
- critical-token statistics use the original token indices, with an explicit compact-to-original index map.

Compaction is enabled automatically for sparse-label training. It has no quality tradeoff because masked rows do not contribute to the current loss.

## 7. Activation Recomputation

### 7.1 User option

Add:

```text
--activation-recompute on|off
```

The default is `off`. The option is available in both `llama-finetune-qlora` and `llama-finetune-qlion`.

The existing `--grad-checkpoint` option is deprecated because it does not perform checkpointing. A value greater than zero maps to activation recomputation with a warning during the transition period. The old persistent-node marking code is removed.

### 7.2 Initial recomputation scope

The first implementation recomputes pure dense FFN regions in Gemma4 layers. It checkpoints the FFN input (`attn_out`) and final layer output while allowing the following intermediates to be released after forward:

- FFN normalization output;
- up and gate projection outputs;
- activation and elementwise product outputs;
- down projection input and output before the final residual.

During backward, the FFN forward region is rebuilt from `attn_out`, its backward graph is executed, and only these values survive across region boundaries:

- the gradient with respect to `attn_out`;
- accumulated gradients for trainable LoRA or QLion state;
- the layer-boundary tensors required by the surrounding backward graph.

Attention is not recomputed in the first version. Its graph writes or consumes KV state, and repeating those operations without a side-effect-free attention mode could change semantics. This limitation is explicit rather than risking duplicate KV writes or detached Q/K/V gradients.

MoE FFN and per-layer embedding branches are not recomputed in the first implementation unless their region is proven side-effect-free and covered by tests. Unsupported model graphs reject `--activation-recompute on` with a clear error. They do not silently run the old persistent-node behavior.

### 7.3 Scheduler lifetime contract

Recomputed FFN intermediates are allocated from a reusable region workspace. The scheduler must not mark them as graph outputs or preserve them across unrelated layers. The workspace is reused in reverse layer order, so its capacity is bounded by the largest single recomputed FFN region rather than the sum across layers.

Recomputation trades time for memory but must not intentionally change precision, dropout state, targets, or optimizer ordering.

## 8. Public Configuration and Compatibility

Add optimizer parameters for:

- quantized frozen-weight kernels, enabled automatically for supported CUDA Q4_0 training;
- activation recomputation, default off;
- bounded fallback workspace in MiB, default 64 and not exposed on the command line in the first version unless measurement shows a need.

Graph cache identity includes both the quantized-kernel policy and recomputation mode.

Compatibility behavior:

- CUDA V100 plus frozen Q4_0: new fused path;
- CUDA with unsupported Q4_0 layout: bounded tiled F16 fallback;
- other CUDA quantization types: current path;
- CPU and other backends: current path;
- unsupported activation-recompute graph: initialization error only when the option is on;
- option off: existing optimizer graph behavior, except the incorrect persistent-node checkpoint block is removed.

## 9. Correctness Tests

Tests use existing test files and infrastructure. No new test executable is required.

### 9.1 CUDA operation tests

Extend backend operation tests with Q4_0 `OUT_PROD` cases covering:

- tied-output-matrix dimensions scaled to practical test sizes;
- non-multiple edge tiles where the ggml shape is still Q4_0 valid;
- beta zero and accumulation;
- standard batch broadcasting used by the model;
- direct-kernel layout;
- forced bounded-fallback layout.

Compare against the CPU reference or the existing full-dequant implementation. Required normalized mean squared error is at most `5e-4`, with finite output and no out-of-bounds sanitizer error.

Add a dispatch test proving that a Q4_0 multiplication with more than 96 columns honors `GGML_PREC_Q4` on Volta-capable CUDA builds.

### 9.2 Training equivalence tests

For a deterministic one-step run, compare optimized off/on variants:

- sparse-output compaction off versus on;
- activation recomputation off versus on;
- combined mode versus the current numerical reference.

Check finite loss, active-label count, gradient norm, and adapter update norm. Compaction should match within normal floating-point ordering tolerance. Recomputation may change reduction order but must remain within an agreed relative tolerance and must update the same parameter set.

Run the existing optimizer, QAT, and flash-attention backward tests to catch QLoRA and QLion regressions.

## 10. Memory and Performance Validation

Use the supplied model:

```text
/mnt/openwebui/AIKAR/Lumen-3.2-MoXXf-2B/Lumen-3.2-Flare-base-Q4_0.gguf
```

Primary benchmark settings match the established baseline:

- rank 4 LoRA;
- context and batch 4096;
- microbatch 512;
- flash attention on;
- CUDA V100;
- fixed seed and identical training sample.

Record for each configuration:

- CUDA peak allocated/reserved memory;
- scheduler allocation;
- persistent model, adapter, optimizer, KV, and fallback-workspace buffers;
- step time;
- loss and adapter update norm.

Required memory properties:

- no allocation proportional to the full weight tensor in F16 or F32;
- bounded fallback workspace no larger than the configured cap;
- all-token logits disappear when only a subset is supervised;
- activation-recompute peak is no greater than recomputation-off peak.

Targets, not correctness gates:

- recomputation off: at most 7.5 GiB peak for the primary benchmark;
- recomputation on: about 6 GiB or less;
- scheduler memory scales with active output rows for the output stage and with one FFN region for recomputed intermediates.

Performance regressions are reported separately. They do not justify silently restoring full-weight dequantization. Users can disable activation recomputation when speed is more important than its additional memory saving.

## 11. Diagnostics

At optimizer initialization, log once:

- whether the CUDA Q4_0 forced-MMQ policy is active;
- whether direct Q4_0 `OUT_PROD` is supported for the built graph;
- bounded fallback workspace cap;
- supervised output row count per microbatch when compaction is active;
- activation recomputation mode and supported region count.

Debug builds may count direct-kernel and fallback launches. Normal builds should not log per operation.

## 12. Non-Goals

This first implementation does not:

- add fused backward kernels for every quantization format;
- optimize non-CUDA backends;
- recompute attention or operations with KV side effects;
- change LoRA rank, optimizer state precision, or training math by default;
- add lossy activation quantization;
- make cuBLAS hold a complete dequantized frozen matrix;
- change ordinary inference dispatch heuristics.

## 13. Delivery Order

Implementation should be divided into reviewable stages:

1. Correct the precision hint for source 0 and force existing Q4_0 MMQ only in training.
2. Add and verify direct Q4_0 `OUT_PROD` plus bounded fallback.
3. Add supervised-output compaction and equivalence tests.
4. Remove fake checkpoint persistence and add the recomputation option.
5. Add dense Gemma4 FFN region recomputation.
6. Run the full correctness and memory benchmark matrix.

Each stage must leave the tree buildable and testable. No stage will be committed or submitted automatically.
