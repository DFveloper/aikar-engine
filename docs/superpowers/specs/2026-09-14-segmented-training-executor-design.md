# Segmented Training Executor Design

## Goal

Train a rank-64 QLoRA adapter at context 16384 on a 16 GiB CUDA device without allocating the complete forward and backward activation graph on that device. Keep the current QLoRA and GRPO trainer as the owner of datasets, losses, LoRA tensors, optimizer state, checkpoints, and phase switching.

## Root Cause

The current weight-streaming scheduler bounds copies of host-resident model weights, but `ggml_opt_eval()` still reserves the complete CUDA activation and attention workspace for each training graph. With a 512-token microbatch, the live graph grows as the KV prefix grows. The measured allocation rises from about 9 GiB near the start of a 16K window to 12.7 GiB near token 12800 and fails on a 16 GiB V100.

Graph metadata is already host-resident. Moving metadata does not reduce VRAM. The executor must shorten device tensor lifetimes by allocating and executing independent graph segments.

## Scope

- Add an optional segmented execution mode to the existing ggml optimizer path.
- Keep the complete forward and backward graph metadata on the host.
- Execute generic contiguous graph segments. Do not add a Gemma-specific trainer.
- Store segment boundary activations and boundary gradients in host buffers.
- Recompute forward operations during reverse traversal instead of preserving all intermediate activations.
- Keep model weights, LoRA ownership, gradient accumulation, optimizer updates, QAT conversion, and checkpoint formats unchanged.
- Implement synchronous transfer and execution first.
- Define interfaces required for later double-buffered activation and weight prefetch, but do not implement asynchronous prefetch in the first version.
- Preserve the existing non-segmented path unless phase offload requests segmented execution.

## Execution Model

### Graph planning

`ggml_opt` continues to build the normal forward graph and expanded backward graph. A planner analyzes tensor dependencies and divides the forward node order into contiguous segments. It prefers cuts with the smallest live boundary set while keeping the estimated device allocation below a configurable budget.

A tensor belongs to a segment when its producing node belongs to that segment. A tensor is a live-in when it is produced outside the segment and consumed inside it. A tensor is a live-out when it is produced inside the segment and consumed later. Parameters, optimizer state, persistent KV tensors, and immutable graph inputs are external state and are not copied into the activation checkpoint store.

The backward planner derives reverse segments from the full autograd graph. Each reverse segment contains:

- the forward nodes required to rematerialize that segment;
- the backward nodes that consume those rematerialized tensors;
- incoming boundary gradients from the next reverse segment;
- outgoing boundary gradients for the previous reverse segment;
- parameter gradient contributions produced by the segment.

The planner is dependency based. Tensor names may be used only for diagnostics and must not determine correctness.

### Forward traversal

For each segment in order:

1. Allocate a fresh segment-local scheduler graph.
2. Restore live-in activations from host checkpoints.
3. Stream required host weights with the existing synchronous weight path.
4. Execute the segment.
5. Copy live-out activations to host checkpoint buffers.
6. Synchronize and release segment-local device allocations.

Only live graph inputs, persistent model state, one segment workspace, and its staged weights occupy device memory.

### Loss and backward traversal

The final forward segment computes the existing loss unchanged. Its loss gradient seeds the reverse traversal.

For each reverse segment:

1. Restore the segment's forward live-ins.
2. Recompute the segment forward nodes.
3. Restore incoming boundary gradients.
4. Execute the segment backward nodes.
5. Accumulate LoRA parameter gradients into the existing authoritative gradient accumulators.
6. Save outgoing boundary gradients in host buffers.
7. Release segment-local device allocations.

The optimizer step runs only after every reverse segment completes. Model parameters do not change during recomputation, so repeated KV writes for the same layer and positions are idempotent. The executor must synchronize before reusing or releasing buffers containing KV copy sources.

## Memory Ownership

Host memory owns:

- full graph metadata;
- immutable model weights selected for streaming;
- activation checkpoint buffers;
- boundary gradient buffers;
- LoRA gradient accumulators when their normal backend is the CPU;
- optimizer state already placed by the existing optimizer.

Device memory owns:

- persistent runtime inputs and KV cache selected by the existing context;
- one segment's temporary tensors;
- synchronous streamed-weight copies for that segment;
- current boundary activation and gradient tensors.

Checkpoint buffers use the backend device host-buffer type when available and ordinary aligned host buffers otherwise. The first version performs explicit synchronization before every device allocation is released.

## Budget and Automatic Planning

Expose these optimizer settings internally:

```cpp
struct ggml_opt_segmented_params {
    bool enabled;
    size_t device_budget;
    size_t min_segment_nodes;
};
```

Phase offload enables segmented execution automatically. A zero device budget selects:

```text
device budget = current free VRAM - max(1536 MiB, 15% of total VRAM)
```

The reserve margin covers CUDA runtime allocations, KV growth, and allocator variation. The planner measures each candidate segment with the real backend allocator. If a candidate exceeds the budget, it moves the cut earlier and measures again. If one operation cannot fit, execution stops before training with the tensor name, operation, required bytes, available bytes, and suggested smaller microbatch.

The target V100 configuration must plan against the worst KV prefix for the complete 16384-token window, not only the first 512-token microbatch.

## Failure Handling

- Allocation failure returns an optimizer status and never proceeds to `set_inputs()` with unallocated tensors.
- Host checkpoint allocation failure reports the required host bytes.
- Unsupported graph operations disable segmented mode before the first optimizer update and report the operation.
- A failure after any parameter gradient was accumulated invalidates the current optimizer window; no partial optimizer update is allowed.
- The GRPO coordinator keeps the last valid adapter and may terminate cleanly through the existing error IPC path.

## Progress and Diagnostics

Report once per new graph shape:

- number of forward and backward segments;
- device budget and maximum measured segment allocation;
- host activation and gradient checkpoint bytes;
- largest live boundary;
- worst KV prefix used during planning.

Report per optimizer window:

- forward segment progress;
- backward segment progress;
- recompute time;
- host-to-device and device-to-host transfer time;
- optimizer time;
- peak allocated device bytes.

The Python wrapper maps these messages to the existing tqdm display.

## Compatibility

- Ordinary inference is unchanged.
- Ordinary QLoRA and QAT use the existing full-graph executor unless segmented mode is explicitly enabled.
- GRPO phase offload uses segmented mode by default.
- Existing LoRA GGUF files and resume checkpoints remain compatible.
- Existing synchronous weight streaming is reused inside each segment.
- The later asynchronous implementation may add two activation slots and two weight slots without changing the planner or optimizer API.

## Verification

Do not add a new file under `tests/`. Extend existing optimizer tests.

1. Compare full-graph and segmented forward results on a small multi-layer graph.
2. Compare input, parameter, and boundary gradients.
3. Compare one AdamW update and one QAT update.
4. Force at least three segments and verify host checkpoint restoration.
5. Verify that allocation failure returns normally and prevents input writes and optimizer updates.
6. Verify repeated KV-like state writes remain deterministic during rematerialization.
7. Run `test-opt` and `test-qat` on CPU and CUDA.
8. Run deterministic one-step GRPO comparisons at context 1024.
9. Run the target rank-64, context-16384, microbatch-512 configuration through all 128 training microbatches and one optimizer update on the V100.

## Success Criteria

- The target run completes one full 16K optimizer window without CUDA OOM.
- Peak device allocation stays below the automatic budget and leaves at least the configured reserve margin.
- Segmented and full-graph parameter updates match within the existing backend tolerance.
- No optimizer update occurs after a partial or failed backward traversal.
- The first synchronous version requires no model-specific layer implementation.
- Planner and executor interfaces admit double-buffered asynchronous prefetch as a later change.
