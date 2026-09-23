# Async Weight Prefetch Design

## Goal

Overlap host-to-device weight transfer for split N+1 with CUDA computation for split N during phase-offloaded QLoRA training. Extend the existing ggml backend scheduler instead of adding a separate trainer or model-specific layer executor.

## Scope

- Apply only when training enables weight streaming.
- Target one physical CUDA device with a CPU fallback backend.
- Preserve the current scheduler path when async prefetch is disabled or unsupported.
- Preserve the current GRPO phase coordinator, LoRA ownership, optimizer state, and checkpoint behavior.
- Keep graph construction and autograd model-independent.
- Do not add activation offload or tiled single-tensor matmul in this change.

## Current Behavior

The weight-streaming scheduler creates split-local CUDA copies for host weights. Allocation can reuse their storage between graph splits, which bounds VRAM use. Execution processes each split in order:

1. Wait until the destination storage is reusable.
2. Copy all split inputs.
3. Submit the split graph to the backend.
4. Record completion.

The CUDA backend does not implement CPU-to-CUDA through `cpy_tensor_async`. The scheduler therefore falls back to a synchronous tensor copy for host weights. Transfer and compute do not overlap.

## Selected Architecture

Add an optional double-buffered prefetch policy to `ggml_backend_sched`. Graph splitting remains unchanged except that streamed weight copies use two non-aliasing allocation slots. Execution uses a second backend instance for the same CUDA device as an upload stream.

For each CUDA split:

1. Prepare split N in staging slot `N % 2`.
2. Wait for the prior compute-complete event for that slot.
3. Copy host weights into pinned host buffers when direct pinned access is unavailable.
4. Submit H2D copies on the upload backend.
5. Record the slot upload-complete event.
6. Make the compute backend wait for the upload-complete event.
7. Submit split N compute on the compute backend.
8. Record the slot compute-complete event.
9. While split N computes, prepare split N+1 in the other slot.

The executor issues the next upload immediately after submitting current compute. It does not use a host worker thread because CUDA submission is asynchronous once the source resides in pinned memory.

## Scheduler Configuration

Expose one scheduler configuration structure rather than independent booleans:

```cpp
struct ggml_backend_sched_weight_streaming_params {
    bool enabled;
    bool async_prefetch;
    size_t staging_bytes;
};

GGML_API bool ggml_backend_sched_set_weight_streaming(
        ggml_backend_sched_t sched,
        struct ggml_backend_sched_weight_streaming_params params);
```

The setter is valid only while the scheduler is reset and before graph reservation. It returns `false` when the requested mode is unsupported or the requested staging budget cannot contain the largest streamed tensor. A disabled configuration preserves current behavior.

The llama optimizer context stores this structure so scheduler suspend and resume recreate the same policy.

## Staging Budget

Add the qlora option:

```text
--layer-staging-mib N
```

`N = 0` selects automatic sizing. A positive value is a hard upper bound for all device staging slots combined.

Automatic sizing combines the largest streamed tensor with free memory reported immediately before allocation:

```text
available = free VRAM - safety margin
budget = min(default cap, available, two aligned copies of largest streamed tensor)
```

The default cap is 1024 MiB and the safety margin is `max(512 MiB, 10% of total VRAM)`. Automatic mode avoids reserving unused slot capacity because later optimizer graph allocation can exceed the measurement graph. If two copies of the largest tensor cannot fit, async prefetch falls back to synchronous single-slot streaming. If the user supplied a hard upper bound smaller than one aligned copy of the largest tensor, reservation fails with the required minimum MiB.

The scheduler reports the selected mode, staging budget, largest streamed tensor, and fallback reason.

## Split Grouping

Continue using graph splits as the execution unit. Adjacent streamed weights may share a split while their total staged bytes fit one slot. If a group exceeds the per-slot budget, start a new split. This naturally reduces a large transformer layer to tensor-sized work without parsing model-specific layer names.

A single tensor is never tiled. Automatic sizing must accommodate it. A manual limit that cannot accommodate it produces a normal error before training compute begins.

## Pinned Host Memory

On CUDA, register the host weight buffer and submit H2D directly from the mmap data. Registration lasts for the scheduler lifetime and is released after both compute and upload backends synchronize. This avoids copying the complete streamed weight set through a bounded host arena on every optimizer window.

If direct host registration is unavailable or fails, allocate two bounded pinned host arenas through `ggml_backend_dev_host_buffer_type()`. Each arena has the same size as one device staging slot. Tensor layout metadata maps source tensor ranges to offsets in the arena.

The synchronous fallback copies directly from the original host tensor and does not allocate pinned arenas.

## Events and Buffer Ownership

Each slot owns:

- one pinned host arena when direct registration is unavailable;
- one set of split-local destination tensors backed by its device staging allocation;
- one upload-complete event;
- one compute-complete event.

Before overwriting slot S, the upload backend waits for S's compute-complete event. Before computing a split in S, the compute backend waits for S's upload-complete event. Scheduler synchronization waits for both backend instances and all live events.

The upload backend is destroyed after synchronization. Suspend and scheduler destruction release both pinned arenas and both device staging slots.

## MoE Handling

Preserve the existing `MUL_MAT_ID` used-expert selection. In the first async version, dynamic expert selection uses the existing synchronous copy path because expert IDs are known only after prior graph work. Static dense weights use double-buffered prefetch.

The scheduler records synchronous MoE transfer time separately so a later change can add expert-aware asynchronous packing without changing the main prefetch state machine.

## Error Handling and Fallback

Use synchronous streaming when any of these are unavailable:

- backend asynchronous execution capability;
- backend event capability;
- pinned host buffer type;
- a second backend instance for the selected CUDA device;
- enough memory for two slots after the safety margin.

Automatic mode logs the reason and continues synchronously. An explicit nonzero staging limit fails only when one tensor cannot fit. Allocation and event creation failures return normal errors from reservation; they must not reach a `GGML_ASSERT` or CUDA allocation abort.

## Progress and Measurements

For the first optimizer graph and once per completed training step, report:

- selected staging mode and total MiB;
- number of prefetched and synchronous splits;
- host packing time;
- H2D submission and wait time;
- compute time;
- estimated overlap ratio;
- total streamed bytes.

The Python tqdm phase display remains unchanged. Detailed transfer metrics remain stderr log messages.

## Verification

Reuse existing tests and do not create a new file under `tests/`.

1. Extend `tests/test-opt.cpp` with a small two-slot scheduler graph where streamed host weights are reused across several splits.
2. Verify synchronous and async modes produce the same tensor values.
3. Force slot reuse and verify events prevent early overwrite.
4. Verify automatic fallback when events or pinned host buffers are unavailable.
5. Verify a manual budget below the largest tensor returns a normal error.
6. Run `test-opt` and `test-qat`.
7. Run a deterministic one-step GRPO comparison between synchronous and async streaming.
8. Run two phase-offload steps and verify optimizer continuation.
9. Measure wall time, transfer time, overlap, and peak VRAM at context 16384 with microbatch 512.

## Success Criteria

- Existing trainer and optimizer APIs remain the execution owner.
- Dense weight transfer for split N+1 overlaps compute for split N on CUDA.
- Two staging slots remain within the selected total staging budget.
- Async and synchronous modes produce numerically equivalent updates.
- MoE used-expert transfer behavior remains correct.
- Unsupported systems continue through synchronous streaming.
- Manual undersizing produces a normal actionable error.
- No rollout runtime overlaps the resumed training scheduler.
