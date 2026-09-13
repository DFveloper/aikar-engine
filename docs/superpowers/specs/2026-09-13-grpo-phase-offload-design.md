# GRPO Phase Offload Design

## Goal

Run GRPO on one CUDA GPU when the quantized model and the training graph do not fit in VRAM at the same time.

Keep rollout inference fully resident on the GPU. Release the rollout model before training. Keep the canonical model weights, LoRA tensors, and optimizer state in system memory. During training, use the existing ggml operation offload path to copy only the weights needed by the active GPU operations.

The first implementation targets `llama-finetune-qlora`, GRPO mode, and one CUDA device. It must not depend on the Gemma 4 graph layout.

## Constraints

- The target GPU is a Tesla V100 with 16 GiB VRAM.
- The host has about 121 GiB RAM.
- The GPU is connected through PCIe 3.0 x4.
- The model has 30 layers, 128 experts, and uses 8 experts per token.
- The current full-GPU model load takes about 8.4 seconds.
- The current training graph requests about 10.7 GiB of VRAM at a 16384 token training context.
- Rollout and training must not hold their large GPU allocations at the same time.
- Optimizer state must survive every phase change.
- Existing GRPO behavior must remain available when phase offload is not enabled.

## User Interface

Add this option to `llama-finetune-qlora`:

```text
--grpo-phase-offload
```

When the option is absent, the existing single model and context path is unchanged.

When the option is present:

- `-ngl` controls the rollout model placement.
- The training model weights remain in host memory.
- `-dev` selects the CUDA device used by rollout and training operation offload.
- KV cache offload is disabled for the training context.
- Host operation offload remains enabled for the training context.

The Python driver passes `--grpo-phase-offload` when explicitly requested. It displays phase IPC messages but does not own phase resources.

## Runtime Ownership

The GRPO coordinator owns two runtimes.

### Training runtime

The training runtime persists for the full run. It owns:

- a host-resident model loaded with mmap;
- the canonical LoRA tensors;
- the optimizer and its step state;
- the training context;
- a CUDA scheduler that can be suspended and resumed.

The training model uses zero resident GPU layers. The existing ggml scheduler may still run supported operations on CUDA through operation offload.

### Rollout runtime

The rollout runtime exists only during generation for one GRPO step. It owns:

- a model loaded with the rollout `-ngl` value;
- an inference-only context and KV cache;
- a read-only clone of the current LoRA tensors.

Destroying the rollout runtime must release all model, KV cache, graph, and adapter allocations before the training scheduler resumes.

## Phase Sequence

Initialization:

1. Create the initial adapter skeleton if no adapter was supplied.
2. Load the host training model and adapter.
3. Mark the host adapter tensors as trainable.
4. Initialize the optimizer on the host adapter tensors.
5. Suspend the training CUDA scheduler.

Each GRPO step:

1. Create the full-GPU rollout runtime.
2. Validate the rollout and training adapter tensor maps on the first step.
3. Copy the canonical LoRA tensors directly from host to the rollout adapter.
4. Generate all rollout responses.
5. Synchronize CUDA and destroy the rollout runtime.
6. Resume the training CUDA scheduler.
7. Build the step dataset.
8. Run forward, backward, and optimizer update through the host-weight operation offload path.
9. Synchronize CUDA and suspend the training scheduler.
10. Save a checkpoint from the canonical host LoRA tensors when requested.

At completion, save the final adapter from the canonical host LoRA tensors.

## Scheduler Suspension

Suspension is allowed only after an optimizer epoch call has completed and all backends are synchronized.

Suspension releases:

- the backend scheduler;
- scheduler allocator buffers;
- transient graph tensors;
- the optimizer compute metadata cache;
- training CUDA staging and compute buffers.

Suspension preserves:

- the host model;
- canonical LoRA tensors;
- optimizer moment tensors;
- optimizer iteration state;
- learning-rate schedule state.

The optimizer currently stores a raw backend scheduler pointer. Add an internal detach and attach operation so the optimizer can survive scheduler replacement. Clear transient graph pointers before destroying their metadata context.

Resume creates a new training scheduler with the same backend order and training graph capacity, then attaches it to the existing optimizer context.

Do not add a broad stable public API for this tool-specific behavior. Keep the new entry points internal or experimental.

## LoRA Synchronization

Do not save and reload a GGUF adapter at each step.

On the first phase change, compare the host and rollout adapter maps:

- tensor name;
- tensor type;
- tensor dimensions;
- byte size.

Fail before generation if any item differs.

Copy each matching A and B tensor from the host adapter to the rollout adapter. Use asynchronous backend copies when supported and synchronize once after all copies are queued. The rollout adapter is read-only after synchronization.

The host adapter remains the only canonical mutable copy.

## Memory Rules

Only one large CUDA phase may be active.

During rollout, VRAM contains the full model, rollout adapter, KV cache, and inference graph. The training scheduler must be suspended.

During training, VRAM contains the training graph, staging copies, and active CUDA operations. The rollout runtime must not exist.

Before the first real training step, measure required scheduler memory per backend. Log the requested CUDA bytes and the available CUDA bytes. Return a normal error with an actionable message if the reservation does not fit. Do not reach a `GGML_ASSERT` after `cudaMalloc` failure.

Do not silently change context size, microbatch size, rank, or rollout generation settings.

## IPC and Progress

Add phase messages to the existing protocol:

```text
[QLORA:PHASE] rollout_load
[QLORA:PHASE] rollout
[QLORA:PHASE] rollout_release
[QLORA:PHASE] training_resume
[QLORA:PHASE] training
[QLORA:PHASE] training_suspend
```

The Python driver maps these messages to the existing tqdm status line. Unknown phase messages remain non-fatal for compatibility.

Log these measurements for each phase:

- phase wall time;
- model load time;
- LoRA copy time and bytes;
- dataset build time;
- training time;
- scheduler CUDA reservation size;
- host-to-device bytes copied when the backend can report them.

## Failure Handling

Use RAII for both runtimes. An error during rollout must destroy the rollout runtime. An error during training must synchronize and suspend the training scheduler before unwinding.

Convert allocation failure in the new path into a returned error and an IPC `ERROR` message. Include the phase, requested bytes, available bytes, context size, batch size, and microbatch size.

SIGINT behavior remains graceful. A stop request during rollout finishes the active backend call, releases rollout resources, and saves only when the existing stop policy requires it. A stop request during training finishes or aborts at the next supported optimizer boundary, then suspends CUDA resources.

## Compatibility

- The existing path remains unchanged unless `--grpo-phase-offload` is set.
- Existing checkpoint files remain compatible.
- Resume continues to restore the LoRA and schedule fields currently stored by the tool.
- This change does not claim to restore optimizer moments from a process restart.
- CPU-only and multi-GPU phase offload are out of scope for the first implementation.

## Verification

Reuse existing test infrastructure and do not add a new test file without maintainer approval.

Verification stages:

1. Build `llama-finetune-qlora` with CUDA enabled.
2. Run existing relevant optimizer and backend tests.
3. Run a one-step GRPO smoke test with short rollouts.
4. Confirm the legacy path still starts and completes the same smoke test.
5. Compare one deterministic training update between the legacy path and phase offload within the existing numerical tolerance.
6. Sample VRAM through all phase transitions and confirm that rollout and training peaks do not overlap.
7. Run two steps and confirm optimizer iteration and LoRA updates continue across suspend and resume.
8. Test an intentional low-memory reservation failure and confirm a normal error replaces the current abort.

Record phase times and transfer volume for the smoke test before enabling prefetch work.

## Future Layer Streaming Plan

Keep phase coordination separate from the training weight supplier. The first supplier is the existing ggml operation offload scheduler.

The future layer-streaming implementation can replace the supplier while keeping the GRPO coordinator, rollout runtime, canonical LoRA ownership, checkpoints, and IPC unchanged.

The future work is:

1. Add per-split transfer and compute timing.
2. Group training graph splits by transformer layer and direction.
3. Allocate two bounded CUDA staging regions.
4. Copy layer N+1 on a dedicated transfer stream while layer N computes.
5. Use CUDA events before a staging region is reused.
6. Preserve the current used-expert-only copy optimization for `MUL_MAT_ID`.
7. Add activation checkpoint recomputation or host activation offload only if measured activation memory still prevents a useful staging window.
8. Select lookahead depth from measured copy and compute times instead of a fixed model-specific value.

Do not implement prefetch in the first change unless measurements show that the existing scheduler cannot keep the GPU usefully occupied and the added synchronization can be isolated behind the weight supplier boundary.

## Success Criteria

- Full-GPU rollout behavior is retained.
- A 10.7 GiB training graph no longer competes with the full 14 GiB rollout model allocation.
- Canonical LoRA and optimizer state survive at least two phase changes.
- No adapter GGUF is written as part of a normal phase change.
- The legacy GRPO path remains available.
- The first implementation is model-independent within the supported single-CUDA GRPO scope.
