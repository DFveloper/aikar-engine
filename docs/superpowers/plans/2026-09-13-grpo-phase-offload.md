# GRPO Phase Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run full-GPU GRPO rollout and host-resident QLoRA training in alternating CUDA phases without losing optimizer or adapter state.

**Architecture:** Add a reusable optimizer scheduler detach/attach primitive, expose an internal llama training-scheduler suspend/resume boundary, and use those primitives in a tool-local GRPO phase coordinator. The coordinator keeps the training model and canonical adapter alive on the host, creates a temporary full-GPU rollout runtime per step, copies LoRA tensors directly, and ensures the two large CUDA phases never coexist.

**Tech Stack:** C++17, ggml optimizer and backend scheduler, llama internal API, common argument parser, CUDA backend, Python 3 subprocess/tqdm driver

**Spec:** `docs/superpowers/specs/2026-09-13-grpo-phase-offload-design.md`

## Global Constraints

- Target `llama-finetune-qlora`, GRPO mode, and one CUDA device.
- Preserve the current path when `--grpo-phase-offload` is absent.
- Keep rollout inference fully GPU-resident according to `-ngl`.
- Keep training base weights, canonical LoRA tensors, and optimizer state in host memory.
- Do not serialize an adapter during phase transitions.
- Keep the first implementation model-independent and do not add layer prefetch yet.
- Reuse existing tests; do not add a new file under `tests/`.
- Keep new code and comments ASCII-only.

---

### Task 1: Add the phase-offload command-line contract

**Files:**
- Modify: `common/common.h`
- Modify: `common/arg.cpp`
- Modify: `examples/qlora_training/finetune_qlora.cpp`

**Interfaces:**
- Produces: `common_params::grpo_phase_offload` and the `--grpo-phase-offload` parser entry.
- Consumes: existing `common_params::grpo_mode` validation and qlora usage contexts.

- [ ] **Step 1: Add a parser-level failing check**

Run the existing binary before the change:

```bash
build/bin/llama-finetune-qlora --help | rg -- '--grpo-phase-offload'
```

Expected: exit 1 because the option is absent.

- [ ] **Step 2: Add the parameter and parser entry**

Add beside `grpo_mode`:

```cpp
bool grpo_phase_offload = false;
```

Register an argument limited to the qlora training usage context:

```cpp
add_opt(common_arg(
    { "--grpo-phase-offload" },
    "alternate a full-GPU rollout runtime with host-resident GRPO training",
    [](common_params & params) {
        params.grpo_phase_offload = true;
    }
).set_examples({LLAMA_EXAMPLE_FINETUNE_QLORA}));
```

Reject use without GRPO and reject unsupported device layouts with a normal error before model creation.

- [ ] **Step 3: Build and verify the option**

Run:

```bash
cmake --build build --target llama-finetune-qlora -j18
build/bin/llama-finetune-qlora --help | rg -- '--grpo-phase-offload'
```

Expected: build succeeds and help contains the option.

### Task 2: Make optimizer scheduler ownership replaceable

**Files:**
- Modify: `ggml/include/ggml-opt.h`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: `void ggml_opt_set_backend_sched(ggml_opt_context_t opt_ctx, ggml_backend_sched_t backend_sched)`.
- Preserves: optimizer iteration, moment tensors, parameter tensors, and learning-rate state.
- Clears: graph pointers and transient graph metadata owned by the old scheduler.

- [ ] **Step 1: Extend an existing optimizer test with a failing scheduler replacement case**

In `tests/test-opt.cpp`, add a test case that performs one optimizer update, calls `ggml_opt_set_backend_sched(opt, nullptr)`, constructs a second scheduler over the same backends, reattaches it, performs a second update, and checks that iteration and parameter changes continue rather than reset.

The key assertions are:

```cpp
GGML_ASSERT(iter_after_first == 1);
GGML_ASSERT(iter_after_second == 2);
GGML_ASSERT(value_after_second != value_after_first);
```

- [ ] **Step 2: Build the optimizer test to verify the new symbol is missing**

Run:

```bash
cmake --build build --target test-opt -j18
```

Expected: compile or link failure naming `ggml_opt_set_backend_sched`.

- [ ] **Step 3: Implement scheduler replacement**

Declare the function in `ggml-opt.h`. In `ggml-opt.cpp`, synchronize the existing scheduler when non-null, release or null all transient graph references in the optimizer context, assign the new scheduler pointer, and leave persistent optimizer state untouched. Require a non-null scheduler before the next graph allocation or optimizer epoch.

- [ ] **Step 4: Run the focused optimizer test**

Run:

```bash
cmake --build build --target test-opt -j18
build/bin/test-opt
```

Expected: exit 0, including the scheduler replacement case.

### Task 3: Add internal training scheduler suspend and resume

**Files:**
- Modify: `src/llama-context.h`
- Modify: `src/llama-context.cpp`
- Modify: `src/llama-ext.h`
- Modify: `examples/qlora_training/test-qat.cpp`

**Interfaces:**
- Produces: `bool llama_opt_suspend(llama_context * ctx)` and `bool llama_opt_resume(llama_context * ctx)` in the experimental internal header.
- Consumes: `ggml_opt_set_backend_sched` from Task 2.

- [ ] **Step 1: Add a failing two-update lifecycle test to the existing qlora test binary**

Extend `examples/qlora_training/test-qat.cpp` to initialize an optimizer, execute one update, suspend, assert that the training scheduler is absent, resume, execute a second update, and assert that the optimizer iteration and LoRA tensor values continue.

- [ ] **Step 2: Build to verify the lifecycle API is missing**

Run:

```bash
cmake --build build --target test-qat -j18
```

Expected: compile failure naming `llama_opt_suspend` or `llama_opt_resume`.

- [ ] **Step 3: Refactor scheduler construction and implement lifecycle methods**

Extract the scheduler creation portion of `llama_context::opt_init` into a private `opt_create_backend_sched()` helper. Implement suspend as backend synchronization, optimizer detach, compute-cache release, and scheduler destruction. Implement resume as scheduler recreation followed by optimizer reattachment. Return `false` for invalid lifecycle transitions and preserve `opt_ctx`, `opt_params`, `opt_model`, and optimizer state.

- [ ] **Step 4: Run lifecycle and optimizer tests**

Run:

```bash
cmake --build build --target test-qat test-opt -j18
build/bin/test-qat
build/bin/test-opt
```

Expected: both exit 0.

### Task 4: Implement the GRPO phase coordinator and LoRA synchronization

**Files:**
- Modify: `examples/qlora_training/finetune_qlora.cpp`

**Interfaces:**
- Produces: a persistent training runtime, a per-step rollout runtime, adapter map validation, direct adapter copying, and phase IPC.
- Consumes: lifecycle API from Task 3 and `common_init_result_ptr` RAII ownership.

- [ ] **Step 1: Add small deterministic adapter-map checks before the runtime branch**

Implement tool-local helpers with these contracts:

```cpp
static bool validate_lora_maps(const llama_adapter_lora & src, const llama_adapter_lora & dst, std::string & error);
static bool copy_lora_maps(const llama_adapter_lora & src, llama_adapter_lora & dst, size_t & copied_bytes, std::string & error);
static void emit_phase(const char * phase);
```

Validation compares name, type, dimensions, and byte size for every A/B tensor. Copy uses `ggml_backend_tensor_copy` for matching tensors and synchronizes the destination backends once after the batch.

- [ ] **Step 2: Add distinct training and rollout parameter builders**

The training copy of `common_params` must set zero resident GPU layers, enable mmap, disable KV offload, retain operation offload, and keep the selected CUDA device. The rollout copy retains the user `-ngl`, uses an inference-only context, and loads the same adapter skeleton.

- [ ] **Step 3: Add the phase loop without changing the legacy loop**

Branch in `main` only when `params.grpo_phase_offload` is true. The new loop must execute:

```text
rollout_load -> copy adapter -> rollout -> rollout_release
training_resume -> build dataset -> training -> training_suspend
```

Create rollout ownership inside the step scope. Destroy it before `llama_opt_resume(training_ctx)`. Use the training adapter for checkpoint and final save. Keep the temporary initial adapter file until both runtimes no longer need it.

- [ ] **Step 4: Add normal errors and timing logs**

On adapter mismatch, runtime creation failure, or lifecycle failure, emit `[QLORA:ERROR]` and return nonzero after RAII cleanup. Log rollout load time, LoRA copy time and bytes, dataset build time, training time, and phase wall time. Include context, batch, and microbatch sizes in allocation-related failures available to the tool.

- [ ] **Step 5: Build the qlora executable**

Run:

```bash
cmake --build build --target llama-finetune-qlora -j18
```

Expected: exit 0 with no new compiler warnings.

### Task 5: Wire the Python driver and preserve visible progress

**Files:**
- Modify: `/mnt/openwebui/AIKAR/Merge/RL.py`
- Modify: `/mnt/openwebui/AIKAR/Merge/RL.sh`

**Interfaces:**
- Produces: `--phase-offload` in the Python wrapper, forwarding `--grpo-phase-offload`, and tqdm phase labels.
- Consumes: `[QLORA:PHASE] <name>` messages from Task 4.

- [ ] **Step 1: Add a failing command-construction check**

Run the wrapper help and a dry command-construction path, confirming `--phase-offload` is initially absent or not forwarded.

- [ ] **Step 2: Add the wrapper flag and phase mapping**

Add `--phase-offload` as an explicit opt-in. Append `--grpo-phase-offload` to the child command only when set. Map all six spec phase names to the existing tqdm postfix and retain unknown messages as ordinary subprocess log lines.

- [ ] **Step 3: Enable the flag in the requested shell invocation**

Add `--phase-offload` to `RL.sh` without changing the user's rank, context, batch, microbatch, rollout, or model placement values.

- [ ] **Step 4: Verify Python syntax and command output**

Run:

```bash
python -m py_compile /mnt/openwebui/AIKAR/Merge/RL.py
bash -n /mnt/openwebui/AIKAR/Merge/RL.sh
python /mnt/openwebui/AIKAR/Merge/RL.py --help | rg -- '--phase-offload'
```

Expected: all commands exit 0.

### Task 6: Verify memory exclusivity, state continuity, and compatibility

**Files:**
- Modify only if a defect is found: files from Tasks 1-5.

**Interfaces:**
- Verifies the complete feature and legacy behavior.

- [ ] **Step 1: Run static and focused regression checks**

Run:

```bash
git diff --check
cmake --build build --target llama-finetune-qlora test-opt test-qat -j18
build/bin/test-opt
build/bin/test-qat
```

Expected: all exit 0.

- [ ] **Step 2: Run a one-step phase-offload smoke test**

Use the supplied model and dataset with `--n-steps 1`, short rollout tokens, and `--grpo-phase-offload`. Sample `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` during phase messages. Expected: rollout VRAM falls before training resumes, the process completes normally, and an adapter is saved.

- [ ] **Step 3: Run a two-step continuity smoke test**

Run two short phase-offload steps with a fixed seed. Expected: optimizer iteration reaches two and the canonical adapter changes on both steps across two suspend/resume cycles.

- [ ] **Step 4: Run the same one-step smoke test without the flag**

Expected: the legacy path starts without phase IPC and retains its prior behavior.

- [ ] **Step 5: Review the diff against the spec**

Confirm no rollout runtime overlaps a resumed training scheduler, no phase transition writes GGUF, no model-specific layer assumptions were added, all new comments are concise ASCII, and prefetch remains a documented future supplier optimization.
