# QLion Q8_0 and Mixed Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Q8_0-only and Q4_0/Q8_0 or MXFP4/Q8_0 GGUF models with QLion while retaining an 8192-token Flare context on a V100 16 GB GPU.

**Architecture:** Select trainable tensors by an exact allowed type set. Extend existing QLion block update and quantized backward paths for Q8_0. Add optional CPU-resident Q8_0 gradient accumulation so all model layers and optimizer states can stay on CUDA0.

**Tech Stack:** C++17, ggml CPU/CUDA/Vulkan backends, GLSL, CMake, Bash.

**Spec:** `docs/superpowers/specs/2026-09-27-qlion-q8-mixed-16gb-design.md`

## Global Constraints

- The model stays on CUDA0 and training context stays at 8192 tokens.
- Only Q8_0, Q4_0, and MXFP4 quantized model weights are trainable; nonquantized weights remain frozen.
- Keep Q8_0 momentum and Q4_0 residual, without a floating-point master weight.
- Reuse `examples/qlora_training/test-qat.cpp`; do not add a test file under `tests/`.
- Do not modify unrelated working-tree changes, commit, push, or create a PR.

## Review Focus

- A GGUF with an unselected quantized type must fail before model load; Task 1 tests this.
- A selected mixed type absent from the GGUF must fail instead of silently training a subset; Task 1 tests this.
- A tied Q8_0 tensor must have one optimizer state and receive both gradients; Task 2 tests this.
- Host Q8_0 accumulation must preserve two-microbatch updates without a CUDA-resident duplicate; Task 5 tests this.
- Resume with a different quantized type set must fail before restoring optimizer state; Task 1 tests this.

---

### Task 1: Exact GGUF type set and checkpoint metadata

**Files:** Modify `common/arg.cpp`, `common/common.h`, `examples/qlora_training/finetune_qat.cpp`, `tests/test-arg-parser.cpp`.

**Interfaces:** Keep `common_params::qat_quant_type` as the canonical string. Add a local parser in `finetune_qat.cpp` that returns an allowed `ggml_type` set and use it in model inspection and `qat_param_filter`.

- [ ] Add parser assertions for all five accepted `--quant-type` values and rejection of other values in the existing argument parser test.
- [ ] Run `build/bin/test-arg-parser` and confirm the new cases fail before implementation.
- [ ] Update `qat_inspect_model` to count each selected type, reject any other quantized type, and compute memory from actual tensor types; update the filter to select any allowed type. Preserve canonical type-set text in state save/load and keep existing single-type behavior.
- [ ] Add focused input-inspection and resume-mismatch checks using temporary GGUF files through existing test helpers where available. Run `cmake --build build --target test-arg-parser llama-finetune-qlion -j8` and `build/bin/test-arg-parser`.

### Task 2: Q8_0 reference and core optimizer registration

**Files:** Modify `examples/qlora_training/qat.h`, `examples/qlora_training/qat.cpp`, `examples/qlora_training/test-qat.cpp`, `ggml/src/ggml-opt.cpp`, `src/llama-context.cpp`, and restricted quantized backward checks in `ggml/src/ggml-cpu/ops.cpp`.

**Interfaces:** Add `QAT_WEIGHT_Q8_0` to the existing reference format enum; `ggml_opt_qat_register_param` accepts `GGML_TYPE_Q8_0` beside the existing two types.

- [ ] Extend existing zero-gradient, nonfinite-gradient, trajectory, resume, dense, tied, sparse-row, and routed QAT cases to Q8_0; assert one state for a tied alias.
- [ ] Build and run `test-qat` on CPU to confirm Q8_0 cases fail at the existing guards.
- [ ] Reuse ggml Q8_0 type traits in the reference and CPU block update; extend registration, graph discovery, parameter marking, and Q8_0 restricted backward dispatch. Keep unmodified formats on their current paths.
- [ ] Run `cmake --build build --target test-qat -j8` and `build/bin/test-qat` with the CPU backend selected.

### Task 3: CUDA Q8_0 update and backward support

**Files:** Modify `ggml/src/ggml-cuda/opt-step-qlion-qat.cu` and the CUDA backward kernels/dispatch currently restricted to MXFP4/Q4_0.

**Interfaces:** Existing QLion ops dispatch by `weight->type`; Q8_0 uses `block_q8_0` and supports F32 or Q8_0 accumulated gradients.

- [ ] Run the Q8_0 cases in `test-qat` with `QAT_TEST_BACKEND=CUDA0` to establish the failing baseline.
- [ ] Extend the existing CUDA block update for Q8_0 decode and requantization. Dispatch the dense, tied, row, sparse-row, and routed cases by the third weight type. Add Q8_0 to quantized routed backward support.
- [ ] Build `test-qat` and run `QAT_TEST_BACKEND=CUDA0 build/bin/test-qat`; check Q8_0 weight, momentum, and residual against the reference and existing Q4_0/MXFP4 cases for regressions.

### Task 4: Vulkan Q8_0 update and backward support

**Files:** Modify `ggml/src/ggml-vulkan/vulkan-shaders/opt_step_qlion_qat.comp`, shader variant registration, and `ggml/src/ggml-vulkan/ggml-vulkan.cpp` dispatch and support checks.

**Interfaces:** Add Q8_0 shader variants for F32 and Q8_0 dense gradients and the existing routed/row variants. Preserve current MXFP4 and Q4_0 pipelines.

- [ ] Run Q8_0 `test-qat` cases with the available Vulkan backend to establish failure.
- [ ] Add `block_q8_0` weight load/store and pipeline variants; extend type checks and routed backward support.
- [ ] Build the Vulkan backend and `test-qat`, then run the Q8_0 and existing format cases. If this machine's Vulkan device cannot run a variant, record the exact skipped capability and rely on shader compilation plus CPU/CUDA execution.

### Task 5: Optional CPU Q8_0 gradient accumulation

**Files:** Modify `common/arg.cpp`, `common/common.h`, `include/llama.h`, `src/llama-context.cpp`, `ggml/include/ggml-opt.h`, `ggml/src/ggml-opt.cpp`, `examples/qlora_training/finetune_qat.cpp`, `examples/qlora_training/test-qat.cpp`.

**Interfaces:** Add `--qat-grad-accumulator device|cpu`, default `device`. Add a bool to `llama_opt_params` and `ggml_opt_params`, then pass it to `ggml_opt_qat_make_grad_state` so only accumulator and sparse bookkeeping use `ggml_backend_cpu_buffer_type()` when enabled.

- [ ] Add a two-microbatch CUDA+CPU scheduler case asserting the accumulator buffer device is CPU, weight and QLion states remain CUDA, and final weights match device accumulation within quantization tolerance. Add a parser test for the new option.
- [ ] Run the focused tests to confirm failure with the new option absent.
- [ ] Thread the flag through parameter structs and allocator creation/replacement paths, including alias promotion. Keep the default device path unchanged and verify graph scheduling transfers gradients without a second persistent CUDA accumulator.
- [ ] Build and run `test-qat`, `test-arg-parser`, and the CUDA two-microbatch case; inspect actual backend buffer placement.

### Task 6: Flare script and 16 GB acceptance run

**Files:** Create `/mnt/openwebui/AIKAR/Lumen-3.2-Flare/train-qlion-v100-16gb.sh`; modify `examples/qlora_training/README.md`.

**Interfaces:** Environment overrides `MODEL`, `TRAIN_FILE`, and `QAT_OUT`; defaults point to the Flare Q4_0_L GGUF and MoXXf `stage1.sh` dataset. The script uses all GPU layers, `-c 8192 -b 8192 -ub 256`, `--quant-type q4_0,q8_0`, and `--qat-grad-accumulator cpu`.

- [ ] Implement the script with input existence and output collision checks, distinct output path, and the prior stage1 optimizer settings. Document Q4_0_XL and MXFP4+Q8_0 overrides.
- [ ] Run `bash -n` on the script and build `llama-finetune-qlion`, `test-qat`, and `test-arg-parser`.
- [ ] Run one complete optimizer update on CUDA0 with the Flare model and a short 8192-token sample. Measure peak VRAM; verify both a Q4_0 and a Q8_0 tensor changed and that save/resume preserves their types. If peak VRAM exceeds usable 16 GB, reduce only `-ub` and repeat.
- [ ] Run `git diff --check`, inspect `git status --short`, and report measured peak, throughput, and any backend coverage limit. Do not start the full dataset training job.
