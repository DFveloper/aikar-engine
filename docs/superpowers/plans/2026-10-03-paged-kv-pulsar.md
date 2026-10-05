# Paged KV for Pulsar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline execution, or superpowers:subagent-driven-development if the user selects delegated execution. Steps use checkbox syntax for tracking.

**Goal:** Integrate #22569 and deliver optional paged F16 and Q8_KV inference for Pulsar and Pulsar S on CUDA and Vulkan.

**Architecture:** Adapt the upstream page pool and attention operation to normal `llama_decode` and the current `llama_memory_i` interface. Separate full/SWA allocation domains and derive layer-local geometry and packed row strides. Finish CUDA correctness and serving integration before adding native Vulkan shaders with the same operation contract.

**Tech Stack:** C/C++, ggml, CUDA, native Vulkan GLSL/SPIR-V, existing CMake and test tools.

**Spec:** `docs/superpowers/specs/2026-10-03-paged-kv-pulsar-design.md`.

## Global Constraints

- Selected PR head: `0b0f7bd7e3c85bda81645edcd7c2c639c67efec0`.
- Both Q8_KV and F16 are mandatory on CPU, CUDA, and Vulkan, including full-attention and SWA caches.
- Default block size is 16; `--kv-paged` is disabled by default.
- Required model architecture is Gemma4: 30 layers, 16 query heads, SWA dimension 256/KV heads 8/window 1024, full dimension 512/KV heads 2.
- Keep existing uncommitted training, CUDA, ggml, argument, and context changes. Apply the upstream change without creating a commit. Do not publish, push, or create an upstream PR.
- Reuse existing test files. Do not add a new file under `tests/`.
- Support CPU placement and placement across CPU plus one CUDA or Vulkan device. General multi-GPU paging is outside the first delivery.
- Use existing Q8_KV packed format: 64 values, FP16 scale, 66 bytes per block. Do not create a persistent F16 backing cache for Q8_KV.
- Use ASCII code and concise necessary comments. Add no VUDA dependency.

## Review Focus

- A page budget that fits a prompt but not its next decode must fail without corrupting the retained sequence. Task 2 tests allocation rollback.
- Partial removal and a divergent shared-prefix append must not change another sequence's packed cache bytes. Task 4 tests copy-on-write at a partial page.
- A prefill ubatch longer than 1024 tokens must retain all entries needed by its early SWA queries. Tasks 2 and 3 test masking and retention before reclamation.
- Quantized zero blocks, negative ties, and token/head strides must use the existing Q8_KV format on both GPU backends. Tasks 3 and 6 compare packed writes and attention.
- Unsupported default-on server features or explicit Hadamard settings must not be silently ignored. Tasks 4 and 5 test startup validation and supported configuration.

---

### Task 1: Preserve the fork and import the selected source

**Files:** Inspect `AGENTS.md`, `CONTRIBUTING.md`, source files named in later tasks, and the selected PR patch. Store source-patch and baseline artifacts outside tracked source files.

**Interfaces:** Produces the upstream page-manager/cache/operator declarations for Tasks 2 and 3, with upstream names retained where suitable.

- [x] Capture tracked and untracked changes and inspect overlapping hunks before applying any source mutation. Prepare an isolated workspace that includes the current fork changes; do not stash or reset the user's workspace.
- [x] Fetch the selected PR revision and inspect its complete diff. Apply the relevant allocator/cache/operator hunks without committing; record omitted example drivers, external scheduler API, and new test executables. Preserve source attribution.
- [x] Establish separate build directories `build-paged-cuda` and `build-paged-vulkan`. Configure CUDA with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='61;70'` and Vulkan with `-DGGML_VULKAN=ON`. Enable existing tests, CLI, server, and bench targets.
- [x] Run relevant pre-import baseline operator/state checks in the preserved fork snapshot. Record failures, executable paths, backend names, and commands so later failures can be compared to the baseline.

### Task 2: Native decode allocation and Gemma4 page geometry

**Files:** `src/llama-block-manager.h`, `src/llama-block-manager.cpp`, `src/llama-kv-cache-paged.h`, `src/llama-kv-cache-paged.cpp`, `src/llama-memory.h`, `src/llama-model.cpp`, `src/llama-context.cpp`, `src/llama-graph.h`, `src/llama-graph.cpp`, `src/models/gemma4.cpp`; test in `tests/test-state-restore-fragmented.cpp`.

**Interfaces:** Implements existing `llama_memory_i::init_batch(llama_batch_allocr &, uint32_t, bool) -> llama_memory_context_ptr`, `init_full()`, `init_update(llama_context *, bool)`, and memory-context `apply()/next()/get_ubatch()`. Produces per-ubatch write slots, page tables, absolute query/key positions, and layer-local cache tensors consumed by Task 3.

- [x] Add cases to the existing sequence/state test for fragmented page assignment, unequal sequence lengths, different `n_batch`/`n_ubatch`, and an allocation failure after a valid prefill. Assert position bounds and retained output remain unchanged after failed preparation. Run the test to demonstrate the missing behavior before implementing it.
- [x] Adapt the upstream allocator to prepare page changes transactionally through the existing memory-context lifecycle. Remove the requirement that an external paged scheduler supply batch metadata.
- [x] Allocate F16 and Q8_KV tensors by layer geometry, using `ggml_row_size` for token/head byte strides. Maintain separate full/SWA pools and metadata. Assert full geometry 512/2 and SWA geometry 256/8 are honored rather than using a global head count.
- [x] Route Gemma4's existing normalized/rotated Q/K and normalized V through paged attention. Preserve its optional V projection behavior, attention scale, output projection, and final softcapping.
- [x] Implement causal/SWA visibility with absolute positions and existing SWA retention policy. Test a multi-ubatch prompt beyond 1024 tokens and an ubatch itself longer than 1024, including positions on page/window boundaries.
- [x] Run the allocation/lifecycle cases. Assert page reuse never exposes removed sequence data and reported packed allocation bytes equal the actual Q8_KV buffers.

### Task 3: CPU reference and native CUDA F16/Q8_KV operations

**Files:** `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cpu/ops.h`, `ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cuda/pagedattn.cuh`, `ggml/src/ggml-cuda/pagedattn.cu`, `ggml/src/ggml-cuda/ggml-cuda.cu`, and existing CUDA attention/set-rows helpers where reused; test in `tests/test-backend-ops.cpp`.

**Interfaces:** Retain the upstream `GGML_OP_PAGED_ATTN` and `ggml_cuda_op_paged_attn(ggml_backend_cuda_context &, ggml_tensor *)` entrypoint, adapting tensor inputs/parameters to Task 2's explicit positions and per-layer byte strides. Cache-write dependencies must be present in the graph. CPU is the numerical reference for the same operation.

- [x] Add paged operator cases with dimensions 256/512, GQA ratios 2/8, block size 16, reordered pages, single-token and chunked queries, and full/SWA masks for both required cache types. Use existing test assertions and backend tolerances; run before implementation to demonstrate lack of operator support.
- [x] Add packed-write assertions against `quantize_row_q8_kv_ref` for zeros, signed inputs, positive/negative rounding boundaries, and successive token/head/page rows. Ensure packed-byte inspection reads the cache after the write operation.
- [x] Implement CPU writes and attention using existing quantization/dequantization functions, explicit byte strides, stable softmax, and the approved visibility contract.
- [x] Implement CUDA F16 and Q8_KV writes and direct paged reads. Reuse existing Q8_KV row conversion logic and match its rounding/zero behavior. Keep accumulation in FP32 and validate actual launch dimensions on Volta.
- [x] Limit backend `supports_op` to implemented shapes, types, strides, and devices. Verify unsupported formats fail validation rather than launching an F16-only kernel against packed bytes.
- [x] Run `build-paged-cuda/bin/test-backend-ops -b CUDA0 -o 'PAGED_ATTN.*'` and the relevant existing `SET_ROWS`/`FLASH_ATTN_EXT` cases. Require nonzero paged executed-case counts and passing output for both types; report existing regression types/counts separately; check device identity before assigning CUDA0 to V100.

### Task 4: Sequence copying, removal, and state restoration

**Files:** `src/llama-kv-cache-paged.h`, `src/llama-kv-cache-paged.cpp`, `src/llama-block-manager.h`, `src/llama-block-manager.cpp`, `src/llama-context.cpp`; test in `tests/test-state-restore-fragmented.cpp` and `tests/test-save-load-state.cpp`.

**Interfaces:** Completes `seq_rm`, `seq_cp`, `seq_keep`, `seq_pos_min/max`, `clear`, `memory_breakdown`, `state_write`, `state_read`, and `get_can_shift` with the signatures already required by `llama_memory_i`.

- [x] Add failing lifecycle assertions for two sequences sharing a partial page then diverging, partial range removal, clear/reuse, and F16/Q8_KV host state round trips. Assert the untouched sequence's next-token logits and shared packed bytes remain unchanged.
- [x] Implement page reference counts and copy-on-write before mutation. Copy packed bytes directly, using the layer's backend transfer mechanism and explicit graph ordering where needed.
- [x] Implement host state serialization and transactional restoration with validated cache geometry, dtypes, positions, and physical capacity. Reject insufficient-capacity or incompatible-state restoration without partially replacing a live sequence.
- [x] Either implement correct RoPE-aware context shifts or report no shift support and reject incompatible configurations. Validate explicit Hadamard policy, speculative/device-only snapshots, multimodal, and training paths before unsupported execution.
- [x] Run existing state tests with both required models and both formats where their invocation supports external models. Report fixture failures separately from paged failures; do not count skipped tests as passed.

### Task 5: Serving options and required CUDA model validation

**Files:** `common/arg.cpp`, `common/common.h`, `common/common.cpp`, `include/llama.h`, `src/llama-cparams.h`, `src/llama-context.cpp`, `tools/server/server-context.cpp`, `tools/server/README.md`; reuse existing server test modules that own parallel requests and prefix reuse.

**Interfaces:** Adds `bool kv_paged` with default false and the source PR's block-size/page-budget parameters to common and context parameters. Propagates existing `-ctk`, `-ctv`, and SWA cache type arguments without changing their defaults.

- [x] Verify help parsing and startup validation for paged F16/Q8_KV, valid differing full/SWA format domains, invalid mixed K/V domains, unsupported architectures, and invalid/overflowing page settings. Define one clear startup failure for each unsupported configuration.
- [x] Wire normal CLI/server decode to the paged memory implementation. Keep the existing server batching scheduler and support CPU plus one GPU layer placement.
- [x] Run both required GGUFs on CUDA with ordinary and paged caches for F16 and Q8_KV. Use fixed prompts, a fixed sampler, prompts longer than 1024, and multiple ubatches. Capture logits comparisons, greedy agreement, and finite outputs for all four model/format pairs.
- [x] Exercise repeated requests, parallel unequal-length prompts, prefix reuse, cancellation, and deliberate pool exhaustion through the server. Confirm requests remain usable after a refused allocation and backend logs show CUDA paged attention.
- [x] Measure matched prompt/decode throughput and peak VRAM before optimizing. Fix material bottlenecks exposed by these measurements without altering cache semantics. Record placement and settings so CPU offload comparisons remain meaningful.

### Task 6: Native Vulkan F16/Q8_KV page operations

**Files:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt`, `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`; create native paged write/attention shader files under `ggml/src/ggml-vulkan/vulkan-shaders/`; tests in `tests/test-backend-ops.cpp`.

**Interfaces:** Implements `GGML_OP_PAGED_ATTN` in the existing Vulkan backend using Task 3's operator contract and Task 2's metadata. Use existing `vk_device`, pipelines, buffers, command submission, and memory barriers. Q8_KV type ID already exists, but the inspected Vulkan tree has no implemented Q8_KV cache-write/attention path.

- [x] Run Task 3's cases against the discovered Vulkan backend names to record missing support before implementing it.
- [x] Add F16 and Q8_KV write shaders and direct paged-read attention shaders. Encode the 66-byte Q8_KV block layout exactly, using supported packing operations when byte storage differs across devices. Test negative values and rounding ties against the existing CPU format.
- [x] Integrate shader generation, pipeline specialization, descriptors, dispatch, and write/read barriers in the current backend. Provide a portable path usable on V100 and P620, without requiring cooperative matrices.
- [x] Implement accurate Vulkan capability checks. Require both F16 and Q8_KV to execute on GPU; prevent silent CPU attention fallback from satisfying acceptance.
- [x] Build Vulkan targets and run the paged operator matrix on V100 and P620. Require all expected cases to execute and pass; inspect diagnostics for byte-stride and barrier problems.
- [x] Run both required models and both formats through Vulkan with CPU layer placement when needed. Compare to CPU/ordinary same-format references where supported and CUDA same-format results. Record if ordinary Vulkan Q8_KV is unavailable; use the CPU operator and CUDA model references instead of inventing a baseline.

### Task 7: Regression checks and serving instructions

**Files:** `tools/server/README.md`, local validation notes under `docs/`; implementation files only for regressions identified during verification.

**Interfaces:** Produces tested command lines and evidence covering model, cache format, backend, placement, pool capacity, and performance.

- [x] Run `git diff --check`, relevant operator/state/server tests, and affected training checks against the preserved fork baseline. Verify ordinary inference still runs with paged mode off. Summarize pre-existing failures separately.
- [x] Complete the acceptance matrix: Pulsar/Pulsar S x F16/Q8_KV x CUDA/Vulkan. Record backend execution, same-format correctness, long/chunked prompts, repeated/parallel requests, and page-exhaustion recovery.
- [x] Benchmark variable-length concurrent serving with matched context and physical budgets. Record prompt/decode throughput, peak VRAM, actual allocated packed bytes, live/free page counts, and graph-rebuild cost.
- [x] Document the supported commands, GPU/CPU placement, full/SWA type controls, budget controls, and any explicitly rejected features. Include Q8_KV as the main serving example.
- [x] Review the complete diff for preserved existing changes, unsupported no-op methods, stale scheduler assumptions, accidental dtype conversions, and unverified support claims. Report completion only when all required acceptance cells are verified.

## Execution result

Completed inline on 2026-10-04 and applied to the main workspace without commits. All required CUDA/Vulkan x Pulsar/Pulsar S x F16/Q8_KV acceptance cells are verified. See `docs/paged-kv-validation.md` for test counts, lifecycle evidence, measured ordinary-logit differences, throughput/VRAM, rejected features and skipped device-state cases. The initial user work remains preserved. The original server is restored; MuSR remains stopped. No commit or external submission was made.

## Follow-up: image inference

The user subsequently requested image inference. The existing MTMD path now drives image-aware paged attention on CPU/CUDA/Vulkan for F16 and Q8_KV. See [multimodal validation](../../paged-kv-multimodal-validation.md) for verified scope and limits. The initial text-only acceptance above is historical.
