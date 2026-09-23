# Upstream API and CUDA Flash Attention Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize `aikar-engine` with upstream commit `ce8caa6e6` while preserving all fork functionality and porting Q8_KV V100 Flash Attention to the current upstream CUDA architecture.

**Architecture:** Merge upstream without committing, use upstream APIs and implementation structure as the base, then restore fork-only behavior at current extension points. Audit both direct conflicts and silent automatic merges, with special treatment for Q8_KV registration, KV cache behavior, and CUDA Flash Attention dispatch and kernels.

**Tech Stack:** C++17, C, CUDA C++ for sm_70, CMake, ggml, llama.cpp backend tests, Python test utilities

**Spec:** `docs/superpowers/specs/2026-09-21-upstream-api-cuda-fa-sync-design.md`

## Global Constraints

- Use upstream commit `ce8caa6e6` as the API and implementation baseline.
- Preserve all fork features represented by the 313 fork-only commits.
- Preserve Q8_KV type, serialization, KV cache support, backend support, and CLI/API selection.
- Preserve the V100 compute capability 7.0 Q8_KV sparse Flash Attention path.
- Do not restore obsolete upstream APIs to accommodate old fork code.
- Do not add new test files; extend existing tests where coverage is needed.
- Use ASCII only in code and comments.
- Do not commit, push, or create a pull request.

## Review Focus

- Q8_KV enum and trait numbering must remain consistent across C, C++, Python GGUF, serialization, and every backend.
- Unsupported Q8_KV Flash Attention shapes must select a valid upstream-compatible fallback instead of asserting or launching an invalid template.
- V100 sparse Q8_KV decode must use byte strides for global quantized K/V while retaining F16 shared-memory tile strides.
- Upstream API signature changes must propagate to fork-only callers without compatibility regressions in public fork extensions.
- Automatically merged build lists and dispatch tables must retain fork-only operators and generated CUDA instances.

---

### Task 1: Capture the Pre-Merge Preservation Inventory

**Files:**
- Modify: `docs/superpowers/plans/2026-09-21-upstream-api-cuda-fa-sync.md`
- Inspect: `ggml/include/ggml.h`
- Inspect: `include/llama.h`
- Inspect: `ggml/src/ggml-cuda/CMakeLists.txt`
- Inspect: `ggml/src/ggml-cuda/fattn.cu`
- Inspect: `ggml/src/ggml-cuda/fattn-mma-f16.cuh`

**Interfaces:**
- Consumes: Git merge base `2a3005c23f60cb38dab70b8ea2ddbd969bcf3e87`
- Produces: Exact pre-merge HEAD identifier and lists of fork-only files, overlapping files, Q8_KV references, CUDA source registrations, and available build/test configuration

- [ ] **Step 1: Record repository state and immutable identifiers**

Run:

```bash
git status --short
git rev-parse HEAD
git rev-parse upstream/master
git merge-base HEAD upstream/master
```

Expected: clean except for the approved spec and plan documents; HEAD is `7027369e9`; upstream is `ce8caa6e6`.

- [ ] **Step 2: Save comparison inventories outside the repository**

Run:

```bash
git diff --name-only 2a3005c23..HEAD | sort > /tmp/aikar-local-files-before-sync.txt
git diff --name-only 2a3005c23..upstream/master | sort > /tmp/aikar-upstream-files-before-sync.txt
comm -12 /tmp/aikar-local-files-before-sync.txt /tmp/aikar-upstream-files-before-sync.txt > /tmp/aikar-overlap-files-before-sync.txt
rg -l 'Q8_KV|q8_kv' > /tmp/aikar-q8-kv-files-before-sync.txt
```

Expected: all four files are nonempty and the overlap includes both CUDA Flash Attention files.

- [ ] **Step 3: Capture build and generated-instance registrations**

Run:

```bash
rg -n 'Q8_KV|q8_kv|turbo|diffusion|opt-step|out-prod|set-rows' ggml/src/ggml-cuda/CMakeLists.txt ggml/src/ggml-cuda/template-instances CMakeLists.txt > /tmp/aikar-cuda-registrations-before-sync.txt
```

Expected: Q8_KV and fork-only CUDA sources or instances appear in the output.

### Task 2: Merge Upstream and Resolve Non-FA Conflicts

**Files:**
- Modify: `common/CMakeLists.txt`
- Modify: `ggml/src/ggml-cpu/ops.cpp`
- Modify: `ggml/src/ggml-openvino/ggml-quants.cpp`
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- Modify: `src/llama-model.h`
- Modify: `tests/test-backend-ops.cpp`
- Defer: `ggml/src/ggml-cuda/fattn.cu`
- Defer: `ggml/src/ggml-cuda/fattn-mma-f16.cuh`

**Interfaces:**
- Consumes: Upstream APIs from `ce8caa6e6` and fork-only symbols from pre-merge HEAD `7027369e9`
- Produces: A merged, uncommitted tree with upstream structure and all non-FA fork symbols retained

- [ ] **Step 1: Start the non-committing merge**

Run:

```bash
git merge --no-commit --no-ff upstream/master
```

Expected: merge stops with eight known content conflicts and does not create a commit.

- [ ] **Step 2: Resolve common and CPU registrations**

Use the upstream ordering and current source lists in `common/CMakeLists.txt`. Retain every fork-only source that existed in the pre-merge side. In `ggml/src/ggml-cpu/ops.cpp`, use upstream operator dispatch and validation structure, then restore fork-only quantized optimizer, QAT, TurboQuant, diffusion, and training cases at the matching switch sites.

Run:

```bash
rg -n '^(<<<<<<<|=======|>>>>>>>)' common/CMakeLists.txt ggml/src/ggml-cpu/ops.cpp
git diff --check -- common/CMakeLists.txt ggml/src/ggml-cpu/ops.cpp
```

Expected: no conflict markers and no whitespace errors.

- [ ] **Step 3: Resolve OpenVINO and Vulkan Q8_KV handling**

Use current upstream type and capability switches as the base. Restore Q8_KV cases with the same support or rejection semantics as before the merge; do not claim native support where the backend previously used a fallback.

Run:

```bash
rg -n 'Q8_KV|q8_kv' ggml/src/ggml-openvino/ggml-quants.cpp ggml/src/ggml-vulkan/ggml-vulkan.cpp
rg -n '^(<<<<<<<|=======|>>>>>>>)' ggml/src/ggml-openvino/ggml-quants.cpp ggml/src/ggml-vulkan/ggml-vulkan.cpp
```

Expected: Q8_KV cases remain present and conflict markers are absent.

- [ ] **Step 4: Resolve model declarations and backend test matrix**

Use upstream declaration order and signatures in `src/llama-model.h`, then reinsert fork-only model, training, and tensor access declarations. Use the new upstream Flash Attention test matrix shape in `tests/test-backend-ops.cpp`, retaining Q8_KV coverage in that existing file.

Run:

```bash
rg -n '^(<<<<<<<|=======|>>>>>>>)' src/llama-model.h tests/test-backend-ops.cpp
git diff --check -- src/llama-model.h tests/test-backend-ops.cpp
```

Expected: no conflict markers or whitespace errors.

### Task 3: Port Q8_KV to the Current CUDA Flash Attention Dispatcher

**Files:**
- Modify: `ggml/src/ggml-cuda/fattn.cu`
- Modify: `ggml/src/ggml-cuda/fattn-common.cuh`
- Modify: `ggml/src/ggml-cuda/fattn-vec.cuh`
- Modify: `ggml/src/ggml-cuda/fattn-tile.cuh`
- Modify: `ggml/src/ggml-cuda/CMakeLists.txt`
- Modify: `ggml/src/ggml-cuda/template-instances/generate_cu_files.py`
- Modify: `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_kv-q8_kv.cu`
- Test: `tests/test-backend-ops.cpp`

**Interfaces:**
- Consumes: Upstream `ggml_cuda_flash_attn_ext` dispatch and current `launch_fattn` template signatures
- Produces: Dense Q8_KV K/V Flash Attention dispatch plus safe fallback for unsupported shapes

- [ ] **Step 1: Adapt existing Q8_KV backend cases to the upstream test schema**

Retain a dense Q8_KV K/V case and the V100 sparse D256/V256/GQA2 case in the existing Flash Attention parameter matrix. Ensure the unsupported-shape coverage includes at least one non-D256 head dimension or multi-query-token case that must not select sparse MMA.

- [ ] **Step 2: Run the CUDA test build before dispatcher changes**

Run:

```bash
cmake -S . -B build-cuda-sync -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DLLAMA_BUILD_TESTS=ON
cmake --build build-cuda-sync --target test-backend-ops -j2
```

Expected: configure, compile, or link failure identifies stale Q8_KV dispatcher or template interfaces.

- [ ] **Step 3: Restore Q8_KV dense dispatch using current launch signatures**

In `fattn.cu`, add Q8_KV to current type eligibility and template selection without bypassing upstream checks for mask, bias, softcap, head sizes, or GPU architecture. Update `fattn-common.cuh`, `fattn-vec.cuh`, and `fattn-tile.cuh` only where current type traits or loaders require Q8_KV support.

- [ ] **Step 4: Restore generated Q8_KV instances**

Update the generator's current type lists and regenerate or mechanically update the Q8_KV instance using the exact current upstream instance format. Ensure `ggml/src/ggml-cuda/CMakeLists.txt` compiles it once.

Run:

```bash
rg -n 'q8_kv|Q8_KV' ggml/src/ggml-cuda/CMakeLists.txt ggml/src/ggml-cuda/template-instances/generate_cu_files.py ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_kv-q8_kv.cu
```

Expected: one source registration and valid Q8_KV template instantiation.

- [ ] **Step 5: Rebuild the CUDA backend test**

Run:

```bash
cmake --build build-cuda-sync --target test-backend-ops -j2
```

Expected: dense Q8_KV dispatcher and instance code compile for sm_70.

### Task 4: Port the V100 Sparse Q8_KV MMA Kernel

**Files:**
- Modify: `ggml/src/ggml-cuda/fattn-mma-f16.cuh`
- Modify: `ggml/src/ggml-cuda/fattn.cu`
- Modify: `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_32.cu`
- Modify: `ggml/src/ggml-cuda/template-instances/generate_cu_files.py`
- Verify: `src/llama-graph.cpp`
- Test: `tests/test-backend-ops.cpp`

**Interfaces:**
- Consumes: Upstream sparse mask compaction, current MMA launch parameters, `block_q8_kv`, and graph-provided positive `n_kv_max`
- Produces: V100 sparse Q8_KV decode for ncols1=1, D256/V256, GQA2, with dense Q8_KV fallback outside that shape

- [ ] **Step 1: Preserve the graph sparse bound through current upstream parameters**

Inspect `src/llama-graph.cpp` and current Flash Attention op parameters. Keep `min(hparams.n_swa, k->ne[2])` for sliding-window layers and zero for full-attention layers, translated to the current upstream parameter setter if its representation changed.

- [ ] **Step 2: Port sparse eligibility before dense Q8_KV selection**

In `fattn.cu`, express the old V100 Q8_KV predicate through the current upstream sparse-selection helper: sm_70, one query token, Q8_KV K and V, D256/V256, GQA2, mask present, no unsupported bias or softcap, positive sparse bound, and the existing KV-length amortization threshold. Evaluate it before returning a dense Q8_KV vector or tile kernel.

- [ ] **Step 3: Port the quantized sparse loader to the current MMA tile API**

In `fattn-mma-f16.cuh`, load selected `block_q8_kv` rows with byte-based global strides, convert packed int8 values with the stored F16 scale, and write the existing F16 swizzled shared-memory tiles. Retain zero fill for invalid padded K/V slots, negative infinity for invalid mask slots, and logical-head bounds on Q, sink, output, and fixup accesses.

- [ ] **Step 4: Update the ncols1=1, ncols2=32 instance**

Use the current upstream template argument order and generator output. Instantiate the D256/V256 path required by V100 without restoring removed template parameters.

- [ ] **Step 5: Compile the sparse path for sm_70**

Run:

```bash
cmake --build build-cuda-sync --target test-backend-ops -j2
```

Expected: the sparse Q8_KV MMA templates compile without ambiguous overloads, invalid shared-memory types, or launch signature mismatches.

- [ ] **Step 6: Run focused Flash Attention backend cases**

Run:

```bash
build-cuda-sync/bin/test-backend-ops test -b CUDA -o FLASH_ATTN_EXT
```

Expected: supported dense and sparse cases pass; unsupported shapes complete through fallback without assertion.

### Task 5: Audit Automatically Merged APIs and Fork Features

**Files:**
- Modify as required: every path in `/tmp/aikar-overlap-files-before-sync.txt`
- Verify: `include/llama.h`
- Verify: `ggml/include/ggml.h`
- Verify: `common/arg.cpp`
- Verify: `src/llama-context.cpp`
- Verify: `src/llama-model.cpp`
- Verify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Verify: `tools/server/server-context.cpp`

**Interfaces:**
- Consumes: Pre-merge overlap, Q8_KV, and CUDA registration inventories
- Produces: No stale upstream signatures, missing fork call sites, missing enum cases, or omitted build registrations

- [ ] **Step 1: Check every overlap file against both parents**

For each path in `/tmp/aikar-overlap-files-before-sync.txt`, compare the merged result with `HEAD^1` equivalent pre-merge blob `7027369e9:<path>` and `upstream/master:<path>`. Confirm that upstream structural changes and fork-only semantics both remain.

- [ ] **Step 2: Compare Q8_KV coverage before and after**

Run:

```bash
rg -l 'Q8_KV|q8_kv' | sort > /tmp/aikar-q8-kv-files-after-sync.txt
comm -23 /tmp/aikar-q8-kv-files-before-sync.txt /tmp/aikar-q8-kv-files-after-sync.txt
```

Expected: no missing path unless upstream removed the file and its responsibility moved to a verified replacement.

- [ ] **Step 3: Compare CUDA registrations before and after**

Run:

```bash
rg -n 'Q8_KV|q8_kv|turbo|diffusion|opt-step|out-prod|set-rows' ggml/src/ggml-cuda/CMakeLists.txt ggml/src/ggml-cuda/template-instances CMakeLists.txt > /tmp/aikar-cuda-registrations-after-sync.txt
diff -u /tmp/aikar-cuda-registrations-before-sync.txt /tmp/aikar-cuda-registrations-after-sync.txt
```

Expected: differences correspond only to upstream file moves, signature updates, or newly added registrations; every removed fork registration has a verified replacement.

- [ ] **Step 4: Search for unresolved merge and API artifacts**

Run:

```bash
rg -n '^(<<<<<<<|=======|>>>>>>>)' . --glob '!vendor/**'
git diff --check
git status --short
```

Expected: no conflict markers or whitespace errors; all merge conflicts are resolved.

### Task 6: Build and Test the Synchronized Fork

**Files:**
- Modify as required: files exposing compile or test regressions
- Test: existing test targets only

**Interfaces:**
- Consumes: Fully resolved merge and ported Q8_KV CUDA implementation
- Produces: Verified CPU build, sm_70 CUDA build, existing feature tests, and an explicit V100 runtime status

- [ ] **Step 1: Configure and build CPU targets**

Run:

```bash
cmake -S . -B build-cpu-sync -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cpu-sync -j2
```

Expected: full CPU build succeeds.

- [ ] **Step 2: Run focused fork and API tests**

Run:

```bash
ctest --test-dir build-cpu-sync --output-on-failure -R 'test-(arg-parser|backend-ops|chat|llama-archs|model-merge|model-quantize|opt|quant-type-selection|quantize-fns|save-load-state|turbo-quant)'
```

Expected: all selected tests pass.

- [ ] **Step 3: Complete the CUDA sm_70 build**

Run:

```bash
cmake --build build-cuda-sync -j2
```

Expected: full CUDA build succeeds for compute capability 7.0.

- [ ] **Step 4: Run CUDA backend tests**

Run:

```bash
build-cuda-sync/bin/test-backend-ops test -b CUDA
```

Expected: backend tests pass, including Q8_KV Flash Attention.

- [ ] **Step 5: Run V100 model verification when assets are available**

Run the available Gemma 4 GGUF with the synchronized server using:

```bash
model_path=$(rg --files /home/user -g '*.gguf' | rg -i 'gemma.?4' | rg -v '/models/ggml-vocab-' | head -n 1)
test -n "$model_path"
CUDA_VISIBLE_DEVICES=0 build-cuda-sync/bin/llama-server -m "$model_path" -fa on -ctk q8_kv -ctv q8_kv -c 262144 -kvu -np 4
```

Expected: startup, warmup, and a completion succeed. Sliding-window layers select sparse Q8_KV MMA; full-attention and unsupported shapes use valid fallback paths. If no model or V100 is available, report this step as not run rather than passing it.

- [ ] **Step 6: Final integrity check**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors, no unmerged paths, no commit created, and only the synchronized implementation plus approved design and plan documents are present.
