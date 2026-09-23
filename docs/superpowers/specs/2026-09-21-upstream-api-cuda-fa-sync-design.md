# Upstream API and CUDA Flash Attention Sync Design

## Goal

Synchronize the private `aikar-engine` fork with the latest `upstream/master` while preserving its existing features, optimizations, and Q8_KV support. The synchronized tree must use upstream APIs and implementation structure as the baseline, including the latest CUDA Flash Attention architecture.

The target CUDA platform for Q8_KV Flash Attention is NVIDIA V100 with compute capability 7.0.

## Baseline

The upstream baseline is commit `ce8caa6e6` from 2026-09-20. The current fork is 139 upstream commits behind and contains 313 commits that are not in upstream. The working tree is clean before synchronization.

A merge-tree analysis identifies eight direct content conflicts:

- `common/CMakeLists.txt`
- `ggml/src/ggml-cpu/ops.cpp`
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`
- `ggml/src/ggml-cuda/fattn.cu`
- `ggml/src/ggml-openvino/ggml-quants.cpp`
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `src/llama-model.h`
- `tests/test-backend-ops.cpp`

All eight paths require explicit resolution.

## Sync Strategy

Merge `upstream/master` into the current branch without creating a commit. Preserve the existing fork history and resolve conflicts against the upstream side as the structural baseline.

Do not rebase the 313 fork commits. Do not reset to upstream and replay features. Those approaches would require reconstructing a large, interdependent patch stack and would increase the chance of silent feature loss.

For conflicting and automatically merged files:

1. Adopt upstream public APIs, internal interfaces, file layout, and call conventions.
2. Reapply fork behavior through the new upstream interfaces.
3. Remove fork code only when upstream has an equivalent implementation and the fork behavior remains intact.
4. Preserve fork-only features unless they cannot be represented safely in the new structure. Any such case must be reported instead of silently removed.

## Feature Preservation

The synchronization must retain all current fork functionality, including these major areas:

- Q8_KV type definitions, quantization, serialization, KV cache use, CLI selection, and backend support
- V100 Q8_KV Flash Attention optimization
- finetuning, QAT, quantized optimizers, gradient and phase offload, and resume behavior
- TurboQuant and custom quantization paths
- diffusion functionality
- pruning, model merge, aikar bench, and other fork tools
- server extensions and custom request handling
- existing CPU, CUDA, Vulkan, Metal, SYCL, OpenVINO, and RPC adaptations

This list identifies high-risk areas but does not limit preservation to those areas. The complete pre-merge fork diff is the preservation inventory.

## CUDA Flash Attention

Use the latest upstream Flash Attention dispatch, templates, launch signatures, sparse handling, and architecture checks as the base.

Reintroduce Q8_KV support at the narrowest compatible extension points:

- register Q8_KV template instances using the current upstream generation layout
- support dense Q8_KV K/V dispatch without bypassing newer upstream eligibility checks
- port the V100 sparse Q8_KV MMA loader to the current shared-memory and launch interfaces
- retain byte-based global Q8_KV strides and F16 shared-memory tiles
- retain bounds checks for the padded 32-column Volta MMA geometry
- retain the Gemma 4 decode specialization for one query token, D256/V256, GQA ratio 2, and a positive sparse KV bound
- keep unsupported shapes and modes on the current upstream-compatible fallback path

The implementation must not restore obsolete upstream interfaces just to fit the old fork kernel.

## API Compatibility

Public C and C++ headers, common argument handling, CLI behavior, server APIs, model loading, context construction, and backend interfaces follow the upstream baseline.

Fork-specific API additions remain available where they do not conflict with upstream names or semantics. If upstream changed a signature used by fork code, all fork callers move to the upstream signature. Compatibility shims are allowed only when required for an existing fork-specific external interface.

## Conflict and Regression Audit

Direct conflict resolution is not sufficient because Git can automatically merge semantically incompatible edits. After the merge:

- compare every file changed on both sides since the previous merge base
- inspect removed or renamed upstream interfaces for remaining fork callers
- search for Q8_KV registrations, switch cases, traits, serialization sizes, backend capability checks, and generated CUDA instances
- inspect deleted upstream CUDA sources so fork-only operators are not lost through CMake or dispatcher changes
- build with warnings visible and treat missing enum cases, unresolved symbols, and stale signatures as sync defects

## Verification

Verification proceeds from broad compile checks to the target optimization:

1. Configure and build a CPU-only tree.
2. Run the relevant argument parser, quantization, model, state, chat, optimizer, and backend tests already present in the repository.
3. Configure and build CUDA for compute capability 7.0.
4. Run CUDA backend Flash Attention cases, including Q8_KV cases in the existing backend test file.
5. Run a V100 server with `-fa on -ctk q8_kv -ctv q8_kv` and a Gemma 4 model.
6. Verify warmup and completion for sliding-window and full-attention layers.
7. Compare deterministic output or logits with Flash Attention disabled using tolerances appropriate for Q8_KV and F16 MMA accumulation.
8. Confirm that the sparse Q8_KV MMA path is selected only for its supported V100 shape and that other cases use a valid fallback.
9. Compare representative decode performance to the pre-sync implementation when a usable pre-sync binary or measurement is available.

If V100 hardware or a suitable model is unavailable in the execution environment, complete compile and unit coverage and report the missing runtime verification explicitly.

## Safety and Handoff

Do not commit, push, or create a pull request. Leave the synchronized changes in the working tree for human review. Preserve any user changes that appear during implementation and stop if they overlap the sync edits in a way that cannot be resolved safely.
