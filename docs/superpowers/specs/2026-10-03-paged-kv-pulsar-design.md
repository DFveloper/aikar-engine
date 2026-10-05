# Paged KV for Pulsar and Pulsar S

## Agreed objective

Integrate llama.cpp PR #22569 into this fork, expose paged KV as an optional inference mode, finish CUDA support first, and implement Vulkan support next. The user selected this PR, confirmed understanding and fork-only use, and named Pulsar and Pulsar S as the first required models. Broader support is allowed where the same implementation is correct. This is an architectural change across memory management, graph construction, and backend execution.

Q8_KV is the user's main serving format. Both Q8_KV and F16 are mandatory on CPU, CUDA, and Vulkan, including full-attention and SWA caches.

Keep existing uncommitted training, CUDA, ggml, argument, and context changes. Apply the upstream change without creating a commit. Do not publish, push, or create an upstream PR.

## Sources and inspected model facts

- Upstream PR: https://github.com/ggml-org/llama.cpp/pull/22569
- Selected PR head: `0b0f7bd7e3c85bda81645edcd7c2c639c67efec0`.
- Current fork HEAD at investigation: `0a801ac39`.
- Required Pulsar model: `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar-LD-Q4_0_M.gguf`.
- Required Pulsar S model: `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf`.

Both inspected GGUFs use architecture `gemma4`, 30 layers, 16 query heads, and a repeating pattern of five SWA layers followed by one full-attention layer. SWA uses head dimension 256, eight KV heads, and window 1024. Full attention uses head dimension 512 and two KV heads. Neither inspected model uses shared KV layers. Final logit softcapping is 30 and must remain in the existing model graph.

The local Gemma4 graph applies separate Q/K normalization and RoPE, normalizes V, and can obtain V from the unnormalized K projection when the V projection is absent. Preserve these operations and existing attention scale. A paged implementation must never substitute the final rotated K for V.

## Approach choice

Use #22569's block allocator, block-table representation, and CPU/CUDA operation as the source implementation. Adapt its memory integration to the current fork rather than replacing existing memory interfaces. Its external scheduler is not sufficient for CLI/server use: `init_batch` requires external paged batch metadata, several sequence/state methods are empty, geometry is uniform across layers, and Vulkan is absent.

The normal `llama_decode` path must own allocation and create metadata for each ubatch. Keep the existing server's batching and request lifecycle. Do not require callers to use a second scheduler API. Do not import unrelated example drivers or new test executables as a condition of enabling the feature. Keep upstream attribution and record any omitted parts of the source patch.

Vulkan will use native ggml Vulkan dispatch and shaders. VUDA's documented interface launches separately compiled SPIR-V modules; it does not make the selected CUDA kernel directly reusable. Adding a second Vulkan runtime would introduce buffer and synchronization integration work. Share the layout, operation parameters, reference behavior, and tests across backends instead. Source: https://github.com/jgbit/vuda

## Memory and batch design

Store page metadata on the host and backend page tensors on the device selected for each layer. Each layer's tensor uses its own KV-head count and head dimension. Layers may share logical page numbering where lifetimes match, but tensor strides and byte sizes must be derived per layer. Separate full-attention and SWA allocation domains so window eviction cannot release full-attention history.

Support CPU placement and placement across CPU plus one CUDA or Vulkan device. Required models must not depend on all weights fitting in one GPU. General multi-GPU paging and inter-device page migration are outside the first delivery.

Use a fixed physical page budget, with default block size 16. Derive the default pool capacity from context and sequence settings with checked arithmetic; explicit pool configuration may lower it. Report actual buffer bytes, allocated pages, and free capacity. Paged KV reduces fragmentation and shares free capacity across sequences; it does not reduce the bytes of an individual retained KV token.

Build write slots, sequence-to-page tables, logical positions, context lengths, and query positions for each ubatch. Handle arbitrary legal `n_batch`/`n_ubatch` combinations. Allocate transactionally before computation: an allocation failure must not leave partially changed sequence metadata. Releasing a sequence returns its pages. Reusing a page must never expose another sequence's keys or values.

Initially retain SWA history according to the existing `--swa-full` policy and reference cache semantics. Apply the exact causal/window visibility rule using absolute positions, including chunked prefill that spans more than one window. Reclaim pages only when the whole page is outside all retained history needed by an active sequence. Conservatively retaining a boundary page is acceptable. Do not overwrite older entries before all queries in the ubatch that need them have executed.

## Graph and backend contract

Preserve the upstream paged operation where practical, but represent cache writes and reads with explicit graph dependencies so scheduling cannot move a read before its write. Pass tensor strides rather than assuming contiguous model projections. Handle different full/SWA dimensions and layer-specific scale. Keep output in the layout expected by the existing attention output projection.

CPU supplies a correctness reference. CUDA supplies native paged cache writes and attention with stable online softmax and FP32 accumulation. Verify head dimensions 256 and 512 on Volta. Capability checks must match implemented dtypes, shapes, devices, and metadata; claiming all paged operations are supported is insufficient.

Reuse the fork's `GGML_TYPE_Q8_KV` definition and reference quantization. Each 64-element block has an FP16 scale and 64 signed int8 values, totaling 66 bytes. Quantization uses max-absolute-value divided by 127, round-to-nearest with the existing `roundf` behavior, saturation to [-127, 127], and zero-block handling. Keep row boundaries within each token/head and derive byte strides with `ggml_row_size`. Copy-on-write and serialization preserve packed bytes. Both GPU backends quantize page writes and dequantize the attended rows directly without creating a persistent F16 cache. Preserve existing explicit Hadamard policy, or reject a conflicting requested policy before execution rather than ignoring it.

Vulkan implements the same operation in the existing shader build and command submission path. Use existing backend buffers, barriers, descriptor management, and pipeline caching. Test a portable shader path on V100 and P620. Optional device-specific optimization must not be necessary for correct output. GPU inference must use the GPU paged operation; silent CPU attention fallback is not a successful Vulkan implementation.

Allow graph reuse only if all page mappings and logical metadata are refreshed and cache writes remain correct. Otherwise use the existing graph-rebuild mechanism and measure its cost.

## Sequence lifecycle and server behavior

Implement full and partial `seq_rm`, `seq_keep`, and position bounds. Implement sequence copying with page refcounts and copy-on-write for a shared page before mutation. This is required for ordinary shared-prefix batches and server sequence operations even though automatic prefix deduplication is outside scope.

Integrate through `llama_memory_i` so CLI and server continue to call normal memory operations. Test repeated requests, prefix reuse, cancellation, and pool exhaustion. Implement host state save/restore through existing state APIs for the supported paged mode, with validation of shape, type, position, and page budget before restoration. Never inherit upstream no-op state methods.

Context shifting requires both logical-position updates and appropriate K RoPE updates. If it cannot be supported correctly in the first delivery, return `get_can_shift() == false` and reject conflicting CLI/server configurations with a clear error. Do not silently implement position changes without updating K. Device-only state snapshots, speculative decoding, multimodal inference, and training are not part of the initial support claim; conflicting paths must be checked and rejected before execution.

## Options and support boundary

- Add `--kv-paged` through common arguments and public context parameters, disabled by default.
- Expose block size and physical page budget using the upstream option names where they still describe implemented behavior. Do not expose CPU swap controls unless swap-in/out is integrated and verified in normal inference.
- Required formats are F16 K/V and Q8_KV K/V, including full/SWA caches, on CPU, CUDA, and Vulkan. Continue to use existing K/V and SWA cache-type arguments; paged mode does not change their defaults. Each cache domain may select F16 or Q8_KV independently. Mixed K/V within one domain and other unsupported combinations must produce a clear startup error. Model weight quantization remains independent and unchanged.
- CUDA is implemented and verified before Vulkan. CPU reference support is required for numerical comparisons.
- Pulsar and Pulsar S text inference is the required end-to-end acceptance target. Broader Gemma4/dense-attention support may use the same verified geometry and lifecycle checks. Do not use model filenames as capability checks.
- Recurrent, MLA/DSA, unsupported layer-sharing, and other incompatible architectures must fail clearly when paged mode is requested. The ordinary cache remains available with the option off.

## Verification and acceptance

Reuse `tests/test-backend-ops.cpp`, existing state/sequence tests, and server tests; do not add a new file under `tests/`. First run the relevant baseline checks and record pre-existing failures separately.

Backend comparisons must cover noncontiguous physical page order, page boundaries, dimensions 256 and 512, GQA ratios 2 and 8, single-token decode, chunked prefill, causal/window boundaries, and differing logical positions across sequences. Compare numerical output against CPU and ordinary attention with documented existing backend tolerances. Use meaningful cases that expose page-index and mask errors.

Run these checks for both F16 and Q8_KV. Q8_KV tests must check packed row writes against existing CPU quantization, including zero blocks, negative values, rounding boundaries, and head/page byte strides. Compare paged Q8_KV logits to ordinary Q8_KV and paged F16 logits to ordinary F16; do not require quantized results to be identical to F16. Verify that reported Q8_KV pool bytes match the packed allocation rather than an F16 backing store.

Lifecycle checks cover partial removal, shared prefixes followed by divergent writes, sequence release/reuse, metadata rollback after pool exhaustion, and supported state round trips. Compare paged and ordinary logits for fixed prompts, and record greedy-token agreement for both required models. Include prompts beyond the 1024-token SWA window and multiple ubatches.

Build CPU/CUDA and Vulkan configurations. Run actual CUDA and Vulkan execution with both required models on hardware that fits them, allowing CPU layer placement. Use P620 for operator checks and a bounded inference check if model placement permits. Measure prompt/decode throughput, peak VRAM, physical KV budget, and page utilization at matched settings and concurrent variable-length requests. Report regressions and limitations with the measurements; do not assume upstream benchmark claims apply to these models.

Completion requires an available option, a built executable, successful Pulsar/Pulsar S text inference with both F16 and Q8_KV through both GPU backends, numerical and lifecycle evidence, and usage instructions with actual tested commands. A compiled shader or a CPU fallback alone does not establish Vulkan support.

## Integration order

1. Preserve the current work and inspect/apply the selected source patch without a commit.
2. Adapt memory and graph integration for normal decode, per-layer geometry, and Gemma4 SWA.
3. Complete CPU reference, CUDA execution, options, capability validation, and lifecycle behavior.
4. Verify required models on CUDA and address correctness or material performance problems.
5. Add native Vulkan execution using the same contract and verify required models.
6. Run relevant regressions and publish local validation notes and tested commands.

The user approved this design with the correction that both Q8_KV and F16 are required. This document incorporates that correction. No source implementation has been changed at the design stage.
