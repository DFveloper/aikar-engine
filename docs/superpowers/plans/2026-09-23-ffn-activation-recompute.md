# Gemma4 FFN Activation Recomputation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make optional activation recomputation reduce Gemma4 dense-FFN scheduler memory while leaving KV and attention semantics unchanged.

**Architecture:** Annotate pure FFN regions in the model graph, clone their forward nodes for backward rematerialization, and redirect FFN backward dependencies to the clones. The original FFN intermediates then have forward-only lifetimes; one reusable rematerialization workspace is sufficient. Attention, KV writes, MoE, and per-layer embedding branches remain outside the first implementation.

**Tech Stack:** C++17, ggml graph/autograd internals, backend scheduler allocator, Gemma4 graph builder, `test-opt`

**Spec:** `docs/superpowers/specs/2026-09-23-q4-fused-training-memory-design.md`

## Global Constraints

- Expose `--activation-recompute on|off`; default is off.
- First implementation supports only pure dense Gemma4 FFN regions.
- Do not replay attention or KV writes.
- Unsupported graphs fail clearly when recomputation is on.
- Remove the current persistent-node checkpoint behavior.
- Do not add a new test file.
- Keep new source and comments ASCII-only.
- Do not commit, push, or create a PR without explicit user approval.

## Review Focus

- Tied parameters used in multiple FFN regions must accumulate gradients once per use; Task 3 compares every parameter gradient.
- Residual edges must keep `attn_out` live without retaining internal FFN tensors; Task 2 inspects liveness.
- Dynamic QLion callback dependencies must refer to rematerialized nodes correctly; Task 4 adds a QLion update test.
- A graph containing MoE or per-layer embedding inside a requested region must reject the option; Task 1 tests both.
- Graph reuse across on/off modes must be impossible; Task 4 checks cache identity.

---

### Task 1: Add the public option and explicit support validation

**Files:**
- Modify: `common/common.h`
- Modify: `common/arg.cpp`
- Modify: `include/llama.h`
- Modify: `ggml/include/ggml-opt.h`
- Modify: `examples/qlora_training/finetune_qlora.cpp`
- Modify: `examples/qlora_training/finetune_qat.cpp`
- Modify: `examples/qlora_training/README.md`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: `bool activation_recompute` in common, llama, and ggml optimizer parameters.
- Deprecates: `grad_checkpoint_interval`; any value above zero maps to recomputation on with one warning.

- [ ] **Step 1: Add argument parsing tests**

Verify `off`, `on`, invalid values, default off, and deprecated `--grad-checkpoint 48`. The deprecated form must set `activation_recompute = true` and emit one warning.

- [ ] **Step 2: Run parser and optimizer tests to verify RED**

Expected: `--activation-recompute` is unknown.

- [ ] **Step 3: Add the option and parameter propagation**

Parse exactly:

```cpp
if (value == "on")  params.activation_recompute = true;
else if (value == "off") params.activation_recompute = false;
else throw std::invalid_argument("invalid --activation-recompute: expected on or off");
```

Pass the boolean through both trainer initializers into `ggml_opt_params`.

- [ ] **Step 4: Remove fake checkpoint marking**

Delete the block in `ggml_opt.cpp` that marks every Nth F32 forward node as `GGML_TENSOR_FLAG_OUTPUT`. Update API comments and README so no text claims zero-compute checkpointing.

- [ ] **Step 5: Add support validation**

When recomputation is on, require Gemma4, dense FFN regions, and no unsupported per-layer branch inside a region. Return initialization failure with the first unsupported layer and reason. Off mode remains unchanged.

- [ ] **Step 6: Run tests to verify GREEN**

Expected: parser cases pass, unsupported graphs reject on mode, and off mode builds the previous graph without persistent-node flags.

### Task 2: Annotate pure FFN recomputation regions

**Files:**
- Modify: `src/llama-graph.h`
- Modify: `src/llama-graph.cpp`
- Modify: `src/models/gemma4.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: `struct llm_recompute_region { ggml_tensor * input; ggml_tensor * output; int32_t node_begin; int32_t node_end; int32_t layer; }`.
- Produces: `llm_graph_result::get_recompute_regions()`.

- [ ] **Step 1: Add region-boundary tests**

Build a two-layer dense Gemma4 fixture and assert each region starts at the FFN normalization consuming `attn_out`, ends before the residual add, and contains no attention/KV operation.

- [ ] **Step 2: Run the test to verify RED**

Expected: graph result exposes no recomputation regions.

- [ ] **Step 3: Record exact node ranges**

Immediately before dense FFN construction, record `gf->n_nodes` and `attn_out`. Expand or finish the FFN nodes, record the final FFN tensor and end node index, then append the region. Do not annotate MoE or per-layer embedding branches.

- [ ] **Step 4: Add dependency validation**

For every node in the region, require each non-parameter source to be the region input or produced within the region. Reject side-effect operations, views of KV state, and outputs consumed outside the region other than the declared region output.

- [ ] **Step 5: Run region tests to verify GREEN**

Expected: dense regions validate and injected external edges or KV-like writes reject recomputation.

### Task 3: Clone FFN forward nodes for backward rematerialization

**Files:**
- Modify: `ggml/include/ggml-opt.h`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `ggml/src/ggml.c`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: `ggml_opt_set_recompute_regions(ggml_opt_context_t, const ggml_opt_recompute_region *, size_t)`.
- Produces: internal `ggml_opt_clone_region(ctx, graph, region, replacements)`.

- [ ] **Step 1: Add a pure-MLP gradient equivalence test**

Build `norm -> up/gate -> activation -> multiply -> down -> residual`, mark the pure MLP region, and compare loss plus every input/parameter gradient with recomputation off and on.

- [ ] **Step 2: Run the test to verify RED**

Expected: region API is absent and the original activations remain live through backward.

- [ ] **Step 3: Clone region tensors and operations**

Create metadata-only tensor clones in the optimizer compute context. Map the declared region input and parameter leaves to originals; clone each region operation in topological order and rewrite its `src[]` through the map. Give clones stable `recompute_` names for diagnostics only.

- [ ] **Step 4: Build backward from the rematerialized output**

Before backward expansion, replace uses of the original region output in the gradient path with the cloned output while preserving the original forward output for the residual forward path. Seed both paths from the same output gradient and accumulate parameter gradients into the existing `grad_accs` tensors.

- [ ] **Step 5: Shorten original activation lifetimes**

Ensure original internal FFN tensors are not referenced by any backward node and are not graph outputs. Verify through scheduler allocation metadata that their last consumer belongs to forward.

- [ ] **Step 6: Run gradient and liveness tests**

Expected: recompute on/off gradients match within `5e-4`; original intermediates end in forward; cloned intermediates exist only near their backward consumers.

### Task 4: Integrate Gemma4, QLoRA, and QLion execution

**Files:**
- Modify: `src/llama-context.cpp`
- Modify: `src/llama-graph.h`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `examples/qlora_training/test-qat.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Consumes: regions from `llm_graph_result` before `ggml_opt_alloc`.
- Includes: recomputation mode and region signature in optimizer graph reuse/cache identity.

- [ ] **Step 1: Add multi-layer QLoRA equivalence test**

Run one deterministic update with at least two regions and compare loss, trainable parameter set, gradient norm, and update norm between off and on.

- [ ] **Step 2: Add QLion dependency test**

Run one QLion QAT update and assert every registered state receives the same number of gradient contributions in off and on modes.

- [ ] **Step 3: Run tests to verify RED**

Expected: standalone recomputation works but llama optimizer integration does not register regions.

- [ ] **Step 4: Register regions before allocation**

After `ggml_opt_prepare_alloc` and before backward construction, translate each `llm_recompute_region` to the ggml optimizer region structure. Reject an empty region set when on mode was requested.

- [ ] **Step 5: Preserve QLion callback ownership**

Include cloned forward nodes in `qat_forward_nodes`, map cloned parameter aliases back to their original QAT state, and keep dependency-plan ordering stable.

- [ ] **Step 6: Run optimizer suites**

Run `test-opt`, `test-qat`, and the existing flash-attention backward tests. Expected: PASS for on and off combinations.

### Task 5: Verify memory reuse and user-visible behavior

**Files:**
- Modify only when verification identifies a defect in Tasks 1-4.

**Interfaces:**
- Verifies: the scheduler reuses one FFN rematerialization workspace.

- [ ] **Step 1: Build both trainers**

Run `cmake --build build --target llama-finetune-qlora llama-finetune-qlion test-opt test-qat test-backend-ops -j8`.

- [ ] **Step 2: Run deterministic off/on steps**

Use the supplied model, rank 4, context/batch 4096, microbatch 512, and FA on. Record loss, gradient norm, update norm, scheduler MiB, CUDA peak, and wall time.

- [ ] **Step 3: Inspect scheduler tensor lifetimes**

Confirm scheduler memory is bounded by the largest cloned FFN region instead of the sum of all original FFN backward lifetimes. Confirm attention/KV nodes are not cloned.

- [ ] **Step 4: Exercise unsupported paths**

Run on mode with a non-Gemma4 fixture and a Gemma4 MoE fixture. Expected: a clear pre-update error naming the unsupported architecture or layer.

- [ ] **Step 5: Run final checks**

Run `git diff --check`, all optimizer/QAT tests, and save off/on peak and timing data for the integration report.
