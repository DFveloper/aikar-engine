# Supervised Output Compaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allocate LM-head logits and loss gradients only for supervised token rows during QLoRA and QLion training.

**Architecture:** Mark active labels in `batch.logits`, let the existing output-ID input compact Gemma4 hidden rows before the LM head, and pack all target/weight/statistics arrays in that identical order. A zero-label microbatch uses a one-row zero-weight sentinel so optimizer accumulation cadence remains unchanged while output memory stays bounded.

**Tech Stack:** C++17, llama batch splitting, Gemma4 graph builder, ggml sparse cross entropy, `test-opt`

**Spec:** `docs/superpowers/specs/2026-09-23-q4-fused-training-memory-design.md`

## Global Constraints

- All transformer tokens and KV updates remain present.
- Only LM-head, logits, loss, and their gradients are compacted.
- Loss scaling remains based on the original active-label count.
- Compaction is automatic for sparse-label training.
- Do not add a new test file.
- Keep new source and comments ASCII-only.
- Do not commit, push, or create a PR without explicit user approval.

## Review Focus

- Active labels split across microbatch boundaries must keep target order; Task 1 tests two boundaries.
- A zero-label microbatch must not advance optimizer cadence incorrectly; Task 3 compares updates.
- Critical-token metadata must follow compact rows but retain original indices; Task 2 tests mixed weights.
- MTP hidden-state extraction must remain unmasked when configured; Task 3 tests target plus MTP graph shapes.
- Graph reuse must distinguish different active-row counts; Task 1 checks reuse identity.

---

### Task 1: Build compact output maps per microbatch

**Files:**
- Modify: `src/llama-context.cpp`
- Modify: `src/llama-graph.h`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Produces: per-ubatch `std::vector<uint32_t> supervised_rows` in original token order.
- Produces: `n_outputs = max<size_t>(1, supervised_rows.size())` for optimizer graph construction.
- Preserves: `ubatch.n_tokens` for transformer and KV execution.

- [ ] **Step 1: Add a sparse-row graph-shape test**

Build an eight-token training batch with active labels at original rows 1, 4, and 7 and microbatch size 4. Assert first graph logits shape is `[n_vocab, 2]`, second is `[n_vocab, 1]`, and each graph still receives four token positions.

- [ ] **Step 2: Run `test-opt` to verify RED**

Run `cmake --build build --target test-opt -j8 && build/bin/test-opt`. Expected: both graphs expose four logits rows.

- [ ] **Step 3: Set batch output flags from labels**

Replace unconditional `batch.logits[pos_batch] = true` with:

```cpp
batch.logits[pos_batch] = labels_sparse[pos_ctx + pos_batch] >= 0;
```

After `mctx->get_ubatch()`, collect each index where `ubatch.output[i]` is true and set `n_outputs` from that count. Include `n_outputs` in graph reuse identity, because two microbatches with equal token shape but different compact counts have different logits shapes.

- [ ] **Step 4: Preserve a zero-label sentinel row**

When the active count is zero, mark the final ubatch token as a temporary output and record `zero_label_sentinel = true`. Its sparse weight is zero, so it contributes no loss or gradient but avoids unsupported zero-sized tensors.

- [ ] **Step 5: Run graph-shape tests to verify GREEN**

Expected: the active cases expose exactly 2 and 1 rows; an all-masked microbatch exposes one sentinel row while all token positions remain present.

### Task 2: Pack labels, weights, accuracy, and critical metadata

**Files:**
- Modify: `src/llama-context.cpp`
- Modify: `ggml/src/ggml-opt.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Consumes: `supervised_rows` from Task 1.
- Produces: sparse target, sparse weight, accuracy target, span weight, and reward weight tensors sized to `n_outputs`.

- [ ] **Step 1: Add target-alignment and weighted-loss tests**

Use unique targets and weights at rows 1, 4, and 7. Assert compact tensors contain those three values in order and compare weighted loss with the existing uncompressed reference.

- [ ] **Step 2: Run the test to verify RED**

Expected: current host arrays are indexed by token row and do not match compact tensor dimensions.

- [ ] **Step 3: Size optimizer label tensors from logits**

In `ggml_opt_prepare_alloc`, derive sparse label tensor length from `outputs->ne[1]`, not `n_ubatch`. Rebuild when the compact output count changes.

- [ ] **Step 4: Pack all host arrays through one map**

For compact index `j`, use:

```cpp
const uint32_t token_row = supervised_rows[j];
const uint32_t label_row = pos_ctx + pos_batch + token_row;
targets[j] = labels_sparse[label_row];
weights[j] = critical_metadata ? 1.0f : label_scale;
```

Use `label_row` for span/reward metadata and original-index statistics. For the sentinel, set target zero, accuracy target `-1`, and every weight to zero.

- [ ] **Step 5: Run loss-equivalence tests**

Expected: active-label count, unweighted loss, weighted loss, and critical-token selection match the uncompressed reference within `1e-6` on CPU.

### Task 3: Preserve optimizer cadence, MTP behavior, and QLion dependencies

**Files:**
- Modify: `src/llama-context.cpp`
- Modify: `src/models/gemma4.cpp` only if its existing output-ID placement fails a test
- Modify: `examples/qlora_training/test-qat.cpp`
- Modify: `tests/test-opt.cpp`

**Interfaces:**
- Preserves: one `ggml_opt_eval` call per original microbatch, including zero-label sentinels.
- Preserves: unmasked `h_nextn` shape for MTP.

- [ ] **Step 1: Add zero-label accumulation test**

Compare two training windows: one uncompressed window with a zero-weight middle microbatch and one compact window with the sentinel. Assert identical optimizer iteration, parameter update, and gradient accumulator reset.

- [ ] **Step 2: Add MTP and QLion graph tests**

Assert `t_h_nextn->ne[1] == ubatch.n_tokens` when `embeddings_nextn_masked` is false, while `t_logits->ne[1] == n_outputs`. In QLion, assert the dependency plan sees the same trainable tensor set before and after compaction.

- [ ] **Step 3: Run tests to verify RED**

Expected: at least label tensor sizing or statistics indexing fails before integration fixes.

- [ ] **Step 4: Keep sentinel evaluation in the normal optimizer path**

Call `ggml_opt_eval` for every microbatch. Do not special-case optimizer iteration or update execution. Verify the zero sparse weight creates zero parameter-gradient contribution.

- [ ] **Step 5: Protect MTP row ownership**

Keep Gemma4 `h_nextn` before output-row compaction when unmasked MTP embeddings are requested. Apply `ggml_get_rows` only to the LM-head input. If no model change is needed, retain the existing placement and record that in the task review.

- [ ] **Step 6: Run optimizer and QAT tests**

Run:

```bash
cmake --build build --target test-opt test-qat llama-finetune-qlora llama-finetune-qlion -j8
build/bin/test-opt
build/bin/test-qat
```

Expected: PASS with identical update cadence.

### Task 4: Measure output-buffer savings

**Files:**
- Modify only when verification identifies a defect in Tasks 1-3.

**Interfaces:**
- Verifies: logits dimensions and scheduler allocation scale with active rows.

- [ ] **Step 1: Run a prompt-masked benchmark**

Use the supplied model and a fixed training sample with a recorded supervised-row count. Run rank 4, context/batch 4096, microbatch 512, and FA on.

- [ ] **Step 2: Capture graph tensors and scheduler allocation**

Record `result_output`, sparse loss-gradient shapes, scheduler MiB, peak VRAM, loss, and update norm before and after compaction.

- [ ] **Step 3: Check the memory relation**

Expected logits bytes are `262144 * active_rows * 4`, not `262144 * 512 * 4`. The logits gradient follows the same relation.

- [ ] **Step 4: Run final checks**

Run `git diff --check`, `test-opt`, `test-qat`, and one deterministic QLoRA and QLion step. Save results for the integration report.
