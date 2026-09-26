# QLion Sparse Row Accumulation Design

## Goal

Make full-parameter QLion QAT with microbatch accumulation fit on a 16 GB V100 by keeping pure `GET_ROWS_BACK` gradients sparse throughout accumulation, without changing dense or mixed-gradient QLion semantics.

## Scope

- Use sparse accumulation only when a logical QLion parameter has exactly one pure `GET_ROWS_BACK` gradient contribution.
- Keep the existing dense Q8 accumulator for ordinary dense gradients and mixed gradients such as `OUT_PROD + GET_ROWS_BACK`.
- Keep Q8 momentum, Q4 residual, QLion update equations, error feedback, and optimizer-step ordering unchanged.
- Implement CPU and CUDA execution for the sparse accumulation and sparse optimizer-step operations.
- Do not implement tied-output low-rank factor accumulation, Flash Attention changes, CPU offload, or another optimizer.

## Confirmed Gradient Classes

The optimizer classifies a canonical parameter from its actual pending gradient graph, not its tensor name.

1. A single `GET_ROWS_BACK` contribution uses sparse Q8 row accumulation.
2. `GET_ROWS_BACK` plus `OUT_PROD`, or any other dense contribution, uses the existing dense Q8 accumulator.
3. An ordinary dense gradient uses the existing dense Q8 accumulator.

For the target Gemma4 model this means:

- `per_layer_token_embd.weight` uses sparse Q8 row accumulation.
- Tied `token_embd.weight` keeps the existing full Q8 accumulator because its gradient is `OUT_PROD + GET_ROWS_BACK`.
- Attention, FFN, and other dense Q4 parameters keep the existing full Q8 accumulator.

## State Ownership

Parameter registration allocates only persistent Q8 momentum and Q4 residual. Gradient accumulator storage is allocated lazily after the first training graph exposes the parameter's gradient class.

Dense state consists of the existing full-shape Q8 accumulator.

Sparse state consists of:

- a Q8 row accumulator with shape `[row_width, capacity]`;
- an I32 row ID array with `capacity` entries;
- an I32 open-addressed hash table that maps row ID to accumulator slot;
- small counters and temporary per-microbatch indexing storage.

Sparse gradient state has separate context and backend-buffer ownership from momentum and residual. This permits lazy allocation and growth without moving optimizer momentum or residual.

## Sparse Capacity

Initial capacity is the smallest reusable capacity that can contain the graph-derived upper bound on distinct rows in one accumulation period:

```text
min(parameter_rows, opt_period * rows_in_one_microbatch)
```

Capacity may be rounded up for alignment or hash-table load factor. The hash table stays below a 0.5 load factor. Allocation is proportional to reachable touched rows, never unconditionally proportional to vocabulary size.

If a later compatible graph requires more capacity, the optimizer grows the sparse state at an accumulation-period boundary, copies live rows and IDs, rebuilds the hash table, and then releases the old buffer. Capacity exhaustion must fail before execution if safe growth is impossible; kernels must never drop a row.

## Sparse Accumulation Operation

A new graph operation accepts the persistent sparse Q8 rows, row IDs, hash state, current F32 row gradients, and current I32 source IDs.

For each microbatch it performs these logical phases on the selected backend:

1. On period reset, clear row IDs, hash slots, counters, and sparse Q8 rows.
2. Process current IDs in input order and map each distinct row to one persistent slot.
3. Build a deterministic per-microbatch unique-row list and occurrence links.
4. For each unique row and Q8 block, sum duplicate F32 occurrences in input order.
5. Dequantize the previous Q8 accumulated block, add the current microbatch sum, and requantize it to Q8.

The persistent unique-row order is first-seen order across the accumulation period. Row ID 0 is valid; `-1` is the empty sentinel. The accumulation graph executes once per microbatch, but no optimizer update occurs until the normal period boundary.

The expected memory complexity is:

```text
O(capacity * row_width) Q8 data + O(capacity) IDs/hash metadata
```

The expected compute complexity is:

```text
O(input_rows * row_width) accumulation + O(capacity * row_width) optimizer step
```

There is no scan over vocabulary rows and no quadratic scan over all accumulated rows.

## Sparse Optimizer Step

A new sparse-row QLion optimizer operation consumes the compact Q8 rows and row IDs. It iterates slots in deterministic first-seen order and applies the existing QLion block update exactly once for every valid unique row.

The operation uses the same quantized parameter, Q8 momentum, Q4 residual, optimizer parameters, clipping, weight decay, and error-feedback implementation as the existing row update. Only gradient storage and row selection differ.

After the step, sparse state remains allocated for reuse. The next period's first accumulation operation resets its logical contents.

## Dense and Mixed Paths

The dense accumulation operation and dense Q8 accumulator remain unchanged for all non-pure-row parameters.

The tied Gemma4 token embedding remains on the existing specialized mixed accumulation path:

```text
OUT_PROD + GET_ROWS_BACK -> full Q8 accumulator -> one QLion step
```

This deliberately preserves its dense output-head contribution and current sequential Q8 accumulation semantics. Removing its approximately 0.398 GiB accumulator requires a separate low-rank factor subsystem and is outside this change.

## Partial Periods and Reset

The optimizer keeps the configured `opt_period` as its maximum period and capacity bound, and tracks a current period length separately. A boundary-only API may set the next period length to any value from 1 through the configured maximum while `opt_i == 0`. Graph build, gradient scaling, and the optimizer-step boundary use the current period length. After that optimizer step, the current period returns to the configured maximum.

High-level epoch code uses this override when it knows that the final logical period contains fewer physical microbatches. The production QAT packer currently constructs complete logical windows, so its normal behavior does not change.

Tests execute shorter periods through this API to verify correct scaling, update timing, and reset. An asynchronous interruption does not apply an incomplete optimizer update implicitly; callers must know the final period length before its first microbatch. Existing checkpoints remain restricted to completed optimizer-step boundaries.

## Numerical Semantics

Dense accumulation currently performs sequential Q8 accumulation: each microbatch adds to the dequantized prior Q8 value and requantizes. Sparse accumulation will use the same rule independently per touched row.

Tests compare:

- compact sparse accumulation against an explicit dense Q8 reference using the same microbatch order;
- effective-batch non-accumulation against microbatch accumulation with a strict dequantized tolerance that accounts for the existing intermediate Q8 quantization;
- updated quantized parameter, Q8 momentum, and Q4 residual;
- repeated rows within one microbatch and across microbatches;
- period reset and a shorter manually stepped period;
- dense and tied mixed-gradient regression behavior.

CPU and CUDA implementations must produce the same quantized sparse accumulator and optimizer state for deterministic test inputs where the backend quantizers are bit-equivalent. Otherwise tests use a documented strict dequantized tolerance and report the maximum observed difference.

## Memory Verification

Unit tests inspect the selected accumulator shape and graph operations to prove that a pure row parameter does not allocate:

- a full-shape Q8 accumulator;
- a full-shape F32 `GET_ROWS_BACK` gradient;
- a dense QLion accumulation operation.

The production run reports dense and sparse gradient-state bytes separately. The expected target-model reduction is approximately 2.324 GiB from `per_layer_token_embd.weight`, reducing persistent accumulation memory from approximately 14.20 GiB to approximately 11.88 GiB.

## Production Verification

The primary CUDA test uses the supplied 4.63B Gemma4 GGUF on CUDA0 with `-c 128 -b 128 -ub 64`, Flash Attention on, and activation recompute on. It must complete at least two optimizer steps without the previous 9,203.26 MiB scheduler allocation.

Additional fresh-process runs cover sequence lengths 128, 256, 512, and 1024 where feasible. Each run records sparse state bytes, scheduler buffer, CUDA process peak, retained memory after a step, and whether the second accumulation period reuses state without stale rows.

## Files Expected to Change

- `ggml/include/ggml.h`: declare sparse QLion graph operations.
- `ggml/include/ggml-opt.h`: expose minimal sparse-state memory inspection if existing accessors are insufficient.
- `ggml/src/ggml.c`: construct sparse accumulation and sparse optimizer-step nodes.
- `ggml/src/ggml-opt.cpp`: lazy gradient-state allocation, graph classification, growth, reset, and sparse graph selection.
- `ggml/src/ggml-cpu/ops.cpp` and CPU dispatch files: execute sparse accumulation and step.
- `ggml/src/ggml-cuda/opt-step-qlion-qat.cu`, its header, and CUDA dispatch: execute deterministic sparse accumulation and step.
- Backend metadata or cloning switches that enumerate ggml operations.
- `examples/qlora_training/test-qat.cpp`: numerical, reset, duplicate, partial-period, capacity, memory, dense, and mixed regression tests.
- `examples/qlora_training/finetune_qat.cpp`: report dense and sparse accumulator bytes separately.

## Non-Goals

- Removing tied output storage duplication.
- Removing the tied token embedding full Q8 accumulator.
- Changing activation recompute or Flash Attention.
- Supporting microbatch-by-microbatch optimizer updates.
- Changing QLion momentum, residual, clipping, weight decay, or error-feedback equations.
- Hard-coding model names, vocabulary size, tensor shapes, GPU model, or sequence length.
