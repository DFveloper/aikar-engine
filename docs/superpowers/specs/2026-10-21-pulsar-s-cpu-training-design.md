# Pulsar S CPU training and inference optimization

## Objective

Improve CPU fine-tuning throughput on the local Pulsar S model, prioritizing training while measuring inference performance and preventing inference regressions. Preserve model precision, training objective, trainable parameters, effective batch, and checkpoint compatibility.

The user approved this scope in conversation on 2026-10-21. This document records the concrete design for review before implementation planning.

## Observed environment

- Intel Xeon E5-2696 v3, 18 physical cores, 36 logical CPUs, AVX2/FMA/F16C, one NUMA node, 121 GiB RAM.
- Repository HEAD at inspection: `521084952`. The existing CPU binary reports `3c9d973b6`, so it cannot serve as the final source-matched baseline.
- Existing `build-sync-cpu` is Release with `GGML_NATIVE=ON` and actual `-march=native` compiler commands. Explicit AVX options in its cache do not establish that SIMD is disabled.
- Available original model: `/mnt/data/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf`, 13,121,388,640 bytes. The model is already on `/mnt/data`; no move is necessary.
- Another available model, `Lumen-3.5-Pulsar_S-exp1-LD-Q4_0_XL.gguf`, has a different size and must not be silently substituted into a before/after comparison.
- Existing external `stage1.sh` uses QLoRA/Q4_0 fake quantization, rank 64, alpha 32, attention Q/K/V/O targets, context and batch 8192, microbatch 64, seed 3407, activation recomputation, and F16 KV training. It requests CUDA and is a reference for semantics, not a CPU launch command.
- The external curriculum runner uses different targets and ranks across stages. No active trainer was found during inspection. The inspected scripts do not establish which command the user currently uses.

## Approach and alternatives

Use source-matched CPU profiling followed by small changes to existing kernels or graph construction. Configuration experiments and code changes are measured separately. This is preferred because the dominant cost and supported training path must be established on current source.

A configuration-only sweep is a useful control, but cannot establish a code optimization. Fusing model Q/K/V tensors immediately is also insufficient: it changes projection shapes and adapter names and can change which parameters are trained. A new training subsystem or training algorithm is outside scope.

## Execution design

1. Build current source into separate CPU-only baseline and candidate directories using identical Release and ISA options. Preserve existing builds and services. Record compiler, flags, model metadata, input hashes, affinity, thread count, and precision options.
2. Select the existing supported QLoRA path as the initial workload. Check CPU support for the reference fake quantization, attention backward, and KV policy before starting measurements. If an option is unsupported, document it and establish a supported control; do not present an altered workload as an equivalent optimization.
3. Use a fixed subset of the real local training data with deterministic ordering and unchanged loss masking. Record supervised and total tokens. Start with a resource-bounded diagnostic workload, then validate the retained change on the representative context and batch. Any reduced-context result is labeled diagnostic.
4. Measure initialization, tokenization, graph setup, forward/backward, accumulation, and optimizer time separately where existing instrumentation allows. Use available CPU profiling tools to identify dominant symbols and operations. Run performance experiments sequentially.
5. Review quantized matmul and input-gradient paths, MoE dispatch, temporary conversion/repacking, and thread synchronization. Choose one measured bottleneck at a time, reuse current infrastructure, and retain only improvements that survive end-to-end measurement.
6. Measure inference prefill and decode on the same model and build options as a secondary validation. Include a short prompt and a longer prompt sized to available execution resources.

## Fused QKV decision

The repository already supports physical fused QKV in `src/models/gemma4.cpp` and provides `scripts/convert_gguf_to_fused_qkv.py`. Training currently materializes the sliced Q/K/V activations. These copies may be required by backward operators and must not be removed solely to reduce graph nodes.

The initial default remains the original separate-projection model. A fused experiment is eligible only after proving that target matching, adapter parameterization, gradient flow, and saved/resumed adapters preserve the requested training behavior. Changing a Q-only adapter into a joint QKV adapter is not an equivalent optimization. If preserving behavior requires a new architecture or adapter format, present that as a separate design instead of expanding this change.

Any converted model is a new file on `/mnt/data`. Verify metadata and all weight bytes using the existing converter; preserve the original and checkpoints. Existing scales, mixed quantization, reused K-as-V, and missing KV projections retain their existing semantics or use the original fallback.

## Expected code scope

Potential implementation locations are existing files under `ggml/src/ggml-cpu/`, `ggml/src/ggml-opt.cpp`, `src/llama-graph.cpp`, `src/models/gemma4.cpp`, and `examples/qlora_training/`. Exact files and algorithms are selected only after profiling. Prefer an existing operator test file over introducing a new test framework. No public API, file-format change, new dependency, or new backend is planned.

## Validation and acceptance

- Obtain source-matched baseline and candidate runs with identical model bytes, data, seed, optimizer, targets, precision, context, batch, microbatch, and affinity.
- Run multiple actual optimizer updates. After warmup, measure each retained performance case at least three times and report median, range, peak RSS, and actual token counts. Separate initialization and configuration effects from steady-state code effects.
- Compare loss, gradients, and parameter updates from identical initial states using existing operator tolerances. Check NaN/Inf. Test adapter save/load and resume when touched; a short test does not establish long-term convergence.
- Run relevant existing backend, optimizer, QAT, quantization, and tiled matmul tests according to the changed path. Cover tails, strides, repeated expert routes, signed extrema, and thread counts where applicable. Maintain generic fallback behavior.
- Compare inference outputs or logits numerically. Use fixed-data perplexity if a suitable workload fits the execution budget. Plausible generated text is not proof of equivalence.
- Retain a candidate only when representative end-to-end improvement exceeds observed variability without material correctness or performance regression. Reject unsupported or ineffective experiments and report the evidence.

## Deliverables and boundaries

Deliver a focused uncommitted patch, reproducible build/run commands, raw logs in a separate results directory, training and inference timing/memory tables, correctness results, and explicit unverified cases. Do not commit, push, create a PR, overwrite original models/checkpoints, stop existing services, or launch an unattended full curriculum.

The user reviews this written specification before the implementation plan is produced, as required by the active brainstorming workflow. Implementation details follow measured evidence rather than assuming that fused QKV is the winning optimization.
