# Native TurboQuant matching the published Google algorithms

## User intent and scope

Implement Google's published TurboQuant algorithm natively for Pulsar S, rather than continuing to tune the existing third-party Turbo3/Turbo4 approximation. Preserve the fresh F16 references and compare the new implementation with F16, Q8_KV and Q8_KV Hadamard. Build with `-j36`. The Pulsar server may be stopped for validation. No commits, pushes, PRs or production-launcher changes are authorized.

This is an architectural change: head-wide transforms, residual storage and the attention estimator must agree. The existing Turbo packed layouts cannot represent the new algorithm.

## Source of truth and fidelity

Source: https://arxiv.org/html/2504.19874v1 . The unversioned HTML address currently resolves to this version. Relevant definitions are section 2.2, section 3.1 / Algorithm 1, and section 3.2 / Algorithm 2.

The Google blog and paper inspected here do not link an official implementation. The complete `google-research/google-research` master tree inspected on this task contains no paths matching TurboQuant, PolarQuant or QJL. This is a bounded source search, not proof that no official implementation exists elsewhere. The deliverable is a reproduction of the published algorithms; byte-for-byte identity with Google's internal kernels, seeds or experiments cannot be claimed from these sources.

The paper's downstream 2.5/3.5-bit experiments also describe mixed precision for outlier channels. Their complete channel-selection, indexing and packing rules are not specified by Algorithms 1 and 2. Do not invent these details or label uniform 3-bit/4-bit profiles as an exact reproduction of those experiments.

## Chosen algorithm

Use Algorithm 2 for K and Algorithm 1 for V. Provide uniform 3-bit and 4-bit profiles, plus independently selectable K and V widths for diagnostics. This role assignment follows the attention objectives: K is used for inner products, while V is reconstructed for a weighted sum. The exact combined KV policy is an integration choice and must be identified as such.

Use the complete per-head vector, after the model's usual K normalization and RoPE. Support D256 and D512 first, covering the dimensions exercised by Pulsar S. Reject unsupported dimensions explicitly. No 128-element independent blocks, signed-Hadamard substitution or Q8 query approximation are allowed in the faithful path.

### MSE quantizer

For a nonzero vector `input`, let `input_norm = ||input||_2`, `unit = input/input_norm`, and `rotated = rotation*unit`.

Generate `rotation` from an IID standard Gaussian square matrix using QR and a diagonal-sign correction to produce the Haar distribution. Verify orthogonality. K and V may use different rotations. Generate them once for a context, reuse them and retain their identities during cache serialization.

For dimension `head_dim`, obtain Lloyd-Max centroids by minimizing the scalar error under the exact spherical-coordinate density proportional to `(1-coordinate^2)^((head_dim-3)/2)` on `[-1,1]`. Solve in double precision with numerical integration, symmetric initialization and a convergence criterion. Validate numerical convergence with a stricter integration resolution and alternative initialization before freezing the tables. The codebook is independent of model data.

Each coordinate stores the nearest centroid index; ties use the lower index. Store the original input norm in FP32. Reconstruct `input_norm*transpose(rotation)*centroids`. Do not normalize the reconstructed centroid vector to length one: the existing codec's reconstructed-norm correction is not Algorithm 1.

### K residual QJL

For total coordinate width `total_bits`, the MSE base uses `total_bits-1` bits. In original coordinates compute `base = input_norm*transpose(rotation)*centroids`, `residual = input-base`, and `residual_norm = ||residual||_2`.

Generate a separate IID standard Gaussian `projection` of shape `[head_dim,head_dim]`, independent of the rotation. Store `residual_signs = sign(projection*residual)` as one bit per coordinate. A nonnegative projection stores the positive sign. Store the residual norm in FP32.

For the original FP32 query `query`, estimate the dot product as:

```text
dot(query, base)
    + residual_norm * sqrt(pi/2) / head_dim
      * dot(projection*query, residual_signs)
```

This is Algorithm 2 expressed as a dot estimator. The query's rotation and Gaussian projection are computed once per query/head and reused over all cache rows. The attention scale multiplies the completed estimate, and the existing model's softcap, masks and sinks are applied in their original order.

Computing the QJL residual in rotated coordinates is permitted only with a correspondingly independent Gaussian projection and a matching query transform. The reference implementation above fixes original coordinates to avoid ambiguity; any later algebraic optimization must demonstrate equivalence against it.

Zero input stores zero input and residual norms with canonical zero indices/sign bits. Zero residual contributes exactly zero. Finite-input norm computation must avoid intermediate overflow. Reject nonfinite inputs in reference tests and define a runtime diagnostic rather than propagating an invalid cache silently.

### V weighted sum

V uses all `total_bits` for MSE centroids and its original FP32 norm. Accumulate weighted reconstructed V vectors in rotated coordinates and apply the inverse V rotation once to the completed output. This is algebraically equivalent to decoding every V row before accumulation. No QJL is applied to V in the initial profile.

| Profile | K base | K residual | V |
| --- | --- | --- | --- |
| 3-bit | 2-bit Lloyd-Max | 1-bit Gaussian QJL | 3-bit Lloyd-Max |
| 4-bit | 3-bit Lloyd-Max | 1-bit Gaussian QJL | 4-bit Lloyd-Max |

## Storage and integration

Keep the existing `turbo3` and `turbo4` formats and their readers unchanged. The faithful path has a separate explicit KV policy and versioned storage. It must never be detected by reinterpreting an existing `signs` field.

Prefer byte tensors (`GGML_TYPE_I8`) for packed per-head rows and existing ggml tensors for FP32 transforms/codebooks, rather than adding dimension-specific general quantization types. The packed row stride is explicit; a logical vector dimension is supplied to the new operations independently of the physical byte width.

K row layout: FP32 original norm, FP32 residual norm, contiguous packed base indices, contiguous packed residual signs. V row layout: FP32 original norm and contiguous packed indices. Specify little-endian state serialization and exact bit ordering; pad the row stride only if the backend requires it and include padding in memory reports. Cache state headers record format version, dimensions, bit widths, transform/codebook identities and serialized values needed for reconstruction. Reject incompatible restores.

Before stride padding, K costs `total_bits + 64/head_dim` bits per element; V costs `total_bits + 32/head_dim`. Report both, their actual allocated bytes, transform storage, temporary workspace and any cache-index metadata. A payload-width label is not the true memory cost.

Add narrowly scoped native pack and packed-attention operations with CPU and CUDA implementations. Extend the existing KV cache policy, graph construction and allocator to route only explicitly selected layers through these operations. Ordinary cache types continue through their existing paths. Unsupported backends or cache modes fail at initialization; silently falling back to an F16 cache is prohibited.

Proposed CLI: `--kv-turboquant-global 3|4` for the previously measured local-F16/global-compressed policy, and `--kv-turboquant 3|4` for all KV layers. Explicit conventional cache-type overrides on selected layers conflict with the new policy and produce an argument error. Expose a reproducible seed and log the effective algorithm, per-layer dimensions, K/V widths and byte counts.

The correctness implementation may reconstruct a bounded attention tile into temporary FP32 workspace. It must not keep a permanent F16/F32 duplicate of the whole cache or expand the entire cache on each token. The optimized CUDA path consumes packed rows directly, combines base and residual scores before softmax and accumulates V in its rotated basis. Preserve FP32 accumulation on V100.

Use the existing per-row cache indexing/copy infrastructure where possible. Validate clear, sequence remove/copy, cache sharing and state save/restore with the new physical strides. Unsupported context-shift re-encoding must be rejected until its original-basis reconstruction and re-quantization are implemented and tested.

## Work sequencing

1. Add the independent double-precision reference and mathematical checks in existing `test-turbo-quant.cpp`.
2. Implement and validate the native CPU codec, transforms, codebooks, bit packing and estimator.
3. Add ordinary CUDA pack/attention, query-transform reuse and the explicit global policy; validate the real S model against fresh logits.
4. Optimize packed CUDA kernels only after they match the reference; measure PP/TG and allocated memory on matched runs.
5. Extend the existing Paged KV worktree after the ordinary path is validated. Do not integrate its unrelated changes into the root checkout or overwrite the existing fast Q8_KV package.

Paged implementation must read block tables and packed row strides directly, preserve the block allocator's ownership rules, and cover partial blocks, multiple sequences and local masks. Keep the validated Q8_KV launcher available while the new format is experimental. Paged Turbo acceptance requires the same quality checks as ordinary Turbo, plus the existing Paged operator matrix and matched single-request measurements.

## Verification and acceptance

Reuse existing test files. Builds use `-j36`. No automatic commit or production deployment.

Mathematical checks: QR orthogonality and inverse rotation; numerical Lloyd-Max convergence; MSE levels under independent rotations; QJL bias/variance over independent Gaussian projections and fixed residual/query cases; exact original-norm and residual-norm use; zero vectors; packed bit ordering and memory accounting. Statistical checks use predetermined seeds, sufficient sample counts and a confidence interval derived from the observed sampling variance. A single fixed projection cannot prove unbiasedness.

Backend checks: CPU/CUDA packed bytes where numerical ties are excluded; completed attention outputs against the independent double oracle; D256/D512, single and multiquery, long caches, causal/local masking, GQA, noncontiguous source strides and partial tiles. Transform/softmax tolerances are specified separately from quantization error, with measured error bounds recorded before enforcing them.

Model checks: use `/home/user/lumen-kv-bench-20261005/Pulsar-S-f16.logits`, the same corpus, 34 chunks, context/batch 512, all GPU and zero CPU MoE layers. Report mean, 99.9% and maximum KLD, RMS probability delta and top-1 agreement. Run K-only and V-only diagnostics. Q8_KV-grade quality means mean KLD no greater than 0.051338 and top-1 agreement no lower than 92.341% on this matched benchmark. F16-like 100% agreement remains a separate aspirational gate, not a consequence of the paper's theorem.

Performance checks: preserve the previous single-request 8K pool/4K prompt configuration, sampling parameters, warmup and repetition count. Measure PP and TG with the same runtime and model placement; target PP above 1000 tok/s and TG around 100. Also report query-transform time and cache-write time, because dense QR-derived rotations and Gaussian projections have greater arithmetic cost than the existing WHT. Fidelity is established before replacing dense transforms with any approximation.

## Risks and limits

The published unbiasedness is an expectation over the random projection, not a guarantee of identical logits or downstream results for a fixed seed. FP32 norm/matrix storage introduces numerical error that must be measured. Google's downstream mixed-precision experiments and hardware timings cannot be reproduced exactly from this paper alone.

Both K and V contributed substantial errors in the previous Pulsar S tests. QJL corrects the K estimator's bias; faithful V rotation/codebooks and norm handling are independently necessary. Neither Q8-level quality nor the V100 performance targets may be claimed before measurement. If a gate fails, retain the implementation as experimental and report the failure.
