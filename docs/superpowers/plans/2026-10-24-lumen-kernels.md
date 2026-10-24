# Lumen CPU kernel experiment

Implement the kernel work authorized after the no-spec audit. Keep the production server running on its existing binary. Preserve model weights, expert routing, KV precision, context length, and non-speculative decoding.

1. Build an immutable baseline in a separate build directory and record the existing dirty worktree.
2. Extend existing MoE tests for 18-thread, 1408-row and ragged-tail workloads. Measure the current kernel before changing row ownership.
3. Balance tiled MoE output rows at microtile boundaries. Compare correctness and model timings; retain only a measured improvement.
4. Evaluate batched paged-attention value accumulation using the existing SIMD GEMM. Exercise causal masks, missing pages, sequence boundaries, and tails against the existing independent oracle.
5. Compare fixed-input prefill/decode and model logits against the baseline, inspect applicable vllm.cpp CPU techniques, and publish measurements with limitations.

No production deployment, commit, push, or PR is included. Experiments run sequentially to avoid competing CPU loads. Timing thresholds are not CI tests.

## Experiment ledger

- Baseline libraries and server copied to `build-lumen-kernels/baseline/bin`; original tracked diff saved as `results/input-worktree.patch`.
- Ruling: work in the existing private fork with a separate build and input-file snapshots, because the current uncommitted server changes are the workload under test. A clean worktree would omit them.
- Ruling: use measured kernel latency as the optimization target and numerical regression tests as correctness gates. Do not add flaky CI timing assertions or invent a functional failure for a performance change.
- Added 3/18-thread IQ3 tests with 17/1408/1423 output rows and an optional production-shape MoE benchmark. Baseline numerical tests passed.
- First balanced model run overlapped backend-test compilation. Exclude it from performance conclusions; rerun baseline and candidate after compilation finishes. The initial 4096-token baseline also overlapped the short operator tests and must be repeated.
- IQ3-only clean comparison: 72.7354 to 78.0154 prefill tokens/s, 11.2872 to 11.3873 decode tokens/s. All 17,039,360 captured logits are bitwise equal. This is an intermediate result, not the final combined comparison.
- Fresh review found direct GEMM's `0 * Inf` difference. `test-paged-zero.cpp` reproduced baseline PASS / first candidate FAIL; the finite-value fallback restores PASS. Include the finite scan in final timings.
- Initial combined attention candidate passed 195/195 independent-oracle cases. After the finite-value fallback, rerun the added tails/missing-page cases and standalone nonfinite regression. The fallback does not change finite arithmetic.
- The first attention performance command matched no cases because the new fixtures were in the evaluation list rather than the performance list. Move fixtures to the performance list, rebuild, and discard empty timing logs. Final logs must contain four actual timings.
- Final paired measurements completed: prefill +9.53% at 512 tokens and +14.23% at 4096, decode effectively unchanged. Full logits match bitwise for both depths; HTTP median response time falls 5.20% and all three response texts match.
- Final verification completed: CTest 3/3, final focused attention 21/21, isolated nonfinite regression passes. Added the analytic regression as a permanent CPU-gated CTest target; checked CPU initialization to address review portability feedback.
- Ruling: retain the small kernel changes based on repeated full-model prefill improvements, but publish the IQ3_XXS and small local-attention microbenchmark regressions. Do not claim universal operator gains or substantial single-request decode acceleration.
- Production checksums and PID/health are unchanged, experiment port closed, unrelated tracked changes preserved. No production restart, commit, push, or PR performed. All five plan steps are complete.
- vllm.cpp inspected at `7f842ee9b3c7e9b651283de48f42cd76a0ad5d30`. CPU activation quantization once and output-row ownership already exist in this fork. Its CPU SwiGLU composite is not an applicable fused Gemma GELU replacement.
