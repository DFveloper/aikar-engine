# Pulsar S KV precision and single-request performance

The model for the S measurements is `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf`. The previous report used the different, unpruned `Lumen-3.1-Pulsar-Q4_0_XL.gguf`. These results must not be merged into one table.

## Reference logits

New F16 references are at `/home/user/lumen-kv-bench-20261005/Pulsar-S-f16.logits` and `Pulsar-f16.logits`. Each file is 4545715972 bytes and covers 34 chunks of 512 tokens. Generation uses `build/bin/llama-perplexity -ngl all -dev CUDA0 -kvu -ctlk f16 -ctlv f16 -ctgk f16 -ctgv f16 -c 512 -b 512 -f /mnt/openwebui/AIKAR/Lumen-Lumen3.txt --save-all-logits FILE`; CPU MoE layers are 0 for S and 5 for the unpruned model.

The `/mnt/openwebui` filesystem is full. The two attempted new files under that mount are truncated and must not be used. Changing batch size did not fix the disk exhaustion. The historical logits are unchanged. The logits writer does not currently report stream write failures.

## Hadamard quality, Pulsar S

Comparison uses the same model, corpus, placement, batch and new S reference, with `--kl-divergence-base FILE --kl-divergence`. Global-only configurations keep local K/V in F16. Both K and V use the listed format and rotation policy.

| Cache policy | Mean KLD | 99.9% KLD | Maximum KLD | RMS delta-p | Same top p |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16 | -0.000000 | 0.000051 | 0.000057 | 0.001% | 100.000% |
| Q8_KV global, no rotation | 0.051338 | 1.705909 | 3.678325 | 6.323% | 92.341% |
| Q8_KV Hadamard global | 0.049189 | 1.496377 | 5.843662 | 5.863% | 92.745% |
| Q4_0 Hadamard global | 0.087796 | 3.203274 | 9.895951 | 8.525% | 90.427% |
| Q8_KV Hadamard local + global | 0.075776 | 2.503627 | 9.705384 | 7.668% | 91.142% |
| Q4_0 Hadamard local + global | 0.199797 | 5.337494 | 8.622211 | 12.824% | 85.894% |

Same top p is top-1 agreement, not top-k set overlap. Distribution drift and downstream task accuracy are different metrics. These measurements do not reproduce the claimed historical Turbo3 100% result.

## Google TurboQuant versus this codec

Sources: https://arxiv.org/html/2504.19874v1 and https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/ . The paper describes separate MSE-optimal and unbiased inner-product algorithms. For a total b-bit inner-product budget, the latter uses a b-1-bit MSE quantizer followed by a 1-bit QJL sketch of its residual. Unbiasedness is an expectation property, not identical logits or guaranteed top-1 agreement on every input.

The current `ggml-turbo-quant.c` uses independent 128-element blocks, fixed signed Walsh-Hadamard rotation, an eight-centroid scalar quantizer for Turbo3 and sixteen centroids for Turbo4. It corrects each reconstructed block's norm and stores the scale in FP16. Turbo3's `signs` field is the high bit of each three-bit centroid index. It is not a QJL residual sketch. Turbo4 also has no QJL sketch. These layouts cost 3.125 and 4.125 bits per element respectively, including the scale.

The external TheTom reference also labels its current Turbo3 implementation as three-bit PolarQuant with WHT. Its legacy Turbo4 path has QJL; that does not make this fork's Turbo3 the paper's unbiased inner-product algorithm.

To close the algorithmic gap, K needs a separately specified residual sketch, residual magnitude, query projection and corrected dot estimator. V should retain an MSE-oriented reconstruction. A QJL design changes the packed layout and backend readers, so it must use a distinct type/version and independent CPU/CUDA tests. Merely reinterpreting `signs`, changing centroids, or copying a similarly named type is incorrect. A full-head rotation and scale calibration can be investigated independently, with real attention-distribution measurements.

The CUDA vector path also quantized Q to Q8 before calculating Turbo dot products. The focused precision fix retains the original FP32 Q on CUDA while preserving the existing K/V layout. This removes an extra implementation error source; it does not remove the codec's intrinsic quantization error.

## Final Turbo precision results

The native change and `test-turbo-quant` were rebuilt with `-j36`. The tighter CUDA reference check failed before the change at approximately 0.00017 maximum absolute attention error. After the change, tested Turbo K cases have errors below 0.00000015 against the dense reconstruction of the same quantized cache. This is kernel correctness, not equivalence to an unquantized F16 cache. Existing CPU and Vulkan tests also pass; their arithmetic paths are unchanged. Coverage includes D256/D512, single-query decode, 32-query prefill and 16384 cached tokens.

All measurements below use local F16 K/V and the global types shown. The same complete S reference and 34 chunks are used throughout.

| Global K / V | Mean KLD | RMS delta-p | Same top p |
| --- | ---: | ---: | ---: |
| Turbo3 / Turbo3, before | 0.163282 | 11.205% | 86.217% |
| Turbo3 / Turbo3, FP32 Q | 0.161496 | 10.893% | 86.217% |
| Turbo4 / Turbo4, before | 0.095972 | 8.586% | 89.677% |
| Turbo4 / Turbo4, FP32 Q | 0.093140 | 8.421% | 89.781% |
| F16 / Turbo3 | 0.116314 | 9.178% | 88.351% |
| F16 / Turbo4 | 0.077094 | 7.655% | 90.911% |
| Turbo3 / F16 | 0.118525 | 9.766% | 89.135% |
| Turbo4 / F16 | 0.077830 | 7.685% | 90.634% |

Neither the full Turbo formats nor these mixed configurations reach the measured Q8_KV baseline. Retaining F16 on one side costs more than Q8_KV on both sides and still has worse quality in this evaluation. The native correction is useful but cannot be presented as a solution to the entire quality target. The global Q8_KV Hadamard result remains the best measured compressed configuration in this table by mean KLD and top-1 agreement.

Example final comparison command:

```sh
env CUDA_VISIBLE_DEVICES=0 build/bin/llama-perplexity \
    -m /mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar_S-LD-Q4_0_XL.gguf \
    -ngl all -ncmoe 0 -dev CUDA0 -kvu -ctlk f16 -ctlv f16 \
    -ctgk turbo4 -ctgv turbo4 -c 512 -b 512 -t 36 -tb 36 \
    -f /mnt/openwebui/AIKAR/Lumen-Lumen3.txt \
    --kl-divergence-base /home/user/lumen-kv-bench-20261005/Pulsar-S-f16.logits \
    --kl-divergence
```

For the conventional global Hadamard tests, replace the global types with `q8_kv` or `q4_0` and add `--k-cache-hadamard-global --v-cache-hadamard-global`. For local + global rotation, also set both local types and add `--k-cache-hadamard-local --v-cache-hadamard-local`.

The historical 100% result remains unverified. The old reference also differs from an F16 replay of the unpruned model on this runtime (mean KLD 0.062875, top-1 92.907%). That establishes a reference/runtime mismatch for that replay, not the original cause or proof that historical Turbo quantization was active. Recovery needs the original model revision, binary revision, command, reference-generation settings and actual per-layer cache types. Current F16 replay against the new S reference reaches 100%, while current Turbo does not.

Closing the remaining gap requires measuring K and V changes independently. QJL can address K inner-product bias, but the F16-K/Turbo4-V row shows that fixing K alone cannot solve the measured V error. Full-head rotation, MSE scale/codebook calibration and selective precision are separate V experiments. The following native implementation now supplies the versioned QJL format. Paged Turbo remains deferred after the ordinary quality gates failed.

## Native published Google algorithms

The experimental `--kv-turboquant-global 3|4` option selects native TurboQuant for global K/V and retains local F16. `--kv-turboquant 3|4` selects both roles. `--kv-turboquant-seed` controls the immutable transforms. This is a reproduction of published Algorithms 1/2, not a claim of identity with unpublished Google source or its mixed 2.5/3.5-bit experiment.

K and V use independent full-head Haar rotations from Gaussian QR with positive diagonal, for D256/D512. Spherical-density Lloyd-Max codebooks quantize K with b-1 MSE bits and an independent one-bit Gaussian QJL residual sketch; V uses all b bits for MSE. Original and residual norms are FP32. Queries remain FP32, and the K estimator applies the residual correction before softmax. V is accumulated in its rotated basis and inversely rotated once. No reconstructed-norm correction is applied.

The physical I8 rows are separately versioned: K stores `8 + D*b/8` bytes and V stores `4 + D*b/8` bytes. D512 K3/V3 require 200/196 bytes per head, K4/V4 require 264/260; D256 K3/V3 require 104/100, K4/V4 require 136/132. These exclude shared transforms, local F16 and attention workspace. One immutable parameter tensor requires `4*(3*D*D+64)` bytes: 0.750244 MiB for D256 and 3.000244 MiB for D512. Existing conventional Turbo layouts remain separate.

The initial candidate's matched global-only measurements use the complete fresh S reference, all 34 chunks, context/batch 512 and no CPU MoE layers. Seeds were declared before comparison. This table predates the final shared-scratch fix and split-V optimization; it is retained as the initial seed experiment, not as final-revision measurements for all seeds.

| Native global K/V bits | Seed | Mean KLD | 99.9% KLD | Maximum KLD | RMS delta-p | Same top p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 42 | 0.262732 | 5.171185 | 12.544744 | 14.195% | 82.318% |
| 3 | 43 | 0.261226 | 4.801270 | 9.172847 | 14.238% | 82.295% |
| 3 | 44 | 0.277497 | 6.064390 | 11.586484 | 15.043% | 82.653% |
| 4 | 42 | 0.152636 | 3.847349 | 12.903497 | 10.952% | 86.909% |
| 4 | 43 | 0.147664 | 4.338174 | 9.452913 | 10.502% | 87.036% |
| 4 | 44 | 0.160256 | 4.666890 | 10.531395 | 11.330% | 86.044% |

All six fail the fixed Q8_KV gate: mean KLD <= 0.051338 and top-1 >= 92.341%. Native4 seed42 has maximum KLD 12.903497, 99.9% KLD 3.847349 and RMS delta-p 10.952%. Native3 seed42 has maximum KLD 12.544744, 99.9% KLD 5.171185 and RMS delta-p 14.195%. Published unbiased dot estimation does not guarantee low softmax/logit error for this model. The exact source of the excess model error is unresolved; native K-only/V-only model routing has not been measured.

The final revision's repeated global4/seed42 comparison completes all 34 chunks with mean KLD 0.151633, 99.9% KLD 3.804432, maximum KLD 11.515713, RMS delta-p 10.605% and top-1 87.186%. It still fails both Q8-quality thresholds. This rerun includes both review fixes and split-V changes, so the small quality change cannot be attributed solely to the V optimization. The other five profiles have not been rerun on the final revision.

Final-revision F16 replay completes the same 34 chunks with mean KLD -0.000000, maximum KLD 0.000057, 99.9% KLD 0.000051, RMS delta-p 0.001% and top-1 100.000%. This validates the fresh reference for the current runtime; it does not establish how the historical Turbo3 result was produced. Native logs, cache-free timing JSON, nvprof output, review regression logs, builds and rerun comparisons are preserved under the benchmark folder's `google-native` subdirectory. The temporary server is stopped and the V100 is idle. No production launcher or validated Paged Q8 staging binary was replaced.

CPU/CUDA tests compare against an independent long-double original-basis reference that does not call production pack/unpack. Coverage includes all four K/V width combinations, GQA, two streams, reversed scatter, noncontiguous queries, partial masks, all-masked queries, sinks, softcap and a 257-row tail. Packed bytes are checked independently. Codebook stationarity and Haar orthogonality are also checked. Native state tests pass all nine existing cases. Both global/local policy and transform headers are validated before either payload is restored. CUDA racecheck reports zero errors and zero warnings after fixing shared reduction-scratch reuse. These are codec/operator checks, not proof of Q8-grade model quality.

Ordinary native global4 at context 8192 allocates 43.94 MiB in the global cache buffer, including the 3.00 MiB parameter tensor, and 450.00 MiB for local F16. CUDA model storage is 12498.50 MiB; reserved CUDA compute storage is 375.67 MiB and host compute storage 53.17 MiB. Attention additionally uses the CUDA pool: `4*queries*2*D` bytes for transformed queries and `4*min(queries,64)*nKV` for scores. These totals should not be confused with raw packed-row compression ratios or a measured peak resident-memory bound.

Cache-free ordinary speed measurement disables `--cache-ram` and `--ctx-checkpoints`, erases slot 0 before every request using `--slot-save-path`, and requires `cache_n=0` with the full requested prompt count. Temperature is zero, seed 1234, GPU sampling enabled, 256 generated tokens, one slot, context 8192 and batch/ubatch 1280. Builds and backend tests do not overlap these measurements. Before V-kernel optimization, three measured repeats give short TG 61.50/61.37/61.54 tok/s and 4K PP about 330, TG about 27 tok/s. Earlier trials with checkpoint/prefix reuse are excluded.

An nvprof run with a cache-free 4K prompt and 64 generated tokens attributes 49.44% of GPU activity to serial V accumulation (9.35729 seconds over 5445 calls), 13.89% to packed scores (2.62932 seconds), and less than 2% each to K packing, V packing and query transforms. The focused CUDA optimization splits V accumulation into 256-token blocks after one shared softmax and merges partials before the inverse V rotation. It preserves the published codec and estimator but changes F32 sum order. Its extra pooled workspace is `4*min(queries,64)*ceil(nKV/256)*D` bytes, only for nKV >= 512. At D512/nKV8192, scores plus partials require 6 MiB per 64-query chunk, in addition to the transformed queries. The original path remains for shorter caches. New 513/1025-row tests exercise split tails and masked/sink behavior.

The expanded oracle suite and focused racecheck on all three new CUDA kernels pass with zero hazards, exit 0. Maximum absolute output error is 2.38419e-7 for D512/1025 rows and 1.49012e-7 for D256/513 rows. Root CUDA61/70 builds complete with -j36; execution is verified on the V100, not the P620. Full sanitizer instrumentation of the enlarged packing cases was interrupted because of runtime; earlier unchanged packing cases had passed racecheck.

With the split kernel, three measured cache-free repeats give 4K PP 579.59/579.89/580.26 tok/s and TG 53.13/53.10/53.04, versus about 330/27 before, approximately 1.76x PP and 1.97x TG. Short TG remains about 61.55. Native throughput still fails the requested PP >1000 / TG around 100 target. The independently validated Paged Q8_KV package below remains the measured route to that target; native Paged Turbo has not been integrated.

Implementation decisions and limits:

- Work remains in the approved detached `/tmp/aikar-google-turboquant` checkout and is synchronized to the root workspace without commits or new branches; history is unchanged and the checkout must be retained.
- Constructors live in the internal `ggml-turboquant.h`; external callers must include that header. The public operator enum and context policy still change and ABI compatibility is not established.
- CPU attention uses O(D) vectors per thread instead of planner workspace; repeated allocation can reduce CPU speed. CUDA uses the existing pool.
- Native model routing is explicitly experimental and opt-in after failed quality gates; selecting it can materially degrade logits. It is not a default replacement.
- Native Paged integration and native K-only/V-only model diagnostics are deferred after failed ordinary quality gates; these remain outstanding work, and prior mixed conventional Turbo results are not native diagnostics.
- Prefix/checkpoint-reusing and compilation-overlapping speed trials are excluded; the accepted ordinary measurements pay the full prompt cost.
- Exact model-quality cause, peak resident pooled memory, unpublished Google source identity, other platforms, ABI compatibility and concurrent state restore are unproven. Native Paged correctness/performance is untested. Calculated workspace sizes and measured ordinary speed do not establish those claims.

## Single-request Paged KV

Reproducible launch: `bash scripts/serve-pulsar-s-paged-single.sh`. It uses the existing `build-paged-decode/staging-bin` package, one slot, all model weights on V100, logical and physical context 8192, Q8_KV, block size 16, batch/ubatch 1280 and GPU sampling. It does not change the production launcher or its 131072-token physical budget.

At temperature 1, top-k 64, top-p 0.95, seed 1234 and 256 generated tokens, three single-request repetitions measured:

| Prompt tokens | PP tok/s | Server TG tok/s | Steady stream TG tok/s |
| --- | ---: | ---: | ---: |
| 50 | 221 on first uncached run | 106.13, 106.51, 106.69 | 106.94, 106.60, 106.86 |
| 4096 | 1502.89, 1497.86, 1503.86 | 96.69, 96.12, 96.45 | 97.38, 96.49, 96.89 |

The first 4K run reused 34 prefix tokens; the next two evaluated all 4096. Short repeat prompts reused 49 tokens, so those repeat PP values are excluded. Baseline `build-paged-fused` with CPU sampling, at temperature zero and 128 generated tokens, measured 70.31 TG tok/s at 50 tokens and 48.30 at 4K. Because binary and sampling settings differ, this is not an isolated kernel speedup claim.

These measurements meet PP >1000 for a 4K single request and TG around 100 at short/4K context. They do not establish the same performance at production pool size, twelve slots or 128K context. FP16 logits precision and paged throughput are separate acceptance checks.

Package SHA256: server `8950c90804081c556ce3bb3f49e3183cc5a07ded95d61aa0ee3e01e5e2c0b1a2`; CUDA library `418d6709ee9ee26a68599c36c506efbdf0fc5d45e3805f979e1bd8ea3a227aa3`.

The staged CUDA library passed all 82 existing `PAGED_ATTN.*` backend cases using `build-paged-decode/device-bin/test-backend-ops -b CUDA0 -o 'PAGED_ATTN.*'` with `LD_LIBRARY_PATH` pointing to `staging-bin`. Raw comparisons, performance JSON and validation logs are preserved in `/home/user/lumen-kv-bench-20261005`. The production Pulsar server remains stopped after the authorized tests.
