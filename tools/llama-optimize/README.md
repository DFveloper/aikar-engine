# llama-optimize PoC

This tool generates packed GSQ/PTQ candidates and optimizes precision logits using a native complete-sequence model graph. Global CE/KL backward currently supports dense Gemma 4. Other architectures and routed experts fail explicitly. Model quality and full experiment completion require saved real-model reports, not synthetic self-tests.

Reference implementations used for the numerical checks: [GSQ](https://github.com/IST-DASLab/GSQ/tree/03fc16484c369e3127225615d5e03e8d3a6043e3) and [RCO](https://github.com/IST-DASLab/RCO/tree/9a1e09c07d468109cbe60a1b87d5036034a79d10).

GSQ uses FP32 logits, scales and Lion moments on the host. GGML performs linear reconstruction and its weight gradient on CPU or CUDA0. The quantizer uses four values `[-2,-1,0,1]`, 64-element groups, seeded Gumbel samples, analytic gradients, two-beta Lion, and hard assignment. The native RTN initializer uses maximum ranges; it differs from upstream RTN's MSE scale search. The optional Python bridge runs upstream RTN or full-Hessian GPTQ on captured inputs. The bridge requires PyTorch, Transformers and lion-pytorch; native execution requires none of these Python packages.

Q2_0 output packs assignments directly: integer code `c=3-argmax(logits)`, stored FP16 scale `d=-scale`, decoder `d*(c-1)`. There is no float weight requantization between optimized assignments and the file. Other GSQ output types are not supported. PTQ candidates support Q2_0, Q4_0 and Q8_0.

`rco-proxy` uses candidate interpolation with hard Gumbel straight-through gradients and an independent linear reconstruction objective. It uses tangent projection, existing GGML FP32 AdamW with zero decay, retraction and first-moment transport in upstream order. It is not the upstream global CE/KL objective. Candidate costs include exact tensor block storage and GGUF alignment. Fixed metadata and unchanged tensors count towards the final file budget. Discrete assignment uses a sparse exact-byte dynamic program capped at one million retained states.

`rco-global` uses complete-model shifted CE or full-vocabulary BF16 teacher-to-student KL. Native backward receives the actual logits derivative; precision gradients contract weight adjoints with lazy-decoded candidates. Hard weights come from exact-byte knapsack assignment; soft Gumbel probabilities supply the STE derivative. Projection, zero-decay Adam, retraction and first-moment projection follow upstream order. Temperature decreases exponentially from 1 to 0.05. The byte cap covers the entire destination GGUF, including fixed tensors and metadata.

Teacher inference runs on CPU from the original BF16 GGUF. Packed candidates stay in filesystem storage, mixed FP32 parameter tensors stay on CPU, and CUDA uses existing weight streaming. A tmpfs staging directory also consumes host RAM outside the process RSS; include it when sizing host memory. No complete GPU teacher or GPU candidate collection is required. Select `float32_math=true` for gradient validation and training: BF16 activation rounding and the native CPU GELU lookup otherwise make finite differences unreliable. This mode uses FP32 student matmuls and continuous tanh GELU. It does not change inference defaults or replace the global objective with a layerwise objective. CPU reference reductions accumulate in FP64 with FP32 outputs. Activation recomputation is currently disabled in this adapter; report the measured context-dependent memory limit.

## Build and numerical checks

```sh
cmake -S . -B build-opt -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-opt --target llama-optimize test-quantize-fns -j 6
python3 tools/llama-optimize/test_reference.py build-opt/bin/llama-optimize /path/to/GSQ /path/to/RCO
build-opt/bin/test-quantize-fns
```

For CUDA, configure with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70`. GPU model loading explicitly selects CUDA0, with no automatic Vulkan or second-GPU fallback. Set `CUDA_VISIBLE_DEVICES=0` on the PoC host. A CPU build supports `GPU_LAYERS=0` and `BACKEND=CPU`.

## Commands

```text
llama-optimize self-test
llama-optimize capture MODEL TEXT TENSOR CAPTURE GPU_LAYERS
llama-optimize capture-store MODEL MANIFEST TOKEN_IDS DIRECTORY REPORT
llama-optimize linear-test CAPTURE BACKEND OUTPUT
llama-optimize gsq MODEL CAPTURE TENSOR OUTPUT STEPS BACKEND REPORT [PRIOR]
llama-optimize gsq-pack MODEL CAPTURE TENSOR PACKED_OUTPUT STEPS BACKEND REPORT [PRIOR]
llama-optimize ptq MODEL CAPTURE TENSOR OUTPUT TYPE
llama-optimize rco-proxy MODEL CONFIG OUTPUT TENSOR_BYTE_BUDGET STEPS BACKEND REPORT
llama-optimize candidate-store MODEL CONFIG DIRECTORY REPORT
llama-optimize candidate-validate MODEL MANIFEST
llama-optimize candidate-publish MODEL MANIFEST ASSIGNMENT OUTPUT TOTAL_BYTE_BUDGET REPORT
llama-optimize candidate-verify MODEL MANIFEST ASSIGNMENT OUTPUT REPORT
llama-optimize global-check MODEL TOKEN_IDS CONFIG REPORT GPU_LAYERS
llama-optimize rco-global MODEL MANIFEST TOKEN_WINDOWS TEACHER_CACHE OUTPUT TOTAL_BYTE_BUDGET STEPS BACKEND REPORT [CONFIG]
llama-optimize eval MODEL TEXT REFERENCE REPORT GPU_LAYERS
llama-optimize corpus-eval MODEL TOKEN_WINDOWS TEACHER_CACHE REPORT GPU_LAYERS MODE(write/read)
llama-optimize infer MODEL TEXT GENERATION_TOKENS REPORT GPU_LAYERS
llama-optimize infer-tokens MODEL TOKEN_IDS GENERATION_TOKENS REPORT GPU_LAYERS
```

`eval` accepts `REFERENCE=-` to save teacher logits as `REPORT.logits`. Other evaluations require identical tokenizer output and compare full-vocabulary teacher KL, logits MSE and next-token PPL. Limits are 2..512 prompt tokens and 1..32 generated tokens. Calibration and evaluation text must be separate. Short authored prose is a smoke test, not a benchmark.

The RCO configuration is JSON with `candidate_directory` and a `tensors` array. Each entry has `tensor` and `capture`. Optional `gsq_model` replaces that entry's Q2_0 candidate with exact bytes from a compatible optimized GGUF. The candidate directory must not exist. Each candidate is stored as a packed `.bin` with a JSON manifest.

```sh
python3 tools/llama-optimize/make_prior.py --upstream /path/to/GSQ --capture linear.capture --output linear-gptq.prior --method gptq
python3 tools/llama-optimize/test_linear.py linear.capture linear-test.bin
python3 tools/llama-optimize/run_experiment.py /path/to/log-prefix build/bin/llama-optimize self-test
```

The measurement wrapper polls total device-0 VRAM, compute-process VRAM and process RSS every 0.2 seconds. These are observed peaks; short-lived allocations can be missed. Run GPU jobs sequentially to avoid attributing other jobs' VRAM to a test.

## Limits

Only dense 2D candidates are accepted. Expert tensor packing/backward, activation recomputation in this adapter, upstream sequential GSQ model replay and MoE pruning are not implemented. Native weight replacement also exposes the routed MatMul hook for a future tested model adapter; unsupported expert graphs are not silently flattened. Outputs replace selected active tensors; the others retain their original bytes. Large fixed BF16 embeddings and stored unused shared-KV tensors count towards the byte budget. The original general.file_type is retained; actual per-tensor types describe the mixed file.

Capture and prior files use a versioned magic, dimensions and contiguous native-endian FP32 arrays on the tested little-endian host. They are local experiment artifacts, not portable GGUF or IMatrix replacements. Only load files created for the specified tensor and original model. Failed materialization leaves `.partial` for inspection, and reruns refuse to overwrite it. Original models and existing outputs are never overwritten.

The existing Q2_0 PTQ codec accepts an importance argument but does not use its values. Supplying an IMatrix therefore does not create a different Q2_0 baseline. Full-Hessian GPTQ requires cross-channel activation products and cannot be reconstructed from the IMatrix diagonal.

## Global workflow

`prepare_data.py` preserves source text and writes immutable calibration/held-out ID windows plus split hashes. Use those IDs for teacher, student, captures and evaluation. A split within one conversation still has topic correlation. No Lumen-specific template is blindly applied to Gemma.

Candidate configuration uses `tensors: [{tensor, candidates: [{type, producer, ...}]}]`. PTQ encoding supports the existing codecs. GSQ imports require `packed_file` and `sha256`, or an existing compatible `model`. Weighted Q4_0 encoding uses `producer=imatrix-ptq`, `imatrix_file` (one FP32 importance per input column), and `imatrix_sha256`. Q2_0 is not a weighted candidate. All source/payload hashes, tensor shapes and aligned byte counts are verified on load. Routed operations require a separate tested expert packing/backward adapter.

`capture-store` captures all indexed active matrices during one supplied token window and releases each temporary tensor payload after writing. `generate_gsq_candidates.py` runs existing native GSQ-RTN and the pinned full-Hessian GPTQ prior bridge, seals payload hashes and assembles a producer-aware manifest. Native RTN remains the existing max-range variant, not upstream MSE scale search. The Python prior bridge also supports upstream RTN explicitly. Captures replay original BF16 inputs; this is a layerwise GSQ calibration mode, not upstream sequential quantized-model replay.

Global RCO configuration supports `objective` (`ce` or `kl`), `float32_math`, `learning_rate`, `seed`, `batches_per_step` (0 means all windows), `gumbel_samples`, `stop_after`, `resume`, `publish`, `monitor_window` and `teacher_cache_max_bytes`. Validated FP32 student math is the default and required training mode. Precision logits start uniformly at zero and are retracted to the actual-byte manifold. This differs from the upstream CLI default sensitivity initialization and nominal-bit uniform initialization; optimizer order and geometry remain independently checked. The cache limit evicts only validated payloads from the same source and recomputes their identical teacher forwards when needed. It does not change the objective. Per-step snapshots store logits, Adam moments, RNG state and settings/data fingerprints. `monitor_window` measures initial/final loss on the same calibration window; it is not a held-out metric.

`global-check` retains raw two-point and five-point finite-difference estimates with the same FP32 absolute/relative tolerances. Large steps expose nonlinear truncation and small steps expose forward rounding. `compare_global_gradients.py` compares CUDA adjoints to a CPU graph already validated by finite differences and records whether direct CUDA differences passed separately. Do not interpret adjoint parity as a direct GPU finite-difference pass.

`candidate-verify` streams the published file and checks every selected payload against its sealed candidate, every fixed payload against the source, all tensor types/shapes, normalized metadata and the exact file size. This checks GSQ preservation without decoding or requantizing its weights.

`prepare_suite.py` selects GSQ Q2 candidates using calibration reconstruction only and prepares static producer controls. `evaluate_suite.py` publishes, verifies and evaluates one temporary baseline at a time, then removes that generated file after saving its results. Persistent input models are retained. CUDA0 with 99 GPU layers is the default; `--gpu-layers 0` selects CPU evaluation. Existing result directories require identical configuration, token IDs and backend settings.

The discrete objective maximizes the sum of selected precision logits under a byte cap; it does not force capacity filling. Report the resulting unused bytes and size differences from uniform Q4 controls. Manifold retraction requires a target strictly between the minimum and maximum candidate costs; discrete publication also supports boundary assignments.

`corpus-eval` processes every supplied window, includes tail targets, and weights CE/KL by target count. `write` saves the BF16 reference logits; `read` validates original source and cache hashes, exact IDs, context and backend before scoring a student. It reports held-out CE/PPL, teacher KL, explicitly defined logit reconstruction MSE, actual file bytes, evaluation prefill throughput and peak process RSS. `infer-tokens` uses identical supplied prompt IDs for a separate native decode benchmark. Its short generation limit is not the quality evaluation token count.
