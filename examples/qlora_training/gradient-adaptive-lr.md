# Gradient-adaptive QLion LR scaling

Measurements below use LR=1 with the native QLion/QAT operations. They establish small-problem behavior, not convergence on a pretrained production model.

## Synthetic quantized regression

Run `QAT_ADAPTIVE_BENCH=1 build/bin/test-qat`. This optional benchmark runs 400 native Q4_0/Q8_0/Q4_0 updates on 544 weights with beta=0.9, no weight decay, fixed tau=1 or 16 warmup samples. The diagonal quadratic has targets `0.125 + 0.01*sin(i)` and curvature from 0.25 to 1 across 17 quantization blocks. It computes gradients from decoded quantized weights, not a floating-point master copy.

Tail loss is the mean over steps 300-399. Amplitude is the RMS across coordinates of each coordinate's tail maximum minus minimum. Convergence is the first step of 20 consecutive losses below 0.001; `none` means this did not happen within 400 updates.

| Unit | Tau estimation | Tail loss | Amplitude RMS | Convergence step |
| --- | --- | ---: | ---: | ---: |
| off | n/a | 0.336710 | 4.08214 | none |
| layer | fixed 1 | 0.0000144393 | 0.0248434 | 49 |
| 256 | fixed 1 | 0.0000144058 | 0.0249546 | 48 |
| 32 | fixed 1 | 0.0000142127 | 0.0245296 | 46 |
| layer | warmup 16 | 0.0682268 | 2.09699 | none |
| 256 | warmup 16 | 0.0825038 | 2.05044 | none |
| 32 | warmup 16 | 0.113757 | 2.42186 | none |

The fixed threshold suppresses persistent oscillation in this example. Warmup reduces the tail loss but does not achieve the convergence criterion. Estimating tau from momentum magnitude alone does not identify the curvature or a universally stable step size.

## Native trainer smoke comparison

A generated Llama fixture has two blocks, embedding size 256, FFN size 384, two attention heads, and 64 tensors. It was converted using `llama-quantize --pure ... Q4_0`; the resulting file is about 0.66 MiB. The dataset contains 16 identical records:

```json
{"text":"The cat sits on the mat. The dog runs in the park. One plus one is two. "}
```

Each run starts from the same Q4_0 file and uses the following settings, changing only scaling mode and tau estimation:

```bash
llama-finetune-qlion --model tiny-q4.gguf --train-file train.jsonl \
  --quant-type q4_0 --optimizer qlion -c 32 -b 32 -ub 16 \
  -lr 1 --lr-scheduler constant --epochs 20 --val-split 0 -t 2 \
  --qat-lr-scale layer --qat-lr-tau 0.01 --qat-out trained.gguf
```

There are eight logical optimizer updates per epoch, 160 total. The table uses unweighted epoch training loss, not the progress bar EMA. Amplitude is the peak-to-peak epoch loss over the last five epochs. Stable epoch is the first epoch whose mean loss and all remaining epoch means are below 0.2. This is a training/overfitting smoke test without held-out validation.

| Unit | Tau estimation | Last 5 epochs mean loss | Last 5 epochs amplitude | Stable epoch |
| --- | --- | ---: | ---: | ---: |
| off | n/a | 0.847368 | 0.864575 | none |
| layer | fixed 0.01 | 0.104767 | 0.00837073 | 2 |
| 256 | fixed 0.01 | 0.140258 | 0.0225440 | 15 |
| 32 | fixed 0.01 | 0.128992 | 0.0243590 | 14 |
| 32 | warmup 16 | 0.581704 | 0.217210 | none |

The warmup run matches the original optimizer for its first two epochs, as expected for 16 unscaled updates. For this fixture, layer-wise fixed tau gives the best stability. These numbers do not select a default tau for other models.

## Validation

- CPU `test-qat` covers independent RMS and update references, all three units, the partial 256-element group, zero momentum/gradient, NaN/Inf gradients and nonfinite momentum scales, large thresholds, Q4_0/MXFP4/Q8_0 weights, Q8_0 gradients, fast state scaling, routed/row/sparse-row/tied updates, and a tied-alias graph whose warmup count advances once per logical update and then freezes.
- `test-backend-ops -o QLION_QAT_STATS` checks 32, 256, 544, and 65536 elements against CPU. CPU, Vulkan llvmpipe, and CUDA/Vulkan on a Quadro P620 execute these cases successfully.
- The Vulkan adaptive tests also execute dense F32/Q8_0-gradient, routed, and row updates. Existing unsupported sparse-row and tied Vulkan operations retain scheduler fallback to CPU.
- CUDA kernels compile for SM70 and SM61. The focused adaptive tests and the full CUDA QAT suite pass on a Quadro P620 (SM61), including dense, routed, row, sparse-row, and tied updates. These correctness tests do not establish GPU training performance.
- A two-epoch CPU trainer run with `-b 32 -ub 16`, warmup=2, and a checkpoint after eight updates produces identical bytes in all 64 final model tensors when resumed. Changing the scaling unit during resume is rejected.
- The pre-existing full Vulkan suite fails `batched router backward differs from reference` on llvmpipe and the Quadro P620. This was also reproduced with the original `HEAD` version of `test-qat.cpp`; the focused adaptive tests pass.

For a Vulkan build, run the focused tests with `GGML_VK_VISIBLE_DEVICES=0 QAT_TEST_ADAPTIVE_ONLY=Vulkan build/bin/test-qat`. The filter fails if no matching backend is present. Use `QAT_TEST_BACKEND=Vulkan` for the existing full backend suite.
