# QLion2 offline logit distillation CPU validation

Date: 2026-11-04. This is a local working-tree implementation on the AIKAR fork. Existing unrelated edits were retained. No commit or push was performed.

## Implementation

- llama-extract-logits shares the native trainer's JSONL rendering, conversation split, packing and next-token masks. It reads tokenizer and chat template metadata from the student, checks teacher token-ID compatibility, and exports unclipped top-512 FP16 log-probabilities plus FP32 tail log-mass.
- llama-finetune-qlion --distill-cache uses only coarse forward KL over the teacher-selected 512 token IDs and one tail category. No hard-label CE is added. Ground-truth IDs only determine supervised positions and reference accuracy. Temperature comes from the cache and the loss uses T*T scaling.
- The loss uses existing GGML operations, stable logsumexp for both full vocabulary and tail, and automatic differentiation. It materializes vocabulary-sized intermediates; no fused low-memory KD kernel was added.
- The loader verifies context, vocabulary fingerprint, train/validation boundary, all packed input IDs and masks, file length, teacher ID ranges/uniqueness and probability normalization. It renormalizes after FP16 decoding. Dataset auxiliary offsets preserve cache alignment under shuffling.
- Context rounding initially rejected an unaligned 128-token training attempt. Extraction now checks runtime context alignment too. The valid experiment uses 256 tokens. Earlier diagnostic files are under diagnostics/.

## Inputs and machine

Student: /mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/Lumen-3.8-Pulsar-base.gguf (lumen4exp, 30 layers, shared_kv_layers=19).

Teacher: /mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/Test.gguf (gemma4, 30 layers).

Data: /mnt/openwebui/AIKAR/train_ko.jsonl. The loader expanded 317 conversations to 483 assistant targets. The full conversation split was 301 train / 16 validation conversations. Only the first two packed windows from each partition were selected for these smoke tests.

Machine: Intel Xeon E5-2696 v3, 18 cores / 36 logical CPUs, about 121 GiB RAM. The tools used 8 CPU threads and no GPU offload. NVIDIA driver access was unavailable, so P620 execution was not tested.

## Teacher extraction

Settings: context=256, microbatch=256, T=1, top-k=512, validation fraction=0.05, maximum 2 windows per partition, thinking preserved, response-only masks.

Cache: smoke-top512-c256.dat, 2,696,932 bytes. Four windows contain 874 supervised records: 420 train and 454 validation. Each record is 3,076 bytes, plus window token/mask arrays and the header. These counts include packing overlap and are not unique-token counts.

Extraction processing took 42.595 seconds after teacher initialization, about 54.805 seconds including initialization and dataset preparation according to the tool log. Mean teacher tail mass was 0.014184289 (1.4184%); maximum was 0.33824516 (33.8245%). This small sample does not establish that top-512 is sufficient for every data domain.

## Full-window training

Settings: QLion2, Q4_0 selected weights, learning rate 1e-4, T=1, context=batch=microbatch=256, shuffle enabled, three epochs. Only blk.29.attn_q.weight and blk.29.attn_output.weight were trained; other tensors stayed frozen. There were 6 optimizer updates and 1,260 supervised-position presentations across the three epochs.

| Checkpoint | Optimizer step | Validation coarse KL | Reference token accuracy |
| --- | ---: | ---: | ---: |
| Before training | 0 | 2.68925476 | 26.4317% |
| Epoch 1 | 2 | 2.39438766 | 28.8546% |
| Epoch 2 | 4 | 2.27938831 | 28.6344% |
| Epoch 3 | 6 | 2.17374378 | 29.7357% |

Validation KL decreased by 19.1693%. This is a pipeline smoke test on two validation windows, not evidence of general capability recovery, long-context quality, or a new PPL result. KL is not NLL and exp(KL) is not perplexity. The user's prior PPL measurements were not repeated here.

The recorded compute workspace was 1,733,842,944 bytes. This is workspace allocation, not peak RSS. Full-file base hashing is performed by the existing replacement-adapter exporter before and after training and adds startup/save time.

## Microbatch training

A separate run used the same cache and targets, microbatch=64, one epoch, two optimizer updates, and Q8_0 gradient accumulation. It used no initial validation pass. The first prompt-only microbatch produced zero loss and was not sent to the teacher-row callback. Reference accuracy is undefined when a microbatch has no supervised tokens.

The epoch training KL was 2.67241672 and final validation KL was 2.39630082. The full-window first-epoch values were 2.67241663 and 2.39438766. These runs are not asserted to be numerically identical: microbatching uses Q8 accumulation and the existing detached past-KV gradient behavior.

Compute workspace was 433,985,024 bytes. Q8_0 gradient accumulator storage was 49,020,928 bytes. Maximum RSS from /usr/bin/time was 15,600,380 KiB, about 14.88 GiB. The process exited successfully. Wall time includes model hashing and adapter saving; it is not a training-throughput benchmark.

## Saved artifact and reload

The main output is smoke-qlion2-adapter.gguf, 25,953,440 bytes. This is a QLion quantized replacement adapter, not a LoRA adapter or a complete model.

| Tensor | Payload bytes | Bytes changed from original base |
| --- | ---: | ---: |
| blk.29.attn_q.weight | 12,976,128 | 3,346,074 |
| blk.29.attn_output.weight | 12,976,128 | 3,336,951 |

The existing core loader loaded the original base plus this adapter and completed CPU inference. All 262,144 output logits were finite. Relative to the original base on the same short "hi" input, all output logits changed; maximum absolute difference was 3.72312838 and RMS difference was 0.69424801. This verifies that the serialized adapter is applied; it is not a quality evaluation.

The microbatch run wrote microbatch-qlion2-adapter.gguf separately. Neither original model was an output destination. Their modification times remained 2026-11-04 17:20:59 for the student and 2026-11-03 20:20:25 for the teacher, in Asia/Seoul time.

## Checks

- CPU build of llama-extract-logits, llama-finetune-qlion, test-logit-distillation and test-qlion-loader passed.
- The distillation test compares loss and finite-difference gradients against a double-precision reference at T=1 and T=2, including masked rows and a 1,000-logit gap. It also checks tiny unclipped teacher tails, header/version errors, duplicate IDs and normalized FP16 cache reloads.
- Existing test-opt passed: 153/153 and 54/54 checks, 3/3 backend/optimizer groups.
- Existing test-critical-token-sft passed.
- Actual extraction, full-window training, masked microbatch training, shuffling, adapter payload inspection and core-loader inference passed.
- Shell syntax and git diff --check passed. GPU execution, distributed training, MTP and critical-token weighting with KD were not validated; the latter two modes are rejected by this workflow.

See README.md and the separate extract.sh / train.sh scripts for build and reproduction instructions. Runtime logs and validation JSON files are stored beside this report. The smoke cache requires CTX=256 and MAX_WINDOWS=2; default script settings intentionally use a separate teacher-top512.dat cache.
