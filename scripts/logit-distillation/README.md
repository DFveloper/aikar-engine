# QLion2 logit distillation

This workflow extracts teacher top-k log-probabilities and one tail log-mass, then trains the student with pure forward KL over those k+1 categories. There is no hard-label CE term. Ground-truth IDs only select the existing supervised positions and provide reference accuracy; displayed loss is KL, and exponentiating it does not give perplexity.

Build the CPU tools:

```bash
cmake -S /home/user/aikar-engine -B /home/user/aikar-engine/build-distill-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build /home/user/aikar-engine/build-distill-cpu --target llama-extract-logits llama-finetune-qlion test-logit-distillation -j 8
/home/user/aikar-engine/build-distill-cpu/bin/test-logit-distillation
```

The copies of extract.sh and train.sh beside the models use these defaults:

- Teacher: /mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/Test.gguf
- Student: /mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/Lumen-3.8-Pulsar-base.gguf
- JSONL: Korean knowledge/reasoning prepared by prepare_ko.py from pinned public train splits. DATA selects an existing file. DATA_MODE=mixed restores the historical Korean-first-turn/English 1:1 mixture.
- Context 512, microbatch 128, 8 CPU threads, T=1, top-k=512, conversation validation fraction 0.05.
- QLion2 updates attention Q/K/V/output tensors only in blocks 11, 17, 23 and 29 (zero-based), while other parameters stay frozen. targets.sh contains the shared target list. Set TARGETS or pass --freeze-layers to change the selected parameters.
- Training writes a separate QLion adapter. Original model files are not output destinations.

Run extraction first, then training. Use the same CTX, MAX_WINDOWS, VAL_SPLIT, DATA and prompt/thinking settings in both stages. MAX_WINDOWS limits each partition separately; zero selects all packed windows. Context must match the runtime allocation, normally a multiple of 256. The training cache records its temperature, so training does not reinterpret it with a different temperature.

Small CPU example, with a separate output cache:

```bash
cd /mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/logit-distillation
CTX=256 UBATCH=256 MAX_WINDOWS=2 CACHE="$PWD/teacher-top512-c256.dat" ./extract.sh
CTX=256 UBATCH=256 MAX_WINDOWS=2 CACHE="$PWD/teacher-top512-c256.dat" EPOCHS=3 LR=0.0001 TARGETS=attn_q,attn_output OUTPUT="$PWD/qlion2-adapter.gguf" ./train.sh --freeze-layers 29
```

Extraction uses the student's vocabulary, chat template, conversation split, packing and next-token masks. It checks teacher token IDs before inference. It does not use llama-perplexity's clipped UINT16 format, and it processes all supervised positions, including positions in the first half of a window. Each packed window has an independent KV stream. No new BOS substitution is performed beyond the shared dataset packer.

QLKD0001 is a versioned little-endian DAT format. Its header records context, vocabulary size, k, temperature, token-ID fingerprint, window counts, train/validation boundary and JSON provenance. Each window stores context-sized input and label I32 arrays, followed by one teacher record per non-masked position. Each teacher record has k I32 IDs, k FP16 log-probabilities and one FP32 tail log-mass: 6*k+4 bytes, or 3,076 bytes for k=512. A reader renormalizes the k+1 probabilities after FP16 decoding. It checks inputs, masks, partitions, dimensions, file length, duplicate IDs and probability values. Cache row offsets are shuffled with dataset windows.

The loss uses the same teacher-selected IDs on the student and computes logsumexp over the full vocabulary and over the remaining vocabulary. Its scale is T*T. This preserves the individual top-k probabilities and total tail mass; it does not supervise relative probabilities inside the tail. A fused CPU forward/backward operation avoids the reference graph's vocabulary-sized intermediates. Full student logits and their gradient are still needed. Set GGML_DISTILL_REFERENCE=1 to compare against the original graph implementation.

Critical-token weighting and MTP are rejected with distillation. Native QLion training already rejects GRPO. The staging callback API must be configured before optimizer initialization. GPU_LAYERS=0 is the local default; the Runpod container selects CUDA offload. See runpod/README.md for A100 build and validation commands. With multiple microbatches, the existing trainer detaches past KV gradients. Use UBATCH=CTX when full-sequence KV gradients are needed.

Keep validation conversations separate before packing. The small example selects only two packed windows per partition and cannot establish general model quality or long-context recovery. Evaluate held-out task metrics and PPL separately before increasing training scale.

## Production recovery

The separate production wrappers use production.env. All wrappers use the same prepared DATA. The default now prepares Korean knowledge/reasoning with prepare_ko.py; see runpod/README.md for sources, counts, filtering and the measured ten-hour budget workflow. DATA_MODE=mixed preserves the historical Korean-first-turn/English 1:1 policy described in DATA_AUDIT.md. DATA overrides preparation entirely. Keep data and sampling settings fixed across extraction and training. Source files are unchanged and JSON sidecars record provenance.

Production uses 32 sampled held-out validation windows and top-512 plus tail at T=1 with pure KL. PROD_MAX_WINDOWS caps training windows; zero selects all. Local context/microbatch defaults are 512; the Runpod image starts with 1024 for both. The budget tool chooses the window cap and checkpoint interval from actual GPU throughput. Attention targets remain blocks 11, 17, 23 and 29, with three epochs and cosine learning rate 1e-5 to 1e-6 after 50 warmup steps. These are the second through fifth global blocks; shared KV source block 5 stays frozen. These settings have not established recovery quality.

Production uploads completed caches after extraction. At each SAVE_EVERY checkpoint it also exports a QLion adapter and queues that adapter plus optimizer state for immediate background publication in one public model repository. Full-model resume checkpoints remain local. Supply HF_TOKEN at runtime; the account namespace comes from whoami. HF_DATASET_REPO and HF_MODEL_REPO set repository names. HF_UPLOAD=0 disables publishing. No token is required for --dry-run. See runpod/README.md for layout and retries.

For this shared-KV model, all four selected blocks reuse KV from block 5. Their stored K projections do not participate in the forward graph, and no separate V projection exists. Thus eight Q/output tensors receive effective updates; four selected but unused K tensors remain unchanged. The shared KV source in block 5 stays frozen.

```bash
cd /mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/logit-distillation
./production-extract.sh --dry-run
./production-train.sh --dry-run
PROD_DIR=/path/with/enough/space ./production-extract.sh
PROD_DIR=/path/with/enough/space RUN_ID=recovery-01 ./production-train.sh
```

Use the same PROD_DIR, PROD_MAX_WINDOWS and packing settings in both stages. Run --dry-run for the actual data/cache plan. Historical reference only: the old Korean-first-turn/English mixture at context 512 had 4493 training windows and a 6609698365-byte cache. Those counts do not apply to the new Korean knowledge/reasoning data or the context-1024 budget profile. Use a fresh PROD_DIR whenever inputs change.

Training retains full model/state pairs at SAVE_EVERY and completion. The budget profile chooses an interval for at most nine pairs; production-train.sh --dry-run reports storage. SAVE_EVERY=0 disables intermediate checkpoints. Old local checkpoints are not automatically deleted after HF upload.

Interrupted extraction resumes complete windows from its .partial file and recomputes a torn final window. Completed caches are verified by SHA-256. Training resume requires an explicit checkpoint, its sibling .qat-state.gguf, the original RUN_ID, and unchanged numerical/data settings:

```bash
PROD_DIR=/path/with/enough/space RUN_ID=recovery-01 \
  RESUME=/path/with/enough/space/runs/recovery-01/final.gguf.step1000.gguf \
  ./production-train.sh
```

Run manifests, commands, logs and status files live under PROD_DIR. Checkpoints include optimizer, scheduler, shuffle and distillation configuration. Keep both files in a checkpoint pair together. Outputs are separate from the source models. This workflow has been exercised on small real CPU runs; full production recovery quality has not been measured.

## Throughput measurement

See THROUGHPUT.md for actual mixed-data CPU rates with attention targets in blocks 11, 17, 23 and 29.

Run bash benchmark.sh for a sequential CPU extraction/training benchmark with production context, threads and attention targets, using two training and two validation windows by default. Override BENCH_WINDOWS for a longer run. It retains the small cache and logs beside the models and removes its own temporary full-model checkpoint after completion. It needs about 15 GiB temporary space for that checkpoint and optimizer state.

Extraction logs input_tok_s for all processed input tokens and supervised_tok_s for teacher rows written. Training logs both rates for the training loop, including forward/backward, optimizer updates, cache staging and graph setup. Epoch-end validation and final checkpoint writes are excluded; any periodic evaluation/checkpoints within a longer epoch are included. These are packed-token presentations, including padding and overlap, rather than unique corpus tokens. Separate .time files record whole-process wall time including loading, validation and serialization. The benchmark uses one epoch and no periodic evaluation or intermediate checkpoints.
