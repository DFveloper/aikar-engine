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
- JSONL: /mnt/openwebui/AIKAR/train_ko.jsonl
- Context 512, microbatch 128, 8 CPU threads, T=1, top-k=512, conversation validation fraction 0.05.
- QLion2 updates attention Q/K/V/output tensors, while other parameters stay frozen. Set TARGETS or pass --freeze-layers to change the selected parameters.
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

The loss uses the same teacher-selected IDs on the student and computes logsumexp over the full vocabulary and over the remaining vocabulary. Its scale is T*T. Tail logits use an additive -1e30 mask on selected IDs. This preserves the individual top-k probabilities and total tail mass; it does not supervise relative probabilities inside the tail. It uses existing GGML graph operations and materializes vocabulary-sized intermediates. This is not a fused or vocabulary-memory-free loss implementation.

Critical-token weighting and MTP are rejected with distillation. Native QLion training already rejects GRPO. The staging callback API must be configured before optimizer initialization. GPU execution is unverified; the supplied scripts explicitly use CPU. With multiple microbatches, the existing trainer detaches past KV gradients. Use UBATCH=CTX when full-sequence KV gradients are needed.

Keep validation conversations separate before packing. The small example selects only two packed windows per partition and cannot establish general model quality or long-context recovery. Evaluate held-out task metrics and PPL separately before increasing training scale.
