# Parallel QLion training implementation plan

Goal: preserve 4096 tokens per sequence while reducing training time on P620, with true parallel sequences, independent gradient accumulation, and CUDA backward support.

Architecture: reuse existing --parallel, KV stream batching and Q8 gradient accumulators. Context/batch CLI sizes remain per sequence in native QAT. Pack independent windows with sequence IDs and reset positions per window. Average sequence gradients, flush partial groups, and checkpoint only at optimizer boundaries. Reuse CUDA/cuBLAS primitives for attention backward and keep numerical stability with FP32 softmax derivatives.

Authorization: the user delegated kernel, refactor and optimization decisions and requested autonomous build/test/debug. Continue with those instructions; no commits or pushes.

- [x] Profile current training; capture CUDA kernel/API timings before changes.
- [x] Add P620 cuBLAS FP32 attention backward fast path with GPU/CPU numerical comparisons, strided tensors, multiple heads/batches, sinks/masks and tiny-gradient tests. Keep generic fallback for unsupported cases.
- [x] Add independent --grad-accum and native QAT --parallel semantics, sequence-separated attention and cache gradients. Preserve per-window checkpoint position, scheduler steps, reward weights, and critical metadata or explicitly reject unsupported combinations.
- [x] Test parallel isolation against sequential evaluation, tail batches, microbatch accumulation, partial accumulation flush, checkpoint resume, invalid sizes, CPU and CUDA learning/finite states.
- [x] Benchmark 4096 single and parallel configurations on P620 within 2 GiB; optimize measured CPU fallback boundaries where practical. Preserve old checkpoints and resume fastest verified configuration in tmux.
- [x] Review diff, run relevant suites, and record commands/results under /mnt/openwebui/AIKAR/QLion2-test.

Ruling: do not reduce sequence length to claim a throughput gain. Different effective batches change optimization, so report throughput and effective batch explicitly.

Review: independent reviewer found weighted tail-loss reporting and public API accumulation with b<c. Added weighted result aggregation/regressions and rejected partial logical windows for multi-batch accumulation. CPU/GPU backward numerical comparisons include tiny gradients and deterministic sink sums across batches. No commits or pushes.

Final results: same 4096/6-update baseline 96.24s ->7.786s (12.36x), fixture VRAM1170->990MiB. Balanced16-window test P1A1=19.56s, P2A1=20.87s, P1A2=22.82s; P1A1 selected for maximum throughput and unchanged effective batch. Parallel2 full4096 fit1766MiB; P4 full did not fit2GiB. CPU/CUDA CTest5/5 each, launcher14/14, CUDA memcheck0errors. CUDA P2A2micro64 loss9.038->8.400 over30updates; all86weightsfinite,79changed. CPU resume86model+237state tensorsbitidentical. Full-corpus run resumedstep5020, savedstep5030 verifiedfinite79momenta, nowtrainingin tmuxfoundation-p620; actual full-corpusVRAMabout1425MiB. Artifacts: /mnt/openwebui/AIKAR/QLion2-test/parallel-training-20261027.
