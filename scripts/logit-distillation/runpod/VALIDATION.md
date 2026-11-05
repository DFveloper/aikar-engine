# CUDA recovery validation

## Immediate adapter checkpoint publication

The streaming-logit draft was removed before release; extraction/cache publication is unchanged. The trainer now uses its existing QLion snapshot/export implementation for each periodic adapter checkpoint as well as the final adapter. The successful checkpoint event follows adapter close. Production enables --qat-adapter-out, monitors save events every second and uses a single background upload worker. It publishes adapter.gguf, matching optimizer state and run.json into one model repository; full model checkpoints stay local. The adapter's embedded base hash remains authoritative, including the resume-checkpoint base after training resume.

A child-process/FIFO test keeps the producer running until the fake Hub acknowledges its adapter upload, proving upload starts before training exits. It also verifies the payload contains an adapter and no full model, and excludes incomplete saves. These tests do not publish real data to Hugging Face.

A real CPU run reuses the historical context-256 smoke cache for one update with SAVE_EVERY=1. It writes a periodic QLion adapter at step 1 and a final adapter, both 77859136 bytes with six changed tensors. Their SHA-256 values match, and the accompanying optimizer state records step 1. The process exits successfully (416.99 seconds total, peak RSS 18013984 KiB). The unchanged exporter rehashes the full base for each adapter; the observed export takes about 91 seconds on this old CPU and pauses training during serialization. Network transfer runs separately in the Python background worker. This is a save/upload integration check, not a recovery-quality result.

## Korean ten-hour profile and startup fix

The current profile targets CUDA 12.8, sm_80 and sm_120 for A100/RTX PRO 6000 Blackwell. Context=microbatch=1024 is a starting benchmark setting, not a measured VRAM guarantee. Local target-GPU end-to-end training is still unverified.

The live Pod diagnostic showed /start.sh running, JUPYTER_PASSWORD unset, no Jupyter process, no listening port and no /jupyter.log. The upstream startup skips Jupyter when no password is supplied. The new wrapper creates a random mode-600 token under /run/aikar-jupyter/token when needed, retains it across restarts and starts the upstream script. An actual local container without JUPYTER_PASSWORD passes authenticated API/kernel checks before and after restart, with token permissions 600 and no token printed to container startup logs. The startup log is linked to /run/aikar-jupyter/jupyter.log with mode 600. Jupyter config, runtime, data and token also use container-local storage; startup never changes workspace file modes. The permissions regression test mounts /workspace read-only, preloads the old workspace log symlink and checks authenticated startup, unauthenticated rejection, private token mode and restart reuse.

Real-source tests read pinned Parquet data. The initial YiSang candidate was excluded after a Korean-reasoning filter left only three eligible rows in one shard. The replacement Korean reasoning 12k source has 11349 eligible rows under the final structural/language filters. KoCommercial has 123134 eligible knowledge rows. A 16-row combined sample checks first-turn parsing and separate Korean reasoning and passes a real-model extraction plan at context 1024. Full preparation samples 6000 reasoning plus 4000 knowledge rows, then normalized-prompt deduplication produces 9996 records (5996 reasoning, 4000 knowledge), 18109010 bytes. Both sources and their revisions are recorded in the generated sidecar.

Regression tests cover Korean-language filtering, JSON-string messages, system/first-turn preservation, reasoning separation, incomplete reasoning rejection, benchmark-derived time/cache bounds and checkpoint retention. The ten-hour estimate reserves two hours, discounts measured compute rates by 20%, caps worst-case cache at 64 GiB and retains at most nine checkpoint sets. It is not a wall-clock kill switch. Actual Pod benchmark rates are required; no A100/Blackwell throughput is inferred from CPU/P620 measurements.

Date: 2026-11-04. CPU baseline commit: b928ce389. CUDA changes and the Korean first-turn policy are subsequent working-tree changes.

## Runpod Ubuntu 24.04 and JupyterLab revision

The revised runtime uses runpod/base:1.0.7-ubuntu2404 at digest 7e45ffb69c0625b22c3fe90886b3192e95e0b9ea44d0861333a874ce37a07187, with NVIDIA CUDA 12.4 runtime libraries and NCCL copied in. The native CUDA tools are unchanged. The default command is the official /start.sh; distill is available from JupyterLab terminals. The earlier minimal CUDA runtime image described below is superseded by this revision.

A local container started with the default command and a random JUPYTER_PASSWORD passes authenticated Jupyter API access and exposes the Python kernel. Unauthenticated API access is rejected. JupyterLab 4.5.9 and NumPy 2.2.6 import successfully. The distill help command, CPU test-logit-distillation and Python mixture regression test pass in this running container. The host driver library is mounted read-only for the CPU binary checks. No GPU device or A100 is available inside this local Docker test. Raw output: /tmp/aikar-runpod-jupyter-test.log.

Upstream references: https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile and https://github.com/runpod/containers/blob/main/container-template/start.sh. JupyterLab is enabled through Runpod's Start Jupyter Notebook setting or JUPYTER_PASSWORD.

Hugging Face uploader tests use a fake Hub API: token-owner namespace discovery, no cached-token fallback, public-repository creation, rejection of an existing private repository, checksum rejection, receipt-based skipping, incomplete checkpoint exclusion, model/state atomic commit and one shared model repository all pass. No HF_TOKEN is present in this local session, so no real dataset/model publication has been performed. Runtime publication uses HF_TOKEN supplied by the Pod environment.

## Tested hardware and builds

The local GPU is a Quadro P620 with 1992 MiB VRAM, compute capability 6.1. Native CUDA 12.4 builds include sm_61 and sm_80. The container targets sm_80 for A100. No A100 is available locally, so full offload, A100 memory use, throughput and recovery quality remain unverified.

The container links against the CUDA driver stub during the build, then relies on NVIDIA Container Toolkit for the real driver at runtime. Driver stubs are not copied into the runtime image. The build directory uses a BuildKit cache for incremental rebuilds.

The default sm_80 Docker image builds successfully as aikar-distill:a100, about 1.57 GB reported image size. Container help, test-logit-distillation and the Python mixture regression test pass. Runtime shared-library resolution has no missing dependencies when the host libcuda.so.1 is mounted read-only. This machine has no NVIDIA Docker runtime configured; this check does not execute container GPU kernels. Native P620 kernels are tested separately below.

## Numerical checks

- CPU test-logit-distillation passes KL reference, finite-difference gradient, mask, temperature and cache checks.
- CUDA DISTILLATION forward matches CPU for vocabularies 8, 64 and 262144, including top-512 at the model's vocabulary size. Finite-difference gradient checks pass for the two small shapes; the harness skips the large shape's finite-difference calculation. Cases include masked rows and temperatures 1 and 2.
- CUDA Q4_1 MUL_MAT_ID_BACK passes five contiguous-weight cases, with input dimensions 32, 96, 256, 288 and 704 and duplicate expert routes. Five strided-weight cases correctly report unsupported and can fall back to CPU. The support gate was tightened after a strided-weight test exposed an assertion.
- CUDA F32-to-FP4_KV and FP4_KV-to-F32 copy cases both match CPU. This enables GPU fake quantization in the full-sequence training graph. Persistent FP4_KV cache SET_ROWS remains CPU-only.
- CUDA quantized microgradient accumulation passes six reset/non-reset cases; combined with the three forward KL cases, that invocation passes nine cases. This does not verify a full GPU optimizer trajectory.
- The broader QAT_TEST_ADAPTIVE_ONLY=CUDA test aborts with an illegal memory access reported in launch_mul_mat_q/mmq.cuh while running compact MoE accumulation trajectories. Its cause has not been established. That suite exercises expert-weight updates, which are outside the selected attention targets. It must not be reported as a passing CUDA regression suite.
- Python mixture regression checks pass deterministic sampling, Korean truncation, reasoning/system preservation, English preservation and cache-policy rejection. Shell syntax, Python compilation and git diff whitespace checks pass.

## Real-model smoke run

The new dataset contains 317 Korean first-turn records and 317 unchanged English conversations sampled with seed 42. Context and microbatch are 256; one training window and one validation window are retained. Teacher top-512 plus tail uses T=1. There is no ground-truth CE. Eight CPU threads are used, with attention targets in blocks 11, 17, 23 and 29.

Teacher extraction with one GPU layer completes: 512 input presentations and 356 supervised rows in 8.727 seconds, or 58.669 input tok/s and 40.793 supervised tok/s. End-to-end process time includes data preparation/model loading and is longer. This short context and partial-offload measurement is not comparable to a sustained A100 production run.

Training with the same partial GPU offload cannot allocate its 10880.60 MiB CUDA compute buffer on the 2 GB P620 and aborts before a training update. CPU training using the exact extracted cache completes one update: 256 input presentations and 191 supervised presentations in 65.478 seconds, or 3.910 input tok/s and 2.917 supervised tok/s. Validation KL is 1.27079320 before the update and 1.29325414 afterward. One update does not establish recovery quality; this smoke run does not improve the held-out loss. CPU compute workspace is 2588866560 bytes.

Local raw logs are /tmp/distill-cuda-kl-grad.log, /tmp/distill-cuda-moe-test-fixed.log, /tmp/distill-cuda-fp4-test.log, /tmp/distill-cuda-optimizer-test.log, /tmp/distill-cuda-qat-test.log and /tmp/distill-firstturn-cpu-train.log. The extraction log and cache are in MODEL_DIR/logit-distillation/benchmark-cuda-firstturn-c256/.

The CPU process exits successfully after writing /tmp/distill-firstturn-smoke-adapter.gguf, about 75 MiB. The adapter contains changed Q weights in all four target blocks and changed output weights in blocks 23 and 29. The Q8 output weights in blocks 11 and 17 do not change quantized payload after this single update. Whole-process wall time is 336.79 seconds, including loading, initial/final validation and adapter generation. Peak RSS is 17952288 KiB. Docker compilation ran concurrently, so these rates are smoke-test measurements under CPU contention.

For production, run preflight and benchmark on the actual A100 before extraction and training. The first-turn context-512 plan has 4493 training windows, 2127247 supervised presentations per epoch, 32 validation windows and a 6609698365-byte cache. The default three epochs retain up to 14 model/state checkpoint pairs and budget about 193.63 GiB after extraction. Recommend an A100 80 GB and a persistent volume with at least 300 GB free, including source models.
