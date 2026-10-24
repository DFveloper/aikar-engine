# Pulsar IQ3 temporary serving: historical measurements

This report records the previous serving session. It is superseded by [the CPU LES and cache-aware FA results](pulsar-iq3-cpu-les-serving-20261024.md). The current launcher disables SWA-full, uses the new CPU build and offers the implemented LES as an experimental option. Descriptions of full SWA, missing LES, old defaults and active PIDs below are historical, not current operating instructions.

Model: `/mnt/openwebui/AIKAR/Adaptive-bit/models/Lumen-3.1-Pulsar-IQ3-Quality.gguf` (11,760,518,400 bytes). Gemma4, 30 layers, 128 experts, 8 selected experts. Host: Xeon E5-2696 v3, 18 physical cores / 36 threads, 121 GiB RAM, Quadro P620 2 GiB, driver 580.178.04.

## Selected configuration

Use CPU with 18 decode and batch threads, native AVX2 compilation, weight repacking, Flash Attention, batch 1024, microbatch 512, F16 KV, context 8192, and one slot. Keep prompt caching enabled for serving. The CPU default uses `--swa-full` to retain SWA history in RAM: the bounded conventional SWA configuration recomputed the entire repeated 128-token prompt with checkpoints disabled. Full SWA avoids that reset condition and costs additional RAM. Thinking is disabled by default; set `PULSAR_IQ3_REASONING=auto` to restore automatic thinking. The default output limit is 1024 tokens and can be overridden with `-n`.

```bash
./scripts/serve-pulsar-iq3-temporary.sh
```

API base: `http://127.0.0.1:11438/v1`. Model alias: `Lumen-3.1-Pulsar-IQ3-Quality`.

Paged KV is available as an optional CPU configuration. It uses Q8_KV and 16-token pages, with the physical page budget derived from the context size. It was slower in the measured workload, so it is not the default.

```bash
PULSAR_IQ3_PAGED=1 ./scripts/serve-pulsar-iq3-temporary.sh
```

Environment overrides: `PULSAR_IQ3_MODEL`, `PULSAR_IQ3_SERVER_BIN`, `PULSAR_IQ3_HOST`, `PULSAR_IQ3_PORT`, `PULSAR_IQ3_CTX`, `PULSAR_IQ3_THREADS`, `PULSAR_IQ3_THREADS_BATCH`, and `PULSAR_IQ3_REASONING`. Additional server arguments can be appended to the command. Context shifting is disabled; requests must fit the context, including generated tokens.

## Measurements

Sequential local HTTP requests, one slot, context 8192, greedy sampling, 512 input tokens, 64 generated tokens, two repetitions. Values are arithmetic means in tokens/second. Each server performs its startup warmup. These are screening results, not broad workload or quality evaluations.

CPU-only build, commit `36a623af6`, build 12164. This initial screen used bounded SWA, before the final cache verification selected full SWA history for conventional KV:

| Configuration | Prompt processing | Generation |
| --- | ---: | ---: |
| F16 KV, microbatch 128 | 64.13 | 10.38 |
| F16 KV, microbatch 512 | 72.89 | 10.35 |
| F16 KV, microbatch 128, physical-core affinity | 65.67 | 10.40 |
| Paged Q8_KV, page 16, microbatch 128 | 53.62 | 10.08 |
| Paged F16, page 128, microbatch 512 | 41.07 | 8.72 |

The larger microbatch improved prompt throughput by about 14% in this screen. Affinity did not produce a meaningful generation improvement. A separate initial thread screen found 18 threads faster than 36 threads; its raw data uses an older CPU binary and must not be treated as a comparison against the final build.

CPU/P620 comparison using the same newly built SM61-capable binary (`build-qlion-adaptive/bin/llama-server`, same source commit):

| Configuration | Prompt processing | Generation |
| --- | ---: | ---: |
| CPU, F16 KV, microbatch 512 | 73.73 | 10.51 |
| P620 dense layers + CPU experts, Paged Q8_KV, microbatch 32 | 34.26 | 9.48 |
| P620 dense layers + CPU experts, CPU F16 KV, microbatch 32 | 36.18 | 8.58 |

The hybrid configurations use `-ngl 99 -ncmoe 30 -dev CUDA0 --load-mode none -b 128 -ub 32`. CPU KV adds `--no-kv-offload`; paged GPU KV adds `--kv-paged --kv-block-size 16 -ctk q8_kv -ctv q8_kv`. The paged GPU configuration used 1940 MiB VRAM in one observation during inference; this is not a peak-memory measurement. All three returned `42` for the arithmetic chat smoke test.

The CPU launcher with bounded SWA was tested with exactly 2048 uncached input tokens and 64 generated tokens, three repetitions. Mean prompt throughput was 68.71 tokens/second; generation was 9.64 tokens/second in every repetition after rounding. The arithmetic chat returned `42`. The optional paged launcher passed a separate smoke measurement at 512 input tokens: 57.10 prompt tokens/second and 10.23 generated tokens/second (one repetition).

Final launcher measurements with full SWA history, two repetitions at each input length, 64 generated tokens, no prompt reuse:

| Input tokens | Prompt processing | Generation |
| --- | ---: | ---: |
| 512 | 73.03 | 10.49 |
| 2048 | 69.57 | 9.33 |

Full SWA has about 3% lower generation throughput at 2048 tokens than the bounded configuration in this screen. It is selected to retain prompt history and avoid full reprocessing on repeated requests. The final configuration also returned `42` in the arithmetic chat smoke test.

## Active server verification

The session started the selected CPU launcher at `127.0.0.1:11438`. The health endpoint returned HTTP 200 after the start/check process exited, and `/v1/models` returned the expected alias. Arithmetic returned `42`; the Korean instruction returned the requested Korean readiness phrase; the OpenAI streaming response included content and its final `[DONE]` event.

For a repeated 128-token native completion request, the final configuration reused 127 tokens and evaluated one token. Prompt latency decreased from 2174.98 ms to 88.46 ms in this smoke check. This is one cache-reuse example, not a general latency guarantee. The process RSS was about 18.0 GiB after validation; P620 memory usage returned to 7 MiB. These are point observations, not peak measurements.

Runtime records are `build-pulsar-iq3-cpu/results/serving.pid`, `serving.log`, and `serving-check.json`. Stop the session-started server before launching another configuration on the same port:

```bash
kill "$(cat build-pulsar-iq3-cpu/results/serving.pid)"
```

## LES and rejected configurations

The source has an expert cache (`--moe-cache-mib`), but it does not implement LES prediction, asynchronous lookahead prefetch, or ready-expert-first execution. The previous LES document at `docs/superpowers/specs/2026-10-18-pulsar35-p620-les-design.md` is a proposed design for a different model file. No LES implementation or speedup is claimed here. The measured paged hybrid has about 108 MiB of device memory remaining, less than an additional 128 MiB expert cache. Expert caching was not benchmarked for this IQ3 model.

The preexisting `build-sync-cuda` targets SM70 and failed on P620 with `no kernel image is available`. Its F16 hybrid also exhausted VRAM when reserving compute buffers. These failed configurations are not valid GPU performance baselines. The SM61 server was then built and measured successfully.

The preexisting CPU server crashed in argument-parser construction. Its common library and server objects had different build dates. A separate consistent CPU build starts successfully; no inference source was changed to address this build artifact problem.

## Reproduction and artifacts

```bash
cmake -S . -B build-pulsar-iq3-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
CCACHE_DIR=/tmp/pulsar-iq3-ccache cmake --build build-pulsar-iq3-cpu --target llama-server llama-bench -j 18
```

Raw logs, commands, timings, outputs, and measurement scripts are in `build-pulsar-iq3-cpu/results/`. `cpu-paged-f16.json` and `cpu-paged-q8.json` are preliminary runs with prompt reuse and must not be used as uncached prefill baselines. Their repeated requests evaluated only one prompt token. Corrected runs explicitly pass `--no-cache-prompt` and assert that every input token was evaluated. Serving intentionally enables caching.

`fixture-short-input.json` records a rejected measurement: the first long-prompt fixture contained only 1442 tokens. The corrected fixture checks available token count before testing, and all three final long measurements evaluated 2048 tokens.

Only the launcher and this note are source additions. Model weights and inference kernels were not modified. No commit, push, or external submission was made.
