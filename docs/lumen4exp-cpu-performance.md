# Lumen4Exp CPU measurements, 2026-11-04

This report records the earlier LES tuning. The current default KV format is documented in [Lumen4Exp](lumen4exp.md#default-fp4-kv-cache).

CPU Flash Attention works with the CPU-resident KV cache. The existing build already enables AVX2, F16C, FMA, OpenMP and CPU weight repacking. Lumen4Exp stores global KV in NVFP4 and decodes it through F32 to F16 for Flash Attention. These LES measurements predate the direct NVFP4 attention optimization; see [the attention report](lumen4exp-nvfp4-cpu-attention.md) for the subsequent change. Cache precision is unchanged.

The existing Gemma4 LES integration now accepts Lumen4Exp, which uses the same router and expert graph. CPU residency, single-sequence, budget, LoRA and training restrictions are retained. LES changes expert preparation and traversal order, not router selections or output slots. It remains opt-in because all measured LES configurations were slower.

## Environment and method

- Intel Xeon E5-2696 v3, 18 physical cores, 36 logical CPUs, one NUMA node.
- Model: `/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/Lumen-3.8-Pulsar-base.gguf`, 14,378,479,104 bytes, Lumen4Exp 26B.A4B Q4_0.
- All comparisons are sequential and use the same rebuilt binary, CPU repacking, `-lm none`, and `-fa on`.
- Server LES comparisons allocate `-c 1000000 -np 1 -kvu`, use `-tb 18`, disable prompt-cache and context-checkpoint storage, and perform a warmup request.
- Each server configuration processes three fixed prompts, each with 128 input tokens and 96 greedy output tokens. Prompt caching is disabled and EOS is ignored. English explanation, Python and Korean prompts are used. The table reports the arithmetic mean of the three server generation rates.
- The 1M KV pool is allocated, but these are short-context workloads. Full 1M-token inference throughput and broad model quality were not evaluated.

## LES results

| Decode threads | LES budget | Generation, tok/s | Change from matching baseline |
| --- | --- | ---: | ---: |
| 18 | Off | 10.644 | baseline |
| 18 | 4 MiB | 10.370 | -2.57% |
| 18 | 16 MiB | 10.249 | -3.71% |
| 18 | 32 MiB | 10.165 | -4.50% |
| 18 | Off, final repeat | 10.497 | -1.38% |
| 12 | Off | 10.684 | baseline |
| 12 | 4 MiB | 10.150 | -4.99% |
| 12 | 16 MiB | 10.240 | -4.16% |
| 12 | 32 MiB | 10.135 | -5.13% |

Each LES configuration produced exactly the same 288 generated token IDs and text as its matching baseline. The final baseline repeat also matched. This establishes output equality for these requests, not byte-identical logits or a general quality evaluation.

With 18 threads and 16 MiB, LES reported 70,080 required experts, 42,905 prediction matches, 14,619 prepared hits and 75,692,763,136 bytes of preparation reads, including the warmup request. This is a 61.2% prediction match rate and a 20.9% prepared-hit rate. Prepared hits only indicate completed software reads; they do not prove hardware-cache residency. Extra reads, speculative misses and worker overhead are consistent with the measured slowdown.

## Thread sweep and serving settings

A separate `llama-bench` sweep used `-p 128 -n 64 -r 3 -fa on -lm none --repack 1`, with 6, 9, 12, 18, 24 and 36 threads. Generation rates were 10.20, 10.89, 11.11, 11.06, 9.26 and 3.33 tok/s respectively. Prompt processing at 18 threads reached 56.38 tok/s versus 50.31 at 12 threads; 24 threads gave 56.88 tok/s but slower generation. Retain 18 threads for both phases rather than enabling SMT for all 36 logical CPUs.

The model's `start.sh` explicitly sets `-lm none -fa on -t 18 -tb 18 --repack`. It retains the 1M context and automatic four-slot unified KV pool. LES is not enabled. To experiment manually, add `-np 1 --cpu-les --cpu-les-prefetch-mib 4`; this changes the server to one slot.

The final launcher configuration passed a real eight-token completion with Flash Attention enabled, four slots and an RSS of 18,198,352 KiB (17.36 GiB). Its compute buffer reserves 8,860.29 MiB; longer active contexts can fault additional pages into RAM. Earlier startup measurements with mmap plus repacking reached 29.56 GiB RSS. File-cache accounting is separate from process RSS, and the initial mmap-backed pages can be reclaimed by the OS.

## Validation and artifacts

`llama-server`, `llama-bench` and `test-cpu-les` built successfully. `test-cpu-les` and `test-arg-parser` passed. The argument parser's HTTP checks required execution outside the filesystem/network sandbox. The final launcher passed `bash -n` and a real completion. No production server was restarted; the new settings apply on its next launch.

Raw artifacts for this session are `/tmp/lumen-cpu-threads.{json,log}`, `/tmp/lumen-les-results/`, `/tmp/lumen-les-benchmark.py`, `/tmp/lumen-final-smoke.{log,json,smaps}`, `/tmp/lumen-les-unit-test.log`, and `/tmp/lumen-les-arg-parser-network.log`. The previous launcher is backed up at `/tmp/lumen-start-before-cpu-tuning.sh`. The benchmark harness records commands, timings, generated tokens and memory snapshots, and stops each diagnostic server after measurement.
