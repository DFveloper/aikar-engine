# Turbo KV quality baseline

This preliminary report uses the unpruned model, not Pulsar S. See `pulsar-s-kv-20261005.md` for the corrected model, new complete references and final results.

The historical command uses `llama-perplexity --kl-divergence` with the saved
logits file and separate local/global cache types. The current saved logits do
not match the checked-out model/runtime exactly: the all-F16 replay reports
mean KLD 0.062875 and 92.907% same-top-p. Therefore the historical 100% result
cannot be claimed as reproduced from this artifact.

Commands used the model at `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar-Q4_0_XL.gguf`, corpus at `/mnt/openwebui/AIKAR/Lumen-Lumen3.txt`, saved logits at `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar-logits`, `CUDA0`, `-b 512`, and `-ncmoe 5`.

| configuration | mean KLD | 99.9% KLD | max KLD | RMS delta-p | same top-p |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16/F16 local+global | 0.062875 | 2.644586 | 7.164707 | 7.192% | 92.907% |
| F16 local, Turbo3 global | 0.146928 | 4.155591 | 8.089011 | 10.799% | 88.085% |
| F16 local, Turbo4 global | 0.098829 | 3.254982 | 7.999873 | 9.152% | 90.819% |
| Turbo3 local+global | 0.461960 | 8.365120 | 12.923449 | 19.453% | 79.631% |
| Turbo4 local+global | 0.214130 | 5.001624 | 8.244025 | 13.813% | 85.306% |
| Q8_KV local+global | 0.077602 | 2.425717 | 7.949293 | 7.838% | 91.984% |

The native CUDA change in `fattn-common.cuh` keeps Turbo block layout and
quantization unchanged. It loads each block norm once per four-element dot
group and indexes the constant centroid table directly instead of calling the
per-element dequantizer repeatedly.

## Native CUDA smoke benchmark

`llama-bench` used two repetitions, `-p 64 -n 32 -b 512 -ub 512`, `CUDA0`,
`-ncmoe 5`, and the rebuilt `build/bin/llama-bench`. Results are tok/s:

| cache | prompt | decode |
| --- | ---: | ---: |
| F16 | 118.42 | 59.72 |
| Q8_KV | 118.50 | 56.15 |
| Turbo3 | 115.46 | 58.29 |
| Turbo4 | 114.20 | 50.38 |

These preliminary rows do not establish a matched cache-policy or before/after comparison. The Turbo3 row used mixed local/global cache types while other rows used uniform types. Use the corrected Pulsar S report for the measured performance targets.
