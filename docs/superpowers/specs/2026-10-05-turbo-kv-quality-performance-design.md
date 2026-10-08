# Turbo KV Quality and Performance

## Goal

Reproduce the historical Pulsar S KV result, then improve Turbo3 or Turbo4 KV performance while preserving FP16-like next-token probability quality. The first implementation target is ordinary CUDA KV on the V100. Paged Turbo KV is a separate follow-up and is only accepted if the ordinary path meets the quality and speed gates.

## Historical baseline

The reference logits are generated with `llama-perplexity --save-all-logits`. Candidate runs use the same model, corpus, batch size, device and model placement with `--kl-divergence-base` and `--kl-divergence`.

The historical high-quality configuration is local F16 for K/V and global Turbo3 for K/V:

```text
-ctlk f16 -ctlv f16 -ctgk turbo3 -ctgv turbo3
```

`Same top p` means the fraction of positions where the candidate and reference assign the highest probability to the same token. It is not top-k set overlap. The historical comparison is teacher-forced by the fixed corpus, so it measures per-token distribution drift and does not claim free-running generation identity.

The available reproduction inputs are the Pulsar S model, `Lumen-Lumen3.txt`, and the saved logits file under `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar`.

## Evaluation matrix

Run the same corpus and saved reference logits for:

- FP16 baseline
- local F16/global Turbo3
- local F16/global Turbo4
- all Turbo3
- all Turbo4
- Q8_KV baseline

Record Mean KLD, 99.9% KLD, maximum KLD, mean/RMS/maximum/minimum delta probability, Same top p, prompt throughput, decode throughput, and peak VRAM. Keep the exact binary, model, arguments, CUDA device and commit in the report.

The primary quality gates are Mean KLD at or below `0.0001` and Same top p at or above `99.9%` for the historical local-F16/global-Turbo candidate. Any candidate that fails these gates remains experimental even if it is faster or smaller. The final thresholds may be tightened after the reproduction run if the historical result is recovered.

## Ordinary CUDA optimization

Keep the existing Turbo block layouts and CPU reference quantization unchanged. Optimize only the CUDA KV write and Flash Attention read paths first.

The first probe targets the vector Flash Attention path on SM70. It will reduce repeated Turbo centroid work by reusing block metadata and packed loads while preserving FP32 accumulation and the existing dequantization formula. Dispatch changes stay restricted to supported head dimensions and Turbo K/V pairs; all other shapes use the current path.

Each kernel change must pass the existing Turbo quantization and backend attention tests, then be measured against FP16, Q8_KV and the unmodified Turbo path at matched prompt lengths and decode spans. No unconditional switch is allowed without a measured improvement and no quality regression.

## Paged follow-up

The current paged cache accepts only matching F16 or Q8_KV K/V types. Paged Turbo support requires a packed page layout, CPU/GPU write quantization, CUDA and Vulkan page reads, attention dispatch, state save/load handling, and page-level numerical tests. This work is not included in the first ordinary-path patch. It begins only if ordinary Turbo meets the quality gate and its memory reduction is needed by the deployed Pulsar S context target.

## Validation

Build the CUDA targets with the repository's existing configuration. Run the existing Turbo quantization tests, backend attention tests for the affected dimensions, and the matched perplexity/logit comparison. Use the historical saved logits for quality and `llama-bench` for speed. Do not overwrite the existing deployed binary or model files; write reports to a new temporary result directory.
