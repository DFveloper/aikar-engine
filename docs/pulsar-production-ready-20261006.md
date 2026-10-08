# Pulsar production-ready binary

The production candidate is assembled in `build-pulsar-production-ready/bin`.
It includes `llama-server` and the matching shared libraries from the clean
Release build. The existing service remains unchanged; switch with
`LUMEN_SERVER_BIN=/home/user/aikar-engine/build-pulsar-production-ready/bin/llama-server`.

## Runtime profile

- 12 slots, logical context 262144
- 4096 physical KV blocks x 32 tokens
- Q8_KV K/V cache
- CUDA fused paged attention enabled
- backend sampling and 64-token prefill chunks enabled
- multimodal projector kept on CPU
- experimental block sizes 64 and 128 are supported by the parser, CUDA
  operator, state format, and independent state tests

## Verification

- Clean Release build completed for `llama-server`, `test-backend-ops`,
  `test-arg-parser`, and `test-save-load-state`.
- CUDA Paged Attention tests passed 80/80 with fused mode off and on while
  the production server was stopped.
- Save/load state tests passed for block sizes 32, 64, and 128.
- End-to-end candidate validation passed health/config checks, greedy output
  checks at 80 and 900 prompt repeats, slot save/restore, four-way generation,
  OpenAI chat, and CPU-mmproj image input.
- Observed candidate throughput: 51.5-51.8 generated tokens/s serial and
  127.6 aggregate tokens/s across four concurrent requests.

Paged KV does not support speculative decoding, training, or partial context
checkpoints. Full prompt state save/restore is supported. The final CUDA test
rerun must be performed with the production process stopped; running it while
the service owns the V100 can fail from resource contention.
