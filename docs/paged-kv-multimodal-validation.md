# Paged KV multimodal extension

Status: completed for Pulsar/Pulsar S image inference on 2026-10-04. Paged KV remains optional and disabled by default.

The first delivery rejected multimodal configuration because paged attention always used causal visibility and `llama_set_causal_attn(false)` threw. Pulsar/Gemma4 image chunks require bidirectional attention in SWA layers; full layers stay causal. The vision projector already supplies embeddings through normal decode, so no separate persistent KV format is needed.

The extension reuses the existing page table, slots and query metadata. A causal setting on the paged operation changes its upper key limit to the populated KV span for image SWA. Absolute-position window masking and packed F16/Q8_KV storage remain in use. The existing MTMD scope switches back to causal mode for text. Non-causal image chunks must fit both logical and physical batch capacity; otherwise MTMD returns an actionable error before changing the attention mode.

Reference: [vLLM attention backends](https://docs.vllm.ai/en/latest/design/attention_backends/) share a KV cache between image-aware bidirectional attention and causal text/decode. The documented Hopper/Blackwell composite backends route image queries to Triton and causal queries to FlashAttention/FlashInfer. [Gemma4 model implementation](https://docs.vllm.ai/en/v0.30.0/api/vllm/model_executor/models/gemma4_mm/) retains image ranges and applies their non-causal visibility only on sliding layers, with window masking. This fork follows the masking model through its existing MTMD chunk boundary, using native CUDA/Vulkan kernels compatible with V100/P620 rather than importing vLLM's hardware-specific backends.

The installed projector is `gemma4v` with a vision encoder. Audio is not validated or claimed with this vision-only file. General architectures, shared-KV Gemma4 variants and position rewrites remain outside the supported scope. Image chunks are append-only, as required by the existing Gemma4 batch validator.

Evidence: `/tmp/aikar-paged-mm-artifacts`. Existing required text validation is recorded separately in `docs/paged-kv-validation.md`.

## Validation

The image fixture is `tools/mtmd/test-1.jpeg`, a newspaper whose headline is "MEN WALK ON MOON". Requests use the installed `chat_template.jinja`, greedy sampling and the OpenAI-compatible image API. The template adds about 1830 text tokens, so the tests use context 8192 and two slots. Decoder placement is one GPU with `-sm none -ngl all`; the projector stays on CPU with `--no-mmproj-offload`.

| Check | Result | Evidence |
| --- | --- | --- |
| Independent CPU operator, F16/Q8_KV | 34/34 | `cpu-green.log` |
| CUDA V100 operator, F16/Q8_KV | 34/34 | `op-CUDA0.log` |
| Vulkan V100 operator, F16/Q8_KV | 34/34 | `op-Vulkan1.log` |
| Vulkan P620 operator, F16/Q8_KV | 34/34 | `op-Vulkan0.log` |
| Pulsar and Pulsar S, F16/Q8_KV, CUDA/Vulkan, 300 image tokens | 8/8 correct headlines | `image-*-normal.json/log` |
| Image prefix reuse and two parallel image requests, Pulsar S Q8_KV | CUDA and Vulkan passed; 2136 cached tokens on repeat | same normal records |
| Pulsar S, F16/Q8_KV, CUDA/Vulkan, 1170 image tokens | 4/4 correct headlines | `image-*-large-all-gpu.json/log` |
| Batch-limit refusal and text recovery | 3/3 HTTP 400 and usable subsequent requests | `image-*-refuse*.json/log` |
| Fresh text state lifecycle, two models/types/backends | 8/8 | `text-matrix-result.log`, `final-state-*.log` |

Ten new attention cases cover bidirectional full/SWA bounds, future image keys, split decode, and a 33-query batch whose visible span exceeds the SWA window. The independent dense reference uses double precision. The remaining 24 cases cover existing causal attention and packed writes. All GPU cases execute on the selected backend.

Repeated-image reuse uses automatic LRU slot selection and a 1024 MiB host prompt cache. It asserts positive cached token counts, reduced prompt work and a correct repeated headline. A same-slot repeat after generated tokens can legitimately recompute because SWA pages required by the previous prompt have expired; no artificial cache hit is forced. Each positive image run also completes a subsequent text request and checks health.

## Image size and placement

Patch-grid rounding turns the tested 280-token image budget into 300 actual image tokens, and the 1120-token budget into 1170 tokens. An image chunk must fit both `-b` and `-ub`; use `-b 512 -ub 512` for the 300-token chunk and `-b 1536 -ub 1536` for the 1170-token chunk on this fixture. The server checks both capacities and returns HTTP 400 with the required size instead of splitting a bidirectional image chunk.

A large-image Pulsar S F16 test with `-ngl 24` returned only `thought` with both ordinary and paged KV. The same paged request with all decoder layers on CUDA correctly returned the headline. Subsequent plain text also produced an incorrect answer in the mixed-placement ordinary baseline. This evidence does not establish the root cause in the existing mixed-placement model path; it does establish that this quality failure is not specific to paged attention. Use the validated all-GPU placement for image serving on this host.

Example options, appended to the model/server command:

```sh
--kv-paged -ctk q8_kv -ctv q8_kv -sm none -dev CUDA0 -ngl all \
  --mmproj /mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/mmproj-BF16.gguf \
  --no-mmproj-offload -c 8192 -np 2 -b 1536 -ub 1536 \
  --image-min-tokens 1120 --image-max-tokens 1120
```

For Vulkan use `-dev Vulkan1`; for F16 use both `-ctk f16 -ctv f16`. These settings describe the test workload, not a production context or concurrency recommendation. The original text benchmarks predate this extension and are not multimodal performance measurements. Ordinary/paged arithmetic can produce different generations; image semantic checks do not imply identical logits or universal image accuracy.

The regressions first failed: the CPU reference rejected ten non-causal cases, the server rejected `--mmproj`, and an oversized image returned generic HTTP 500. The extension retains those red logs. Fixture problems and the ordinary mixed-placement comparison are recorded separately and are not counted as passing product cells.

Combined CUDA/Vulkan builds of server, CLI, completion and existing operator/state targets passed (`build-green.log`, `build-http-green.log`, `build-final.log`). Final read-only review found no concrete blockers. User changes outside task files were unchanged; main-workspace/worktree task sources matched. No commit, push or external submission was made.

The original nonpaged Q8_KV server was restored with identical argv/cwd and verified healthy at port 11435. MuSR remains stopped as authorized. Paged mode was not enabled automatically on the production process.
