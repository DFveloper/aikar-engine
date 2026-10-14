# Pulsar decode overlap and expert cache follow-up

Date: 2026-10-11. Status: measured proposal; execution contracts are not implemented by the Paged KV codec patch.

## Measured cache tradeoff

The current V100 deployment holds all 116 routed experts on GPU. Its expert-weight transfer miss rate is zero. The REAM cache in `tools/aikar-prune/main.cpp` is a calibration layer cache, not a serving expert cache.

One actual Pulsar trace contains 30 layers, 44 prefill tokens and 256 subsequent tokens from a code/Korean explanation prompt. This is one workload, not a traffic distribution. Top-8 tensor views were read using their byte strides. The initial diagnostic read used the logical element count as a physical span, corrupted its temporary host buffer, and was corrected before these results.

For each layer, preload experts ranked by observed prefill frequency, break ties by expert ID, then replay decode in order. Pin all eight selected experts while replacing other entries. There is no future lookahead or speculative prefetch benefit. Packed expert payload is 3345408 bytes per expert per layer. Scale metadata, staging buffers, maps and alignment are additional costs.

| Resident experts per layer | Gross saving MiB | LRU decode hit rate | Misses per model token | Measured H2D lower bound ms/token |
| --- | ---: | ---: | ---: | ---: |
| 116 | 0 | 100% | 0 | 0 |
| 112 | 382.85 | 99.826% | 0.418 | 0.450 |
| 108 | 765.70 | 99.650% | 0.840 | 0.904 |
| 96 | 1914.26 | 99.071% | 2.230 | 2.400 |
| 80 | 3445.66 | 97.809% | 5.258 | 5.658 |
| 64 | 4977.07 | 95.054% | 11.871 | 12.775 |

Pinned host-to-device measurements on this machine gave approximately 3.1 GB/s and 1.076 ms for one payload, 8.607 ms for eight, and 121.508 ms for a full 116-expert layer. Measurements used 20 transfers after warmup and CUDA event timing. They are transfer lower bounds, not predicted serving latency. Request diversity and concurrent top-8 selections can increase misses substantially. Frequency-aware eviction produced similar or slightly worse results in this trace.

Raw evidence: `/tmp/pulsar-paged-262k/artifacts/routes.jsonl`, `cache-replay.json`, `h2d.jsonl`. Replay and capture source are retained next to the artifacts.

## Proposed bounded expert residency

Start with 112 resident experts per layer as an optional VRAM mode, and retain all-GPU residency as the default. A speed claim requires measured exposed miss time, not only the hit rate. Four fewer experts could supply useful 262K headroom, but the net saving must subtract pinned staging, CUDA slot alignment, routing maps and graph overhead.

1. Use GGUF expert slices directly as the authoritative host weights. Store GPU slots for packed gate/up and down weights, with corresponding scale metadata. Loading a context must not first allocate a full GPU expert tensor and then allocate the slot cache. The current tensor-buffer ownership model therefore needs an explicit loader change.
2. Keep logical expert IDs separate from physical slot IDs. Reuse the existing grouped MMID implementation and contiguous per-layer slot tensors. Remap selected IDs once per batch; do not launch one host-controlled matrix kernel per expert. Preserve fused gate/up layout and expert scales.
3. Build a residency snapshot only after routing is available. Pin every selected expert until both gate/up and down consumers finish. Queue misses on a transfer stream into unpinned slots and record a completion event. Compute waits on required miss events; independent shared FFN work may run while copies execute if its input and scratch lifetimes permit it.
4. When a batch selects more experts than the residency budget, use a correct explicit fallback. Start with the all-GPU mode selected at context creation or a bounded chunked MMID execution designed with the allocator. Do not evict in-flight experts or silently drop selections. Concurrent request unions must be tested, especially at 12 slots.
5. Separate stable slot tensors from mutable ID mappings. Decode graphs can reference stable storage only after residency is prepared and events are satisfied. A residency miss must have a defined capture fallback and graph invalidation path. Never mutate mappings belonging to an outstanding graph.
6. On cancellation, stop admitting new work, wait for outstanding transfer/compute events, release pins, and reclaim staging. Model teardown owns all slot and host buffers. KV state persistence does not serialize expert slots because weights are immutable.

Acceptance requires diverse traces at concurrency 1/4/12, exact output parity against all-GPU weights, no use-after-eviction under cancellation, actual net bytes saved, and matched PP/TG/TTFT measurements. The capture above does not establish any of these broader gates.

## Decode overlap scope

Official vLLM DBO overlaps two microbatches around DP/EP all-to-all communication. This deployment has one V100 and no expert-parallel all-to-all. Its synchronization and weight ownership differ from vLLM.

The current server synchronizes logits while sampling after `llama_decode`, then accepts the sampled token and processes its text before scheduling another decode. Same-sequence token dependence remains. A useful first unit is to overlap independent requests' preparation and output processing with a GPU batch, after measuring the exposed host gaps separately from CUDA launch time.

An isolated Q8 loopback-server Nsight capture completed one warmup and one64-token streaming request at concurrency1, with CUDA graphs disabled. Within the two long decode bursts,124completed logit-readback-to-next-kernel intervals measured median0.993ms, mean1.020ms, range0.845-1.452ms. The interval includes CPU sampling/output/scheduling, metadata work and any transfer before the next kernel; profiling overhead is present. This is roughly5% of the observed20ms single-token step and does not establish that overlap can hide the entire interval or produce a large TG gain. Both active decode bursts uploaded two98304-byte block tables per step; the larger page-payload readbacks in the complete trace occur outside those decode bursts. Capture: `artifacts/server-decode-profile.nsys-rep`, `server-profile-summary.json` under `/tmp/pulsar-paged-262k/`.

Use immutable batch metadata and two explicitly owned input/output slots. GPU events mark output readiness and input reuse. Sampling owns sequence RNG and stop state; it must finish before that sequence contributes its next token. Cancellation invalidates future admission but drains outstanding work before releasing slots. Prefix/state mutations and KV allocator commits remain ordered with the compute that consumes them. Keep one serialized fallback for unsupported batch or capture cases.

Pulsar also has independent shared and routed FFN branches that join by addition. Existing CUDA stream/event support can be reused only after proving scheduler allocator lifetime safety for both branches. The current one-row QKV fan-out optimizer does not automatically cover this join. A separate branch-overlap prototype should retain numerical parity, measure stream contention on SM70, and account for the extra live scratch memory.

## References

- vLLM DBO design: https://docs.vllm.ai/en/latest/design/dbo/
- Expert cache proposal: https://github.com/vllm-project/vllm/issues/38256 . Treat this as an open proposal, not a stable vLLM implementation or a verified speedup for this machine.
- Approved codec/capacity plan: `docs/superpowers/plans/2026-10-11-pulsar-paged-kv-262k.md`.
