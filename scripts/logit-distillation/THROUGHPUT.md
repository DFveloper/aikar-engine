# Mixed-data CPU throughput

These measurements used the earlier Korean multi-turn mixture. After the data audit, the default keeps only the first Korean user/assistant pair. The current 634-row mixture has 4950283 bytes, 4493 training windows and a 6609698365-byte teacher cache at context 512. Rates below are historical measurements, not a benchmark of the new corpus. See DATA_AUDIT.md and run benchmark.sh for a new measurement.

Date: 2026-11-04. CPU: Xeon E5-2696 v3, eight threads, no GPU. Context=batch=microbatch=512, parallel sequences=1, gradient accumulation=1, teacher top-512 plus tail at T=1. Training uses pure KL, QLion2, learning rate 1e-5 and static graph reuse.

## Dataset and targets

All 317 nonblank JSONL records from /mnt/openwebui/AIKAR/train_ko.jsonl are retained. Reservoir sampling without replacement selects 317 of 120342 records from /mnt/openwebui/AIKAR/train.jsonl, with seed 42. The deterministic shuffled output has 634 records and 6138194 bytes. This is a 1:1 conversation ratio, not a token ratio. Both extraction and training use the same mixed file, split and masks.

Training targets only attention Q/K/V/output weight names under blk.11, blk.17, blk.23 and blk.29. The runtime registers exactly 12 tensors: Q, K and output for each block. There are no separate V tensors in this model. These blocks reuse global KV from block 5, so their stored K projections are unused in the forward graph. Payload inspection confirms that the eight Q/output tensors change, while the four unused K tensors remain identical. No local layers or block 5 weights are selected. Model metadata marks global blocks as 5, 11, 17, 23, 29, using zero-based indices.

Frozen Q4_1 MoE weights still need to propagate activation gradients between selected attention blocks. The CPU MUL_MAT_ID_BACK support gate now permits Q4_1, using the existing block dequantization kernel. A direct dequantized matrix reference test passes at one and four threads; ten backend cases pass with multiple dimensions, duplicate expert IDs and strided inputs. Q4_1 FFN weights are not optimizer targets.

## Measured rates

| Stage | Input presentations | Supervised presentations | Loop seconds | Input tok/s | Supervised tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher extraction, four windows | 2048 | 1892 | 77.418 | 26.454 | 24.439 |
| Training, two windows / two updates | 1024 | 959 | 290.502 | 3.525 | 3.301 |

Training supervised tok/s is calculated from the cache's 959 training rows divided by the measured training time. The measured binary logs input throughput; the final instrumentation also prints the supervised count and rate. Extraction processing includes inference, top-k encoding and cache writes. Training includes forward/backward, optimizer, cache reads and graph preparation; epoch-end validation, initial model/dataset loading and final saving are excluded. Input presentations include packed prompts, padding and overlapping windows. This short run includes initial graph setup and is not a sustained full-dataset benchmark.

Raw extraction process wall time including loading/preparation is 111.73 seconds, corresponding to 18.33 input tok/s. Raw logs are under benchmark-20261104T103530Z: extract.log, extract.time, train-selected.log and train-selected.time. The earlier train.log records an abandoned all-layer attempt that exposed the missing Q4_1 support gate; its failure is not used as a throughput measurement.

Training exits successfully after two optimizer updates, validation KL 2.13099790 and saving a separate full-model checkpoint plus optimizer state. Whole-process training wall time, including loading, validation and serialization, is 409.87 seconds (2.498 input tok/s). Peak RSS is 21438512 KiB (20.45 GiB); compute workspace is 5634437120 bytes (5.25 GiB). The temporary benchmark checkpoint is removed after inspection; runtime logs and the small teacher cache remain.

## Production plan

The mixed corpus packs to 5562 training windows, with 2513367 supervised training presentations per epoch. Held-out validation uses 32 sampled windows and 14705 supervised positions. Cache size is 7799262856 bytes (7.26 GiB). The default three epochs produce 16686 optimizer updates and retain up to 17 checkpoint sets; the conservative storage budget is about 234.69 GiB after extraction. Full production training has not been started.

The model-directory copies of extract.sh, train.sh, production-extract.sh and production-train.sh all prepare/reuse the mixture by default. Override DATA to explicitly select a different existing file. The shared targets.sh sets the four-block attention selection. Run bash benchmark.sh to repeat extraction and training sequentially with those defaults. Small benchmark checkpoints are temporary; source models are never output destinations.
