# REAM 실행 구성 및 병목 개선

## 실행 스크립트

`tools/aikar-prune/ream.sh`를 `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/ream.sh`에 적용했다. 원래 한 줄 스크립트는 `/tmp/pulsar-ream-before-speed.sh`에 보관했다. 기존 `prune.sh`의 모델, 전체 `../train_ko.jsonl`, context 4096, batch/ubatch 512, CPU/dataset threads 36, 현재 사용자 seed 42, GPU layers 99를 반영했다. Calibration 데이터를 줄이지 않았다. REAM은 앞쪽 routed weights를 8GiB 예산으로 GPU에 캐시하고 나머지를 CPU에 보관하며 Expert/거리 계산에 V100을 사용한다. `-ncmoe 1` 대신 `--cpu-moe`를 명시한 것은 기존 REAM 실행 내부에서도 강제하던 CPU routed weight 배치를 드러낸 것이다.

기존 ream.sh는 GPU Expert 연산 옵션 및 threads를 지정하지 않아 CPU Expert 연산, 기본 4 threads를 사용했다. 새 스크립트는 `--ream-compute-device gpu`, chunk 512를 지정한다. Target 64, group-size 16, logits+weights, sequential, activation sample 상한 32768은 유지한다. `--dry-run`과 추가 CLI 인자를 전달할 수 있다. Script는 cwd와 관계없이 절대 경로로 실행한다. 초기 검증에서는 모델 디렉터리 가용 공간이 3.1GB였으나 현재는 약 19GB다. 사용자가 변경한 기본 저장 위치 `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar`를 유지했다. 임시 입력/Router는 `/tmp/pulsar-ream-PID`, sampled activation은 모델 디렉터리의 `pulsar-ream-activ-PID`를 사용한다.

```bash
cd /mnt/openwebui/AIKAR/Lumen-3.1-Pulsar
./ream.sh --dry-run
./ream.sh
```

환경변수로 `REAM_OUTPUT`, `REAM_OUTPUT_DIR`, `REAM_LOG`, `REAM_WORK_DIR`, `REAM_THREADS`, `REAM_COMPUTE_DEVICE`, `REAM_GPU_LAYERS`, `REAM_CHUNK_SIZE`, `REAM_ACTIVATION_SAMPLES`, `REAM_CALIBRATION`, `PULSAR_PRUNE_BIN`, `PULSAR_MODEL_DIR` 등을 조정할 수 있다. CPU 실행은 `REAM_COMPUTE_DEVICE=cpu REAM_GPU_LAYERS=0 ./ream.sh`로 가능하다. 출력/임시 경로가 기존 파일과 겹치면 엔진이 거부한다. 전체 train_ko로 병합한 새 운영 모델을 이번 검증에서 생성한 것은 아니다.

## 확인한 병목과 수정

- `check(bool, const std::string&)`를 모든 float의 finite 검사에서 호출하면서 매번 긴 메시지의 문자열을 할당했다. Literal 메시지용 `const char *` overload를 추가해 성공 경로에서 문자열 할당을 제거했다. 검사는 그대로 수행한다.
- Expert Forward가 매 chunk마다 GGML context/graph를 생성하고 전체 gate/up/down을 복사했으며 GPU에서는 매번 weights를 다시 전송했다. `aikar_ream_forward_runner`가 Expert당 graph와 weight buffer를 한 번 준비하고 chunk마다 input만 갱신한다. 마지막 짧은 chunk는 zero padding하고 유효한 output만 반환한다. CPU도 graph와 weights를 재사용한다.
- Router softmax denominator를 각 Expert의 각 토큰마다 다시 계산했다. 레이어당 한 번 계산해 임시 `probabilities.f64`에 저장한다. 기존 double 정밀도를 유지하며 Expert 루프에서 재사용한다. 추가 임시 디스크는 token당 `expert_count * 8` bytes이며 layer 종료 후 삭제한다.
- 같은 중심의 hidden/weight feature 정규화를 그룹 내 각 Expert마다 반복했다. 중심 정규화를 그룹당 한 번 수행하고 source 특징과의 거리 계산에서 재사용한다. 정규화하지 않는 distance 경로의 불필요한 vector 복사도 제거했다.
- Layer report 및 stderr에 calibration, expert_features, grouping, merging, distance, hungarian별 초 단위 시간을 추가했다. Distance/Hungarian은 merging 안에 포함되므로 이 항목들을 모두 더하면 안 된다.
- Hungarian의 방문 상태는 bit proxy 대신 byte 배열로 저장한다. 정확한 assignment와 tie 순서는 유지한다. 704 x 704의 모든 값이 같은 비용에서 기존 0.532-0.540초, byte 배열은 0.494-0.523초로 측정되어 큰 배속으로 주장하지 않는다.

정확한 Hungarian, Saliency 가중평균, neuron permutation, 그룹 구성 순서, sequential replay를 유지한다. REAP의 실행 경로는 변경하지 않았다. GPU FP32 연산은 CPU와 작은 수치 차이가 있을 수 있어 병합 후 품질 검증은 계속 필요하다.

## 연산 단위 측정

Lumen과 같은 embedding 2816, intermediate 704, 64-token chunk, CPU 36 threads, 고정 seed의 랜덤 F32 weights로 측정했다. V100은 CUDA graph를 끈 상태로 측정했다. 각 수치는 개별 호출/짧은 반복 실측이며 전체 파이프라인 배속이 아니다. Final GPU 연산 측정 중 별도 Lumen 병합도 실행 중이었다.

| 항목 | 수정 전 CPU | 수정 전 GPU | 최종 CPU | 최종 GPU |
| --- | ---: | ---: | ---: | ---: |
| Expert Forward seconds | 0.123906 | 0.170240 | 0.030065 | 0.001233 |
| Weight feature distance seconds | 0.411411 | 0.423238 | 0.104796 | 0.105460 |

수정 전은 원래 일회성 graph 함수, 최종 Forward는 재사용 runner다. 최종 distance는 재사용할 중심 정규화를 미리 준비하고 source 정규화 비용은 측정에 포함했다. 같은 backend에서 준비된 특징을 사용하는 cost와 기존 정규화 cost의 전체 float vector가 정확히 일치함을 benchmark 및 단위 테스트에서 확인했다. GPU와 CPU의 cost checksum은 같은 출력 정밀도로 699978이며, Forward 첫 원소는 CPU -0.0137185, GPU -0.0137119였다.

이 랜덤 가중치의 Hungarian은 약 0.002초였지만 실제 Lumen의 비용 행렬에서는 더 오래 걸린다. 랜덤 diagonal 우세 비용의 결과를 실제 모델 정렬 시간으로 일반화하지 않는다.

## 검증과 한계

빌드 대상은 `test-moe-prune`, `aikar-prune`다. 기존 test-moe-prune에 반복 실행, tail chunk, 기존 Forward와 hidden/output 일치, empty/NaN 거부, 정규화 특징 재사용 cost 일치 검증을 추가했다. 별도 test 파일을 추가하지 않았다. 합성 8/4 모델의 전체 sequential CPU/GPU 병합 및 별도 프로세스 GGUF 재로드, finite logits, 8-token 생성도 검증했다.

실제 Lumen smoke에서 첫 레이어의 기존 REAP collector와 REAM capture selection/count/sum 비교 오차는 0이었다. 초기 최적화 빌드의 실제 Lumen GPU 병합 시간을 별도로 측정했다. 그 실행은 graph 재사용, 문자열 할당 제거, softmax cache 및 stage timing을 포함하며, 이후 추가된 중심 정규화 cache와 double probability cache, Hungarian byte 배열 변경 전 빌드다. 따라서 최종 코드의 전체 대형 모델 시간을 실측했다고 표시하지 않는다. 최신 코드 전체 train_ko 품질/시간은 아직 실측하지 않았다. 출력 공간이 추정 GGUF 크기보다 부족하면 모델 로딩/병합 전에 거부한다. Tensor 쓰기 오류에는 tensor명, offset, errno를 표시한다.

큰 데이터에서는 모든 Expert의 calibration Forward, 반복하는 전체 모델 sequential replay, sampled hidden matrix 거리 계산, 정확한 Hungarian, activation 임시 디스크 I/O가 계속 필요하다. 데이터 축소나 근사 정렬 없이 이 비용이 모두 사라지는 것은 아니다. 기본 sample 상한의 layer별 activation 임시 디스크 요구량은 약 11.8 GB다. workspace memory limit는 모델/context까지 포함하는 전체 RSS 한도가 아니다.


## 실제 30-layer 병합 시간과 저장 실패

같은 원본 Lumen, 같은 19-token smoke calibration, seed 42, CPU threads 8, context 64, batch/ubatch 32, samples/chunk 32로 이전 CPU 구현과 초기 최적화 GPU 빌드를 비교했다. 초기 GPU 빌드의 연산은 30개 레이어 모두 완료했다.

- 이전 layer elapsed 합계: 3259.274초.
- 최적화 calibration/features/grouping/merging 합계: 834.381초.
- 위 병합 연산 시간 기준 배속: 3.906배.
- 최적화 stage 총 시간: calibration 37.393초, Expert features 126.330초, grouping 0.734초, merging 669.925초. Merging 안의 distance는 317.651초, Hungarian은 190.629초다.
- 전체 프로세스 elapsed: 14:11.61. 마지막 `/tmp/lumen-ream-fast.gguf` 저장이 write 오류로 실패하여 exit code는 1이다. 이 시간을 성공한 전체 모델 생성 시간으로 비교하지 않는다. 임시 출력은 정리되었고 새 대형 GGUF의 재로드 검증은 수행할 수 없었다.
- Peak RSS: 15,288,592 KiB. 실행 중 GPU 메모리 관측은 791 MiB였으나 peak VRAM으로 측정한 값은 아니다.

`/tmp` 저장 실패 당시 기존 오류 메시지는 errno/실패 tensor를 기록하지 않아 정확한 원인을 입증하지 못했다. 실패 후 df에서 `/tmp`는 17GB 여유 공간을 보였다. 따라서 이 실패를 단순 디스크 부족으로 단정하지 않는다. 별도로 모델 디렉터리의 3.1GB 가용 공간은 기본 출력에 부족함을 확인해 스크립트의 기본 저장 경로를 수정했다.

최종 CPU/GPU 합성 모델에서는 별도 프로세스 reload, finite logits, 8-token 생성이 모두 성공했다. CPU 합성 모델의 그룹 매핑 및 PPL 223821.35948313778은 이전 구현과 일치했다. GPU PPL은 223815.04202570627이었다. 이 값들은 기능 검사이며 품질 결론이 아니다. 세부 로그는 `/tmp/lumen-ream-fast.log`, `/tmp/ream-speed-tests.log`, `/tmp/ream-validation/fast-final-{cpu,gpu}-metrics.json`에 있다.

최종 빌드는 성공했고 관련 CTest는 2/2 통과, 1.90초였다. `bash -n`, 배포 스크립트 내용 비교, `git diff --check`를 통과했다. 출력 공간 부족 경로를 실제로 호출해 available=3278004224, estimated output=7906608128로 병합 전에 거부함을 확인했다. Commit/push는 하지 않았다.

## 추가 최적화: 전체 Calibration 반복 비용

현재 사용자가 조정한 `ream.sh`의 seed 42 및 모델 디렉터리 출력 경로를 보존했다. 전체 train_ko는 실제 tokenizer/chat template 기준 317 records, 793610 tokens이며 context window별 평가 입력은 793230 tokens다. Dataset이나 activation sample 상한 32768을 줄이지 않았다.

새 Calibration 전용 경로는 Gemma4 graph를 분석 대상 레이어의 `l_out`에서 끝낸다. 뒤쪽 레이어, 최종 vocabulary projection, CPU PPL 계산은 실행하지 않는다. 이 graph는 REAM 내부 context flag로만 선택하며 기본값 -1인 일반 추론과 REAP은 기존 경로다. Flag 변경 시 KV를 비우고 graph 재사용을 막는다. `--ream-full-forward`로 전체 Forward 기준 경로를 선택할 수 있다. 별도 `verify`의 PPL 및 생성은 계속 전체 Forward로 실행한다.

기존 shared REAP saliency 교차 검증은 전체 dataset 추가 pass 대신 첫 batch에서 수행한다. 첫 레이어 REAM capture가 같은 토큰 수에 도달했을 때 expert별 selection/count/sum을 비교한다. 실제 병합용 saliency 및 sequential calibration은 모든 records를 그대로 사용한다.

`ream_expert_cache`는 앞쪽 레이어의 원래 양자화 Expert 가중치를 GPU에 유지한다. 기본 `--ream-expert-cache-mib 8192`, 0은 비활성화다. 모델 전체를 FP32로 복원하지 않는다. 현재 할당량, VRAM 가용량 및 최소 2GiB/예상 workspace headroom을 검사하며 들어가지 않는 레이어는 CPU에 남긴다. 병합 weight 갱신과 최종 export도 같은 cached tensor를 사용한다. CPU 원본 buffer는 유지하므로 RAM을 줄이는 최적화는 아니다.

Lumen/V100, CPU threads 36, batch/ubatch 128, 256 tokens, KV를 유지하는 두 batch의 별도 harness 실측:

| 대상 레이어 | 기존 전체 Forward (초) | 대상까지 Forward, CPU Experts (초) | GPU Expert cache 적용 (초) |
| --- | ---: | ---: | ---: |
| 0 | 4.789960 | 0.141673 | 미측정 |
| 15 | 4.794650 | 2.233920 | 0.259605 |
| 29 | 4.811880 | 4.588230 | 아래 추가 측정 참조 |

Layer 15 캐시 별도 실행의 전체 Forward 기준은 4.838590초, 16-layer cached weights는 6851403776 bytes다. 해당 calibration 구간은 전체 Forward 대비 18.64배, CPU Expert 부분 Forward 대비 8.60배다. 전체 병합 배속으로 일반화하지 않는다. Layer 0/15/29의 input, router logits 및 effective expert output은 기준 전체 Forward와 최대 절대 차이 0이었다. GPU cache layer 15도 차이 0이었다. 부분 경로에서 정상 경로로 되돌린 Forward 역시 검증했다.

합성 30-layer 전체 sequential 실행에서 CPU prefix와 CPU full reference의 saliency 배열 및 그룹 매핑은 모든 레이어에서 정확히 일치했다. Calibration stage 합계는 0.825206초 대 1.262505초였다. GPU cache 전체 30층 실행도 완료했으며 CPU/GPU 결과 모두 별도 프로세스 GGUF reload, finite logits, 8-token 생성에 성공했다. 이 합성 수치는 실제 Lumen 품질 지표가 아니다. GPU/CPU 결과는 연산 수치 차이 때문에 같다고 가정하지 않는다.

전체 Calibration 임시 input/router/probability는 약 10.15GB, sampled hidden은 약 11.81GB가 동시에 필요하다. 현재 /tmp 가용 공간 약 16GB에 전부 저장하면 실패하므로 `--ream-activation-dir`로 sampled hidden 저장을 분리했다. Script 기본은 input/router를 /tmp, sampled hidden을 모델 디렉터리의 고유 임시 폴더에 저장한다. 각 layer 후 파일을 지우며 정상/예외 종료 시 생성한 임시 폴더를 정리한다. `REAM_ACTIVATION_DIR`로 변경 가능하다. 모델 출력은 마지막 layer 이후 activation이 해제된 뒤 저장한다.

정확한 Hungarian 및 모든 Expert의 all-token feature 계산은 그대로 남는다. 최신 코드로 전체 79만 토큰 병합 완료 시간은 아직 실측하지 않았으므로 기존 3시간 예측을 새 완료 시간으로 환산하지 않는다.

임시 GPU cache harness가 처음에 fused gate/up을 처리하지 못해 중단되었고, root 소유 12GB core dump를 생성했다. 실제 REAM cache는 fused 텐서를 처리한다. Harness를 수정하고 재실행해 성공했다. `/var/lib/systemd/coredump/core.ream-cache-chec.1000.d014dea195684130b87ec1f504e8968b.3510144.1791549509000000.zst` 삭제는 sudo 인증이 필요해 수행하지 못했다. 빌드 공간 확보를 위해 재생성 가능한 `build/tests/test-models`를 `/tmp/aikar-test-models-space-recovery-20261009`로 옮겼다.

추가 마지막 레이어 cache 실측: full 4.833070초, prefix + 앞쪽 20-layer cache 1.988200초, cached bytes 8564254720, 최대 capture 차이 0. 해당 구간은 약 2.43배다. 마지막 레이어는 대부분의 앞쪽 레이어를 실행해야 하므로 첫/중간 레이어와 배속이 다르다.

최종 추가 검증: `cmake --build build --target aikar-prune test-moe-prune -j 4` 성공, 관련 CTest 2/2 통과(2.23초). 분리 activation 경로를 사용한 GPU synthetic 30-layer 병합은 완료했고 임시 두 폴더가 제거됨을 확인했다. 새 프로세스 verify는 finite_logits=true, generated_tokens=8이었다. 배포 script의 실제 Lumen dry-run, `bash -n`, 배포본 내용 일치, `git diff --check`도 통과했다. 최신 전체 Lumen 병합 완료/최신 Lumen GGUF 재로드까지 수행한 결과로 표시하지 않는다. 상세 로그는 `/tmp/ream-validation/prefix-v3-gpu.log`, `prefix-v3-reload.json`, `/tmp/ream-prefix-script-dry-run.log` 및 `/tmp/ream-{prefix,cache}-check-lumen-*.log`다.


## GGUF write 실패 원인 재현

기존 `/tmp/lumen-ream-fast.gguf` 저장 실패와 동일한 export 경로를 최신 진단 메시지로 재현했다. 기존 병합 결과 64-expert GGUF를 identity export하여 병합 계산을 반복하지 않았고, `/tmp` 저장은 5.68초 후 `blk.12.ffn_gate_up_exps.weight`, offset 3480082048에서 `errno=122 (Disk quota exceeded)`로 실패했다. 실패한 테스트 파일은 정리했다.

호스트 `/tmp`는 `tmpfs ... usrquota`다. 현재 UID 1000의 kernel quota 조회는 hard/soft limit 50933305 KiB = 52155704320 bytes, 사용량 48592175104 bytes, 잔여 3563529216 bytes였다. 따라서 `df`의 파일시스템 전체 여유 약 16GB와 달리 이 사용자는 약 3.56GB만 더 쓸 수 있다. 7.89GB GGUF 및 약 10.15GB calibration input/router 모두 이 한도를 넘는다. Inode quota는 0(한도 없음), 파일 크기 ulimit는 unlimited였다. `/mnt/openwebui`는 quota mount option이 없는 ext4이고 현재 전체 여유는 약 46GB다.

처음 실패에는 errno가 기록되지 않았으므로 과거 순간의 값을 소급 증명할 수는 없다. 다만 같은 저장 경로에서 quota 오류를 직접 재현했으며, quota로 인해 `df`가 충분해 보여도 저장이 실패하는 원인을 확인했다. 이후 발생한 root core dump는 다른 파일시스템의 별도 사건이고 현재 정리되어 있다.

기본 script의 input/router work 경로도 `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/pulsar-ream-work-PID`로 변경했다. Sampled activation은 같은 모델 디스크의 별도 고유 폴더를 계속 사용한다. 전체 임시 파일 약 22GB를 확보할 수 있으며 layer 종료 후 제거하고 마지막에 약 7.89GB 출력만 남긴다. `REAM_WORK_DIR`를 /tmp로 재지정하려면 사용자 quota 잔여까지 확인해야 한다. 현재 엔진의 `filesystem::space()` 사전 검사는 사용자 quota를 반영하지 않으므로 이 한계는 남는다. SSD I/O는 tmpfs와 성능이 다를 수 있지만 quota 초과로 실패하는 기본 구성을 피한다.

사용자의 최신 모델 디렉터리 `ream.log`도 확인했다. 이번 오류는 최종 GGUF export가 아니라 layer 0 Calibration 중 `record 115/317, processed 288842 tokens` 이후 `REAM calibration disk write failed`다. Input/router 저장은 token당 11776 bytes여서 마지막 진행 로그 시점에 이미 약 3.40GB를 저장했으며, /tmp 사용자 잔여 quota 약 3.56GB와 실패 시점이 일치한다. 전체 793230 tokens의 input/router는 9341076480 bytes이므로 현재 quota로 끝까지 저장할 수 없다. Probability cache를 더하면 총 10153344000 bytes다.

같은 identity GGUF export를 모델 디스크에서 재시험한 결과 exit 0, 10.764567초, output 7884996736 bytes로 성공했고 metadata reload 검사도 통과했다. 임시 테스트 GGUF와 report는 바로 정리했다. 이 검증은 저장 경로만 검사한 것이며 최신 전체 Calibration 병합 완료를 뜻하지 않는다. 로그: `/tmp/ream-write-diagnostic.log`, `/tmp/ream-write-diagnostic-disk.log`.


## 2026-10-10 all-token Expert bottleneck optimization

The dataset, seed 42, 32768 activation samples, all 128 Experts, Top-K 8, sequential replay and Hungarian assignment are retained. `ream.sh` now uses GPU F16 feature projections with FP32 accumulation, chunk 4096, a 6144 MiB feature input cache and a 12288 MiB original-quantization calibration Expert cache. The engine CLI still defaults to F32 features. Use `REAM_FEATURE_PRECISION=f32` for the F32 path; `--ream-full-expert-forward` retains the full all-token Down projection reference. Merged weights still use FP32 accumulation and the original output format.

For a linear Down projection, the router-weighted mean of outputs equals Down applied to the router-weighted hidden mean. The new path accumulates hidden means and executes Down once per Expert. GPU chunk reduction copies only weighted sums and sampled hidden activations to the CPU. Input activations are converted/uploaded once per layer within the VRAM budget. Calibration Expert weights are copied back, including merged values, before feature extraction to free GPU memory; they are cached again for the next layer. Router probabilities are stored as expert-major F64, reducing a full layer's repeated probability reads from 103970242560 bytes to 812267520 bytes. No probability values or tokens are removed.

Actual Lumen layer 0 captured inputs, 793230 tokens, 32768 samples, V100:

| Feature stage | Experts evaluated | Seconds |
| --- | ---: | ---: |
| Prior full all-token Down reference, chunk 512 | 8 | 126.372 |
| Hidden mean, F16 cached input, GPU summary, chunk 512 | 8 | 20.315 |
| Same, chunk 4096 | 8 | 19.2383 |
| Same plus expert-major probability reads | 8 | 16.3814 |
| Latest feature stage, all Experts | 128 | 77.1994 |

The 8-Expert comparison is 7.71x faster. Do not extrapolate it to whole-model runtime. Benchmarks use the same captured inputs/probabilities; probability creation and calibration are outside these feature-stage timings. All 128 feature-stage peak RSS was 591976 KiB; its GPU input cache was 4475322368 bytes. This excludes full model RAM and calibration VRAM. The numerical F16/GPU reduction change has relative L2 mean difference 5.0969e-5, maximum absolute difference 1.02445e-7 over the 8 actual Experts compared with the F32 full reference. It is not a perplexity or quality result. Reordering probability reads leaves means and all 8 sampled activation files bitwise identical at the same precision/chunk.

Bulk GPU output capture feeds the unchanged shared REAP saliency collector. Actual 65536-token layer 0 saliency benchmark: 13.38247 seconds before bulk copy, 11.53958 after; all 128 selection counts, REAP counts and REAP sums exactly equal. The Router Gram calculation measured only 3.47338 seconds for all 793230 tokens, so its algorithm was retained rather than introducing lower precision changes.

All 30 actual Lumen layers' original quantized Expert weights fit the 12 GiB cache: 12846382080 bytes, approximately 14680 MiB total observed GPU use during the layer 29 calibration check. That check processed 65536 actual tokens in 47.28727 seconds at batch/ubatch 512. It used unmerged source weights and /dev/null capture, so this is a calibration compute measurement, not a complete sequential merge or disk-I/O measurement.

CPU synthetic 30-layer sequential merge after the probability layout change has all group mappings and tensor bytes identical to the previous CPU linear implementation. Latest GPU synthetic sequential merge completed, and a separate process reloaded the output with finite logits and 8 generated tokens. Different calibration batch sizes, CPU/GPU placement and F16 chunk sizes can change numerical results; only identical configurations should be compared for bitwise equality.

The remaining structural cost is sequential prefix replay: later layers repeat the already merged preceding layers. Cached single-layer execution needs correct Gemma4 per-layer embeddings and shared KV handling and is not implemented. Latest full-dataset 30-layer merge wall time and full-size output reload have not been measured. The original REAP run was 5597.09 seconds (93.28 minutes). REAM being equally fast or faster overall is not established by these stage benchmarks.

Logs: `/tmp/ream-feature-stage-{full,summary,summary4096,transposed,all}.log`, `/tmp/ream-bulk-calibration-{fast,slow}.log`, `/tmp/ream-last-calibration.log`, `/tmp/ream-validation/final-transposed-reload.json`, `/tmp/ream-latest-build.log`. Default input/router and sampled activation temporary directories are both on the model's ext4 filesystem; `/tmp` still has the per-user quota limitation described above. Test-generated large feature files are removed after comparison; user calibration captures are retained.

Final checks for this optimization: build succeeded, related CTest 2/2 passed, script syntax and deployed copy match passed, and `git diff --check` passed. GPU synthetic comparison with identical n_gpu_layers=99, batch/ubatch=8 and feature chunk=8 has all 30 saliency arrays and group mappings equal before/after the probability layout change. A changed n_gpu_layers=0 or calibration batch configuration is not an equality test. Latest model was reloaded in a separate process with finite logits and 8 generated tokens.

Additional calibration batch experiment, layer 29, 65536 tokens: batch/ubatch 1024 took 37.63021 seconds versus 47.28727 at 512 (1.26x). Saliency arrays differ with the changed batch processing, and no quality evaluation was performed; the script retains the REAP-like calibration batch/ubatch 512. `REAM_BATCH_SIZE=1024 REAM_UBATCH_SIZE=1024` enables this explicit alternative. Test-generated large feature files/probability cache were cleaned; original captures were retained, and the model filesystem then had 34859917312 available bytes. No commit or push.

Final metadata/script-only rebuild also succeeded. Latest targeted CTest 2/2 passed. Deployed dry-run asserts feature chunk 4096, calibration batch 512, target 64 and Top-K 8, and preserves the normal execution log; dry-run logs now use ream-dry-run.log. Execution settings are included in provenance for reproducible comparisons.
