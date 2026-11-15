# Training and Docker audit, 2026-11-15

현재 실행은 중단되지 않았다. optimizer step과 저장 파일이 증가하지만, 동일한 73개 검증 window의 KL은 계속 악화된다. Docker 이미지에는 별도로 재현되는 optimizer 결함과 V100 아키텍처 누락이 있다. 현재 실행은 Docker 밖에서 실행되므로 이 이미지 결함이 현재 품질 악화의 직접 원인은 아니다.

## 현재 실행에서 확인한 사실

- Trainer PID: 3324216; controller PID: 3313753. 조사 중 optimizer step 102에서 103으로 진행했다.
- Run: `/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/logit-distillation/interleaved/runs/20261114T142347Z-3313753`.
- 실행 파일: 모델 디렉터리의 `recovery/llama-finetune-qo`. 저장소의 기본 `llama-finetune-qlion`과 다르다.
- 실제 로드된 라이브러리: `.cache/ga16-runtime/libggml-base.so.0.27.0`, 저장소 `build-joint/bin/libllama.so.0.7.0`, `build/bin/libggml-cuda.so.0.27.0` 및 CPU backend. `/proc/3324216/maps`에서 확인했다.
- 시작 가중치는 shared-KV step136 adapter지만 `--fresh-optimizer`로 momentum, residual, schedule, 데이터 위치를 초기화했다. 단순한 step136 연속 학습이 아니다.
- QLion2, LR 1e-6, minimum 1e-7, tau 0.001, context 16384, microbatch 768, accumulation 6. step100 state는 실제로 dataset window600, optimizer/scheduler step100을 기록한다.
- Q/output 8개 텐서만 학습한다. `blk.5.attn_k.weight`는 adapter에 보존되지만 학습 대상에서 빠져 있다.
- validation guard와 자동 rollback/LR backoff는 꺼져 있다. `best_validation` 상태 필드는 과거 실행에서 복사된 값이며 이번 실행의 실시간 best라고 볼 수 없다.

| Snapshot step | Full validation KL |
| --- | --- |
| 0 | 2.51044141 |
| 16 | 2.51165409 |
| 32 | 2.51766669 |
| 48 | 2.52710657 |
| 64 | 2.53986860 |
| 80 | 2.55544983 |

step80은 시작 대비 1.79285% 악화됐다. 비동기 검증이므로 step80 결과는 실제 학습 step92에서 보고됐다. 실시간 progress bar의 loss는 마지막 physical microbatch 값이다. 서로 다른 데이터의 이 값을 연속 비교해서 학습 개선 여부를 판단하면 안 된다. `loss=0, acc=nan`도 supervision 없는 microbatch에서 발생하며, 그것만으로 학습 발산을 뜻하지 않는다.

## 실제 갱신 병목

완료된 step96/100 adapter의 packed tensor bytes 전체를 비교했다. optimizer momentum은 257개 block마다 하나를 표본 추출했다. 아래 LR 배율은 step100 momentum으로 계산한 `RMS(m)/(RMS(m)+0.001)`의 표본 중앙값으로, 매 optimizer update의 실제 배율을 계측한 값은 아니다.

| Tensor | Changed bytes, step96 -> 100 | Estimated LR factor |
| --- | ---: | ---: |
| blk.11.attn_q.weight | 1 | 0.01007 |
| blk.17.attn_q.weight | 1 | 0.00782 |
| blk.23.attn_q.weight | 27 | 0.01094 |
| blk.29.attn_q.weight | 635 | 0.03458 |
| blk.11.attn_output.weight | 38776 | 0.11675 |
| blk.17.attn_output.weight | 79883 | 0.25106 |
| blk.23.attn_output.weight | 26560 | 0.40567 |
| blk.29.attn_output.weight | 39553 | 0.64428 |
| blk.5.attn_k.weight | 0 | frozen |

표본 momentum 원소의 약 96.8~98.6%는 0이 아니다. gradient 경로 전체가 끊긴 상황이라는 증거는 없다. Q 텐서는 작은 adaptive LR와 Q4 양자화 때문에 실제 가중치 변화가 매우 제한되고, output은 계속 변한다. F32 residual도 state에서 확인했다. packed byte 변경 수는 변경된 실수 가중치 개수가 아니다. scale byte 하나가 바뀌면 block 전체의 값이 달라질 수 있다.

이는 Q 경로의 학습 정체와 output 중심 갱신을 직접 뒷받침한다. 다만 검증 악화의 단독 원인을 확정하려면 동일한 입력과 시작 상태에서 대조 실험이 필요하다. 현재 증거만으로 LR 전체를 높이거나 낮추는 것이 해결책이라고 결론낼 수 없다. 또한 context 16384/microbatch768은 과거 KV를 유지하되 이전 microbatch의 backward graph를 detach하는 경로다. full-context gradient와 동일하지 않다 (`examples/qlora_training/README.md:614`).

## Docker 이미지 대비 차이

비교한 것은 로컬 `aikar-distill:recovery-jupyter`이다. RepoDigests에는 `ghcr.io/dfveloper/aikar-distill@sha256:56ded6caedc568166ad8eec29096093bfbb3cbf94713a202e98d5d65bef29aed`가 기록돼 있다. 생성 시각은 2026-11-04 23:41:53 +09:00이다. 원격 registry의 현재 tag가 같은 digest인지는 이번 조사에서 확인하지 않았다.

| 항목 | Docker 이미지 | 현재 로컬 실행/저장소 |
| --- | --- | --- |
| CUDA cubin | sm_80, sm_120a | 실제 GPU는 V100 sm_70; 로컬 전용 빌드 사용 |
| 실행 파일 | /opt/aikar-engine/bin/llama-finetune-qlion | 모델 디렉터리의 llama-finetune-qo |
| 라이브러리 | 이미지에 포함된 한 세트 | ga16-runtime + build-joint + build 혼합 |
| 기본 targets.sh | 블록 11/17/23/29의 Q/K/V/output 16개 이름 | Q/output 8개 이름 |
| Joint validation API | llama_opt_joint_validation_attach 심볼 없음 | 실제 활성화됨 |
| 무효 갱신 시 가중치 보존 | CPU 회귀 검사 실패 | 현재 및 실제 실행 라이브러리 조합 모두 통과 |
| 입력 경로 기본값 | /workspace | /mnt/openwebui/AIKAR/Lumen-3.8-Pulsar |

이미지의 GPU 대상은 현재 Runpod Dockerfile의 기본 `CUDA_ARCHITECTURES=80;120`과 일치하며 V100용 sm_70 코드가 없다. 저장소 `scripts/logit-distillation/v100.sh`는 별도로 sm_70을 빌드한다. 현재 Dockerfile도 기본값 그대로 재빌드하면 V100 대상은 추가되지 않는다.

이미지와 저장소의 `scripts/logit-distillation` 파일을 SHA-256으로 비교했다. 달라진 파일은 README.md, targets.sh, production.py, THROUGHPUT.md, benchmark.sh, hf_upload.py, test_hf_upload.py다. 저장소에만 ALGORITHM-OPTIMIZATION.md, V100.md, v100.sh가 있다. runpod/Dockerfile 자체와 train.sh는 이미지에 복사된 버전과 현재 저장소가 같다. Dockerfile이 같아도 COPY로 들어가는 C++ 소스와 빌드 결과는 같지 않다.

## 재현한 검사

기존 `examples/qlora_training/test-qat.cpp`의 `test_preserve_weight_scale` 함수를 독립 CPU harness로 실행했다. 새 optimizer 로직을 작성하지 않았다. 소스와 실행 파일은 `/tmp/aikar-training-audit-20261115/scale-check.cpp`, `scale-check`에 있다.

- 현재 build/bin 라이브러리: PASS.
- 현재 학습과 같은 LD_LIBRARY_PATH 조합: PASS.
- Docker 이미지 라이브러리: FAIL, `zero LR changed a valid noncanonical weight block`.
- 현재 라이브러리의 test-logit-distillation: KL reference, finite differences, masks, temperature/cache, compact encoding 및 paired-forward CPU 검사 통과.
- 실제 실행 라이브러리 조합의 test-logit-distillation: 같은 CPU 검사 통과.
- 이미지 자체의 test-logit-distillation: 이미지에 있던 KL reference, finite differences, masks, temperature/cache CPU 검사 통과.

이미지의 기본 distillation 검사가 통과해도 optimizer의 zero-update 결함을 검출하지 못한다. GPU는 진행 중인 학습이 점유하고 있어 이번 조사에서는 추가 GPU 실험을 실행하지 않았다. 이미지 결함의 GPU 재현 여부와 현재 품질 악화의 인과관계는 확정하지 않았다.

## 다음 검증의 우선순위

1. 현재 seed와 동일한 고정 training/validation subset에서 QLion2 tau만 바꾼 실험을 비교한다. Q/output별 실제 dequantized delta와 검증 KL을 함께 측정한다. 전체 LR 변경과 섞지 않는다.
2. 같은 조건에서 공유 K 학습 여부만 비교한다. 현재 frozen K와 잘못된 target 이름 추가를 구분한다.
3. Docker로 옮길 때는 최신 소스의 optimizer 수정을 포함하고 V100이면 sm_70을 명시해 재빌드한다. 위 가중치 보존 검사를 이미지 검증에 포함한다.

이번 조사에서는 기존 코드, 실행 설정, 학습 프로세스를 변경하지 않았다. 증거 요약은 `run-summary.json`, tensor별 값은 `checkpoint-analysis.json`에 보관했다.
