# Lumen 3.1 Pulsar REAM 구현 및 검증 보고서

## 결과와 검증 범위

실제 GGUF의 architecture는 gemma4이며, MoE는 30개 레이어, 레이어당 128 Experts, Top-K 8이다. 실제 실행에서는 128개를 64개로 병합했다. 요청의 116개를 64개로 줄이는 경우는 합성 단위 테스트로 검증했다. 원본 파일을 덮어쓰지 않았으며 commit/push는 하지 않았다.

실제 출력은 `.cache/ream-validation-20261009/lumen-ream-64.gguf`이다. 병합 프로세스 종료 후 별도 `verify` 프로세스에서 GGUF 재로드, Forward, 8-token 생성, finite logits를 확인했다. 전체 모델 재학습이나 PyTorch 학습 파이프라인은 추가하지 않았다.

핵심 병합 경로는 동작하지만 공식 구현의 비트 단위 재현이나 충분한 데이터에서의 품질 검증까지 완료한 상태는 아니다. V100 16GB에서 실제 Lumen GPU 병합 전체 실행 및 peak VRAM 측정은 완료하지 않았다.

## 변경 파일과 주요 함수

| 파일 | 역할 |
| --- | --- |
| `tools/aikar-prune/ream.h` | Expert, layout, group 자료구조와 REAM API |
| `tools/aikar-prune/ream.cpp` | `aikar_ream_pseudo_group`, `aikar_ream_validate_groups`, `aikar_ream_hungarian`, `aikar_ream_permute`, `aikar_ream_accumulate`, `aikar_ream_forward`, `aikar_ream_inspect`, `aikar_ream_read_expert`, `aikar_ream_encode`, `aikar_ream_export` |
| `tools/aikar-prune/main.cpp` | CLI, `route_callback`, `run_ream`, `ream_group_distances`, `ream_set_expert`, `run_verify` |
| `tools/aikar-prune/CMakeLists.txt` | REAM 소스 연결 |
| `tools/CMakeLists.txt` | GGUF/내부 모델 인터페이스 연결 |
| `tests/test-moe-prune.cpp` | 기존 테스트에 grouping/alignment/merge/GGUF 검증 추가 |
| `tests/CMakeLists.txt` | 기존 test-moe-prune에 REAM 소스 연결 |
| `docs/superpowers/plans/2026-10-09-ream.md` | 구현 설계와 reference 차이 기록 |

작업 트리에는 이 작업과 관계없는 기존 변경도 있다. 위 목록은 REAM 변경 범위이며 다른 변경을 되돌리지 않았다.

## REAP 재사용

- 기존 `aikar_dataset_load`, chat template/tokenization, calibration `evaluate`를 재사용한다.
- `common_moe_prune_selected_ids`, `record_selection`, `common_moe_prune_collect_output`, `reap_score`를 공유한다.
- Saliency는 기존 엔진의 normalized Top-K gate와 effective Expert 출력 norm을 사용하며, 선택된 토큰에서의 평균이다. 선택되지 않은 Expert의 원래 score는 0이다.
- `aikar_hard_prune_publish`와 GGUF API를 이용해 임시 파일을 게시한다. REAM의 가중치 병합은 compact export 전에 수행한다.
- 기존 REAP `analyze`, `profiles`, `hard` 실행 경로는 기본 `--method reap`으로 유지된다. REAM 전용 로딩 설정은 REAM에만 적용한다.

## 구현된 REAM 경로

1. Saliency 순으로 중심 Expert를 선정한다. 중심을 미리 할당한 뒤 중요한 중심부터 distance 순으로 미할당 Expert를 group-size 상한까지 배정한다. 모든 Expert의 중복/누락, 중심 존재, 정확한 목표 그룹 수를 검사한다.
2. Calibration의 all-token Expert 출력에 router softmax 기여도를 적용해 평균 출력 특징을 만든다. Router logit의 정규화 거리도 결합한다. 단순 cosine clustering으로 대체하지 않는다.
3. Hidden activation 및 gate/up/down weight 특징의 비용을 결합하고 정확한 Hungarian assignment를 계산한다. `logits`, `weights`, `logits+weights`를 선택할 수 있다. Greedy 대체 경로는 없다.
4. gate/up의 intermediate 행과 down의 intermediate 열에 같은 permutation을 적용한다. Fused gate_up도 처리한다.
5. FP32 누산으로 saliency 가중평균한다. 양자화/F16/BF16은 기존 GGML 변환 함수를 사용한다. 단독 그룹은 다시 encode하지 않아 원래 byte를 보존한다. Per-expert down scale은 effective down weight에 흡수하고 병합 중심의 scale을 1로 만든다.
6. 중심 Router 행을 보존하고 중심 ID를 정렬하여 연속 index로 export한다. Top-K, shared Expert, 비대상 텐서는 보존한다. GGUF에 `aikar.ream.mapping`, `aikar.ream.provenance`를 기록하고 sidecar report도 저장한다.
7. 한 레이어를 변경한 뒤 기존 engine context를 재생성하고 calibration을 다시 Forward한다. 이미 병합된 앞 레이어의 출력을 다음 레이어 분석에 사용한다.
8. 각 실행의 첫 레이어에서 동일 모델/동일 runtime 설정의 기존 REAP 수집 결과와 REAM capture 수집 결과를 비교한다. 선택 및 측정 횟수는 정확히 같아야 하고 saliency 합계의 상대 오차 허용치는 1e-5이다. 결과는 provenance에 기록한다.

## 공식 구현과 차이 및 제한

- 공식 구현의 weight PCA는 사용하지 않는다. 전체 weight feature의 비용을 계산하므로 차원 축소에 따른 근사 대신 비용과 시간이 증가한다.
- Expert별 독립 activation sample 대신 seed에 따른 공통 deterministic sample을 사용한다. Expert 간 같은 입력으로 neuron 비용을 비교한다.
- 출력 특징은 토큰 가중 평균이다. 공식 코드의 batch별 평균을 동일 비중으로 합치는 방식과 입력 batch 길이가 다를 때 차이가 날 수 있다.
- 기존 엔진 REAP의 gate 처리 의미를 유지한다. 공식 Python 코드와 saliency가 수치적으로 동일하다고 주장하지 않는다.
- zero score는 공식 코드의 양수 최소값 기반 대체를 반영하며 all-zero에는 명시적인 균등 fallback을 적용한다. 따라서 zero인 Expert를 병합에서 무조건 무시하지 않는다.
- 지원 아키텍처는 현재 Gemma4 GELU routed FFN이다. 다른 아키텍처, heterogeneous FFN 크기, MTP, 지원하지 않는 Expert/Router bias 및 shape는 오류로 중단한다.
- 지원 Expert 포맷은 F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K이다. 나머지는 조용히 변환하지 않고 거부한다. 모든 포맷의 대형 모델 통합 실행까지 수행한 것은 아니다.
- RAM workspace 제한은 전체 프로세스 RSS 제한이 아니다. 모델 로딩, context, mmap page residency 및 backend allocator 메모리가 추가된다. OOM 상황별 실측 검증은 남아 있다.

## 실행 명령

아래 명령은 실제 검증 환경의 경로를 사용한다. 출력 및 work 경로는 기존 파일/디렉터리와 겹치면 안 된다. 실사용에서는 짧은 smoke calibration 대신 충분한 대표 데이터를 지정해야 한다.

```bash
build/bin/aikar-prune hard --method ream \
  --model /mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar-Q4_0_XL.gguf \
  --ream-calibration /tmp/ream-validation/calibration.jsonl \
  --target-experts 64 --ream-group-size 16 --ream-merging logits+weights \
  --ream-activation-samples 32 --ream-chunk-size 32 \
  --ctx-size 64 --batch-size 32 --ubatch-size 32 --threads 8 \
  --n-gpu-layers 0 --ream-compute-device cpu \
  --output /tmp/lumen-ream-64-new.gguf --dry-run
```

실제 병합은 동일 명령에서 `--dry-run`을 제거한다. Sequential은 항상 켜져 있다. `--ream-work-dir /tmp/lumen-ream-work-new`로 전용 임시 경로를 지정할 수 있으며 정상 종료/예외에서는 정리한다. 강제 프로세스 종료 시 임시 파일이 남을 수 있다. `--ream-max-memory-mib`는 별도 작업 버퍼 추정치의 상한이다.

```bash
build/bin/aikar-prune verify \
  --model .cache/ream-validation-20261009/lumen-ream-64.gguf \
  --dataset /tmp/ream-validation/calibration.jsonl \
  --reference-logits .cache/ream-validation-20261009/original.logits \
  --ctx-size 64 --batch-size 32 --ubatch-size 32 --threads 4 \
  --n-gpu-layers 0 --output /tmp/lumen-ream-verify-new.json

build/bin/aikar-prune hard --method reap \
  --model /mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.1-Pulsar-Q4_0_XL.gguf \
  --profile .cache/ream-validation-20261009/reap-analysis/profile-050.json \
  --output /tmp/lumen-reap-64-new.gguf
```

REAP profile 경로는 실제 `profiles`/`analyze` 결과의 파일을 지정한다. REAP은 기존 profile 기반 방식이며 `--target-experts`가 기존 알고리즘을 바꾸지 않는다.

## 테스트 결과

`cmake --build build --target test-moe-prune aikar-prune -j 4` 성공. 최신 `ctest --test-dir build -R 'test-moe-prune|test-arg-parser' --output-on-failure`는 2/2 통과, 1.98초였다. `git diff --check`도 통과했다.

기존 test-moe-prune 안에서 8/4, 116/64, 116/116 grouping, 전체 mapping 중복/누락, Hungarian, neuron permutation 출력 불변성, saliency weighted merge, zero/NaN/Inf, F32/F16/BF16/Q4_0 GGUF 저장 및 재로드, router/scale shape, bias/MTP 거부를 검증한다. 새 테스트 파일은 추가하지 않았다.

합성 8/4 모델의 최신 전체 30-layer sequential 병합과 별도 프로세스 재로드가 성공했다. Finite logits와 8-token 생성도 확인했다. 합성 GPU backend 병합/재로드도 이전 검증에서 성공했다.

실제 Lumen의 최신 첫 레이어 Saliency 진단에서 REAP baseline과 REAM capture의 선택 횟수가 같고 최대 상대 sum 오차가 0이었다. 진단 실행은 이 비교를 확인한 뒤 30초 제한으로 중단했다. 최종 대형 GGUF는 진단 추가 전 완료한 55분 실행의 결과이며, 진단 추가 후 전체 대형 병합을 다시 수행하지 않았다. 기존 REAP cache는 로딩/runtime 설정이 다른 결과이므로 그 cache와의 수치 동일성은 입증하지 않았다.

전체 CTest는 성공하지 않았다. 기록된 실패는 test-chat abort, test-generate-models segfault, 이에 의존하는 recurrent/save-load 테스트 미실행, test-jinja-py 및 test-turbo-quant timeout이다. REAM 대상 테스트에서는 재현되지 않았지만 이 작업으로 인한 것인지 baseline 전체 실행으로 분리 입증하지 않았으므로 전체 회귀 검증 성공으로 표시하지 않는다.

## 실제 원본 / REAP / REAM 비교

같은 원본 모델과 같은 calibration을 사용했다. Calibration은 user `Hi`, assistant `Hello` 한 record로 19 tokens이며 assistant 평가 token은 단 1개다. 아래 PPL/KLD와 속도는 기능 smoke test 측정값이다. 품질 순위나 운영 성능 결론으로 해석할 수 없다.

| 측정 항목 | 원본 | REAP | REAM |
| --- | ---: | ---: | ---: |
| Experts / Top-K | 128 / 8 | 64 / 8 | 64 / 8 |
| GGUF bytes | 14,329,791,488 | 7,884,973,600 | 7,885,092,768 |
| Assistant PPL | 80,608,355,171,022.23 | 156,207,800.67 | 44,768,357.91 |
| 원본 대비 logit KLD | 기준 | 4.31038845 | 4.29596349 |
| Finite logits / 생성 | true / 8 tokens | true / 8 tokens | true / 8 tokens |
| 평가 prompt tokens/s | 16.14 | 7.57 | 7.34 |
| 생성 tokens/s | 7.60 | 4.09 | 3.98 |
| verify peak RSS (KiB) | 28,157,892 | 8,394,412 | 8,394,180 |
| 압축 peak RSS (KiB) | 해당 없음 | 15,560,712 | 15,077,712 |
| 압축 elapsed | 해당 없음 | 4:28.70 | 55:02.27 |
| Peak VRAM | 미측정 | 미측정 | 미측정 |

REAP의 calibration 분석 시간과 peak RSS는 별도로 3:05.48, 28,163,664 KiB이다. 표의 REAP 압축 시간은 hard export 단계만 나타낸다. REAM은 calibration과 sequential 병합을 포함하므로 두 시간의 직접적인 알고리즘 속도 비교에는 주의가 필요하다. Prompt 속도는 짧은 calibration Forward 측정으로 전용 prefill benchmark가 아니다. 원본 로딩과 압축 전용 로딩 설정이 달라 RSS/속도 비교에도 한계가 있다.

숫자의 근거는 `.cache/ream-validation-20261009/{original,reap,ream}-metrics.json`, 각 단계의 `.log`, 그리고 GGUF `.report.json`이다. REAM의 다른 실행에서 나온 PPL 값과 섞지 않고 최종 metrics 파일 값을 사용했다.

## 메모리와 성능 병목

- 전체 모델을 FP32로 복원하지 않는다. 모델은 CPU에 원래 형식으로 로드하고 필요한 Expert만 FP32로 읽는다. CPU Expert 가중치는 실제 모델 전체 로딩의 RAM을 필요로 한다.
- Expert 입력/router capture와 sampled hidden activation은 전용 디스크 임시 파일에 저장한다. 전체 Expert의 모든 토큰 출력은 RAM에 저장하지 않는다.
- 기본 32,768 samples 기준 작업 RAM 추정치는 820,477,952 bytes, layer별 sampled activation 디스크 상한은 11,811,160,064 bytes다. 실제 smoke는 32 samples 상한을 사용했다.
- Lumen intermediate 크기 704의 Hungarian cost matrix는 1,982,464 bytes다. 메모리보다 정확한 Hungarian 반복과 full-weight 거리 계산 시간이 주요 병목이다.
- Chunk별 Expert Forward, source Expert dequantize, calibration replay, 양자화 및 디스크 I/O도 비용이 발생한다. 레이어 종료 시 activation 파일을 해제한다.
- GPU 선택 시 GGML backend를 재사용하며 device workspace 추정과 free VRAM을 확인한다. 이것만으로 전체 context/backend 동시 메모리까지 16GB에 맞는다는 보장은 없다.

## 추가 검증이 필요한 사항

대표 calibration과 별도 held-out 평가 데이터를 사용한 재압축/품질 측정, 반복 seed 재현성 및 batch 길이 영향, 공식 Python 구현과의 중간 결과 비교, 모든 지원 dtype의 Forward 통합 실행, V100 16GB의 실제 peak VRAM/RAM 및 긴 context prefill/generation benchmark가 필요하다. MTP와 다른 아키텍처는 현재 미구현으로 명확하게 거부한다. 전체 CTest 실패의 baseline 비교 및 OOM/디스크 부족의 실제 장애 검증도 남아 있다.
