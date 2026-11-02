# 계획 A: QLion 대상 선택과 어댑터

설계: [QLion Gemma 4 학습 및 어댑터 설계](../specs/2026-11-01-qlion-gemma4-training-design.md). 기존 1~3단계에 해당한다. 각 단계는 이전 단계의 검증 후 진행한다. QLion2 QLoRA와 FP32 LoRA 어댑터 resume은 범위 밖이다.

1. **기준선**: 6M dense 모델 `artifacts/gemma4-tiny-6l-q4.gguf`와 `synthetic_chat.jsonl`로 QLion 학습과 전체 모델 저장을 재현한다. 모델 경로의 루트는 `/mnt/openwebui/AIKAR/QLion2-test`다. 원본 텐서 타입, 형상, 바이트, tied 관계, 출력 및 학습 전후 변경량을 기록한다. `test-opt`, `test-qat` 결과와 장치 구성을 기록한다.
2. **대상 선택**: 기존 `--qat-targets`와 QLion의 `--freeze-layers`를 교집합으로 적용한다. 선택된 텐서와 상태 메모리를 실제 등록 기준으로 보고한다. tied embedding/output은 기본 선택에서 제외하고 명시적 선택 시 dense gradient와 예상 크기를 경고한다. 0건, 잘못된 범위, tied alias 및 비선택 텐서 불변성을 시험한다.
3. **어댑터와 export**: 학습 시작 시 원본 파일과 대체 대상 텐서 해시를 확보하고, 종료 시 원본 바이트와 비교해 실제 변경된 텐서만 원래 양자화 타입으로 저장한다. v1은 전체 텐서 대체이며 선택적 행/전문가 인덱스 필드를 예약하고 미지원 부분 대체는 거부한다. 저장 크기 및 tied 중복을 검사한다. `llama-export-qlion`은 원본과 대상 텐서 해시를 검증하고 서로 겹치지 않는 어댑터를 병합한다. 전체 해시가 다를 때는 명시적 호환성 검사로 모든 대상 텐서 해시와 형상 등이 일치하는 경우만 허용하고 경고한다.

**완료 조건**: 같은 학습 실행에서 저장한 전체 모델과 병합 모델의 텐서 바이트가 일치하고 추론이 성공한다. 변경 없는 텐서는 어댑터에 없어야 한다. 파일 해시 캐시의 오래된 항목, 메타데이터만 변경된 기본 모델, 대상 바이트 변경, 겹치는 어댑터, tied alias, 잘린 GGUF, 중복 이름, 범위 및 크기 오류, 저장 실패를 시험한다. 전체 파일 해시 일치 여부와 무관하게 대상 텐서의 원본 해시를 검증한다. optimizer state와 재개 동작은 기존 체크포인트에 그대로 둔다.

## 2026-11-02 구현 및 검증

초기 결과는 [1차 보고서](../reports/2026-11-02-qlion2-improvements.md), CPU MoE 그래프 재사용, fused/compact 전문가 gradient 및 nonempty 26B 어댑터 검증은 [후속 보고서](../reports/2026-11-02-qlion2-followup.md)에 기록했다. P620 성능 목표와 일반 microbatch static graph는 미검증/미구현이며, 26B의 Q8 gradient 용량 상한은 여전히 전체 전문가 수에 도달한다.

## 2026-11-03 P620 후속 검증

[P620 검증 보고서](../reports/2026-11-03-qlion2-p620.md)에 CUDA 라우팅 재현성 수정, 대형 양자화 OUT_PROD의 F32 작업 버퍼 제한, 실제 26B 혼합 CPU/GPU 학습을 기록했다. P620의 메모리 비율 목표는 통과했지만 시간 비율 목표는 미달이다. 일반 microbatch static graph 및 나머지 런타임 범위는 미완료 상태를 유지한다.

## 2026-11-03 나머지 세 범위 구현

속도 개선을 제외한 일반 microbatch 고정 KV/mask 그래프 재사용, CPU/CUDA 실제 라우팅 기반 Q8 전문가 누적, core loader의 mmap/lazy overlay 및 repacking 통합을 구현했다. 검증과 지원 제한은 [세 범위 구현 보고서](../reports/2026-11-03-qlion2-three-scopes.md)에 기록한다. compact 모드는 추가 forward로 라우팅을 확인하며 static graph 재사용을 비활성화한다. P620 step 시간 목표는 이번 범위에 포함하지 않는다.
