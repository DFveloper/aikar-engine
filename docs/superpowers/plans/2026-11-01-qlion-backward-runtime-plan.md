# 계획 B: QLion backward 및 런타임

설계: [QLion Gemma 4 학습 및 어댑터 설계](../specs/2026-11-01-qlion-gemma4-training-design.md). 기존 4~6단계와 후속 런타임 로딩을 다룬다. 계획 A의 어댑터 포맷 계약을 사용한다.

1. **활성 행과 backward**: 기존 6M Gemma 4 dense 모델과 별도로 약 6M 파라미터의 Gemma 4 MoE 모델을 생성한다. Q4 양자화 GGUF와 기존 테스트 데이터를 사용하고, 전문가 수와 토큰당 활성 전문가 수, shared FFN, 모델 구조 및 생성 명령을 기록한다. 이 dense 및 MoE fixture에서 `attn_k`, `attn_v`, router, shared FFN 및 전문가 gate/up/down gradient를 각각 수치 참조와 비교한다. 중복 라우팅, 패딩, 미선택 전문가, gradient accumulation, KV 경계와 flash attention을 검증한다. 기존 sparse 경로를 활용하되 실제 연산과 메모리가 활성 범위에 비례하는지 측정한다. tied embedding/output은 출력 gradient가 전체 어휘에 닿으므로 dense 경로를 사용한다.
2. **P620 성능 목표**: Quadro P620에서 실제 학습을 실행하고 기존 6M dense 모델과 MoE 모델을 비교한다. 총 파라미터 수를 가능한 한 맞추고 실제 활성 파라미터 수와 전문가 구성의 차이를 함께 기록한다. 데이터, 토크나이저, 컨텍스트, 유효 토큰 수, 배치/마이크로배치, 양자화 타입, 학습 대상, optimizer, 스레드, 오프로딩 및 backend 설정을 동일하게 맞추되 구조적으로 불가능한 차이는 명시한다. 워밍업과 초기 모델 로딩을 제외한 여러 optimizer step의 중앙값 및 변동 범위, 유효 학습 토큰/초, step 시간, 프로세스 peak RAM, GPU peak VRAM을 측정한다. MoE가 dense와 비슷하거나 더 나은 시간 및 메모리 효율을 보이는 것을 목표로 한다. 비교 기준과 허용 오차는 측정 전에 기록하며 미달 시 라우팅, 활성 전문가 gradient와 메모리 할당을 프로파일링해 개선하고 재측정한다. 하드웨어 용량이나 정확성 제약으로 목표를 달성하지 못하면 수치와 원인을 보고하고 통과로 표시하지 않는다. 시간이나 메모리 개선이 학습 대상 누락 또는 gradient 중단에서 나온 것이 아닌지 수치 검증으로 확인한다.
3. **Static graph**: 고정 KV 용량과 마스크 후보를 구현하고 동적 그래프와 여러 step의 loss, gradient, 업데이트를 비교한다. 길이 및 배치 변경, 마지막 배치, optimizer state, KV 초기화, 지원하지 않는 형상과 fallback을 시험한다. peak 메모리와 시간을 함께 측정한다.
4. **통합 검증**: `test-opt`, `test-qat`, 관련 backend 연산 테스트 및 소형 end-to-end 테스트를 실행한다. 실제 26B A4B에서 대상 선택, K/V 및 MoE gradient, RAM/VRAM, step 시간, 어댑터 저장/병합과 추론을 시험한다. V100 실기기 검증은 범위에서 제외하며 P620에서 확인하지 못한 조합은 미검증으로 기록한다.
5. **후속 런타임 로딩 및 부분 대체**: 병합 파일 없이 검증된 전체 텐서 대체를 로더에서 선택한다. 양자화 행 또는 전문가 단위 대체 포맷의 인덱스, 단위별 원본 해시와 데이터 순서를 정의하고 기존 예약 필드를 활용한다. 로더 overlay/조립, 메모리 매핑과 장치 버퍼 수명을 설계한다. 서로 겹치지 않는 다중 어댑터는 허용하고 tied alias를 포함해 범위가 겹치면 거부한다. v1 전체 텐서 어댑터와의 호환성 및 미지원 판독기의 안전한 거부를 시험한다.

## 2026-11-02 구현 및 검증

초기 결과는 [1차 보고서](../reports/2026-11-02-qlion2-improvements.md), CPU MoE 그래프 재사용, fused/compact 전문가 gradient 및 nonempty 26B 어댑터 검증은 [후속 보고서](../reports/2026-11-02-qlion2-followup.md)에 기록했다. P620 성능 목표와 일반 microbatch static graph는 미검증/미구현이며, 26B의 Q8 gradient 용량 상한은 여전히 전체 전문가 수에 도달한다.

## 2026-11-03 P620 후속 검증

[P620 검증 보고서](../reports/2026-11-03-qlion2-p620.md)에 CUDA 라우팅 재현성 수정, 대형 양자화 OUT_PROD의 F32 작업 버퍼 제한, 실제 26B 혼합 CPU/GPU 학습을 기록했다. P620의 메모리 비율 목표는 통과했지만 시간 비율 목표는 미달이다. 일반 microbatch static graph 및 나머지 런타임 범위는 미완료 상태를 유지한다.

## 2026-11-03 나머지 세 범위 구현

속도 개선을 제외한 일반 microbatch 고정 KV/mask 그래프 재사용, CPU/CUDA 실제 라우팅 기반 Q8 전문가 누적, core loader의 mmap/lazy overlay 및 repacking 통합을 구현했다. 검증과 지원 제한은 [세 범위 구현 보고서](../reports/2026-11-03-qlion2-three-scopes.md)에 기록한다. compact 모드는 추가 forward로 라우팅을 확인하며 static graph 재사용을 비활성화한다. P620 step 시간 목표는 이번 범위에 포함하지 않는다.
