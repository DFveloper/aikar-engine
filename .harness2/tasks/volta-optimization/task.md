# NVIDIA Volta 모델 추론 및 학습 최적화

이 저장소의 CUDA 백엔드에서 NVIDIA Volta GPU를 대상으로 모델 추론과 학습을 최적화하라. V100/SM70에서 실제 실행 시간과 VRAM 사용량을 개선하되, 계산 정확성, 모델 품질, 학습 의미와 다른 GPU의 기존 동작을 유지해야 한다. 프로파일링으로 병목을 확인하고 기존 CUDA 경로 안에서 작고 검토 가능한 개선을 수행하라.

## 작업 범위

- 추론: prefill/decode, 양자화 dense/MoE GEMM 및 GEMV, attention 및 paged KV cache, dequantization, kernel launch와 CPU-GPU 동기화 비용을 조사한다.
- 학습: 기존 QLoRA SFT와 QLion/QAT 등 실제 지원되는 경로에서 forward, backward, routed expert gradient, gradient accumulation, optimizer update, activation 및 workspace 비용을 조사한다.
- `ggml/src/ggml-cuda/`, `ggml/src/ggml-opt.cpp`, `src/`, `examples/qlora_training/`를 중심으로 기존 kernel, dispatch, backend buffer와 graph 실행 구조를 재사용한다.
- 로컬 Lumen/Pulsar 및 Pulsar S 모델이 있으면 우선 사용하되, 실행 가능한 모델과 학습 모드를 현재 환경에서 확인한다. 새 학습 알고리즘이나 별도 추론 엔진 구현은 범위에 포함하지 않는다.

## 진행 방법

1. 저장소 지침과 `skills/code-review/SKILL.md`를 읽는다. 기존 변경 사항을 보존하고, 관련 과거 최적화 기록과 현재 구현의 차이를 확인한다.
2. 실제 GPU 모델, compute capability, VRAM 용량, 드라이버, CUDA toolkit 및 컴파일러를 확인한다. V100의 VRAM 용량이나 GPU 번호를 가정하지 않는다. SM70 빌드 지원 여부를 확인하고 필요한 경우 `CMAKE_CUDA_ARCHITECTURES=70`을 명시한다.
3. 현재 상태로 별도의 baseline 빌드와 실행 로그를 확보한다. 모델/데이터 식별 정보, offload 범위, KV 형식, context, batch/microbatch, CUDA graph 및 attention 옵션을 기록한다. 다른 GPU 작업의 간섭과 power/clock 상태도 확인한다.
4. 추론은 prefill과 decode, 작은/큰 batch, 짧은/긴 KV 길이를 구분한다. 학습은 고정 데이터로 실제 optimizer update를 여러 번 실행해 기준 시간을 확보한다. VRAM 한도 안의 대표 사례를 선택한다.
5. 사용 가능한 profiler 또는 CUDA event 측정으로 kernel 시간, launch 수, host-device 복사, 동기화, register spill, occupancy, 메모리 및 임시 버퍼 비용을 조사한다. microbenchmark와 전체 모델 실행의 병목이 일치하는지 확인한다.
6. 측정 근거에 따라 tile 및 dispatch 조정, 중복 변환/복사 제거, 안전한 연산 fusion, workspace 재사용 등을 검토한다. Volta에서 지원하는 명령과 Tensor Core 경로만 사용하고, 다른 아키텍처용 기능을 무조건 활성화하지 않는다.
7. 하나의 가설씩 검증한다. 지원 shape, dtype, stride, alignment, device 조건을 dispatch에 명시하고 그 밖의 입력은 기존 fallback으로 보낸다. 전체 모델에서 이득이 없거나 회귀하는 실험은 최종 변경에서 제외한다.

## 검증 기준

- 추론은 `llama-bench`의 prefill/decode tok/s와 peak VRAM을 보고한다. 서버 실행에 영향을 주면 같은 prompt와 출력 token 수로 TTFT와 전체 요청 시간도 비교한다.
- 학습은 동일 초기 상태, seed, 데이터 순서, supervised token 수, context, batch/microbatch, optimizer에서 step 시간, 학습 tok/s와 peak VRAM을 비교한다. 초기화, graph capture, rollout 등 별도 단계는 정상 상태의 update 시간과 구분한다.
- 워밍업 후 각 사례를 최소 3회 순차 측정해 중앙값과 변동 범위를 보고한다. 비동기 CUDA 작업은 event 또는 적절한 동기화로 실제 완료 시간을 측정한다. profiler가 켜진 시간과 일반 실행 시간을 섞지 않는다.
- 변경 연산에 맞춰 `test-backend-ops`, `test-opt`, `test-qat` 등 기존 테스트를 선택하고 실제 CUDA 실행을 확인한다. 빌드 성공이나 CPU fallback만으로 Volta 검증을 통과 처리하지 않는다.
- 최적화 경로와 기준 경로의 출력을 기존 수치 허용 오차로 비교한다. tile 경계와 tail, strided 입력, MoE expert ID 매핑, 여러 query/KV 길이 등 변경 경로의 경계를 검증한다. attention/KV를 바꾸면 causal/SWA mask, page 경계와 상태 저장/복원 등 해당 기능도 확인한다.
- 동일 입력의 logits와 가능한 경우 고정 검증 데이터의 perplexity를 비교한다. 학습은 loss, gradient와 parameter update를 비교하고 NaN/Inf 및 누적 오차를 확인한다. checkpoint 경로에 영향을 주면 저장/재개 후 update와 결과물 로딩을 검증한다.
- 메모리 접근이나 동기화를 바꾸면 가능한 경우 Compute Sanitizer로 작은 대표 사례를 검증한다. 도구가 없거나 실행하지 못한 검사는 미검증으로 기록한다.
- SM70 전용 최적화는 capability guard와 fallback을 확인한다. 다른 GPU에서 실행하지 않았다면 해당 GPU의 성능이나 정확성까지 검증했다고 주장하지 않는다.
- 성공은 대표 end-to-end 작업에서 측정 변동을 넘는 개선이 있고 검증 범위에서 유의미한 정확성 또는 성능 회귀가 없는 경우다. kernel만 빨라진 결과, 더 적은 학습량, 축소한 context를 전체 작업의 가속으로 보고하지 않는다.

## 제약 및 결과물

- 메모리를 줄이기 위해 모델 전체를 FP16/F32로 상시 복제하거나 CPU로 숨겨진 offload를 추가하지 않는다. 필요한 임시 메모리와 전송 비용은 모두 측정에 포함한다.
- 양자화 형식, KV 정밀도, 학습 objective, optimizer, effective batch를 임의로 바꾸지 않는다. 설정 및 정밀도 실험은 동일 조건의 코드 최적화 결과와 구분하고 품질 및 메모리 trade-off를 보고한다.
- stream, event, graph capture 또는 buffer lifetime을 바꾸면 데이터 의존성과 소유권을 설명한다. 동기화를 제거했다는 이유만으로 안전하다고 가정하지 않는다.
- 기존 사용자 변경과 모델/데이터/checkpoint 원본을 보존한다. 검증은 별도 출력 경로에서 수행한다. 운영 GPU 작업을 종료하거나 서비스를 재시작하지 않는다.
- 큰 구조 변경이나 새 실행 패턴은 근거와 설계를 먼저 제시해 사용자 확인을 받는다. commit, push, PR 생성은 수행하지 않는다.
- Volta GPU, 호환 toolkit, 모델 또는 데이터가 없으면 가능한 빌드/연산 검증까지만 수행하고 실제 GPU 검증의 차단 사유와 재현 명령을 기록한다. 다른 아키텍처 측정이나 이론값을 V100 실측으로 대체하지 않는다.
- 최종 결과에는 병목과 변경 파일, dispatch 및 fallback 조건, 재현 가능한 빌드/실행 명령, 추론/학습별 변경 전후 성능과 peak VRAM 표, 정확성 결과, 원시 로그 위치, 미검증 항목을 포함한다.
