# Lumen CPU 서버 최적화 분석 및 프로파일링

측정일: 2026-10-24. 대상은 실제 실행 중인 `Lumen-3.1-Pulsar-IQ3-Quality`, CPU 서버 `127.0.0.1:11438`이다. MTP, draft 모델, ngram speculation, expert 생략, 문맥 축소, 모델 재양자화 없이 조사했다. 운영 서버와 기존 소스 변경은 유지했으며, 실험 서버는 별도 포트 `18148`에서 실행하고 종료했다.

## 판단

- **반복 대화의 지연 시간:** 대화가 바뀌어도 KV 상태를 보존하는 설정이 가장 큰 검증된 개선이다. 512토큰 입력을 A/B/A 순서로 요청했을 때 마지막 A의 4토큰 응답 시간이 7.28초에서 0.35~0.51초로 줄었다. 생성 자체가 20배 빨라진 결과가 아니라 입력 재계산을 제거한 결과다.
- **여러 요청의 총 처리량:** 고정 토큰 디코딩에서 4개 요청 배칭은 2.05배, 8개는 2.41배였다. 실제 HTTP 4개 요청의 총 처리량 개선은 25%였다. 요청당 생성 속도와 먼저 도착한 요청의 지연 시간은 나빠진다.
- **단일 요청의 생성 속도:** 현재 약 11.5 tok/s에서 2배 이상을 달성할 검증된 소프트웨어 변경은 찾지 못했다. 가중치 연산과 메모리 전송이 지배적이며, KV 압축이나 스레드 증가만으로 해결할 수 있는 상황이 아니다.
- **커널 개발 우선순위:** dense Q4 GEMV/GEMM, IQ3의 작은 배치 및 작업 분배, 긴 문맥의 paged attention 순으로 검증할 가치가 있다. 아래 후보의 성능 개선은 아직 구현하거나 측정한 결과가 아니다.

## 환경과 측정 방법

| 항목 | 확인한 상태 |
| --- | --- |
| CPU | Xeon E5-2696 v3, 18코어/36스레드, AVX2/FMA/F16C, L3 45 MiB |
| RAM | 약 121 GiB, swap 없음 |
| GPU | Quadro P620 2 GiB, 운영 모델은 CPU 실행 |
| 모델 | `/mnt/openwebui/AIKAR/Adaptive-bit/models/Lumen-3.1-Pulsar-IQ3-Quality.gguf`, 11,760,518,400 bytes |
| 구조 | Gemma4, 30층, hidden 2816, expert 128개 중 8개 선택, 어휘 262144개 |
| 실행 파일 | `build-pulsar-throughput/bin/llama-server` |
| 운영 설정 | 18 decode/batch threads, batch 1024, microbatch 512, context 8192, slot 1, paged F16 KV, bounded SWA, LES 꺼짐 |
| 기존 최적화 | native Release, AVX2, CPU repack, tiled IQ3 prefill, 최적화된 paged attention이 이미 활성화됨 |
| 소스 기준 | `36a623af60241c1bf31b20ee94aa086261563349` 및 기존 미커밋 변경. 바이너리 해시는 `runtime.json`에 저장 |

CPU의 `perf_event_paranoid=4` 때문에 `perf stat`의 task-clock/cycles/instructions가 거부됐다. 호스트 보안 설정은 변경하지 않았다. 대신 기존 평가 콜백으로 연산별 시간을 수집하고, 콜백 없는 고정 토큰 실행과 HTTP 실행을 별도로 측정했다. 콜백은 그래프를 분리하고 fusion/동기화 비용에 영향을 주므로, 프로파일링된 tok/s와 비중을 운영 성능이나 개선율로 그대로 해석하면 안 된다.

고정 토큰 실험은 요청마다 512토큰을 먼저 처리하고 64단계 디코딩했다. 배칭 실험은 요청별로 다른 고정 토큰을 사용하며, 1/2/4/8개 시퀀스를 각각 두 번 측정했다. 입력 처리는 시퀀스별로 수행하고 디코딩만 묶었다. HTTP 실험은 자연어/코드/구조화 출력 관련 반복 입력을 사용했다. 공유 호스트의 소규모 측정이며 통계적 신뢰구간이나 일반 품질 보증은 아니다.

## 프로파일링으로 확인한 병목

18스레드, 고정 토큰 24단계의 연산 시간 비중이다. 입력 처리와 디코딩을 구분했다.

| 연산 | 입력 512 prefill | 512 이후 decode | 입력 4096 prefill | 4096 이후 decode |
| --- | ---: | ---: | ---: | ---: |
| Dense Q4_0 matmul | 39.83% | 44.01% | 33.74% | 39.26% |
| Expert IQ3_XXS matmul | 20.49% | 12.10% | 16.60% | 11.48% |
| Expert IQ3_S matmul | 6.10% | 4.62% | 5.29% | 4.48% |
| Expert Q4_0 matmul | 13.73% | 9.27% | 11.34% | 8.36% |
| Dense Q8_0 matmul | 7.78% | 6.04% | 6.39% | 5.44% |
| Paged attention | 5.95% | 6.14% | 21.74% | 15.72% |

F32 router까지 포함하면 행렬곱 전체가 짧은 문맥에서 prefill 88.2%, decode 77.8%다. 4K에서는 각각 73.6%, 70.6%다. HTTP/JSON 처리가 주 병목이라는 근거는 없다.

출력 projection은 `2816 x 262144` Q4_0 가중치 약 415 MB를 읽는다. 짧은 문맥의 계측 실행에서 이것만 약 13.44 ms/token이었다. 코드 위치는 [Gemma4 출력 projection](../src/models/gemma4.cpp#L475), CPU 경로는 [Q4 GEMV](../ggml/src/ggml-cpu/arch/x86/repack.cpp#L1483)와 [Q4 GEMM](../ggml/src/ggml-cpu/arch/x86/repack.cpp#L2061)이다. 단순히 빠른 SIMD 경로가 빠져 있는 상태가 아니다.

토큰당 논리적 행렬 가중치 범위는 2,107,111,424 bytes이며, 선택 expert 부분은 약 0.642 GB다. 이는 **실제 DRAM 카운터가 아니라 텐서 크기 합산**이다. 캐시 재사용, 중복 읽기, activation, KV 전송은 별도다.

1 GiB 버퍼를 8번 읽은 새 순차 읽기 실험은 6/12/18/24/36스레드에서 각각 30.25/30.94/31.74/31.47/28.80 GB/s였다. 가중치 범위를 한 번씩 읽는 단순 모델에서는 14.4~15.1 tok/s이고, attention과 계산 비용은 아직 제외되어 있다. 절대적인 하드웨어 상한을 증명한 것은 아니지만, 현재 11.5 tok/s를 20~50으로 올린다고 약속할 근거는 없다. 같은 합산으로도 20 tok/s에는 42.1 GB/s, 50에는 105.4 GB/s가 필요하다.

6스레드에서도 순차 대역폭은 거의 포화되지만 생성은 8.42 tok/s에 그쳤다. 따라서 메모리만의 문제로 단정해서도 안 된다. 양자화 해독, 정수 내적, 작업 분배와 동기화에도 개선 여지가 있다.

## 실제로 효과가 확인된 설정

### 대화 전환 후 캐시 보존

운영 launcher는 [단일 슬롯과 RAM 캐시 비활성화](../scripts/serve-pulsar-iq3-temporary.sh#L72)를 선택한다. `--cache-prompt` 덕분에 같은 슬롯의 즉시 반복은 빠르지만 A 다음에 B가 오면 A 상태는 유지되지 않는다.

별도 서버에서 입력 512토큰, 출력 4토큰, greedy, 요청 A/B/A/A를 실행했다. 각 설정에서 첫 A, 대화 전환 뒤 A, 즉시 반복 A의 출력 문자열은 일치했다.

| 설정 | B 다음 A의 재사용 토큰 | B 다음 A의 입력 계산 | B 다음 A의 응답 전체 | 즉시 반복 A의 응답 전체 |
| --- | ---: | ---: | ---: | ---: |
| 현재 정책: RAM 캐시 없음 | 1 | 7.018초 | 7.284초 | 0.352초 |
| RAM 캐시 1024 MiB + idle slot 저장 | 511 | 0.091초 | 0.511초 | 0.374초 |
| Paged resident prefix 캐시 256 MiB | 511 | 0.089초 | 0.355초 | 0.398초 |

SWA window 1024를 넘는 입력 1536토큰도 RAM 캐시로 추가 확인했다. 첫 A는 23.093초, 다른 B를 처리한 뒤 돌아온 A는 0.788초였고 1535토큰을 재사용해 1토큰만 평가했다. 즉시 반복은 0.378초였다. 첫 A와 두 번의 재요청 A의 4토큰 출력은 일치했다. 이는 한 가지 bounded-SWA 복원 사례의 검증이며 임의의 대화 이력 전체에 대한 동등성 보증은 아니다. 첫 요청과 복원 요청의 비교이므로 독립적인 RAM 캐시 비활성화 1536토큰 A/B/A 대조군을 측정한 것으로 해석하면 안 된다.

RAM 캐시는 `--cache-ram 1024 --cache-idle-slots`로 활성화할 수 있다. Resident 캐시는 `--cache-prefix-gpu --cache-prefix-gpu-mib 256`이다. 후자는 이름에 GPU가 있지만, 이 포크의 기존 paged KV 경로를 사용하며 이번 실험은 `-dev none -ngl 0`인 CPU 실행이었다. [캐시 생성](../tools/server/server-context.cpp#L1727), [prefix 복원](../tools/server/server-context.cpp#L2021), [페이지 보존](../src/llama-kv-cache-paged.cpp#L382)에서 이를 확인했다.

효과는 재사용 가능한 입력 비중에 달려 있다. 긴 출력의 디코딩 자체를 빠르게 하지는 않는다. 두 정책을 결합한 성능은 측정하지 않았다. Resident 캐시는 기존 KV pool의 페이지를 점유하므로 MiB 예산만 늘린다고 실제 페이지 여유가 늘지는 않는다. 여러 긴 SWA 대화를 보존할 때는 pool 압박과 eviction을 확인해야 한다. 메모리 여유가 큰 현재 호스트에서는 RAM 캐시가 함께 검토할 실용적인 대안이다.

### 여러 요청의 디코딩 배칭

| 동시 시퀀스 | 총 decode tok/s, 2회 평균 | 요청당 tok/s | 단일 대비 총 처리량 |
| ---: | ---: | ---: | ---: |
| 1 | 11.49 | 11.49 | 1.00배 |
| 2 | 15.55 | 7.77 | 1.35배 |
| 4 | 23.51 | 5.88 | 2.05배 |
| 8 | 27.74 | 3.47 | 2.41배 |

여러 요청이 공통 dense 가중치를 함께 사용하는 효과다. 한 요청의 다음 토큰을 추측하는 방법은 사용하지 않았다. 단일 사용자 응답 속도가 목표이면 배칭이 해결책이 아니다.

실제 HTTP에서는 4개 서로 다른 512토큰 입력을 동시에 제출하고 각각 64토큰을 생성했다. `-np 1 -c 8192`는 총 51.232초, `-np 4 -c 32768`은 40.929초였다. 출력 전체 처리량은 5.00에서 6.25 tok/s로 25.2% 증가했고, 총 완료 시간은 20.1% 감소했다. 입력 처리와 대기 시간을 포함하면 decode만의 2.05배보다 작다.

개별 완료 시간은 단일 슬롯에서 12.8/25.7/38.5/51.2초, 4슬롯에서 40.6/40.8/40.9/40.9초였다. 먼저 도착한 요청의 지연 시간은 악화됐다. API별 generation timing에는 다른 요청의 prefill 대기가 포함될 수 있으므로 이를 순수 커널 속도로 해석하면 안 된다.

HTTP 출력은 4개 중 2개가 슬롯 수에 따라 달랐다. 동일 가중치라도 GEMV/GEMM의 부동소수점 연산 순서가 달라질 수 있다. 이번 측정은 배칭의 품질 동등성이나 bitwise 동등성을 인증하지 않는다. 고정 토큰 실험은 이 출력 차이의 영향을 피하기 위해 사용했다.

Paged unified KV에서는 `-c`가 공유 총 용량이다. 여러 요청 각각에 8K를 제공하려면 총 페이지 용량을 함께 늘리고, 필요하면 기존 `--kv-unified-per-slot 8192`로 요청당 상한을 설정해야 한다. 상한을 생략한 HTTP 실험 서버는 슬롯당 최대 32768을 보고했으며, 실제 사용한 입력은 각 512토큰이었다.

## 코드 변경이 필요한 후보와 기대 한계

### 1. Dense Q4/Q8 GEMV와 GEMM

단일 생성에서 가장 큰 비중이다. 기존 repack과 AVX2 경로를 기반으로 출력 행 여러 개의 activation 로드 재사용, 정수 내적의 instruction scheduling, 행 분할과 메모리 전송을 검토한다. 공유 FFN의 up/gate는 [별도 행렬곱](../src/llama-graph.cpp#L2082)이므로 동일 입력 양자화와 실행 진입 비용을 합치는 후보도 있다. Expert gate/up은 이미 fused 텐서이므로 같은 개선을 새로 제안할 부분이 아니다.

출력 projection의 논리적 읽기 속도는 이미 순차 대역폭에 가깝다. 이를 SIMD 변경만으로 2배 줄일 것으로 기대해서는 안 된다. 각 출력 행의 정수 계산과 FP 누적 순서를 유지할 수 있는 변경부터 독립 oracle 및 전체 logits 비교로 검증하는 편이 적합하다.

### 2. IQ3의 작은 배치 커널과 expert 작업 분배

[IQ3_XXS AVX2 내적](../ggml/src/ggml-cpu/arch/x86/quants.c#L3305)은 LUT와 sign/scale 해독을 수행한다. [IQ3_S](../ggml/src/ggml-cpu/arch/x86/quants.c#L3429)도 전용 SIMD 경로가 있다. 작은 다중 행 내적, activation 재사용, 2~4 routed rows에 맞는 커널을 측정할 가치가 있다. 현재 [tiled 진입 기준](../ggml/src/ggml-cpu/tiled/tiled.cpp#L1202)은 expert당 8행 이상이다. 이미 존재하는 tiled prefill을 새로 구현할 필요는 없다.

모델의 IQ3 gate/up 출력은 1408행이다. [64행 그룹과 정적 분할](../ggml/src/ggml-cpu/tiled/tiled.cpp#L865)은 22개 그룹을 18스레드에 나눈다. 4스레드는 그룹 2개, 나머지는 1개를 맡으므로 균등한 그룹 비용을 가정하면 불균형이 생긴다. 이론적인 분할 효율 약 61%는 실제 CPU utilization 측정값이 아니다. 행 그룹 크기 16/32/64, expert별 routed rows 수, 기존 정적 분할을 비교하고, 필요할 때만 expert와 행 그룹을 합친 작업 분배를 검토한다. 단일 토큰의 generic 경로에는 이 분석을 그대로 적용할 수 없다.

IQ3 전체를 미리 큰 정수/실수 형식으로 풀어 두는 방법은 우선순위가 낮다. LUT 비용을 줄여도 expert 전송량이 커져 현재 메모리 대역폭 조건에서는 손해가 날 수 있다. 재양자화 없이 같은 코드와 scale을 보존하더라도 이 전송 비용은 사라지지 않는다.

### 3. 긴 문맥용 CPU paged attention

[현재 구현](../ggml/src/ggml-cpu/ops.cpp#L256)은 K/V 타일과 query/GQA 재사용을 이미 갖췄지만, QK는 개별 dot, PV는 key별 vector MAD 루프다. Prefill에서 여러 query를 기존 `simd_gemm` 기반 타일로 처리하는 후보가 있다. [일반 CPU flash attention](../ggml/src/ggml-cpu/ops.cpp#L9939)에 재사용 가능한 패턴이 있다.

4K에서 attention 비중은 prefill 21.74%, decode 15.72%다. 해당 연산만 2배가 되어도 나머지가 동일하면 전체 개선은 각각 약 12.2%, 8.5%다. 짧은 문맥의 6% 구간만 최적화해서 전체를 크게 개선할 수는 없다. Causal/SWA mask, 누락 페이지, GQA, 여러 시퀀스와 online softmax 안정성을 유지해야 하며, 누적 순서 변경의 수치 영향도 확인해야 한다.

### 4. 동시 요청의 prefill 간섭 줄이기

서버는 생성 토큰을 먼저 배치에 넣어도 남은 공간을 prompt로 채우고 배치 전체를 실행한다. 따라서 큰 prefill이 먼저 생성 중인 요청의 스트리밍을 멈출 수 있다. 기존 [text prefill budget](../tools/server/server-context.cpp#L4113)과 `--prefill-chunk-size`가 이미 있다. CPU에서 64/128/256토큰을 비교할 가치가 있다. 이번 CPU 조사에서는 이 옵션의 성능을 측정하지 않았으며, 과거 CUDA 결과를 CPU 개선율로 인용하지 않는다. 생성 토큰이 있을 때만 추가 text prefill을 제한하므로 단순히 `-ub`만 줄이는 것과도 다르다.

## 효과가 없거나 이미 적용된 접근

| 조건 | 입력 처리 tok/s | 단일 decode tok/s |
| --- | ---: | ---: |
| 6스레드 | 30.86 | 8.42 |
| 12스레드 | 57.09 | 10.80 |
| 18스레드, 2회 범위 | 70.84~72.18 | 11.47~11.55 |
| 24스레드 | 70.81 | 10.54 |
| 36스레드 | 52.53 | 7.04 |
| 18스레드, 물리 코어 0~17 고정 | 73.25 | 11.47 |
| 18스레드, 기존 fusion 끄기 | 70.32 | 11.45 |
| 18스레드, 기존 tiled matmul 끄기 | 44.56 | 11.73 |

스레드 증가와 코어 고정에서 단일 생성의 유의미한 이득을 확인하지 못했다. Tiled를 끈 한 번의 decode 수치가 약간 높지만 해당 단일 토큰 shape는 원래 tiled 진입 기준을 만족하지 않는다. 이를 tiled 비활성화의 생성 개선으로 해석하면 안 된다. Prefill은 기존 tiled 경로 덕분에 약 1.6배 빠르므로 끄는 것은 권하지 않는다. Fusion 끄기 역시 새 개선안이 아니다.

KV 정밀도 감소, LES, P620 부분 offload의 이전 결과는 [기존 보고서](pulsar-iq3-throughput-20261024.md)에 있다. 이번 조사에서 다시 측정한 결과와 구분해야 한다. 이번에는 모델 정밀도나 expert 수를 줄이는 방법을 채택하지 않았다. P620의 2 GiB에는 현재 모델 전체를 상주시킬 수 없다. 단일 생성의 큰 배수 개선이 필수라면 더 높은 실제 메모리 대역폭 또는 모델 전체를 수용하는 가속기에서 다시 측정하는 방향이 필요하다.

## 자료 및 재현

원본 자료는 `build-pulsar-throughput/results/no-spec-audit/`에 있다. 이 디렉터리는 ignored build artifact이며 별도 보관이 필요하다.

- `runtime.json`: 운영 명령행, 소스 기준, 관련 환경 설정, 실행 파일/라이브러리 SHA-256.
- `profile-both18.log`, `profile-both4096.log`: 입력 처리 및 디코딩의 연산별 시간.
- `fixed-t*.log`, `batch-s*-r*.log`: 콜백 없는 실행과 두 번의 배칭 측정.
- `http-*.json`, `http-measure.log`: 응답 내용, 평가/재사용 토큰 수, 서버 timing과 wall time.
- `http-cache-long.json`, `http-long-measure.log`: 1536토큰 RAM 캐시 복원 추가 검증.
- `bandwidth-fresh.log`, `perf-unavailable.log`: 순차 읽기 측정과 perf 거부 기록.
- `audit.cpp`, `http_audit.py`, `summarize.py`, `summary.json`: 측정 코드와 합산 결과.

`audit.cpp`는 기존 `build-pulsar-cpu-les/results/phase3/profile.cpp`의 콜백을 재사용한다. 다음 명령은 운영 라이브러리를 다시 빌드하지 않고 실험 실행 파일만 만든다.

```bash
c++ -O2 -std=c++17 -Iinclude -Iggml/include build-pulsar-throughput/results/no-spec-audit/audit.cpp -Lbuild-pulsar-throughput/bin -Wl,-rpath,/home/user/aikar-engine/build-pulsar-throughput/bin -lllama -lggml -lggml-base -o build-pulsar-throughput/results/no-spec-audit/audit
LD_LIBRARY_PATH=build-pulsar-throughput/bin build-pulsar-throughput/results/no-spec-audit/audit 18 0 64 4 512 512
LD_LIBRARY_PATH=build-pulsar-throughput/bin build-pulsar-throughput/results/no-spec-audit/audit 18 1 24 1 4096 512
python3 build-pulsar-throughput/results/no-spec-audit/http_audit.py
PYTHONPATH=build-pulsar-throughput/results/no-spec-audit python3 -c 'from http_audit import run; run("http-cache-long", 1, 1024, depth=1536)'
python3 build-pulsar-throughput/results/no-spec-audit/summarize.py
c++ -O3 -march=native -fopenmp -include initializer_list build-pulsar-cpu-les/results/phase3/bandwidth.cpp -o build-pulsar-throughput/results/no-spec-audit/bandwidth
build-pulsar-throughput/results/no-spec-audit/bandwidth
```

CPU 측정은 서로 겹치지 않게 순서대로 실행해야 한다. 제품 코드의 최적화 구현이나 운영 설정 변경은 이번 작업에 포함하지 않았다. 실험 종료 후 운영 PID `1253152`의 명령행과 health `ok`를 확인했으며, 실험 서버는 남아 있지 않았다.
