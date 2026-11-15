# Validation and chat audit, 2026-11-15

검증 데이터가 깨졌다는 증거는 찾지 못했다. 다만 검증의 출처 비중이 학습과 다르고, teacher-forced KL만으로 실제 챗 품질을 판단하기 어렵다는 문제가 확인됐다. 실제 생성은 학습 시작 seed와 step104 모두 두 질문에서 반복 출력으로 실패했다. 원본 base도 같은 현상이 있고, 26B teacher는 같은 질문의 답을 생성했다.

## 검증 데이터 검사

실행 manifest의 세 cache에서 전체 6057 training / 73 validation window를 읽었다. 토큰과 mask는 전수 검사했고 teacher 확률은 2028개 행을 표본 검사했다. cache 전체의 확률 payload를 전수 검사한 것은 아니다.

| Source | Train windows | Validation windows | Train share | Validation share | Validation supervised tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 26B Korean CPT | 2913 | 32 | 48.09% | 43.84% | 524105 |
| 26B Benchmaxx | 243 | 9 | 4.01% | 12.33% | 104790 |
| 12B Agentic | 2901 | 32 | 47.89% | 43.84% | 72329 |
| Total | 6057 | 73 | 100% | 100% | 701224 |

- 세 cache의 tokenizer fingerprint, vocabulary, context, top-k, temperature가 일치한다. 실제 native fixture 로딩도 모델 tokenizer와의 일치 검사를 통과했다.
- 잘못된 token/label ID, 빈 validation window, 파일 길이 불일치가 없다.
- token과 label을 합친 전체 window hash 기준 train/validation 중복 및 validation 내부 중복이 없다. 부분 문장이나 같은 문서 일부의 중복까지 없다는 뜻은 아니다.
- 사용 가능한 원본 Korean CPT/Benchmaxx JSONL에서 마지막 5% record를 held-out으로 보는 보조 검사도 normalized input 중복이 없다. 실제 packed partition에 대한 완전한 conversation lineage 검사는 아니다. Agentic 원본 경로는 현재 존재하지 않는다.
- 다음 token과 label이 다른 validation 위치 451개는 전부 다음 입력이 BOS인 sample 경계다. `pack_dataset_partition`은 각 sample의 마지막 token을 정답으로만 사용하고 다음 sample 앞에 BOS를 삽입한다. 전체 데이터가 한 칸 밀린 오류는 아니다.
- 표본 teacher 행에 중복/범위 밖 ID, NaN/Inf, 양의 log-probability가 없다. top-k + tail 확률 합의 최대 오차는 0.000414 미만으로, 저장된 FP16 log-probability 정밀도와 부합한다. 원래 teacher forward를 다시 실행해 모든 행을 대조한 검사는 아니다.

## 검증의 한계

Benchmaxx가 window 기준으로 학습보다 약 3.07배 큰 비중을 차지한다. 현재 joint KL은 window별 supervised 평균을 평균하므로 이 비중 차이가 metric에도 반영된다. 출처별 metric과 학습 비중으로 재가중한 metric을 함께 보는 것이 필요하다.

분할 코드는 conversation 순서의 앞 95%를 학습, 뒤 5%를 검증으로 사용한다. 검증 window 제한은 이 held-out 구간에서 일정 간격으로 고른다. 무작위 대표 표본을 보장하지 않으므로 원본 정렬에 따른 주제/시기 편향 가능성도 있다. 원본이 실제로 어떤 기준으로 정렬됐는지까지 검증한 것은 아니다.

Korean CPT는 일반 문서이며, Agentic 표본은 영어 도구 사용 기록과 잘린 이전 문맥을 포함한다. Benchmaxx에는 중간 추론이 포함된다. 이 혼합 검증을 한국어 일반 챗의 정확도나 지시 이행률로 해석할 수 없다. `text-samples.json`에 실제 검증 표본을 보관했다.

검증에서는 정답 문맥을 입력하고 다음 token을 평가한다. 실제 대화에서는 모델이 앞서 생성한 token을 다시 입력한다. 따라서 teacher-forced KL이나 token accuracy 개선이 자유 생성의 반복 문제 해결을 보장하지 않는다. 다만 같은 고정 validation에서 KL이 악화되는 현상 자체가 이 편향 때문에 허위로 만들어졌다는 증거도 없다.

## Validation accuracy 추가

기존 snapshot joint validation은 loss만 계산했다. `src/llama-joint-validation.cpp`에 기존 ggml argmax/count-equal 연산으로 정답 token 수를 집계하도록 추가했다. prompt, padding 및 supervision 없는 microbatch의 sentinel은 제외한다. `llama_joint_validation_status`는 correct와 supervised counter를 전달한다. loss 계산은 변경하지 않았다.

이 accuracy는 **모델의 top-1 예측과 데이터 정답 token의 일치율**이다. teacher top-1과의 일치율이나 질문 단위 정답률이 아니다. 서로 다른 window 길이/정답 수에 대해 token 수로 가중된다.

실제 3출처 cache에서 각각 고정 validation window 하나를 뽑아 직렬 및 joint 검증을 비교했다.

| Path | KL | Accuracy |
| --- | ---: | ---: |
| Existing serial evaluation | 2.13477307 | 31.2938% |
| Snapshot joint evaluation | 2.13274486 | 31.3207% |

Joint 결과는 `correct=10463 supervised=33406`이다. 독립적으로 cache labels에서 센 33406개와 정확히 일치한다. 여기에는 supervision 없는 microbatch가 포함되어 sentinel 제외도 실제 모델에서 확인했다. 두 경로의 Q4 output head 정밀도 차이로 수치가 완전히 같지는 않다. 이는 작은 고정 표본 결과이며 전체 73-window 결과가 아니다.

현재 recovery trainer의 joint 완료 로그에 `acc=...% correct=... supervised=...`를 추가했다. 실제 대시보드는 이미 이 형식을 파싱하고 validation accuracy 카드와 그래프에 표시하므로 frontend 변경은 필요하지 않았다. 기존 loss-only 로그의 과거 acc는 복원할 수 없어 임의로 채우지 않았다.

업데이트된 실행 파일과 libllama는 `build-validation-runtime`에 있다. 모델 디렉터리 `train.sh`의 trainer 및 library 경로를 이 빌드로 연결했다. 작은 원본 변경은 `recovery-validation.patch`, `train-launcher.patch`에 보관했다. 외부 recovery 원본 파일은 바꾸지 않고 복사본을 빌드했다.

## 실제 챗 질문 두 개

질문 1:

> 연필 3자루와 공책 2권을 샀습니다. 연필은 한 자루에 700원, 공책은 한 권에 1,800원입니다. 10,000원을 냈다면 거스름돈은 얼마인가요? 계산 과정을 짧게 설명해 주세요.

기대 답은 4300원이다.

질문 2:

> 다음 내용을 JSON 객체로 바꿔 주세요. 다른 설명이나 코드 블록 없이 JSON만 출력하세요. 이름은 민수, 나이는 29세, 거주지는 부산입니다. 키는 name, age, city를 사용하고 age는 숫자로 쓰세요.

기대 값은 `{"name":"민수","age":29,"city":"부산"}`이다.

| Model | Arithmetic | JSON |
| --- | --- | --- |
| Original base, no adapter | 숫자 반복, 답변 미완성 | 숫자 반복, JSON 미생성 |
| Training seed, shared-KV step136 | 숫자 반복, 답변 미완성 | 문자 반복, JSON 미생성 |
| Current Q/O step104 | 숫자 반복, 답변 미완성 | 숫자 반복, JSON 미생성 |
| 26B teacher | 4300원과 올바른 계산 과정 | 기대한 JSON 값 생성 |

Seed와 step104는 같은 두 질문, 모델 내 Jinja template, greedy decoding, context2048, 생성 한도192로 비교했다. 둘 다 반복으로 한도에 도달했다. Base에서는 한도512로도 같은 현상이 있었다. 기존 control vector와 layer range 11~29를 추가한 base 질문1에서도 반복이 재현됐다. 이 추가 검사는 모든 adapter/control-vector 조합을 평가한 것은 아니다.

Teacher 대조는 생성 한도를 늘려 산술1024, JSON512에서 최종 답변까지 확인했다. 초기 한도192에서는 teacher도 추론 도중 잘렸지만 입력과 계산 내용은 올바르게 작성했다. 따라서 표를 동일 token budget의 정량 벤치마크로 해석하면 안 된다. CLI에 `--reasoning off`를 전달했지만 실제 출력에는 thought channel이 남아 있어 이를 thinking-off 평가라고 부르지 않는다.

원본 base에서부터 재현되므로 최근 step0~104 학습만으로 생긴 현상이라고 단정할 수 없다. 이 두 질문은 짧은 smoke test이며 광범위한 챗 benchmark가 아니다. 모든 실제 명령, stderr, 생성 원문은 `chat-*.command.json`, `chat-*.log`, `chat-*.txt`에 있다.

## 검사 및 재개

- 기존 CPU distillation / paired-forward 검사 통과.
- 실제 두 V100의 serial/joint accuracy 비교 통과; 분모 33406 전수 대조 통과.
- 실제 live dashboard 기존 테스트 25개 통과.
- 학습 launcher 기존 테스트 10개 통과.
- `git diff --check` 통과.

학습은 테스트 동안 중지한 뒤 step104의 가중치, momentum, F32 residual, schedule, dataset window624를 복원했다. optimizer를 초기화하지 않았다. 기존 LR, accumulation, targets, validation guard 설정을 유지했다. 새 run은 `resumed-run.json`, 정확한 재개 명령은 `resume-command.json`에 있다. 전체 73-window 초기 validation을 먼저 수행하도록 설정했다.

전체 초기 검증도 완료했다. Step104의 73-window 결과는 **KL 2.59529904, token top-1 accuracy 25.7541%**다. 이는 위의 3-window smoke test와 다른 집합이다. 실제 `/api/legacy-status?ema=100` 응답에서 validation 카드와 그래프에 전달되는 accuracy가 25.7541이고 estimated=false인 것을 확인했다. 응답은 `live-validation-ui.json`에 보관했다. 초기 검증은 기존 직렬 경로이며, 이후 주기적 검증은 accuracy가 추가된 snapshot joint 경로를 사용한다. 두 경로의 projection 정밀도 차이를 고려해서 KL을 비교해야 한다.
