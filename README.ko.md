[日本語](README.md) | [한국어](README.ko.md)

# Self-Improving LLM Prompts with Arize Prompt Learning

Hugging Face 데이터셋, OpenAI 모델, Arize Prompt Learning을 사용해 LLM
프롬프트를 자연어 피드백으로 반복 개선하는 실험 저장소입니다.

두 가지 과제를 실험합니다.

1. `dair-ai/emotion`을 이용한 감정 분류
2. `kdave/Indian_Financial_News`를 이용한 금융 뉴스 요약

공통 파이프라인은 다음과 같습니다.

```text
현재 프롬프트로 Agent 실행
          ↓
결과 평가 및 자연어 피드백 생성
          ↓
Arize PromptLearningOptimizer.optimize()
          ↓
GPT-4o가 개선 프롬프트 생성
          ↓
다음 Iteration에서 재평가
```

## 준비

현재 Arize Prompt Learning의 공식 저장소 패키지는 Python 3.12 이상을
요구합니다. 이 프로젝트의 스크립트 자체는 Python 3.10 문법과 호환되지만,
최신 SDK를 함께 설치하려면 Python 3.12+ 가상환경을 사용하세요.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## 실행

기본 설정은 상위 30개 샘플, 최대 3회 평가, 목표 정확도 95%입니다.

```bash
python self_improving_emotion_classifier.py
```

비용을 줄여 빠르게 확인하려면 다음처럼 실행할 수 있습니다.

```bash
python self_improving_emotion_classifier.py --samples 5 --iterations 1
```

최종 프롬프트는 기본적으로 `optimized_prompt.txt`에 저장됩니다. `--output`,
`--iterations`, `--target-accuracy` 옵션으로 설정을 변경할 수 있습니다.

## 실험 1: 감정 분류 결과

빠른 검증을 위해 `dair-ai/emotion`의 첫 5개 샘플로 실행한 결과, 초기
프롬프트는 2개를 맞혀 정확도 40%를 기록했습니다.

| 입력의 핵심 표현 | 정답 | 초기 예측 | 결과 |
|---|---|---|---|
| `humiliated` | sadness | sadness | 정답 |
| `hopeless`에서 `hopeful`로 변화 | sadness | joy | 오답 |
| `greedy`, `wrong` | anger | sadness | 오답 |
| 벽난로를 회상하는 문장 | love | sadness | 오답 |
| `grouchy` | anger | anger | 정답 |

초기 모델은 문장 전체의 감정 대신 `hopeful`, `wrong`, `nostalgic` 같은 일부
단어에 민감하게 반응했으며, `joy`, `anger`, `love`를 `sadness`와 혼동했습니다.

Arize Prompt Learning은 피드백을 바탕으로 다음 지침을 추가했습니다.

```text
Consider the overall emotional tone of the text rather than focusing on
isolated words.
```

또한 세 개의 오답을 포함한 실제 샘플을 Few-shot 예제로 추가했습니다. 생성된
전체 프롬프트는
[`artifacts/emotion/optimized_prompt.txt`](artifacts/emotion/optimized_prompt.txt)에서
확인할 수 있습니다.

이 실행에서는 개선 프롬프트를 독립된 Test 데이터로 다시 검증하지 않았습니다.
따라서 40% 이후의 일반화 성능이 개선됐다고 결론 내리지는 않습니다. 추후에는
최적화용 Train과 평가용 Test를 분리해야 합니다.

## 금융 뉴스 요약 파이프라인

`self_improving_financial_summarizer.py`는 금융 기사를 GPT-4o-mini로 요약하고,
GPT-4o Evaluator의 자연어 비평을 이용해 Arize Prompt Learning으로 프롬프트를
개선합니다. 기본 실행은 20개 기사와 3회의 평가 회차를 사용하므로 먼저 아래의
저비용 확인 명령을 권장합니다.

```bash
python self_improving_financial_summarizer.py \
  --samples 2 \
  --iterations 2 \
  --evaluator-model gpt-4o-mini
```

전체 기본 설정으로 실행하려면 다음 명령을 사용합니다.

```bash
python self_improving_financial_summarizer.py
```

최종 프롬프트는 `financial_optimized_prompt.txt`, 회차별 프롬프트·평가 결과·
점수 이력·Arize 메타 프롬프트는 `financial_results/`에 저장됩니다.

## 실험 2: 금융 뉴스 요약 결과

금융 기사 20개를 대상으로 3회 평가했습니다. GPT-4o Evaluator는 각 요약을
다음 세 기준으로 1~5점 평가하고 구체적인 자연어 피드백을 생성했습니다.

- 주요 재무 수치 포함 여부
- 불필요한 내용이 없는 간결성
- 원문에 없는 내용을 만들지 않는 충실성

### 회차별 점수

| Iteration | 종합 점수 | 재무 수치 | 간결성 | 충실성 |
|---|---:|---:|---:|---:|
| 1 | 83.67 | 3.85 | 3.95 | 4.75 |
| 2 | 83.67 | 4.00 | 3.95 | 4.60 |
| 3 | 83.33 | 3.95 | 3.90 | 4.65 |

Iteration 2에서는 주요 재무 수치, 경제적 요인, 영향과 시사점을 강조하고 근거
없는 내용을 제외하라는 지침이 추가됐습니다. 그 결과 재무 수치 점수는
`3.85 → 4.00`으로 상승했지만 충실성이 `4.75 → 4.60`으로 하락해 종합
점수는 동일했습니다.

Iteration 3에서는 금융시장 및 경제정책의 중요한 변화를 강조하는 지침이
추가됐지만, 종합 점수는 `83.33`으로 소폭 하락했습니다.

### 개선되지 않은 원인

이번 실험에서는 재무 수치를 강조하는 지침이 모든 기사에 동일하게 적용되면서
부작용이 발생했습니다.

- 수치가 많은 기사에서는 주변 수치와 기업 소식까지 과도하게 나열함
- 수치가 없는 기사에서도 재무 지표가 존재하는 것처럼 구조화함
- 정보량이 증가하면서 간결성과 충실성이 낮아짐
- Evaluator가 원문에 수치가 없는 경우에도 재무 수치 점수를 낮게 부여함

예를 들어 한 샘플은 `66.67 → 46.67 → 73.33`, 다른 샘플은
`66.67 → 66.67 → 40.00`으로 변했습니다. 전체 평균만 보면 작은 변화지만,
개별 기사에서는 프롬프트 변경으로 큰 성능 하락이 발생했습니다.

Iteration 1과 2가 같은 최고 점수를 기록했기 때문에, 동점일 때 먼저 평가된
프롬프트를 유지하는 현재 로직은 초기 프롬프트를 최종 결과로 선택했습니다.

```text
Summarize the following financial news article:

Article: {article}

Summary:
```

따라서 이번 금융 요약 실험의 결론은 “프롬프트가 개선됐다”가 아니라, 재무
수치를 더 잘 포함하도록 최적화됐지만 다른 평가 기준과의 트레이드오프로 전체
성능 향상에는 실패했다는 것입니다.

회차별 프롬프트와 전체 집계 점수는
[`artifacts/financial/`](artifacts/financial/)에서 확인할 수 있습니다.

## 두 실험에서 얻은 결론

| 구분 | 감정 분류 | 금융 뉴스 요약 |
|---|---|---|
| 평가 방식 | 정답 레이블과 직접 비교 | GPT-4o Evaluator |
| 피드백 생성 | Python 비교 로직 | Evaluator LLM |
| 주요 개선 | 전체 감정 판단 규칙과 Few-shot 예제 | 재무 수치와 충실성 관련 규칙 |
| 주요 한계 | 동일 데이터 재사용 및 과적합 | Evaluator 편향과 평가 기준 간 트레이드오프 |
| 확인된 결과 | 구체적인 개선 프롬프트 생성 | 전체 점수는 개선되지 않음 |

Prompt Learning은 성능 향상을 보장하는 도구가 아니라, 실패 원인을 자연어
규칙으로 변환하고 프롬프트 수정 과정을 자동화하는 도구에 가깝습니다. 특히
자유 형식 생성 과제에서는 Optimizer보다 먼저 Evaluator가 사람의 판단과
일치하는지 검증해야 합니다.

신뢰할 수 있는 실험을 위해 다음 개선이 필요합니다.

- Train, Validation, Test 데이터 분리
- 원문에 재무 수치가 없을 때 `not_applicable` 처리
- 평가 기준별 독립 Evaluator 사용
- 일부 샘플에 대한 사람 평가와 LLM 평가 비교
- 점수가 낮아진 프롬프트를 적용하지 않는 롤백
- API 비용, 토큰 수, 처리 시간 기록

## 공개 실험 산출물

재현성과 결과 비교를 위해 공개 가능한 프롬프트 및 집계 점수는 `artifacts/`에
보관합니다.

```text
artifacts/
├── emotion/
│   └── optimized_prompt.txt
└── financial/
    ├── optimized_prompt.txt
    ├── iteration_1_prompt.txt
    ├── iteration_2_prompt.txt
    ├── iteration_3_prompt.txt
    └── iteration_history.csv
```

기사 원문과 참고 요약이 포함된 `iteration_N_results.csv` 및 Arize가 생성한
메타프롬프트는 데이터 재배포를 피하기 위해 저장소에서 제외합니다.

이는 결과를 숨기기 위한 것이 아닙니다. 저장소에는 회차별 프롬프트와 집계
점수를 공개해 실험 과정과 결론을 확인할 수 있게 하고, 제3자 기사 전문이
중복 저장된 원시 실행 파일만 제외합니다.
