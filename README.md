# Self-Improving Emotion Classifier

Hugging Face의 `dair-ai/emotion` 샘플을 GPT-4o-mini로 분류하고, 자연어
피드백을 Arize Prompt Learning에 전달해 프롬프트를 반복 개선하는 독립형
스크립트입니다.

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
