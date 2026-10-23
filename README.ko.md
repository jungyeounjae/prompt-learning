[日本語](README.md) | [한국어](README.ko.md)

# 평가 기반 프롬프트 최적화 실험

프롬프트 변경은 평균 점수를 높이면서 특정 입력이나 평가 기준을 악화시킬 수
있습니다. 이 도구는 감정 분류와 금융 뉴스 요약에서 Train으로 후보를 만들고,
독립된 Validation의 회귀 방지 조건으로 채택 여부를 결정한 뒤 최종 프롬프트만
Test에서 한 번 평가합니다. 로컬 CLI와 산출물로 실험을 재현하고 검토합니다.

현재 공통 파이프라인으로 검증한 실제 모델 실행은 없습니다. 성능 향상, 실측
비용·지연시간, Phoenix 화면의 근거도 없습니다.
[실험 상태와 기존 산출물](artifacts/experiment-summary.md)을 확인하세요.
자동 테스트는 외부 서비스를 대체한 동작 검증이며 모델 품질 측정이 아닙니다.

## 설치와 오프라인 확인

저장소 루트에서 Python 3.12 이상으로 실행하세요. 의존성의 기준은
[pyproject.toml](pyproject.toml)입니다. [requirements.txt](requirements.txt)는
같은 프로젝트를 설치하는 호환 파일입니다. 최초 설치에는 네트워크와 상위 Git
저장소 접근이 필요합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' build
# 호환용 기본 설치: python -m pip install -r requirements.txt
```

다음 도움말은 API 키나 네트워크 없이 실행할 수 있습니다.

```bash offline
python -m prompt_learning.cli --help
python -m prompt_learning.cli run --help
```

설치 후 `prompt-learning --help`도 사용할 수 있습니다.

## 모델 실험 실행

`OPENAI_API_KEY`를 환경변수 또는 루트의 로컬 `.env`에 설정하세요. 아래는 키를
명령 이력에 남기지 않고 입력하는 예입니다. 실험은 유료 API와 데이터셋
다운로드를 사용합니다.

```bash
read -r -s OPENAI_API_KEY
export OPENAI_API_KEY
export PHOENIX_ENABLED=false
python -m prompt_learning.cli run --config configs/emotion.yaml
python -m prompt_learning.cli run --config configs/financial.yaml
```

기본 설정은 seed 42, 60/20/20 분할, 최대 후보 3개입니다. 감정은 레이블별
20개씩 총 120개, 금융은 중복 제거 후 60개입니다. 추론은 `gpt-4o-mini`,
금융 평가와 후보 생성은 `gpt-4o`입니다. 감정 평가는 LLM을 사용하지 않습니다.

작은 실행을 확인하려면 먼저 Hugging Face의 현재 commit을 조회해 고정하세요.
같은 실험을 재현할 때는 저장된 설정의 commit을 재사용해야 합니다.

```bash
export EMOTION_DATASET_REVISION="$(python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("dair-ai/emotion").sha)')"
export FINANCIAL_DATASET_REVISION="$(python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("kdave/Indian_Financial_News").sha)')"
```

아래 설정 생성 자체는 오프라인입니다. 위에서 조회한 revision이 필요합니다.
원본 설정을 보존하면서 감정은 레이블별 5개씩 총 30개, 금융은 5개로 줄입니다.
모든 모델을 `gpt-4o-mini`, 후보는 1회로 바꾸고 분할과 승인 기준은 유지합니다.

```bash offline
python - <<'PY'
import os
from pathlib import Path
import yaml

destination = Path('runs/configs')
destination.mkdir(parents=True, exist_ok=True)
for task, count_key in (('emotion', 'samples_per_label'), ('financial', 'sample_count')):
    config = yaml.safe_load(Path(f'configs/{task}.yaml').read_text())
    config['dataset']['revision'] = os.environ[f'{task.upper()}_DATASET_REVISION']
    config['dataset'][count_key] = 5
    config['models'].update(inference='gpt-4o-mini', evaluator='gpt-4o-mini', meta='gpt-4o-mini')
    config['max_iterations'] = 1
    config['phoenix']['enabled'] = False
    config['cache_path'] = f'.cache/{task}-smoke.sqlite3'
    (destination / f'{task}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

```bash
export PHOENIX_ENABLED=false
python -m prompt_learning.cli run --config runs/configs/emotion.yaml
python -m prompt_learning.cli run --config runs/configs/financial.yaml
```

이 크기는 실행 확인용입니다. 금융은 3/1/1개로 분할되므로 일반화 성능을
추정하기에는 너무 작습니다. 기본 `revision: main`, 모델 이름의 alias, 고정되지
않은 상위 의존성은 변할 수 있습니다. 엄밀한 재현에는 revision, 모델 snapshot,
의존성 목록, 설정 파일을 함께 보관하세요.

## 평가와 후보 선택

Train은 후보 생성, Validation은 선택, Test는 최종 평가에만 사용합니다.
Train 점수는 최종 성능이 아닙니다. Test 결과를 보고 후보나 임계값을 조정하면
독립적인 평가가 아니므로 새로운 미사용 데이터가 필요합니다.

| 과제 | Validation 승인 조건 |
|---|---|
| 감정 분류 | Macro F1 비감소, Accuracy 하락 1%p 이내, invalid 비율 비증가, 클래스별 Recall 하락 5%p 미만 |
| 금융 요약 | 종합 점수·평균 Faithfulness 비감소, 심각한 환각 수 비증가, 20점 이상 하락한 샘플 비율 10% 이내 |

모든 지표가 같으면 짧은 프롬프트를 선택하고 길이도 같으면 기존 프롬프트를
유지합니다. 금융은 재무 수치·간결성·충실성을 1~5점으로 평가합니다. 중요한
재무 수치가 없는 기사에서는 해당 기준을 `not_applicable`로 분모에서 제외합니다.
빈 출력, 40~800자 범위, JSON 형식, 원문에 없는 통화·백분율도 검사합니다.
거절한 후보도 보관하므로 평균 점수로 드러나지 않는 회귀를 확인할 수 있습니다.
임계값은 구현된 승인 정책으로 고정되며 임의 변경값은 CLI가 거절합니다.

## Phoenix와 로컬 산출물

Phoenix는 선택 사항입니다. `PHOENIX_ENABLED=false`면 관측 연결을 하지 않습니다.
활성화된 상태에서 연결에 실패하면 경고를 남기고 로컬 실험을 계속합니다.

```bash
python -m pip install -e '.[phoenix]'
docker compose config
docker compose up -d
export PHOENIX_ENABLED=true
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
python -m prompt_learning.cli run --config runs/configs/emotion.yaml
docker compose stop
```

UI는 `http://localhost:6006`입니다. Compose 기본 image는 `latest`이므로
재현할 때는 `PHOENIX_IMAGE`에 검증한 tag/digest를 지정하세요. 데이터는 Docker
volume에 남습니다. Phoenix에는 hash/ID와 집계 정보만 보내며 원문과 참고 요약은
보내지 않는 설계입니다. 실제 서버 호환성과 화면은 아직 검증하지 않았습니다.

기본 저장 위치는 `runs/<task>/<run-id>/`입니다. manifest에는 UTC 시각, 설정,
revision, 모델, seed, 입력 hash를 기록합니다. 프롬프트, 후보별 Validation 지표,
최종 Test 지표, 거절 이유, 회귀 목록, 보고서를 저장합니다. 감정 분류에는
혼동행렬 이미지도 생성합니다. `runs/`와 `.cache/`는 Git에서 제외합니다.
[아키텍처와 산출물 명세](docs/architecture.md)에 상세 내용이 있습니다.

## 한계와 검증

LLM 평가 편향, 작은 표본의 분산, Validation에 대한 반복 적합이 남습니다.
금융 평가는 사람의 검토와 비교하세요. Test는 최종 선택 프롬프트만 평가하므로
같은 Test에서 초기 프롬프트와의 차이는 이 실행만으로 측정하지 않습니다.
가격 설정은 기본값이 비어 있어 비용을 알 수 없습니다. 현재 CLI 로컬 산출물에는
완전한 토큰·비용·호출 지연시간 장부가 없으며 meta model은 공통 gateway 밖에서
호출합니다. 실행 시간을 API 지연시간이나 실비로 보고하지 마세요. 공개 전에는
프롬프트에 들어간 예시와 스크린샷도 점검하세요.

```bash
python -m pytest tests/test_documentation.py -v
python -m pytest -v
python -m build
docker compose config
git diff --check
```

기존 `self_improving_*.py`는 비권장 호환 래퍼입니다. 예전 감정 `--samples 5`
예시와 개별 출력 파일 지정은 사용할 수 없습니다. 새 CLI와 설정으로 이전하세요.
구현된 기능과 미검증 항목은 [실험 상태](artifacts/experiment-summary.md)를 참고하세요.
