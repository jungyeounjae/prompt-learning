# Evaluation-Driven Prompt Optimization 설계 명세

## 1. 목적

이 프로젝트는 수작업 프롬프트 튜닝의 비재현성과 성능 회귀 문제를 해결하는
로컬 실행형 프롬프트 최적화 실험 도구다. Arize Prompt Learning은 평가 피드백으로
후보 프롬프트를 생성하고, 격리된 Validation 데이터와 명시적인 승인 기준은
후보의 채택 여부를 결정한다. Phoenix는 실행 과정, 평가 결과, 비용과 지연시간을
관측하고 실험을 비교하는 선택적 계층으로 사용한다.

외부 웹 서비스 배포는 범위에 포함하지 않는다. 사용자는 CLI로 실험을 실행하고,
로컬 artifact와 Phoenix UI에서 결과를 검토한다.

## 2. 성공 기준

- 동일한 설정과 seed로 동일한 데이터 split을 재현할 수 있다.
- Train 데이터만 후보 프롬프트 생성에 사용한다.
- 후보의 채택 여부는 Validation 데이터와 명시적인 gate로 결정한다.
- Test 데이터는 최종 선택된 프롬프트를 평가할 때 한 번만 사용한다.
- 감정 분류와 금융 뉴스 요약이 하나의 공통 실험 흐름을 사용한다.
- Phoenix가 실행되지 않아도 실험과 로컬 artifact 생성이 가능하다.
- 후보가 거절된 이유를 지표와 샘플 단위 회귀 사례로 설명할 수 있다.
- 외부 API 없이 핵심 로직을 검증하는 자동화 테스트를 제공한다.
- README만으로 문제, 설계 판단, 실험 결과와 한계를 이해할 수 있다.

## 3. 범위

### 포함

- 감정 분류 및 금융 뉴스 요약 데이터 adapter
- Train 60%, Validation 20%, Test 20% 분리
- 과제별 추론과 평가
- Prompt Learning 후보 생성
- Validation gate와 후보 선택
- Phoenix tracing, dataset 및 experiment 연동
- 토큰, 비용, 지연시간 기록
- hash 기반 로컬 응답 캐시
- CLI와 YAML 설정
- JSON, CSV, 텍스트 및 이미지 artifact
- 단위 테스트, 통합 테스트 및 문서

### 제외

- 외부 서버 배포
- 별도 웹 프론트엔드
- 사용자 인증과 멀티테넌시
- 장기 실행 작업 큐와 스케줄러
- 운영 프롬프트의 자동 무승인 배포
- 대규모 모델 학습 또는 fine-tuning

## 4. 프로젝트 구조

```text
prompt-learning/
├── src/prompt_learning/
│   ├── config.py
│   ├── models.py
│   ├── cli.py
│   ├── datasets/
│   │   ├── base.py
│   │   ├── emotion.py
│   │   └── financial_news.py
│   ├── tasks/
│   │   ├── base.py
│   │   ├── emotion.py
│   │   └── financial_news.py
│   ├── evaluators/
│   │   ├── classification.py
│   │   └── summarization.py
│   ├── optimization/
│   │   ├── optimizer.py
│   │   └── selection.py
│   ├── observability/
│   │   └── phoenix.py
│   └── pipeline/
│       └── experiment.py
├── configs/
│   ├── emotion.yaml
│   └── financial.yaml
├── tests/
├── artifacts/
├── docker-compose.yml
└── README.ko.md
```

각 모듈은 한 가지 책임을 가진다.

- `datasets`: 원천 데이터를 정규화하고 재현 가능한 split을 생성한다.
- `tasks`: 프롬프트를 렌더링하고 모델 추론을 수행한다.
- `evaluators`: 과제별 지표와 자연어 피드백을 생성한다.
- `optimization`: 후보 프롬프트 생성과 승인 정책을 담당한다.
- `observability`: Phoenix와의 통신을 캡슐화한다.
- `pipeline`: 컴포넌트를 연결하고 실험 lifecycle을 관리한다.
- `config.py`: YAML을 검증된 실행 설정으로 변환한다.
- `models.py`: 컴포넌트 사이에서 공유하는 데이터 구조를 정의한다.
- `cli.py`: 설정 파일을 받아 실험을 실행하는 단일 진입점이다.

## 5. 핵심 데이터 모델과 인터페이스

공통 데이터 구조는 과제 구현과 파이프라인의 결합을 줄이기 위해 명시적인
dataclass 또는 동등한 typed model로 정의한다.

- `DatasetSplits`: `train`, `validation`, `test` DataFrame
- `PromptCandidate`: prompt text, version, hash, source iteration
- `MetricResult`: aggregate metrics, per-example metrics, feedback
- `SelectionDecision`: accepted 여부, 이유, 회귀 샘플 목록
- `RunManifest`: 설정, 데이터 revision, 모델, seed, 시간 및 artifact 경로

과제 adapter는 다음 책임을 구현한다.

- 데이터 로드와 정규화
- 입력 placeholder 검증 및 프롬프트 렌더링
- 모델 호출
- 결과 정규화
- 평가 실행

파이프라인은 구체적인 과제 내부를 알지 않고 공통 인터페이스만 호출한다.

## 6. 데이터 흐름

```text
YAML 설정 로드 및 검증
        ↓
데이터셋 로드 및 고정 split 생성
        ↓
초기 프롬프트 Validation 평가
        ↓
Train 추론 및 실패 피드백 생성
        ↓
Prompt Learning 후보 생성
        ↓
후보 Validation 평가
        ↓
Validation gate 승인 또는 거절
        ↓
승인된 후보만 다음 iteration의 기준 프롬프트가 됨
        ↓
최종 선택 프롬프트 Test 평가 1회
        ↓
로컬 artifact 및 Phoenix 기록
```

Train 점수는 후보를 생성하기 위한 진단 정보로만 사용하며 최종 성능으로
보고하지 않는다. Test 결과는 후보 생성이나 선택에 다시 사용하지 않는다.

## 7. 데이터 분리 정책

- 기본 비율은 Train 60%, Validation 20%, Test 20%다.
- 모든 split은 설정의 고정 seed를 사용한다.
- 기본 seed는 `42`다.
- 감정 분류의 기본 샘플은 레이블마다 20개씩 균형 표집한 120개이며,
  레이블별 비율을 유지하는 stratified split을 사용한다.
- 감정 분류는 각 레이블에 최소 5개가 없으면 실행을 중단한다.
- 금융 요약은 동일한 원문이 여러 split에 포함되지 않도록 원문 hash로 중복을
  제거한 후 분리한다.
- 금융 요약의 기본 샘플은 중복 제거 후 60개이며, 유효한 고유 기사가 5개
  미만이면 실행을 중단한다.
- 데이터셋 이름, revision, split, seed와 각 subset의 입력 hash를 manifest에
  저장한다.

## 8. 감정 분류 평가

감정 분류는 결정적 평가를 사용한다.

- Accuracy
- Macro F1
- 클래스별 Precision, Recall, F1
- Confusion Matrix
- Invalid Output Rate

출력은 허용된 여섯 레이블 중 하나로 정규화한다. 정확히 하나의 레이블을
결정할 수 없는 응답은 `invalid`로 처리하며 오답으로 집계한다.

감정 분류 후보는 다음 조건을 모두 만족해야 한다.

- Macro F1이 기준 프롬프트보다 낮아지지 않는다.
- Accuracy 하락이 1 percentage point를 초과하지 않는다.
- Invalid Output Rate가 증가하지 않는다.
- 어떤 클래스의 Recall도 5 percentage point 이상 하락하지 않는다.

모든 지표가 동점이면 더 짧은 프롬프트를 선택하고, 길이도 같으면 기존
프롬프트를 유지한다.

## 9. 금융 뉴스 요약 평가

금융 요약은 서로 독립된 세 평가 기준을 사용한다.

- `financial_figures`: 원문에 존재하는 중요한 재무 수치의 포함 여부
- `conciseness`: 반복, filler 및 불필요한 세부 내용의 부재
- `faithfulness`: 원문에 없는 주장이나 왜곡의 부재

각 기준은 1~5 정수 점수, 판정 label과 근거를 반환한다. 원문에 평가할 중요한
재무 수치가 없으면 `financial_figures`를 `not_applicable`로 표시하고 종합
점수의 분모에서 제외한다. 적용 가능한 기준의 평균에 20을 곱해 0~100
종합점수로 변환한다.

다음 결정적 검사를 LLM 평가와 별도로 수행한다.

- 빈 요약
- 기본 40~800자 범위를 벗어난 요약
- 평가 응답 JSON schema 위반
- 원문에 나타나지 않은 통화 또는 백분율 표현

금융 요약 후보는 다음 조건을 모두 만족해야 한다.

- 적용 가능한 항목으로 계산한 종합 점수가 낮아지지 않는다.
- 평균 Faithfulness가 낮아지지 않는다.
- 심각한 환각 판정 샘플 수가 증가하지 않는다. Faithfulness 1점 또는 원문에
  없는 통화·백분율 표현이 발견된 경우를 심각한 환각으로 정의한다.
- 점수가 20점 이상 하락한 샘플의 비율이 10%를 초과하지 않는다.

모든 지표가 동점이면 더 짧은 프롬프트를 선택하고, 길이도 같으면 기존
프롬프트를 유지한다.

## 10. Prompt Learning과 후보 lifecycle

각 iteration은 현재 기준 프롬프트로 Train 데이터를 실행하고 평가 피드백을
생성한다. Prompt Learning은 이 Train 결과만 받아 하나의 후보를 생성한다.
후보는 필수 placeholder와 비어 있지 않은 내용을 검증한 후 Validation에서
평가한다.

Gate를 통과한 후보는 다음 iteration의 기준 프롬프트가 된다. 거절된 후보는
폐기하지 않고 prompt, 평가 결과, 거절 이유와 회귀 샘플을 artifact에 남긴다.
연속으로 후보가 거절되어도 설정된 최대 iteration까지 새로운 후보 생성을
계속하며, 각 iteration은 당시의 기준 프롬프트에서 시작한다.

## 11. Phoenix 관측 설계

Phoenix는 OpenTelemetry/OpenInference 기반의 선택적 관측 계층이다.

```text
experiment_run
├── dataset_load
├── baseline_validation
├── iteration_N
│   ├── train_inference
│   ├── feedback_generation
│   ├── prompt_optimization
│   ├── candidate_validation
│   └── selection
└── final_test
```

trace 및 span에는 가능한 경우 다음 속성을 기록한다.

- task, run ID, iteration
- dataset revision, subset 및 input hash
- prompt version 및 SHA-256 hash
- 모델, temperature 및 주요 호출 매개변수
- input/output token, 예상 비용 및 지연시간
- 평가 점수, 후보 채택 여부 및 거절 이유
- 오류 유형과 재시도 횟수

`PHOENIX_ENABLED=false`이면 관측 호출을 수행하지 않는다. 활성 상태에서 서버
연결이 실패하면 경고를 남기고 로컬 실행을 계속한다. 관측 실패는 추론 결과나
선택 결정을 변경할 수 없다.

Phoenix 서버는 Docker Compose로 로컬 실행한다. 프로젝트는 Phoenix 서버의
영속 데이터를 Git에 포함하지 않는다.

## 12. 오류 처리와 재시도

- 인증 오류, 잘못된 요청 및 설정 오류는 즉시 중단한다.
- rate limit과 일시적인 서버 오류는 exponential backoff로 최대 3회 재시도한다.
- 빈 모델 응답과 schema를 위반한 평가 응답은 설정된 제한 안에서 재요청한다.
- 샘플 실패를 성공처럼 제외하지 않고 실패 상태와 오류를 결과에 기록한다.
- 한 subset의 실패율이 5%를 초과하면 해당 평가를 실패 처리한다.
- 필수 placeholder가 없는 후보는 모델 호출 전에 거절한다.
- artifact 저장 실패는 실험 재현성을 해치므로 실행을 실패 처리한다.

hash 기반 캐시는 Python 표준 라이브러리의 SQLite로 구현한다. cache key는 task,
모델, 호출 매개변수, prompt text 및 입력 text의 SHA-256으로 구성하고, 값에는
모델 출력과 토큰 사용량만 저장한다. 캐시 결과도 실제 호출과 동일한 출력 검증을
통과해야 한다. API key, 원문과 전체 prompt는 cache metadata에 기록하지 않는다.

## 13. 설정과 CLI

YAML 설정은 다음 항목을 포함한다.

- task와 dataset 식별자 및 revision
- seed와 split 비율
- 샘플 수와 최대 iteration
- inference, evaluator 및 meta model
- temperature와 재시도 정책
- 과제별 평가 및 gate 임계값
- cache, Phoenix와 artifact 설정

CLI의 기본 형태는 다음과 같다.

```bash
python -m prompt_learning.cli run --config configs/emotion.yaml
python -m prompt_learning.cli run --config configs/financial.yaml
```

잘못된 설정은 API 또는 데이터셋 호출 전에 검증하고, 사용자가 수정할 항목을
포함한 오류 메시지를 출력한다.

## 14. Artifact와 보고서

각 실행은 충돌하지 않는 run ID를 사용한다.

```text
artifacts/<task>/<run-id>/
├── manifest.json
├── baseline_prompt.txt
├── selected_prompt.txt
├── iteration_history.csv
├── validation_metrics.json
├── test_metrics.json
├── regressions.csv
├── confusion_matrix.png
└── report.md
```

분류에만 의미가 있는 confusion matrix는 감정 분류 실행에서만 생성한다.
금융 요약 보고서에는 평가 기준별 점수와 `not_applicable` 수를 포함한다.

제3자 데이터셋 원문과 reference 전문은 공개 artifact에 복제하지 않는다.
입력은 비가역 hash와 데이터셋 내 식별자로 참조하고, 사용자가 명시적으로 만든
로컬 결과에는 원문 포함 여부를 설정할 수 있다.

## 15. 테스트 전략

기본 테스트는 네트워크와 유료 API를 사용하지 않는다. fake model client와 fake
Phoenix adapter를 주입한다.

- split 재현성, 비율과 stratification
- 중복 데이터의 split 간 누출 방지
- prompt placeholder 검증과 안전한 렌더링
- 감정 출력 정규화와 invalid 처리
- 분류 지표와 gate의 승인, 거절 및 동점 처리
- 금융 평가 schema 및 `not_applicable` 집계
- 금융 gate의 환각 및 샘플 단위 회귀 차단
- 재시도 가능한 오류와 즉시 실패 오류 구분
- 캐시 hit/miss와 cache 결과 검증
- Phoenix 비활성화와 연결 실패 fallback
- manifest와 필수 artifact 생성
- CLI smoke test

실제 Hugging Face 및 OpenAI를 사용하는 테스트는 별도의 integration marker와
명시적인 환경변수가 있을 때만 실행한다.

## 16. 사용자 문서

README는 다음 순서로 재구성한다.

1. 해결하려는 문제와 한 문장 소개
2. 핵심 결과 및 후보 거절 사례
3. 아키텍처와 데이터 흐름
4. 재현 가능한 로컬 실행 방법
5. Train/Validation/Test 및 평가 설계
6. Phoenix trace와 experiment 화면
7. 감정 분류와 금융 요약 결과 비교
8. 비용, 지연시간과 실패 분석
9. 한계와 후속 과제

프로젝트의 결론은 항상 실험 결과와 일치해야 한다. 성능이 개선되지 않은 경우
이를 숨기거나 `self-improving`이라고 표현하지 않고, gate가 회귀 후보를 차단한
결과와 평가 설계에서 얻은 교훈을 설명한다.

## 17. 구현 순서

1. 테스트 기반과 typed data model을 마련한다.
2. 재현 가능한 데이터 split을 구현한다.
3. 기존 감정 분류 흐름을 공통 interface로 이전한다.
4. 분류 evaluator와 selection gate를 구현한다.
5. 금융 요약 흐름과 다기준 evaluator를 이전한다.
6. 금융 selection gate를 구현한다.
7. 공통 experiment pipeline과 CLI를 연결한다.
8. artifact, manifest, cache와 오류 처리를 추가한다.
9. Phoenix tracing과 experiment 연동을 추가한다.
10. 실제 실험을 실행하고 README 및 보고서를 갱신한다.

## 18. 완료 조건

- 전체 기본 테스트가 네트워크 없이 통과한다.
- 두 YAML 설정으로 각 과제를 CLI에서 실행할 수 있다.
- Phoenix를 끈 실행과 연결 실패 fallback 실행이 성공한다.
- 각 실행이 manifest, 프롬프트, metric, history 및 report를 생성한다.
- Test 데이터가 후보 생성이나 Validation gate에 사용되지 않는다.
- 거절된 후보와 이유를 artifact에서 확인할 수 있다.
- README에 실제 실행 결과, Phoenix 화면과 재현 명령이 포함된다.
