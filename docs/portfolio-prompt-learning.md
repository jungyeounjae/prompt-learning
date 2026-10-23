# Portfolio slide draft — Prompt Learning Lab

This draft is prepared for `docs/portfolio-ja.pptx` as project 02. The Phoenix
screen is intentionally a placeholder; capture approval is required before
placing a real screenshot in the deck.

## Slide 3 — Works entry

**02 Prompt Learning Lab**  
LLM 프롬프트를 데이터 기반으로 개선·검증하는 실험 플랫폼  
2026.10 ~ 진행 중 / 개인 개발 / Python · OpenAI API · Hugging Face · Phoenix

## Slide 17 — 프로젝트 개요

**프롬프트 변경을 감으로 판단하지 않고, 동일한 데이터와 평가 기준으로 반복 검증하는 실험 시스템**

- baseline 프롬프트와 후보 프롬프트를 자동 비교
- Train / Validation / Test를 분리해 평가 오염 방지
- 실행 결과를 파일과 Phoenix trace로 재현 가능하게 보존

## Slide 18 — 문제

- 프롬프트가 좋아졌는지 정량적으로 판단하기 어렵다.
- 후보별 성능 저하와 회귀 사례를 놓치기 쉽다.
- Test 데이터를 선택 과정에 사용하면 최종 평가가 오염된다.
- 모델 호출, 지연시간, 실패 사례를 한곳에서 추적하기 어렵다.

## Slide 19 — 실행 흐름

Dataset Load → Baseline Validation → Train Inference → Feedback Generation → Prompt Optimization → Candidate Validation → Selection → Final Test

후보는 Validation 결과가 승인 기준을 통과할 때만 선택되며, Test는 최종 선택 이후 한 번만 실행한다.

## Slide 20 — 아키텍처

CLI → ExperimentRunner

- Dataset Loader
- Model Gateway: retry / cache / token / latency
- Task Runner
- Evaluator
- Prompt Optimizer
- Candidate Selector
- Artifact Writer
- Phoenix Observer

## Slide 21 — 핵심 설계

- 동일한 seed로 데이터 분할 재현
- 후보 프롬프트를 Validation에서만 비교
- accuracy, macro F1, invalid output rate를 함께 평가
- 성능 저하·회귀·실패율을 기준으로 후보 자동 거절
- Test 결과는 선택 이후 독립적으로 기록

## Slide 22 — Phoenix 관측

**[Phoenix 화면 캡처 자리 — 실제 캡처 후 삽입]**

- dataset_load
- baseline_validation
- candidate_validation
- selection
- final_test

Attributes: accuracy / macro F1 / invalid output rate / prompt hash / prompt version / latency

Annotation은 별도 사람 평가나 evaluator가 추가한 판정이며, 현재 실행에서는 Attributes가 주요 평가 데이터다.

## Slide 23 — 실패와 회귀 관리

- `failure_cases.csv`: 실패한 Test 사례의 안전한 식별 정보
- `regressions.csv`: 후보 적용 후 성능이 떨어진 사례
- `iteration_history.csv`: 후보 생성·수용·거절 이력
- 원문과 민감한 출력은 산출물에 복사하지 않는다.

## Slide 24 — 반복 실행 비교

seed 42 / 123 / 2024를 사용해 실행 결과를 비교한다.

- Validation/Test accuracy
- Validation/Test macro F1
- 평균·최솟값·최댓값·표준편차

단일 실행의 우연한 결과가 아니라 seed 변화에도 유지되는 경향인지 확인한다.

## Slide 25 — 회고와 한계

- 프롬프트 개선을 정량적인 실험 단위로 관리할 수 있게 되었다.
- Train / Validation / Test 분리로 Test 오염을 방지했다.
- Phoenix로 실행 흐름과 평가 지표를 추적할 수 있다.
- 외부 모델 응답은 seed·네트워크·모델 상태에 따라 달라질 수 있다.
- 비용 비교는 검증된 모델 가격을 설정한 실행에서만 해석한다.
- 현재는 로컬 실행 중심이며, 공개 배포를 목표로 하지 않는다.
