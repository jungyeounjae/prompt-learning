# Evaluation-Driven Prompt Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 두 실험 스크립트를 Train/Validation/Test 분리, 검증 게이트, 로컬 artifact, 선택적 Phoenix 관측을 갖춘 재현 가능한 CLI 실험 도구로 재구성한다.

**Architecture:** `src/prompt_learning` 패키지에서 데이터 adapter, 과제 실행기, evaluator, 후보 선택기, artifact writer와 observability adapter를 분리하고 `ExperimentRunner`가 이들을 조합한다. Prompt Learning은 Train 피드백으로 후보만 생성하고, Validation gate가 채택을 결정하며, Test는 최종 프롬프트에 한 번만 사용한다. Phoenix는 실패해도 실험 결과를 바꾸지 않는 선택적 adapter로 둔다.

**Tech Stack:** Python 3.12+, pandas, Hugging Face datasets, OpenAI Python SDK, Arize Prompt Learning, scikit-learn, PyYAML, matplotlib, SQLite 표준 라이브러리, OpenTelemetry/OpenInference, Phoenix, pytest

**Spec:** `docs/superpowers/specs/2026-10-22-evaluation-driven-prompt-optimization-design.md`

## Global Constraints

- 외부 웹 서비스, 별도 프론트엔드, 인증, 작업 큐는 구현하지 않는다.
- 기본 seed는 `42`, split은 Train 60%, Validation 20%, Test 20%다.
- 감정 분류 기본 표본은 클래스별 20개인 총 120개이고 클래스별 최소 5개가 필요하다.
- 금융 요약 기본 표본은 중복 제거 후 60개이고 고유 기사 최소 5개가 필요하다.
- Test subset은 후보 생성과 선택에 사용하지 않고 최종 프롬프트에 한 번만 사용한다.
- Phoenix 비활성화 또는 연결 실패가 로컬 실험의 성공 여부와 선택 결과를 바꾸면 안 된다.
- API key, 원문과 전체 prompt는 cache metadata에 저장하지 않는다.
- 기본 테스트는 네트워크나 유료 API를 호출하지 않는다.
- 기존 `arize-phoenix-evals==2.7.0` pin은 Prompt Learning 호환성 검증 전까지 유지한다.
- 제3자 기사 원문과 reference 전문을 Git에 포함되는 공개 artifact에 저장하지 않는다.

## Review Focus

- 클래스별 표본이 5개 미만인 감정 데이터는 불완전한 split 대신 설명 가능한 `ValueError`를 내야 한다; Task 2의 `test_emotion_split_rejects_underrepresented_label`이 고정한다.
- 원문 hash가 같은 금융 기사는 서로 다른 subset에 들어가면 안 된다; Task 2의 `test_financial_split_prevents_duplicate_leakage`가 고정한다.
- 캐시에 손상된 응답이 있으면 성공으로 반환하지 않고 삭제 후 실제 호출을 시도해야 한다; Task 3의 `test_gateway_discards_invalid_cached_value`가 고정한다.
- Validation에서 좋아도 Test 실행 전후에 후보를 다시 선택하면 안 된다; Task 9의 `test_runner_calls_test_once_after_selection`이 고정한다.
- Phoenix exporter가 초기화 중 또는 span 종료 중 실패해도 동일한 선택 결과와 artifact가 남아야 한다; Task 8의 fallback 테스트 두 개가 고정한다.

---

## File Map

- `pyproject.toml`: package metadata, runtime/dev dependencies, pytest 설정과 CLI entry point
- `src/prompt_learning/models.py`: 모든 컴포넌트가 공유하는 typed dataclass
- `src/prompt_learning/config.py`: YAML 로드와 fail-fast 설정 검증
- `src/prompt_learning/datasets/`: 과제별 로드, 정규화, split
- `src/prompt_learning/gateway.py`: 재시도, 토큰 metadata와 SQLite 캐시를 포함한 모델 호출 경계
- `src/prompt_learning/tasks/`: 프롬프트 렌더링과 과제별 출력 정규화
- `src/prompt_learning/evaluators/`: 분류 및 요약 metric/feedback
- `src/prompt_learning/optimization/`: Prompt Learning wrapper와 Validation gate
- `src/prompt_learning/artifacts.py`: 실행 디렉터리, manifest, metric, report 저장
- `src/prompt_learning/observability/phoenix.py`: no-op 및 Phoenix observer
- `src/prompt_learning/pipeline/experiment.py`: 전체 experiment lifecycle
- `src/prompt_learning/cli.py`: `prompt-learning run --config ...` 진입점
- `configs/`: 두 과제의 실행 가능한 기본 설정
- `tests/`: 네트워크 없는 단위 및 통합 테스트
- `docker-compose.yml`: 로컬 Phoenix 서버
- `README.ko.md`, `README.md`: 문제, 설계와 재현 절차

### Task 1: Package Foundation, Shared Models, and Config Validation

**Files:**
- Create: `pyproject.toml`
- Create: `src/prompt_learning/__init__.py`
- Create: `src/prompt_learning/models.py`
- Create: `src/prompt_learning/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `DatasetSplits`, `PromptCandidate`, `MetricResult`, `SelectionDecision`, `RunManifest`, `ExperimentConfig`
- Produces: `load_config(path: Path) -> ExperimentConfig`
- Consumes: no project interfaces

- [ ] **Step 1: Write failing config and model tests**

Create tests named `test_load_config_returns_typed_config`, `test_split_ratios_must_sum_to_one`, `test_rejects_unknown_task`, and `test_rejects_non_positive_sample_count`. Use a temporary YAML file and assert exact defaults: seed `42`, retry count `3`, failure rate `0.05`, and Phoenix disabled unless explicitly enabled.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL because `prompt_learning.config` does not exist.

- [ ] **Step 3: Add packaging and typed shared models**

Configure a `src` layout and the console script `prompt-learning = prompt_learning.cli:main`. Add runtime dependencies already present in `requirements.txt` plus `PyYAML`, `scikit-learn`, and `matplotlib`; add `pytest` as a dev dependency. Define immutable dataclasses where mutation is not required and keep DataFrames only in `DatasetSplits` and per-example metric payloads.

- [ ] **Step 4: Implement `load_config(path: Path) -> ExperimentConfig`**

Validate required sections before any external calls. Support only `emotion` and `financial` tasks, require split ratios to total `1.0` within floating-point tolerance, and emit messages naming the invalid key and value.

- [ ] **Step 5: Run tests and verify pass**

Run: `python -m pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/prompt_learning tests/test_config.py
git commit -m "feat: add typed experiment configuration"
```

### Task 2: Reproducible Dataset Adapters and Split Isolation

**Files:**
- Create: `src/prompt_learning/datasets/__init__.py`
- Create: `src/prompt_learning/datasets/base.py`
- Create: `src/prompt_learning/datasets/emotion.py`
- Create: `src/prompt_learning/datasets/financial_news.py`
- Create: `tests/datasets/test_emotion.py`
- Create: `tests/datasets/test_financial_news.py`

**Interfaces:**
- Consumes: `DatasetSplits`, dataset section of `ExperimentConfig`
- Produces: `split_emotion(frame: pd.DataFrame, *, seed: int, samples_per_label: int = 20) -> DatasetSplits`
- Produces: `split_financial(frame: pd.DataFrame, *, seed: int, sample_count: int = 60) -> DatasetSplits`
- Produces: `load_emotion_dataset(config: ExperimentConfig) -> DatasetSplits`
- Produces: `load_financial_dataset(config: ExperimentConfig) -> DatasetSplits`

- [ ] **Step 1: Write failing emotion split tests**

Add `test_emotion_split_is_reproducible_and_stratified`, `test_emotion_subsets_are_disjoint`, and `test_emotion_split_rejects_underrepresented_label`. Assert a synthetic six-class dataset yields 60/20/20 proportions, stable row IDs for the same seed, distinct IDs across subsets, and a `ValueError` naming the deficient label when it has fewer than five rows.

- [ ] **Step 2: Run emotion tests and verify failure**

Run: `python -m pytest tests/datasets/test_emotion.py -v`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement emotion normalization, balanced sampling, and two-stage stratified split**

Normalize to `example_id`, `text`, and `expected_label`. Map integer labels through the existing six-label mapping. Select `samples_per_label` deterministically, then split 60/40 and split the remainder 50/50 with the same seed.

- [ ] **Step 4: Write failing financial split tests**

Add `test_financial_split_is_reproducible`, `test_financial_split_prevents_duplicate_leakage`, `test_financial_split_rejects_fewer_than_five_unique_articles`, and `test_financial_alias_columns_are_normalized`. Assert hash-disjoint subsets after duplicate article removal.

- [ ] **Step 5: Run financial tests and verify failure**

Run: `python -m pytest tests/datasets/test_financial_news.py -v`

Expected: FAIL because the financial adapter does not exist.

- [ ] **Step 6: Implement financial normalization, deduplication, and split**

Normalize aliases to `example_id`, `article`, and `reference_summary`; use a normalized article SHA-256 as the duplicate identity. Shuffle with the seed and allocate counts by a deterministic largest-remainder calculation so every retained row belongs to exactly one subset.

- [ ] **Step 7: Run dataset tests and verify pass**

Run: `python -m pytest tests/datasets -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/prompt_learning/datasets tests/datasets
git commit -m "feat: add reproducible isolated dataset splits"
```

### Task 3: Model Gateway, Retry Policy, and SQLite Cache

**Files:**
- Create: `src/prompt_learning/gateway.py`
- Create: `tests/test_gateway.py`

**Interfaces:**
- Consumes: model name, messages, temperature, output validator
- Produces: `ModelResponse(text: str, input_tokens: int, output_tokens: int, latency_ms: float, cached: bool)` in `models.py`
- Produces: `ModelGateway.complete(request: ModelRequest, validator: Callable[[str], bool]) -> ModelResponse`
- Produces: `estimate_cost_usd(response: ModelResponse, pricing: ModelPricing) -> Decimal`

- [ ] **Step 1: Write failing gateway tests**

Cover `test_gateway_retries_transient_error_three_times`, `test_gateway_does_not_retry_authentication_error`, `test_gateway_caches_by_hashed_request`, `test_cache_does_not_store_api_key_or_raw_input_metadata`, `test_gateway_discards_invalid_cached_value`, and `test_cost_uses_separate_input_and_output_rates`. Use a fake callable client and a temporary SQLite path. Assert cost with `Decimal` arithmetic from per-million-token rates configured in YAML; do not hard-code vendor prices in Python.

- [ ] **Step 2: Run gateway tests and verify failure**

Run: `python -m pytest tests/test_gateway.py -v`

Expected: FAIL because `ModelGateway` is undefined.

- [ ] **Step 3: Implement request hashing and SQLite cache**

Hash task, model, parameters, prompt and input into one SHA-256 key. Store only key, response text, token counts, and timestamps. Never persist request text or credentials in cache columns.

- [ ] **Step 4: Implement bounded exponential retry**

Retry rate limits, connection failures and 5xx-equivalent OpenAI errors at most three times. Authentication and invalid-request errors fail immediately. Validate cached and live text with the same callback; delete an invalid cache row before attempting a live request.

- [ ] **Step 5: Run gateway tests and verify pass**

Run: `python -m pytest tests/test_gateway.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/prompt_learning/models.py src/prompt_learning/gateway.py tests/test_gateway.py
git commit -m "feat: add resilient cached model gateway"
```

### Task 4: Emotion Task, Metrics, Feedback, and Selection Gate

**Files:**
- Create: `src/prompt_learning/tasks/__init__.py`
- Create: `src/prompt_learning/tasks/base.py`
- Create: `src/prompt_learning/tasks/emotion.py`
- Create: `src/prompt_learning/evaluators/__init__.py`
- Create: `src/prompt_learning/evaluators/classification.py`
- Create: `src/prompt_learning/optimization/__init__.py`
- Create: `src/prompt_learning/optimization/selection.py`
- Create: `tests/tasks/test_emotion.py`
- Create: `tests/evaluators/test_classification.py`
- Create: `tests/optimization/test_emotion_selection.py`

**Interfaces:**
- Consumes: `ModelGateway`, an emotion DataFrame, `PromptCandidate`
- Produces: `render_emotion_prompt(template: str, text: str) -> str`
- Produces: `normalize_emotion_output(raw: str) -> str`
- Produces: `evaluate_classification(frame: pd.DataFrame) -> MetricResult`
- Produces: `select_emotion_candidate(baseline: MetricResult, candidate: MetricResult, baseline_prompt: str, candidate_prompt: str) -> SelectionDecision`

- [ ] **Step 1: Write failing prompt and normalization tests**

Test literal braces in user text, missing `{text}`, one allowed label embedded in a short sentence, multiple labels, and an unsupported label. Preserve the existing behavior that ambiguous output becomes `invalid`.

- [ ] **Step 2: Run task tests and verify failure**

Run: `python -m pytest tests/tasks/test_emotion.py -v`

Expected: FAIL because the task functions do not exist.

- [ ] **Step 3: Implement the emotion task boundary**

Use `str.replace` for the required placeholder and delegate API behavior to `ModelGateway`. Return an output DataFrame containing `example_id`, expected label, raw output, normalized output, token counts, latency and error status.

- [ ] **Step 4: Write failing classification metric and feedback tests**

Assert exact accuracy, Macro F1, per-class recall, invalid rate, confusion matrix labels, and natural-language feedback for correct, confused, and invalid examples.

- [ ] **Step 5: Implement `evaluate_classification`**

Use scikit-learn metrics with the six labels in the existing canonical order. Count `invalid` as incorrect without adding it as a target class. Return aggregate and per-example data through `MetricResult`.

- [ ] **Step 6: Write failing emotion gate tests**

Cover candidate acceptance, Macro F1 regression, accuracy regression beyond `0.01`, invalid-rate increase, per-class recall drop of `0.05` or more, shorter-prompt tie-break, and exact tie retaining baseline.

- [ ] **Step 7: Implement `select_emotion_candidate`**

Return all failed rules in `SelectionDecision.reasons`, not only the first. Interpret the recall rule as rejecting a drop greater than or equal to `0.05`, matching the approved design wording.

- [ ] **Step 8: Run focused tests and verify pass**

Run: `python -m pytest tests/tasks/test_emotion.py tests/evaluators/test_classification.py tests/optimization/test_emotion_selection.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/prompt_learning/tasks src/prompt_learning/evaluators src/prompt_learning/optimization tests/tasks tests/evaluators tests/optimization
git commit -m "feat: add emotion evaluation and validation gate"
```

### Task 5: Financial Task, Structured Evaluation, and Selection Gate

**Files:**
- Create: `src/prompt_learning/tasks/financial_news.py`
- Create: `src/prompt_learning/evaluators/summarization.py`
- Create: `tests/tasks/test_financial_news.py`
- Create: `tests/evaluators/test_summarization.py`
- Create: `tests/optimization/test_financial_selection.py`

**Interfaces:**
- Consumes: `ModelGateway`, financial DataFrame, `PromptCandidate`
- Produces: `render_financial_prompt(template: str, article: str) -> str`
- Produces: `parse_summary_evaluation(raw: str) -> SummaryEvaluation`
- Produces: `evaluate_summaries(frame: pd.DataFrame, evaluator: Callable[..., str]) -> MetricResult`
- Produces: `select_financial_candidate(baseline: MetricResult, candidate: MetricResult, baseline_prompt: str, candidate_prompt: str) -> SelectionDecision`

- [ ] **Step 1: Write failing financial task tests**

Test missing `{article}`, literal braces in articles, empty generation, and preservation of reference data outside the rendered inference prompt.

- [ ] **Step 2: Implement the financial task boundary**

Render only the article into the summarizer prompt. Return output, token, latency and error columns while keeping evaluator inputs separate from model inputs.

- [ ] **Step 3: Write failing structured evaluator tests**

Cover valid three-score JSON, `financial_figures=not_applicable`, invalid score range, missing rationale, 40/800-character boundaries, unsupported currency/percentage detection, and aggregate scoring that excludes N/A. Assert the aggregate is the mean of applicable 1–5 scores multiplied by `20`.

- [ ] **Step 4: Implement parsing, deterministic checks, and aggregation**

Use a typed `SummaryEvaluation` with nullable financial score, integer conciseness/faithfulness, labels and rationales. Mark severe hallucination when faithfulness equals `1` or deterministic numeric checks find unsupported currency/percentage expressions.

- [ ] **Step 5: Write failing financial gate tests**

Cover overall regression, any faithfulness regression, increased severe-hallucination count, more than 10% of examples dropping at least 20 points, exactly 10% acceptance, shorter-prompt tie-break, and exact tie retaining baseline.

- [ ] **Step 6: Implement `select_financial_candidate`**

Compare example IDs rather than row position. Include each regressed example and delta in `SelectionDecision.regressions`.

- [ ] **Step 7: Run focused tests and verify pass**

Run: `python -m pytest tests/tasks/test_financial_news.py tests/evaluators/test_summarization.py tests/optimization/test_financial_selection.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/prompt_learning/tasks/financial_news.py src/prompt_learning/evaluators/summarization.py src/prompt_learning/optimization/selection.py tests/tasks/test_financial_news.py tests/evaluators/test_summarization.py tests/optimization/test_financial_selection.py
git commit -m "feat: add financial evaluation and regression gate"
```

### Task 6: Prompt Learning Optimizer Boundary

**Files:**
- Create: `src/prompt_learning/optimization/optimizer.py`
- Create: `tests/optimization/test_optimizer.py`

**Interfaces:**
- Consumes: current `PromptCandidate`, Train `MetricResult`, required placeholder
- Produces: `PromptOptimizer.optimize(current: PromptCandidate, training_result: MetricResult, required_placeholder: str) -> PromptCandidate`

- [ ] **Step 1: Write failing optimizer adapter tests**

Use a fake optimizer factory. Cover passing only Train outputs/feedback, monotonically increasing version numbers, SHA-256 prompt hash, empty optimizer output, and missing required placeholder.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/optimization/test_optimizer.py -v`

Expected: FAIL because `PromptOptimizer` does not exist.

- [ ] **Step 3: Implement the adapter around `PromptLearningOptimizer`**

Keep the third-party import inside the default factory so unit tests import without constructing the SDK. Reject invalid candidates before they reach Validation and return an explicit invalid-candidate result or typed exception that the pipeline can archive.

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m pytest tests/optimization/test_optimizer.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/prompt_learning/optimization/optimizer.py tests/optimization/test_optimizer.py
git commit -m "feat: isolate prompt learning candidate generation"
```

### Task 7: Artifact Writer and Reproducible Run Report

**Files:**
- Create: `src/prompt_learning/artifacts.py`
- Create: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: `RunManifest`, prompts, metric results, decisions
- Produces: `ArtifactWriter.start_run(task: str, run_id: str) -> Path`
- Produces: `ArtifactWriter.write_iteration(...) -> None`
- Produces: `ArtifactWriter.finalize(...) -> Path`

- [ ] **Step 1: Write failing artifact tests**

Test collision-resistant explicit run IDs, required files, valid JSON, UTF-8 CSV, classification-only confusion matrix, financial N/A counts, redacted source/reference text, and write failure propagation.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_artifacts.py -v`

Expected: FAIL because `ArtifactWriter` is undefined.

- [ ] **Step 3: Implement atomic artifact writes**

Write to a sibling temporary file and replace the destination only after serialization succeeds. Produce `manifest.json`, baseline and selected prompt, history CSV, Validation/Test metrics, regressions CSV and Markdown report. Generate a confusion matrix PNG only for classification.

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m pytest tests/test_artifacts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/prompt_learning/artifacts.py tests/test_artifacts.py
git commit -m "feat: persist reproducible experiment artifacts"
```

### Task 8: Optional Phoenix Observability Adapter

**Files:**
- Create: `src/prompt_learning/observability/__init__.py`
- Create: `src/prompt_learning/observability/phoenix.py`
- Create: `tests/observability/test_phoenix.py`
- Modify: `pyproject.toml`
- Create: `docker-compose.yml`

**Interfaces:**
- Consumes: `ExperimentConfig`, run/span attributes and evaluation annotations
- Produces: `Observer.start_run(attributes: Mapping[str, object]) -> ContextManager[RunSpan]`
- Produces: `RunSpan.child(name: str, attributes: Mapping[str, object]) -> ContextManager[RunSpan]`
- Produces: `Observer.sync_dataset(name: str, version: str, examples: Sequence[DatasetExample]) -> str | None`
- Produces: `Observer.record_experiment(dataset_id: str | None, run: ExperimentResult) -> None`
- Produces: `build_observer(config: ExperimentConfig) -> Observer`

- [ ] **Step 1: Write failing no-op and fallback tests**

Add `test_disabled_phoenix_returns_noop_observer`, `test_initialization_failure_returns_noop_and_warns`, `test_span_export_failure_does_not_change_wrapped_result`, `test_observer_records_required_safe_attributes`, `test_dataset_sync_uses_hashes_and_reference_ids_only`, and `test_experiment_record_failure_only_warns`. Explicitly assert raw article, reference summary, full prompt and API key values are absent.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/observability/test_phoenix.py -v`

Expected: FAIL because the observer does not exist.

- [ ] **Step 3: Add compatible lightweight Phoenix/OpenInference dependencies and local server**

Keep Phoenix client/tracing dependencies in a `phoenix` optional dependency group so the base package remains runnable. Do not change `arize-phoenix-evals==2.7.0` until the complete dependency resolver and Prompt Learning import smoke test pass. Add a Compose service with a named volume, ports documented through environment defaults, and no repository-mounted trace data.

- [ ] **Step 4: Implement no-op and Phoenix observers**

Catch setup/export errors at the adapter boundary, log one actionable warning, and preserve wrapped return values. Record only hashes/IDs for source examples and prompts, plus model/token/cost/latency/decision metadata. Synchronize the three split identities as versioned Phoenix datasets and attach baseline/candidate/final runs as experiments when the connected Phoenix client supports those operations; degrade each unsupported or failed operation independently to a warning and local artifact.

- [ ] **Step 5: Run tests and dependency smoke checks**

Run: `python -m pytest tests/observability/test_phoenix.py -v`

Run: `python -c "from optimizer_sdk.prompt_learning_optimizer import PromptLearningOptimizer; import prompt_learning.observability.phoenix"`

Expected: tests PASS and imports exit `0`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml docker-compose.yml src/prompt_learning/observability tests/observability
git commit -m "feat: add optional Phoenix experiment tracing"
```

### Task 9: Experiment Runner and Test Isolation Guarantees

**Files:**
- Create: `src/prompt_learning/pipeline/__init__.py`
- Create: `src/prompt_learning/pipeline/experiment.py`
- Create: `tests/pipeline/test_experiment.py`

**Interfaces:**
- Consumes: dataset loader, task runner, evaluator, optimizer, selector, observer, artifact writer
- Produces: `ExperimentRunner.run(config: ExperimentConfig) -> ExperimentResult`

- [ ] **Step 1: Write failing lifecycle tests with spies**

Add `test_runner_uses_train_only_for_optimizer`, `test_runner_uses_validation_only_for_selection`, `test_runner_calls_test_once_after_selection`, `test_rejected_candidate_does_not_replace_baseline`, `test_runner_continues_after_rejection_until_iteration_limit`, `test_failure_rate_above_five_percent_aborts_subset`, and `test_phoenix_failure_preserves_selection_and_artifacts`.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/pipeline/test_experiment.py -v`

Expected: FAIL because `ExperimentRunner` does not exist.

- [ ] **Step 3: Implement baseline Validation and iterative Train/Validation loop**

Validate the baseline once, generate each candidate from Train feedback, validate the candidate, archive the decision, and update the current prompt only when accepted. Do not optimize after the final configured iteration.

- [ ] **Step 4: Implement one-time final Test and failure accounting**

Run Test exactly once after candidate selection completes. Count failed examples in the subset denominator and abort when failures exceed `0.05`; never silently drop failed rows from metrics.

- [ ] **Step 5: Add experiment-level spans and artifact finalization**

Use the approved span tree and ensure `ArtifactWriter.finalize` executes for successful experiments even when the observer has degraded to no-op.

- [ ] **Step 6: Run pipeline tests and verify pass**

Run: `python -m pytest tests/pipeline/test_experiment.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/prompt_learning/pipeline tests/pipeline
git commit -m "feat: orchestrate guarded prompt optimization experiments"
```

### Task 10: CLI, Default Configurations, and Offline Smoke Test

**Files:**
- Create: `src/prompt_learning/cli.py`
- Create: `configs/emotion.yaml`
- Create: `configs/financial.yaml`
- Create: `tests/test_cli.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `load_config`, task component factories, `ExperimentRunner`
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write failing CLI tests**

Cover `run --config`, a missing config path, invalid YAML, missing API key before network access, successful fake-run summary, and non-zero exit on an experiment failure. Assert error output identifies the corrective action.

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `python -m pytest tests/test_cli.py -v`

Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement component factories and CLI**

Map `task: emotion` and `task: financial` to their loaders, task runners, evaluators and selectors. Keep dependency construction in the CLI layer so the runner remains testable. Print the run ID, selected prompt path, Validation decision summary and Test report path.

- [ ] **Step 4: Add complete default YAML configurations**

Encode every approved default explicitly, including seed, split, sample counts, three retries, five-percent failure rate, emotion/financial gates, 40–800 character summary range, cache path, artifact root and Phoenix disabled by default.

- [ ] **Step 5: Ignore only generated local state**

Add run directories, SQLite cache files, Phoenix volume bind paths if any, `.env`, and Python build/test caches. Do not ignore the curated existing `artifacts/` evidence wholesale.

- [ ] **Step 6: Run CLI and full offline test suite**

Run: `python -m pytest -v`

Run: `prompt-learning --help`

Expected: all tests PASS and help exits `0` without credentials or network.

- [ ] **Step 7: Commit**

```bash
git add src/prompt_learning/cli.py configs tests/test_cli.py .gitignore
git commit -m "feat: expose reproducible experiment CLI"
```

### Task 11: Legacy Script Transition and Behavior Regression Tests

**Files:**
- Modify: `self_improving_emotion_classifier.py`
- Modify: `self_improving_financial_summarizer.py`
- Create: `tests/test_legacy_entrypoints.py`

**Interfaces:**
- Consumes: new CLI and default config paths
- Produces: deprecated compatibility entry points that delegate to `prompt_learning.cli.main`
- Produces: `emotion_legacy_args(argv: Sequence[str] | None) -> ConfigOverrides`
- Produces: `financial_legacy_args(argv: Sequence[str] | None) -> ConfigOverrides`

- [ ] **Step 1: Write failing compatibility tests**

Assert both legacy scripts print a deprecation message, translate their supported arguments into temporary config overrides without calling a network in `--help`, and return the delegated CLI exit code.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_legacy_entrypoints.py -v`

Expected: FAIL because the scripts still contain independent pipelines.

- [ ] **Step 3: Replace duplicate implementations with thin compatibility wrappers**

Preserve existing user-facing flags where they map cleanly. Route execution to the shared factories instead of retaining duplicate inference, evaluation or optimizer logic. Mark removal as a future major-version concern only in documentation, not as an unimplemented code path.

- [ ] **Step 4: Run compatibility and full tests**

Run: `python -m pytest tests/test_legacy_entrypoints.py -v`

Run: `python -m pytest -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add self_improving_emotion_classifier.py self_improving_financial_summarizer.py tests/test_legacy_entrypoints.py
git commit -m "refactor: route legacy scripts through shared pipeline"
```

### Task 12: User Documentation and Verified Example Artifacts

**Files:**
- Modify: `README.ko.md`
- Modify: `README.md`
- Modify: `requirements.txt`
- Create: `docs/architecture.md`
- Create: `artifacts/experiment-summary.md`
- Test: `tests/test_documentation.py`

**Interfaces:**
- Consumes: actual CLI output, test results, curated experiment metrics and Phoenix screenshots supplied by a real run
- Produces: Korean and Japanese user documentation and an architecture reference for reproducible experiments

- [ ] **Step 1: Write failing documentation checks**

Assert both READMEs contain the canonical CLI commands, Train/Validation/Test warning, Phoenix-optional behavior, artifact path, no unsupported claim that performance improved, and working relative links to committed evidence. Check the dependency declarations remain consistent between `requirements.txt` and `pyproject.toml` or replace `requirements.txt` with a documented generated compatibility file.

- [ ] **Step 2: Run documentation tests and verify failure**

Run: `python -m pytest tests/test_documentation.py -v`

Expected: FAIL because the current README documents the old scripts and result claims.

- [ ] **Step 3: Write architecture and user documentation**

Lead with the problem, guarded optimization design and measured outcome. Document local Phoenix startup, Phoenix-disabled execution, exact CLI commands, artifact meanings, evaluation limitations and why rejected candidates are valuable evidence.

- [ ] **Step 4: Run a low-cost real emotion and financial experiment**

Run with explicit small validation configs derived from the defaults while preserving valid split minima. Record the command, model identifiers, UTC timestamp, dataset revisions, cost, latency and result. Never commit raw third-party article/reference text or credentials.

- [ ] **Step 5: Capture curated evidence**

Copy only aggregate metrics, prompt versions, hashes, regression summaries and redacted Phoenix screenshots into committed experiment artifacts. If the real run cannot be performed due to credentials, do not invent metrics; leave the existing historical results clearly labeled as legacy and document the exact reproduction command.

- [ ] **Step 6: Run final verification**

Run: `python -m pytest -v`

Run: `python -m build`

Run: `docker compose config`

Run: `git diff --check`

Expected: tests PASS, package builds, Compose configuration validates, and no whitespace errors are reported.

- [ ] **Step 7: Commit**

```bash
git add README.ko.md README.md requirements.txt docs/architecture.md artifacts/experiment-summary.md tests/test_documentation.py
git commit -m "docs: document guarded prompt optimization experiments"
```

## Implementation Completion Checklist

## Follow-up progress (2026-10-23)

- [x] Increase emotion samples per label to 100.
- [x] Add reproducible seed configurations for 123 and 2024.
- [x] Add artifact-only per-run comparison CSV and Markdown output.
- [x] Add multi-seed summary statistics (mean, min, max, population standard deviation).
- [x] Document the comparison workflow and its no-fabrication rule.
- [ ] Configure verified model pricing and expose cost in comparison artifacts.
- [ ] Add failure-case dataset exports and Phoenix annotations.

- [ ] Run `python -m pytest -v` and retain the passing summary.
- [ ] Run `python -m build` and confirm both sdist and wheel are created.
- [ ] Run `prompt-learning --help` without an API key or network.
- [ ] Run `docker compose config` and confirm the Phoenix service is valid.
- [ ] Run `git status --short` and account for every remaining change.
- [ ] Compare the implementation against every section of the linked design spec.
- [ ] Use `superpowers:verification-before-completion` before reporting success.
- [ ] Use `superpowers:requesting-code-review` before merging or handing off the branch.
