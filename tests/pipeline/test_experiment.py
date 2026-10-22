"""Exercise orchestration with real metrics/artifacts and offline boundary spies."""

from contextlib import contextmanager
import csv
from dataclasses import replace
from functools import partial
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from prompt_learning.artifacts import ArtifactWriter
from prompt_learning.config import DatasetConfig, ExperimentConfig, ModelConfig
from prompt_learning.evaluators.classification import evaluate_classification
from prompt_learning.evaluators.summarization import evaluate_summaries
from prompt_learning.models import DatasetSplits, MetricResult, ModelResponse, PromptCandidate, SelectionDecision
from prompt_learning.observability.phoenix import NoOpObserver, PhoenixObserver
from prompt_learning.optimization.selection import select_financial_candidate
from prompt_learning.optimization.optimizer import InvalidCandidateError
from prompt_learning.pipeline import ExperimentRunner, SubsetFailureError
from prompt_learning.tasks.emotion import run_emotion_inference
from prompt_learning.tasks.financial_news import run_financial_inference


class ObserverSpy(NoOpObserver):
    def __init__(self):
        self.spans, self.datasets, self.experiments = [], [], []
        self.stack = []

    @contextmanager
    def _span(self, name, attributes):
        self.spans.append((name, dict(attributes), tuple(self.stack)))
        self.stack.append(name)
        try:
            yield self
        finally:
            self.stack.pop()

    def start_run(self, attributes):
        return self._span("experiment_run", attributes)

    def child(self, name, attributes):
        return self._span(name, attributes)

    def sync_dataset(self, name, version, examples):
        self.datasets.append((name, version, examples))
        return name

    def record_experiment(self, dataset_id, run):
        self.experiments.append((dataset_id, run))


def harness(tmp_path, *, accepted=(True, False, True), failures=None, observer=None):
    config = ExperimentConfig(
        task="emotion", dataset=DatasetConfig("offline-emotion", "revision-1"),
        models=ModelConfig("fake-inference", "fake-evaluator", "fake-meta"),
        max_iterations=len(accepted), artifact_root=tmp_path,
    )
    splits = DatasetSplits(*(
        pd.DataFrame({"example_id": [f"{split}-{i}" for i in range(20)],
                      "text": [f"private {split} source {i}" for i in range(20)],
                      "expected_label": ["joy"] * 20})
        for split in ("train", "validation", "test")
    ))
    events, optimizer_calls, selections, evaluations = [], [], [], []
    failures = failures or {}

    class Gateway:
        def complete(self, request, *, validator):
            text = request.messages[0]["content"]
            for split, count in failures.items():
                if f"private {split} source" in text and int(text.rsplit(" ", 1)[1]) < count:
                    raise RuntimeError("private external error")
            return ModelResponse("joy", 2, 1, 0.5)

    def task(frame, prompt, gateway, **kwargs):
        split = frame.example_id.iloc[0].split("-")[0]
        events.append(("inference", split, prompt.text))
        return run_emotion_inference(frame, prompt, gateway, **kwargs)

    def evaluator(frame):
        result = evaluate_classification(frame)
        evaluations.append(result)
        return result

    class Optimizer:
        def optimize(self, current, training_result, required_placeholder):
            optimizer_calls.append((current, training_result, required_placeholder))
            events.append(("optimize", current.text))
            return PromptCandidate(f"candidate-{len(optimizer_calls)} {{text}}", current.version + 1,
                                   len(optimizer_calls))

    def selector(baseline, candidate, baseline_prompt, candidate_prompt):
        selections.append((baseline, candidate, baseline_prompt, candidate_prompt))
        events.append(("selection", candidate_prompt))
        return SelectionDecision(accepted[len(selections) - 1], ("guardrail",))

    writer = ArtifactWriter(tmp_path)
    tracing = observer or ObserverSpy()
    runner = ExperimentRunner(
        dataset_loader=lambda supplied: splits, task_runner=task, evaluator=evaluator,
        optimizer=Optimizer(), selector=selector, gateway=Gateway(),
        baseline_prompt=PromptCandidate("baseline {text}", 0, 0),
        artifact_writer=writer, observer=tracing, run_id_factory=lambda: "run-1",
    )
    return SimpleNamespace(**locals())


def test_runner_uses_train_only_for_optimizer(tmp_path):
    h = harness(tmp_path)
    result = h.runner.run(h.config)
    assert len(h.optimizer_calls) == 3
    for _, metrics, placeholder in h.optimizer_calls:
        assert metrics.per_example.example_id.str.startswith("train-").all()
        assert "validation-" not in metrics.feedback and "test-" not in metrics.feedback
        assert placeholder == "{text}"
    assert result.candidate_attempts == 3
    assert result.manifest.config.max_iterations == 3


def test_runner_uses_validation_only_for_selection(tmp_path):
    h = harness(tmp_path)
    h.runner.run(h.config)
    for baseline, candidate, _, _ in h.selections:
        assert baseline.per_example.example_id.str.startswith("validation-").all()
        assert candidate.per_example.example_id.str.startswith("validation-").all()
    validation = [event for event in h.events if event[:2] == ("inference", "validation")]
    assert [event[2] for event in validation] == [
        "baseline {text}", "candidate-1 {text}", "candidate-2 {text}", "candidate-3 {text}",
    ]


def test_runner_calls_test_once_after_selection(tmp_path):
    h = harness(tmp_path)
    result = h.runner.run(h.config)
    assert h.events[-1] == ("inference", "test", "candidate-3 {text}")
    assert len([event for event in h.events if event[:2] == ("inference", "test")]) == 1
    assert len(h.selections) == 3
    assert result.selected_prompt.text == "candidate-3 {text}"
    assert result.test_metrics.per_example.example_id.str.startswith("test-").all()


def test_rejected_candidate_does_not_replace_baseline(tmp_path):
    h = harness(tmp_path, accepted=(False,))
    result = h.runner.run(h.config)
    assert result.selected_prompt.text == "baseline {text}"
    assert result.validation_metrics is result.baseline_validation_metrics
    assert result.decisions == (SelectionDecision(False, ("guardrail",)),)
    assert (result.artifact_path / "selected_prompt.txt").read_text() == "baseline {text}"
    assert (result.artifact_path / "iteration_1_prompt.txt").read_text() == "candidate-1 {text}"


def test_runner_continues_after_rejection_until_iteration_limit(tmp_path):
    h = harness(tmp_path, accepted=(False, False, True))
    result = h.runner.run(h.config)
    assert [call[0].text for call in h.optimizer_calls] == ["baseline {text}"] * 3
    assert [decision.accepted for decision in result.decisions] == [False, False, True]
    assert len(h.optimizer_calls) == h.config.max_iterations
    assert len([event for event in h.events if event[:2] == ("inference", "train")]) == 3


def test_accepted_candidate_becomes_next_training_and_selection_baseline(tmp_path):
    h = harness(tmp_path, accepted=(True, False, True))
    h.runner.run(h.config)
    assert [call[0].text for call in h.optimizer_calls] == [
        "baseline {text}", "candidate-1 {text}", "candidate-1 {text}",
    ]
    assert h.selections[1][0] is h.selections[0][1]
    assert h.selections[2][0] is h.selections[0][1]


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_failure_rate_above_five_percent_aborts_subset(tmp_path, split):
    h = harness(tmp_path, accepted=(True,), failures={split: 2})
    with pytest.raises(SubsetFailureError) as caught:
        h.runner.run(h.config)
    assert caught.value.split == split
    assert caught.value.failure_count == 2
    assert caught.value.example_count == 20
    assert not (h.writer.path / "manifest.json").exists()


def test_exactly_five_percent_keeps_failed_rows_in_metrics_denominator(tmp_path):
    h = harness(tmp_path, accepted=(True,), failures={"validation": 1})
    result = h.runner.run(h.config)
    assert len(result.validation_metrics.per_example) == 20
    assert result.validation_metrics.aggregate["accuracy"] == 0.95
    assert result.validation_metrics.aggregate["invalid_output_rate"] == 0.05


def test_config_cannot_relax_five_percent_failure_ceiling(tmp_path):
    h = harness(tmp_path, accepted=(True,), failures={"validation": 2})
    with pytest.raises(SubsetFailureError):
        h.runner.run(replace(h.config, failure_rate=1.0))


@pytest.mark.parametrize("boundary", ["task_runner", "evaluator"])
def test_dropped_rows_are_rejected_before_metrics_can_hide_failures(tmp_path, boundary):
    h = harness(tmp_path, accepted=(True,))
    if boundary == "task_runner":
        original = h.runner.task_runner
        h.runner.task_runner = lambda *args, **kwargs: original(*args, **kwargs).iloc[1:]
    else:
        original = h.runner.evaluator

        def drop(frame):
            result = original(frame)
            return MetricResult(result.aggregate, result.per_example.iloc[1:], result.feedback)

        h.runner.evaluator = drop
    with pytest.raises(ValueError, match="retain every example"):
        h.runner.run(h.config)


def test_runner_emits_approved_tree_and_only_safe_metadata(tmp_path):
    h = harness(tmp_path, accepted=(True,))
    result = h.runner.run(h.config)
    assert [(name, parents) for name, _, parents in h.tracing.spans] == [
        ("experiment_run", ()),
        ("dataset_load", ("experiment_run",)),
        ("baseline_validation", ("experiment_run",)),
        ("iteration_1", ("experiment_run",)),
        ("train_inference", ("experiment_run", "iteration_1")),
        ("feedback_generation", ("experiment_run", "iteration_1")),
        ("prompt_optimization", ("experiment_run", "iteration_1")),
        ("candidate_validation", ("experiment_run", "iteration_1")),
        ("selection", ("experiment_run", "iteration_1")),
        ("final_test", ("experiment_run",)),
    ]
    payload = repr((h.tracing.spans, h.tracing.datasets, h.tracing.experiments))
    for private in ("private", "baseline {text}", "candidate-1 {text}", "Correct joy classification"):
        assert private not in payload
    assert [name for name, _, _ in h.tracing.datasets] == [
        "offline-emotion.train", "offline-emotion.validation", "offline-emotion.test",
    ]
    assert [metadata["phase"] for _, metadata in h.tracing.experiments] == [
        "baseline", "candidate", "final",
    ]
    assert all(len(example.input_hash) == 64 for _, _, rows in h.tracing.datasets for example in rows)
    manifest = json.loads((result.artifact_path / "manifest.json").read_text())
    assert manifest["max_iterations"] == 1
    assert set(manifest["input_hashes"]) == {"train", "validation", "test"}
    assert manifest["finished_at"] is not None


@pytest.mark.parametrize("operation", ["enter", "exit", "dataset", "record"])
def test_phoenix_failure_preserves_selection_and_artifacts(tmp_path, caplog, operation):
    class Exporter:
        @contextmanager
        def start_span(self, name, attributes):
            if operation == "enter":
                raise RuntimeError("private credential")
            yield
            if operation == "exit":
                raise RuntimeError("private credential")

    class Client:
        def __init__(self):
            self.datasets = SimpleNamespace(create_dataset=self.create_dataset)
            self.experiments = SimpleNamespace(create=self.create_experiment)

        def create_dataset(self, **kwargs):
            if operation == "dataset":
                raise RuntimeError("private source")
            return {"id": "dataset-1"}

        def create_experiment(self, **kwargs):
            if operation == "record":
                raise RuntimeError("private output")

    h = harness(tmp_path, accepted=(True, False), observer=PhoenixObserver(Exporter(), Client()))
    result = h.runner.run(h.config)
    assert result.selected_prompt.text == "candidate-1 {text}"
    assert [decision.accepted for decision in result.decisions] == [True, False]
    assert (result.artifact_path / "manifest.json").exists()
    assert (result.artifact_path / "test_metrics.json").exists()
    assert (result.artifact_path / "report.md").exists()
    assert "private" not in caplog.text


def test_unhandled_observer_failure_cannot_interrupt_local_work(tmp_path, caplog):
    class BrokenObserver(NoOpObserver):
        def start_run(self, attributes):
            raise RuntimeError("private credential")

        def sync_dataset(self, *args):
            raise RuntimeError("private source")

        def record_experiment(self, *args):
            raise RuntimeError("private output")

    h = harness(tmp_path, accepted=(True,), observer=BrokenObserver())
    result = h.runner.run(h.config)
    assert result.selected_prompt.text == "candidate-1 {text}"
    assert (result.artifact_path / "manifest.json").exists()
    assert "private" not in caplog.text


def test_artifact_failure_remains_a_failed_experiment(tmp_path):
    h = harness(tmp_path, accepted=(True,))

    def fail(*args):
        raise OSError("disk unavailable")

    h.writer.finalize = fail
    with pytest.raises(OSError, match="disk unavailable"):
        h.runner.run(h.config)


@pytest.mark.parametrize("judge_fails", [False, True])
def test_financial_pipeline_uses_real_evaluator_and_counts_judge_failures(tmp_path, judge_fails):
    config = ExperimentConfig(
        task="financial", dataset=DatasetConfig("offline-news", "revision-1"),
        models=ModelConfig("fake-inference", "fake-evaluator", "fake-meta"),
        max_iterations=1, artifact_root=tmp_path,
    )
    summary = "The company described its operating performance and future plans."
    splits = DatasetSplits(*(
        pd.DataFrame({"example_id": [f"{split}-{i}" for i in range(20)],
                      "article": [f"{summary} private {split} article {i}" for i in range(20)],
                      "reference_summary": ["private reference"] * 20})
        for split in ("train", "validation", "test")
    ))
    training_results = []

    class Gateway:
        def complete(self, request, *, validator):
            return ModelResponse(summary, 4, 3, 0.5)

    class Optimizer:
        def optimize(self, current, training_result, required_placeholder):
            assert required_placeholder == "{article}"
            training_results.append(training_result)
            return PromptCandidate("Summarize: {article}", 1, 1)

    def judge(*, article, reference_summary, output):
        if judge_fails and "train article" in article and int(article.rsplit(" ", 1)[1]) < 2:
            raise ValueError("private judge response")
        return json.dumps({name: {"score": 5, "label": "excellent", "rationale": "Supported."}
                           for name in ("financial_figures", "conciseness", "faithfulness")})

    runner = ExperimentRunner(
        dataset_loader=lambda _: splits, task_runner=run_financial_inference,
        evaluator=partial(evaluate_summaries, evaluator=judge), optimizer=Optimizer(),
        selector=select_financial_candidate, gateway=Gateway(),
        baseline_prompt=PromptCandidate("Write a precise financial summary of: {article}", 0, 0),
        artifact_writer=ArtifactWriter(tmp_path), run_id_factory=lambda: "financial-run",
    )
    if judge_fails:
        with pytest.raises(SubsetFailureError) as caught:
            runner.run(config)
        assert caught.value.split == "train"
        assert caught.value.failure_count == 2
        assert caught.value.example_count == 20
        assert not training_results
    else:
        result = runner.run(config)
        assert result.selected_prompt.text == "Summarize: {article}"
        assert result.test_metrics.aggregate["overall_score"] == 100
        assert training_results[0].per_example.example_id.str.startswith("train-").all()
        assert json.loads((result.artifact_path / "test_metrics.json").read_text())["overall_score"] == 100


def test_observer_exit_failure_does_not_hide_application_failure(tmp_path):
    class BrokenSpan(ObserverSpy):
        @contextmanager
        def _span(self, name, attributes):
            try:
                yield self
            finally:
                raise RuntimeError("private telemetry error")

    h = harness(tmp_path, accepted=(True,), observer=BrokenSpan())
    error = ValueError("application configuration failure")

    def fail(config):
        raise error

    h.runner.dataset_loader = fail
    with pytest.raises(ValueError) as caught:
        h.runner.run(h.config)
    assert caught.value is error
    assert not (h.writer.path / "manifest.json").exists()


@pytest.mark.parametrize("reason", ["empty_output", "missing_placeholder", "private generated text"])
def test_invalid_candidate_is_archived_without_validation_and_later_attempt_runs(tmp_path, reason):
    h = harness(tmp_path, accepted=(True, True))
    original = h.runner.optimizer
    calls = []

    class InvalidThenValid:
        def optimize(self, current, training_result, required_placeholder):
            calls.append(current.text)
            if len(calls) == 1:
                raise InvalidCandidateError(reason)
            return original.optimize(current, training_result, required_placeholder)

    h.runner.optimizer = InvalidThenValid()
    result = h.runner.run(h.config)
    assert calls == ["baseline {text}", "baseline {text}"]
    assert result.candidate_attempts == 2
    assert result.selected_prompt.source_iteration == 2
    assert [decision.accepted for decision in result.decisions] == [False, True]
    assert [event[2] for event in h.events if event[:2] == ("inference", "validation")] == [
        "baseline {text}", "candidate-1 {text}",
    ]
    category = reason if reason in ("empty_output", "missing_placeholder") else "invalid_candidate"
    with (result.artifact_path / "iteration_history.csv").open(newline="") as file:
        history = list(csv.DictReader(file))
    assert [row["iteration"] for row in history] == ["0", "1", "2"]
    assert history[1] == {"iteration": "1", "prompt_version": "", "prompt_hash": "",
                          "accepted": "False", "reasons": f"optimization_error:{category}"}
    assert not (result.artifact_path / "iteration_1_prompt.txt").exists()
    assert not (result.artifact_path / "iteration_1_validation_metrics.json").exists()
    assert "private generated text" not in repr(h.tracing.experiments)


@pytest.mark.parametrize("text", ["", "private missing placeholder"])
def test_runner_rejects_malformed_injected_candidate_before_validation(tmp_path, text):
    h = harness(tmp_path, accepted=(True,))

    class Optimizer:
        def optimize(self, *args):
            return PromptCandidate(text, 1, 1)

    h.runner.optimizer = Optimizer()
    result = h.runner.run(h.config)
    assert result.selected_prompt.text == "baseline {text}"
    assert result.decisions[0].accepted is False
    assert len([event for event in h.events if event[:2] == ("inference", "validation")]) == 1
    assert not h.selections
