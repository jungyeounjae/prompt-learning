"""Keep Train optimization, Validation selection, and final Test isolated."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import logging
from math import isfinite
from pathlib import Path
from typing import Callable, ContextManager, Iterator, Protocol
from uuid import uuid4

import pandas as pd

from prompt_learning.artifacts import ArtifactWriter
from prompt_learning.config import ExperimentConfig
from prompt_learning.datasets.base import content_identity
from prompt_learning.gateway import ModelGateway
from prompt_learning.models import DatasetSplits, MetricResult, PromptCandidate, RunManifest, SelectionDecision
from prompt_learning.observability.phoenix import DatasetExample, NoOpObserver, Observer, RunSpan
from prompt_learning.optimization.optimizer import InvalidCandidateError


logger = logging.getLogger(__name__)
_METADATA_METRICS = (
    "accuracy", "macro_f1", "invalid_output_rate", "overall_score", "mean_financial_figures",
    "mean_conciseness", "mean_faithfulness", "financial_figures_not_applicable_count",
    "severe_hallucination_count", "failure_count",
)


class Optimizer(Protocol):
    def optimize(
        self, current: PromptCandidate, training_result: MetricResult, required_placeholder: str,
    ) -> PromptCandidate: ...


class SubsetFailureError(RuntimeError):
    """A subset exceeded its allowed failure rate, retaining its full denominator."""

    def __init__(self, split: str, failure_count: int, example_count: int) -> None:
        self.split = split
        self.failure_count = failure_count
        self.example_count = example_count
        super().__init__(f"{split} subset failed: {failure_count}/{example_count} examples failed")


@dataclass(frozen=True)
class ExperimentResult:
    manifest: RunManifest
    baseline_prompt: PromptCandidate
    selected_prompt: PromptCandidate
    baseline_validation_metrics: MetricResult
    validation_metrics: MetricResult
    test_metrics: MetricResult
    decisions: tuple[SelectionDecision, ...]
    candidate_attempts: int

    @property
    def artifact_path(self) -> Path:
        return self.manifest.artifact_path


def _retain_examples(source: pd.DataFrame, result: pd.DataFrame) -> None:
    if ("example_id" not in result or result.example_id.isna().any()
            or result.example_id.duplicated().any() or len(result) != len(source)
            or set(result.example_id) != set(source.example_id)):
        raise ValueError("Task and evaluator results must retain every example exactly once")


def _failed_ids(frame: pd.DataFrame) -> set:
    failed = pd.Series(False, index=frame.index)
    if "error" in frame:
        failed |= frame.error.notna() & frame.error.ne("")
    if "evaluation_failed" in frame:
        failed |= frame.evaluation_failed.fillna(False).astype(bool)
    if "normalized_output" in frame:
        failed |= frame.normalized_output.eq("invalid")
    return set(frame.loc[failed, "example_id"])


def _metric_metadata(result: MetricResult) -> dict[str, object]:
    return {key: value for key in _METADATA_METRICS
            if type(value := result.aggregate.get(key)) in (int, float) and isfinite(value)}


class ExperimentRunner:
    """Run one experiment using preconfigured, replaceable components.

    ``max_iterations`` counts candidate-generation attempts *after* the separate
    baseline Validation pass. Every attempt starts from the currently accepted
    prompt and performs one Train pass; final Test runs once after all attempts.
    The evaluator is a single-frame callable (bind a financial judge beforehand).
    The observer receives only purpose-built metadata mappings, never this result.
    """

    def __init__(
        self, *, dataset_loader: Callable[[ExperimentConfig], DatasetSplits],
        task_runner: Callable[..., pd.DataFrame], evaluator: Callable[[pd.DataFrame], MetricResult],
        optimizer: Optimizer, selector: Callable[[MetricResult, MetricResult, str, str], SelectionDecision],
        gateway: ModelGateway, baseline_prompt: PromptCandidate, artifact_writer: ArtifactWriter,
        observer: Observer | None = None, run_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.dataset_loader = dataset_loader
        self.task_runner = task_runner
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.selector = selector
        self.gateway = gateway
        self.baseline_prompt = baseline_prompt
        self.artifact_writer = artifact_writer
        self.observer = observer if observer is not None else NoOpObserver()
        self.run_id_factory = run_id_factory or (lambda: uuid4().hex)
        self._observer_warned = False

    def _warn_observer(self) -> None:
        if not self._observer_warned:
            self._observer_warned = True
            logger.warning("Experiment observer failed; continuing with local selection and artifacts.")

    def _observe(self, operation: Callable):
        try:
            return operation()
        except Exception:
            self._warn_observer()
            return None

    @contextmanager
    def _span(self, operation: Callable[[], ContextManager[RunSpan]]) -> Iterator[RunSpan]:
        try:
            manager = operation()
            span = manager.__enter__()
        except Exception:
            self._warn_observer()
            with NoOpObserver().start_run({}) as fallback:
                yield fallback
            return
        try:
            yield span
        finally:
            # Do not expose application exception text to telemetry, or allow an
            # instrumentation exit failure to suppress the application exception.
            try:
                manager.__exit__(None, None, None)
            except Exception:
                self._warn_observer()

    def _infer(
        self, frame: pd.DataFrame, prompt: PromptCandidate, config: ExperimentConfig,
    ) -> pd.DataFrame:
        outputs = self.task_runner(
            frame.copy(deep=True), prompt, self.gateway,
            model=config.models.inference, temperature=config.models.temperature,
        )
        _retain_examples(frame, outputs)
        return outputs

    def _evaluate(
        self, frame: pd.DataFrame, outputs: pd.DataFrame, split: str, config: ExperimentConfig,
    ) -> MetricResult:
        failures = _failed_ids(outputs)
        metrics = self.evaluator(outputs.copy(deep=True))
        _retain_examples(frame, metrics.per_example)
        failures |= _failed_ids(metrics.per_example)
        # A configuration may tighten, but cannot relax, the specified 5% gate.
        if len(failures) / len(frame) > min(config.failure_rate, 0.05):
            raise SubsetFailureError(split, len(failures), len(frame))
        return metrics

    def _run_subset(
        self, frame: pd.DataFrame, prompt: PromptCandidate, split: str, config: ExperimentConfig,
    ) -> MetricResult:
        return self._evaluate(frame, self._infer(frame, prompt, config), split, config)

    def _record(
        self, dataset_id: str | None, run_id: str, phase: str, prompt: PromptCandidate,
        metrics: MetricResult, *, iteration: int, decision: SelectionDecision | None = None,
    ) -> None:
        metadata = {
            "run_id": run_id, "phase": phase, "prompt_hash": prompt.hash,
            "prompt_version": prompt.version, "iteration": iteration, **_metric_metadata(metrics),
        }
        if decision is not None:
            metadata.update(accepted=decision.accepted, regression_count=len(decision.regressions))
        self._observe(lambda: self.observer.record_experiment(dataset_id, metadata))

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        if type(config.max_iterations) is not int or config.max_iterations < 1:
            raise ValueError("max_iterations must be an integer >= 1")
        run_id = self.run_id_factory()
        started_at = datetime.now(timezone.utc)
        path = self.artifact_writer.start_run(config.task, run_id)
        attributes = {
            "run_id": run_id, "task": config.task, "seed": config.seed,
            "dataset_name": config.dataset.name,
            "dataset_revision": config.dataset.revision, "inference_model": config.models.inference,
            "evaluator_model": config.models.evaluator, "meta_model": config.models.meta,
            "train_ratio": config.split.train, "validation_ratio": config.split.validation,
            "test_ratio": config.split.test,
        }
        with self._span(lambda: self.observer.start_run(attributes)) as root:
            with self._span(lambda: root.child("dataset_load", {
                "dataset_name": config.dataset.name, "dataset_revision": config.dataset.revision,
                "configured_sample_count": config.dataset.sample_count,
                "configured_samples_per_label": config.dataset.samples_per_label,
                "train_ratio": config.split.train, "validation_ratio": config.split.validation,
                "test_ratio": config.split.test,
            })):
                splits = self.dataset_loader(config)
                dataset_ids, input_hashes = {}, {}
                seen = set()
                for split in ("train", "validation", "test"):
                    frame = getattr(splits, split)
                    if (frame.empty or "example_id" not in frame or frame.example_id.isna().any()
                            or frame.example_id.duplicated().any()):
                        raise ValueError("Dataset subsets require nonempty, unique example identities")
                    source_column = "text" if config.task == "emotion" else "article"
                    reference_column = "expected_label" if config.task == "emotion" else "reference_summary"
                    examples = [DatasetExample(
                        str(row["example_id"]), content_identity(row[source_column]),
                        sha256(str(row[reference_column]).encode("utf-8")).hexdigest(),
                    ) for row in frame.to_dict("records")]
                    hashes = tuple(example.input_hash for example in examples)
                    if seen.intersection(hashes):
                        raise ValueError("Dataset subsets must have disjoint input identities")
                    seen.update(hashes)
                    input_hashes[split] = hashes
                    dataset_ids[split] = self._observe(lambda: self.observer.sync_dataset(
                        f"{config.dataset.name}.{split}", config.dataset.revision, examples,
                    ))

            manifest = RunManifest(
                run_id, config.task, config, config.dataset.revision,
                {"inference": config.models.inference, "evaluator": config.models.evaluator,
                 "meta": config.models.meta}, config.seed, started_at, path, input_hashes=input_hashes,
                dataset_exclusions=splits.exclusions,
            )
            current = self.baseline_prompt
            with self._span(lambda: root.child("baseline_validation", {
                "split": "validation", "prompt_hash": current.hash, "prompt_version": current.version,
            })):
                baseline_metrics = self._run_subset(splits.validation, current, "validation", config)
            selected_metrics = baseline_metrics
            self.artifact_writer.write_iteration(0, current, baseline_metrics, None)
            self._record(dataset_ids["validation"], run_id, "baseline", current, baseline_metrics, iteration=0)
            decisions = []
            placeholder = "{text}" if config.task == "emotion" else "{article}"
            for iteration in range(1, config.max_iterations + 1):
                with self._span(lambda: root.child(f"iteration_{iteration}", {
                    "iteration": iteration, "prompt_hash": current.hash, "prompt_version": current.version,
                })) as step:
                    with self._span(lambda: step.child("train_inference", {"split": "train"})):
                        outputs = self._infer(splits.train, current, config)
                    with self._span(lambda: step.child("feedback_generation", {"split": "train"})):
                        training_metrics = self._evaluate(splits.train, outputs, "train", config)
                    try:
                        with self._span(lambda: step.child("prompt_optimization", {"model": config.models.meta})):
                            candidate = self.optimizer.optimize(current, training_metrics, placeholder)
                            if not isinstance(candidate.text, str) or not candidate.text.strip():
                                raise InvalidCandidateError("empty_output")
                            if placeholder not in candidate.text:
                                raise InvalidCandidateError("missing_placeholder")
                            candidate = replace(candidate, source_iteration=iteration)
                    except InvalidCandidateError as error:
                        category = (error.reason if error.reason in ("empty_output", "missing_placeholder")
                                    else "invalid_candidate")
                        decision = SelectionDecision(False, (f"optimization_error:{category}",))
                        self.artifact_writer.write_optimization_error(iteration, category)
                        decisions.append(decision)
                        self._observe(lambda: self.observer.record_experiment(dataset_ids["validation"], {
                            "run_id": run_id, "phase": "candidate", "iteration": iteration,
                            "accepted": False, "failure_count": 1,
                        }))
                        continue
                    with self._span(lambda: step.child("candidate_validation", {
                        "split": "validation", "prompt_hash": candidate.hash, "prompt_version": candidate.version,
                    })):
                        candidate_metrics = self._run_subset(splits.validation, candidate, "validation", config)
                    with self._span(lambda: step.child("selection", {"iteration": iteration})):
                        decision = self.selector(selected_metrics, candidate_metrics, current.text, candidate.text)
                    self.artifact_writer.write_iteration(iteration, candidate, candidate_metrics, decision)
                    decisions.append(decision)
                    self._record(dataset_ids["validation"], run_id, "candidate", candidate,
                                 candidate_metrics, iteration=iteration, decision=decision)
                    if decision.accepted:
                        current, selected_metrics = candidate, candidate_metrics

            with self._span(lambda: root.child("final_test", {
                "split": "test", "prompt_hash": current.hash, "prompt_version": current.version,
            })):
                test_metrics = self._run_subset(splits.test, current, "test", config)
            manifest = replace(manifest, finished_at=datetime.now(timezone.utc))
            self.artifact_writer.finalize(manifest, self.baseline_prompt, current, selected_metrics, test_metrics)
            result = ExperimentResult(
                manifest, self.baseline_prompt, current, baseline_metrics, selected_metrics,
                test_metrics, tuple(decisions), len(decisions),
            )
            self._record(dataset_ids["test"], run_id, "final", current,
                         test_metrics, iteration=config.max_iterations)
            return result
