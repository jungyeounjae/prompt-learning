"""Fail-open Phoenix tracing with a deliberately small metadata export contract.

SDK imports are lazy. ``exporter_factory`` provides a ``start_span`` context
manager; ``client_factory`` provides Phoenix Client 3.5 dataset/experiment APIs.
The experiment payload is aggregate metadata, never a re-execution of a task.

Install ``.[phoenix]`` and start ``docker compose up -d`` for local use. YAML
``phoenix.enabled`` defaults to false; ``PHOENIX_ENABLED`` overrides it. The
collector base URL defaults to http://localhost:6006; set
``PHOENIX_COLLECTOR_ENDPOINT`` when changing Compose's ``PHOENIX_HTTP_PORT``.
``PHOENIX_OTLP_ENDPOINT`` can override the complete /v1/traces URL. Authentication
uses ``PHOENIX_API_KEY`` only as a transport credential, never an attribute.

Dataset names include the caller's version (for example news.train@revision-1).
Pass train/validation/test separately. ``record_experiment`` accepts a safe
mapping with run_id, phase (baseline/candidate/final), prompt_hash, and numeric
aggregates; it creates an experiment metadata record without uploading model
outputs, per-example evaluations, or raw reference text. Dataset/experiment
operations require a server compatible with Phoenix Client 3.5. Unsupported
methods and server failures leave local artifacts authoritative. No live-server
compatibility or per-example experiment replay is implied by the offline tests.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import logging
from math import isfinite
import os
import re
from typing import Callable, ContextManager, Iterator, Mapping, Protocol, Sequence

from prompt_learning.config import ExperimentConfig


logger = logging.getLogger(__name__)
_dependency_operation = ContextVar("prompt_learning_phoenix_dependency_operation", default=False)
_DEPENDENCY_LOG_NAMES = (
    "",  # Some dependency/factory diagnostics use the root logger directly.
    "phoenix.client.utils.config",
    "phoenix.client.utils.parse_env_headers",
    "opentelemetry.util.re",
    "opentelemetry.exporter.otlp.proto.http._common",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
)


class _DependencyLogFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.warned = False

    def filter(self, record: logging.LogRecord) -> bool:
        if not _dependency_operation.get():
            return True
        self.warned |= record.levelno >= logging.WARNING
        return False


@contextmanager
def _quiet_dependency_logs() -> Iterator[_DependencyLogFilter]:
    """Hide only the active SDK operation, restoring every logger afterwards.

    Client header parsing and OTLP transports may log credentials before they
    return or raise. Use a unique filter per operation so overlapping calls can
    remove their own filters without disturbing other operations or callers.
    """
    scoped_filter = _DependencyLogFilter()
    dependency_loggers = [logging.getLogger(name) for name in _DEPENDENCY_LOG_NAMES]
    token = _dependency_operation.set(True)
    try:
        for dependency_logger in dependency_loggers:
            dependency_logger.addFilter(scoped_filter)
        yield scoped_filter
    finally:
        for dependency_logger in dependency_loggers:
            dependency_logger.removeFilter(scoped_filter)
        _dependency_operation.reset(token)


@dataclass(frozen=True)
class DatasetExample:
    """Safe dataset identity. Hash source/reference text before constructing this."""

    example_id: str
    input_hash: str
    reference_id: str


# Task runners can supply this mapping without coupling their result model to an SDK.
ExperimentResult = Mapping[str, object]


class RunSpan(Protocol):
    def child(self, name: str, attributes: Mapping[str, object]) -> ContextManager["RunSpan"]: ...


class Observer(Protocol):
    def start_run(self, attributes: Mapping[str, object]) -> ContextManager[RunSpan]: ...

    def sync_dataset(
        self, name: str, version: str, examples: Sequence[DatasetExample],
    ) -> str | None: ...

    def record_experiment(self, dataset_id: str | None, run: ExperimentResult) -> None: ...


class SpanExporter(Protocol):
    def start_span(self, name: str, attributes: Mapping[str, object]) -> ContextManager[object]: ...


_TEXT_FIELDS = frozenset({
    "run_id", "task", "split", "example_id", "reference_id",
    "dataset_id", "dataset_revision", "model", "inference_model", "evaluator_model", "meta_model",
})
_HASH_FIELDS = frozenset({"input_hash", "prompt_hash"})
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_NUMBER_FIELDS = frozenset({
    "iteration", "prompt_version", "seed", "input_tokens", "output_tokens", "total_tokens",
    "cost_usd", "latency_ms", "accuracy", "macro_f1", "invalid_output_rate", "overall_score", "mean_length_score",
    "mean_compression_score", "mean_rouge_l_f1", "mean_faithfulness", "mean_conciseness",
    "mean_financial_figures", "failure_count", "failure_rate", "severe_hallucination_count",
    "financial_figures_not_applicable_count", "example_count", "regression_count",
})
_SPAN_NAMES = frozenset({
    "experiment", "dataset", "inference", "evaluation", "optimization", "selection",
    "baseline", "candidate", "final", "train", "validation", "test",
    "experiment_run", "dataset_load", "baseline_validation", "train_inference",
    "feedback_generation", "prompt_optimization", "candidate_validation", "final_test",
})


def _safe_attributes(attributes: Mapping[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in attributes.items():
        if key in _TEXT_FIELDS and isinstance(value, str):
            safe[key] = value
        elif key in _HASH_FIELDS and isinstance(value, str) and _SHA256.fullmatch(value):
            safe[key] = value
        elif key in _NUMBER_FIELDS and type(value) in (int, float) and isfinite(value):
            safe[key] = value
        elif key in ("accepted", "cached") and type(value) is bool:
            safe[key] = value
        elif key == "phase" and value in ("baseline", "candidate", "final"):
            safe[key] = value
    return safe


class _NoOpRunSpan:
    @contextmanager
    def child(self, name: str, attributes: Mapping[str, object]) -> Iterator[RunSpan]:
        yield self


class NoOpObserver:
    @contextmanager
    def start_run(self, attributes: Mapping[str, object]) -> Iterator[RunSpan]:
        yield _NoOpRunSpan()

    def sync_dataset(
        self, name: str, version: str, examples: Sequence[DatasetExample],
    ) -> str | None:
        return None

    def record_experiment(self, dataset_id: str | None, run: ExperimentResult) -> None:
        return None


class _PhoenixRunSpan:
    def __init__(self, observer: "PhoenixObserver") -> None:
        self.observer = observer

    def child(self, name: str, attributes: Mapping[str, object]) -> ContextManager[RunSpan]:
        return self.observer._span(name, attributes)


class PhoenixObserver:
    def __init__(self, exporter: SpanExporter, client: object) -> None:
        self.exporter = exporter
        self.client = client
        self._warned: set[str] = set()

    def _warn(self, operation: str) -> None:
        if operation not in self._warned:
            self._warned.add(operation)
            # Exception text can contain full HTTP bodies, inputs, or credentials.
            logger.warning(
                "Phoenix %s failed or is unsupported; check the Phoenix endpoint and "
                "installed phoenix extra/client compatibility. Local artifacts remain authoritative.",
                operation,
            )

    @contextmanager
    def _span(self, name: str, attributes: Mapping[str, object]) -> Iterator[RunSpan]:
        try:
            safe_name = (name if name in _SPAN_NAMES or re.fullmatch(r"iteration_[0-9]+", name)
                         else "span-" + sha256(name.encode()).hexdigest())
            manager = self.exporter.start_span(safe_name, _safe_attributes(attributes))
            manager.__enter__()
        except Exception:
            self._warn("span export")
            yield _NoOpRunSpan()
            return
        try:
            yield _PhoenixRunSpan(self)
        finally:
            try:
                # Never hand application exceptions to instrumentation: it may record
                # exception messages/stack traces containing source text or secrets.
                manager.__exit__(None, None, None)
            except Exception:
                self._warn("span export")

    def start_run(self, attributes: Mapping[str, object]) -> ContextManager[RunSpan]:
        return self._span("experiment_run", attributes)

    def sync_dataset(
        self, name: str, version: str, examples: Sequence[DatasetExample],
    ) -> str | None:
        try:
            identities = []
            for example in examples:
                identity = {
                    field: example[field] if isinstance(example, Mapping) else getattr(example, field)
                    for field in ("example_id", "input_hash", "reference_id")
                }
                if any(not isinstance(value, str) or not value for value in identity.values()):
                    raise ValueError("Expected safe string dataset identities")
                if not _SHA256.fullmatch(identity["input_hash"]):
                    raise ValueError("Expected SHA-256 input hash")
                identities.append(identity)
            dataset = self.client.datasets.create_dataset(
                name=f"{name}@{version}",
                inputs=[{"input_hash": row["input_hash"]} for row in identities],
                outputs=[{"reference_id": row["reference_id"]} for row in identities],
                metadata=[{"example_id": row["example_id"], "version": version} for row in identities],
                timeout=5,
            )
            identifier = dataset["id"] if isinstance(dataset, Mapping) else dataset.id
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Expected Phoenix dataset ID")
            return identifier
        except Exception:
            self._warn("dataset synchronization")
            return None

    def record_experiment(self, dataset_id: str | None, run: ExperimentResult) -> None:
        if dataset_id is None:
            return
        try:
            attributes = _safe_attributes(run)
            name = f"{attributes.get('run_id', 'run')}-{attributes.get('phase', 'final')}"
            self.client.experiments.create(
                dataset_id=dataset_id, experiment_name=name,
                experiment_metadata=attributes, timeout=5,
            )
        except Exception:
            self._warn("experiment recording")


def _endpoint() -> str:
    return os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")


def _build_client() -> object:
    from phoenix.client import Client

    return Client(base_url=_endpoint())


def _build_exporter() -> SpanExporter:
    # Keep optional imports here so Phoenix-disabled runs need no tracing packages.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import SpanExportResult

    headers = {}
    if key := os.environ.get("PHOENIX_API_KEY"):
        headers["authorization"] = f"Bearer {key}"
    transport = OTLPSpanExporter(
        endpoint=os.environ.get("PHOENIX_OTLP_ENDPOINT", _endpoint() + "/v1/traces"),
        headers=headers, timeout=5,
    )
    class CheckedProcessor(SpanProcessor):
        def on_end(self, span):
            # SimpleSpanProcessor logs and swallows export errors. Check the transport
            # directly so the adapter owns the single sanitized, actionable warning.
            with _quiet_dependency_logs():
                result = transport.export((span,))
            if result != SpanExportResult.SUCCESS:
                raise RuntimeError("Phoenix span transport failed")

        def shutdown(self):
            transport.shutdown()

        def force_flush(self, timeout_millis=30000):
            return transport.force_flush(timeout_millis)

    provider = TracerProvider(
        resource=Resource({"service.name": "prompt-learning", "openinference.project.name": "prompt-learning"}),
        shutdown_on_exit=False,
    )
    provider.add_span_processor(CheckedProcessor())
    tracer = provider.get_tracer("prompt_learning.observability")

    class TracingBackend:
        def start_span(self, name, attributes):
            return tracer.start_as_current_span(
                name, attributes=attributes, record_exception=False, set_status_on_exception=False,
            )

    return TracingBackend()


def build_observer(
    config: ExperimentConfig, *,
    exporter_factory: Callable[[], SpanExporter] | None = None,
    client_factory: Callable[[], object] | None = None,
) -> Observer:
    """Build only when enabled; PHOENIX_ENABLED overrides the YAML setting.

    Fake factories allow tests and alternate SDK versions without network calls.
    Each adapter operation emits at most one warning, without exception contents.
    """
    enabled = config.phoenix.enabled
    if "PHOENIX_ENABLED" in os.environ:
        enabled = os.environ["PHOENIX_ENABLED"].strip().lower() in ("true", "1", "yes", "on")
    if not enabled:
        return NoOpObserver()
    try:
        with _quiet_dependency_logs() as diagnostics:
            exporter = (exporter_factory or _build_exporter)()
            client = (client_factory or _build_client)()
        if diagnostics.warned:
            logger.warning(
                "Phoenix initialization ignored invalid configuration; check the Phoenix "
                "endpoint and URL-encode PHOENIX_CLIENT_HEADERS values. "
                "Dependency diagnostics were omitted to protect credentials.",
            )
        return PhoenixObserver(exporter, client)
    except Exception:
        logger.warning(
            "Phoenix initialization failed; install the phoenix extra and check the Phoenix "
            "endpoint/client compatibility. Continuing with local artifacts and no tracing.",
        )
        return NoOpObserver()
