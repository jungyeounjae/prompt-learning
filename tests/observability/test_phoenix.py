"""Optional network boundaries are replaced; sanitization and fallbacks stay real."""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from prompt_learning.config import DatasetConfig, ExperimentConfig, ModelConfig, PhoenixConfig
from prompt_learning.observability.phoenix import DatasetExample, NoOpObserver, build_observer


PRIVATE = ("private raw article", "private reference summary", "private full prompt", "sk-private-key")


@pytest.fixture
def config(monkeypatch):
    monkeypatch.delenv("PHOENIX_ENABLED", raising=False)
    return ExperimentConfig(
        task="financial", dataset=DatasetConfig("news", "revision-1"),
        models=ModelConfig("fake-model", "fake-model", "fake-model"),
        phoenix=PhoenixConfig(enabled=True),
    )


class Exporter:
    def __init__(self, failure=None):
        self.failure = failure
        self.spans = []
        self.stack = []

    @contextmanager
    def start_span(self, name, attributes):
        if self.failure == "start":
            raise RuntimeError(PRIVATE[-1])
        self.spans.append((name, dict(attributes), tuple(self.stack)))
        self.stack.append(name)
        try:
            yield
        finally:
            self.stack.pop()
        if self.failure == "end":
            raise RuntimeError(PRIVATE[-1])


class Client:
    def __init__(self, fail_dataset=False, fail_experiment=False):
        self.datasets = SimpleNamespace(create_dataset=self.create_dataset)
        self.experiments = SimpleNamespace(create=self.create_experiment)
        self.dataset_payloads = []
        self.experiment_payloads = []
        self.fail_dataset = fail_dataset
        self.fail_experiment = fail_experiment

    def create_dataset(self, **kwargs):
        if self.fail_dataset:
            raise RuntimeError(PRIVATE[0])
        self.dataset_payloads.append(kwargs)
        return SimpleNamespace(id="dataset-1")

    def create_experiment(self, **kwargs):
        if self.fail_experiment:
            raise RuntimeError(PRIVATE[1])
        self.experiment_payloads.append(kwargs)
        return {"id": "experiment-1"}


def observer(config, exporter=None, client=None):
    return build_observer(config, exporter_factory=lambda: exporter or Exporter(),
                          client_factory=lambda: client or Client())


def assert_private_absent(payload):
    assert all(value not in repr(payload) for value in PRIVATE)


def test_disabled_phoenix_returns_noop_observer(config, monkeypatch):
    def forbidden():
        pytest.fail("disabled observer initialized an external dependency")

    monkeypatch.setenv("PHOENIX_ENABLED", "false")
    disabled = build_observer(config, exporter_factory=forbidden, client_factory=forbidden)
    assert isinstance(disabled, NoOpObserver)
    with disabled.start_run({}) as run:
        with run.child("inference", {}):
            result = 42
    assert result == 42
    assert disabled.sync_dataset("news.train", "v1", []) is None
    assert disabled.record_experiment(None, {}) is None
    monkeypatch.delenv("PHOENIX_ENABLED")
    assert isinstance(build_observer(replace(config, phoenix=PhoenixConfig(False)),
                                    exporter_factory=forbidden), NoOpObserver)


def test_initialization_failure_returns_noop_and_warns(config, caplog):
    def broken():
        raise RuntimeError(PRIVATE[-1])

    failed = build_observer(config, exporter_factory=broken)
    assert isinstance(failed, NoOpObserver)
    assert len(caplog.records) == 1
    assert "install" in caplog.text.lower()
    assert_private_absent(caplog.text)


@pytest.mark.parametrize("failure", ["start", "end"])
def test_span_export_failure_does_not_change_wrapped_result(config, tmp_path, caplog, failure):
    tracing = observer(config, Exporter(failure))

    def work():
        with tracing.start_run({"run_id": "run-1"}) as run:
            with run.child("inference", {}):
                (tmp_path / "result.json").write_text('{"selected": "baseline"}')
                return "baseline"

    assert work() == "baseline"
    assert work() == "baseline"
    assert (tmp_path / "result.json").read_text() == '{"selected": "baseline"}'
    assert len(caplog.records) == 1
    assert "endpoint" in caplog.text.lower()
    assert_private_absent(caplog.text)


def test_observer_records_required_safe_attributes(config):
    exporter = Exporter()
    tracing = observer(config, exporter)
    safe = {"run_id": "run-1", "task": "financial", "split": "validation",
            "example_id": "example-1", "input_hash": "a" * 64, "reference_id": "ref-1",
            "prompt_hash": "b" * 64, "model": "fake-model", "input_tokens": 12,
            "output_tokens": 4, "cost_usd": 0.001, "latency_ms": 20.5,
            "accepted": False, "iteration": 2, "phase": "candidate", "mean_faithfulness": 0.9}
    unsafe = dict(zip(("article", "reference_summary", "prompt", "api_key"), PRIVATE))
    unsafe.update({"metadata": {"api_key": PRIVATE[-1]}, "arbitrary_id": PRIVATE[0]})
    with tracing.start_run({**safe, **unsafe}) as run:
        with run.child("evaluation", {**safe, **unsafe}):
            pass
        with run.child(PRIVATE[0], {}):
            pass
    assert exporter.spans[0][1] == safe
    assert exporter.spans[1] == ("evaluation", safe, ("experiment",))
    assert_private_absent(exporter.spans)


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_dataset_sync_uses_hashes_and_reference_ids_only(config, split):
    client = Client()
    tracing = observer(config, client=client)
    rows = [{"example_id": "example-1", "input_hash": "a" * 64,
             "reference_id": "ref-1", **dict(zip(
                 ("article", "reference_summary", "prompt", "api_key"), PRIVATE))},
            DatasetExample("example-2", "b" * 64, "ref-2")]
    assert tracing.sync_dataset(f"news.{split}", "revision-1", rows) == "dataset-1"
    payload = client.dataset_payloads[0]
    assert payload["name"] == f"news.{split}@revision-1"
    assert payload["inputs"] == [{"input_hash": "a" * 64}, {"input_hash": "b" * 64}]
    assert payload["outputs"] == [{"reference_id": "ref-1"}, {"reference_id": "ref-2"}]
    assert payload["metadata"] == [{"example_id": "example-1", "version": "revision-1"},
                                   {"example_id": "example-2", "version": "revision-1"}]
    assert_private_absent(client.dataset_payloads)


def test_dataset_failure_warns_once_and_does_not_disable_experiments(config, caplog):
    client = Client(fail_dataset=True)
    tracing = observer(config, client=client)
    for _ in range(2):
        assert tracing.sync_dataset("news.train", "v1", []) is None
    tracing.record_experiment("existing-dataset", {"run_id": "run-1", "phase": "baseline"})
    assert len(client.experiment_payloads) == 1
    assert len(caplog.records) == 1
    assert_private_absent(caplog.text)


def test_experiment_record_failure_only_warns(config, caplog):
    client = Client(fail_experiment=True)
    tracing = observer(config, client=client)
    for _ in range(2):
        assert tracing.record_experiment("dataset-1", {"run_id": "run-1"}) is None
    assert tracing.sync_dataset("news.test", "v1", []) == "dataset-1"
    assert len(caplog.records) == 1
    assert_private_absent(caplog.text)


@pytest.mark.parametrize("phase", ["baseline", "candidate", "final"])
def test_experiment_records_only_safe_aggregates(config, phase):
    client = Client()
    tracing = observer(config, client=client)
    tracing.record_experiment("dataset-1", {"run_id": "run-1", "phase": phase,
                              "prompt_hash": "b" * 64, "accuracy": 0.8,
                              "feedback": PRIVATE[0], "prompt": PRIVATE[2],
                              "per_example": [{"reference": PRIVATE[1]}], "api_key": PRIVATE[3]})
    assert client.experiment_payloads == [{
        "dataset_id": "dataset-1", "experiment_name": f"run-1-{phase}",
        "experiment_metadata": {"run_id": "run-1", "phase": phase,
                                "prompt_hash": "b" * 64, "accuracy": 0.8}, "timeout": 5,
    }]
    assert_private_absent(client.experiment_payloads)


def test_user_exception_is_preserved_without_exporting_its_text(config):
    exporter = Exporter("end")
    tracing = observer(config, exporter)
    error = ValueError(PRIVATE[0])
    with pytest.raises(ValueError) as caught:
        with tracing.start_run({}):
            raise error
    assert caught.value is error
    assert_private_absent(exporter.spans)


def test_missing_client_capabilities_degrade_independently(config, caplog):
    tracing = observer(config, client=object())
    for _ in range(2):
        assert tracing.sync_dataset("news.train", "v1", []) is None
        assert tracing.record_experiment("dataset-1", {}) is None
    assert len(caplog.records) == 2
    assert "dataset" in caplog.text.lower()
    assert "experiment" in caplog.text.lower()


def test_missing_dataset_id_skips_experiment_without_warning(config, caplog):
    client = Client()
    assert observer(config, client=client).record_experiment(None, {}) is None
    assert client.experiment_payloads == []
    assert not caplog.records


def test_production_tracer_does_not_export_environment_resources(config, monkeypatch):
    transport_module = pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    from opentelemetry.sdk.trace.export import SpanExportResult

    spans = []

    class Transport:
        def __init__(self, **kwargs):
            pass

        def export(self, batch):
            spans.extend(batch)
            return SpanExportResult.SUCCESS

    monkeypatch.setattr(transport_module, "OTLPSpanExporter", Transport)
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", f"article={PRIVATE[0]},api_key={PRIVATE[-1]}")
    tracing = build_observer(config, client_factory=Client)
    with tracing.start_run({"run_id": "run-1"}) as run:
        with run.child("inference", {"input_tokens": 12}):
            pass
    assert len(spans) == 2
    assert spans[0].parent.span_id == spans[1].context.span_id
    assert_private_absent([dict(span.resource.attributes) for span in spans])


def test_real_transport_error_logs_one_safe_warning(config, monkeypatch, caplog):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    import requests

    def fail_request(self, *args, **kwargs):
        raise ValueError(PRIVATE[-1])

    monkeypatch.setattr(requests.Session, "request", fail_request)
    tracing = build_observer(config, client_factory=Client)
    for _ in range(2):
        with tracing.start_run({}):
            result = 42
    assert result == 42
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert_private_absent(caplog.text)


def test_mislabeled_raw_text_cannot_be_exported_as_hashes(config, caplog):
    exporter, client = Exporter(), Client()
    tracing = observer(config, exporter, client)
    with tracing.start_run({"input_hash": PRIVATE[0], "prompt_hash": PRIVATE[2]}):
        pass
    assert exporter.spans[0][1] == {}
    assert tracing.sync_dataset("news.train", "v1", [
        DatasetExample("example-1", PRIVATE[0], "ref-1"),
    ]) is None
    assert client.dataset_payloads == []
    assert len(caplog.records) == 1
    assert_private_absent([exporter.spans, caplog.text])
