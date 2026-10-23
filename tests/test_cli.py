"""Exercise CLI wiring with real runners and only external boundaries replaced."""

import json
from pathlib import Path
from types import SimpleNamespace

import datasets
import pandas as pd
import pytest
import yaml

from prompt_learning import cli
from prompt_learning.optimization import optimizer


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "The company described its operating performance and future plans."


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PHOENIX_ENABLED", "false")


def config_path(tmp_path, task="emotion", **overrides):
    data = {
        "task": task,
        "dataset": {"name": "offline-data", "revision": "offline-revision"},
        "models": {"inference": "fake-inference", "evaluator": "fake-judge", "meta": "fake-meta"},
        "max_iterations": 1,
        "cache_path": str(tmp_path / "cache.sqlite3"),
        "artifact_root": str(tmp_path / "runs"),
    }
    data.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_help_needs_neither_credentials_nor_clients(monkeypatch, capsys):
    def forbidden(**kwargs):
        pytest.fail("help must not construct a model client")

    monkeypatch.setattr(cli, "OpenAI", forbidden)
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "run" in capsys.readouterr().out


def test_run_requires_config_argument(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["run"])
    assert caught.value.code != 0
    assert "--config" in capsys.readouterr().err


def test_missing_config_explains_how_to_fix_path(tmp_path, capsys):
    missing = tmp_path / "missing.yaml"
    assert cli.main(["run", "--config", str(missing)]) != 0
    error = capsys.readouterr().err
    assert str(missing) in error and "--config" in error


def test_invalid_yaml_explains_what_to_fix(tmp_path, capsys):
    path = tmp_path / "broken.yaml"
    path.write_text("task: [")
    assert cli.main(["run", "--config", str(path)]) != 0
    error = capsys.readouterr().err
    assert "YAML" in error and "Fix" in error


def test_missing_api_key_fails_before_external_access(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("missing key must fail before clients or datasets are accessed")

    monkeypatch.setattr(cli, "OpenAI", forbidden)
    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    assert cli.main(["run", "--config", str(config_path(tmp_path))]) != 0
    assert "export OPENAI_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "cache.sqlite3").exists()


@pytest.mark.parametrize("overrides,field", [
    ({"retry": {"count": 4}}, "retry.count"),
    ({"evaluation": {"accuracy_max_drop": 0.5}}, "evaluation.accuracy_max_drop"),
    ({"split": {"train": 1, "validation": 0, "test": 0}}, "split"),
])
def test_unsupported_settings_fail_before_external_access(tmp_path, monkeypatch, capsys, overrides, field):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")

    def forbidden(**kwargs):
        pytest.fail("invalid settings must fail before constructing clients")

    monkeypatch.setattr(cli, "OpenAI", forbidden)
    assert cli.main(["run", "--config", str(config_path(tmp_path, **overrides))]) != 0
    assert field in capsys.readouterr().err


@pytest.mark.parametrize("task,field", [
    ("emotion", "samples_per_label"), ("financial", "sample_count"),
])
def test_impossible_sample_counts_fail_before_constructing_clients(tmp_path, monkeypatch, capsys, task, field):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")

    def forbidden(**kwargs):
        pytest.fail("sample counts below the dataset minimum must fail before external access")

    monkeypatch.setattr(cli, "OpenAI", forbidden)
    path = config_path(tmp_path, task, dataset={"name": "offline", "revision": "v1", field: 4})
    assert cli.main(["run", "--config", str(path)]) != 0
    assert f"dataset.{field}" in capsys.readouterr().err


@pytest.mark.parametrize("ratios", [
    {"train": 0.98, "validation": 0.01, "test": 0.01},
    {"train": 0.01, "validation": 0.98, "test": 0.01},
    {"train": 0.01, "validation": 0.01, "test": 0.98},
])
def test_financial_empty_split_allocation_fails_before_external_access(tmp_path, monkeypatch, capsys, ratios):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")

    def forbidden(*args, **kwargs):
        pytest.fail("empty requested financial subsets must fail before factories or network")

    monkeypatch.setattr(cli, "build_runner", forbidden)
    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    path = config_path(tmp_path, "financial", split=ratios,
                       dataset={"name": "offline", "revision": "v1", "sample_count": 5})
    assert cli.main(["run", "--config", str(path)]) == 2
    error = capsys.readouterr().err
    assert "split" in error and "dataset.sample_count" in error
    assert "empty" in error and "Increase" in error
    assert not (tmp_path / "cache.sqlite3").exists()


def test_financial_preflight_accepts_nonempty_largest_remainder_allocation(tmp_path, monkeypatch):
    from prompt_learning.config import load_config

    monkeypatch.setattr(cli, "OpenAI", lambda **kwargs: SimpleNamespace())
    path = config_path(tmp_path, "financial",
                       split={"train": 0.1, "validation": 0.45, "test": 0.45},
                       dataset={"name": "offline", "revision": "v1", "sample_count": 5})
    runner = cli.build_runner(load_config(path), api_key="offline-key")
    assert runner.gateway.cache_path.exists()


def offline_boundaries(monkeypatch, task, *, fail=False):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")
    requests = []

    def load_dataset(name, *, revision, split):
        assert (name, revision, split) == ("offline-data", "offline-revision", "train")
        if fail:
            raise RuntimeError("private remote response with secret")
        if task == "emotion":
            frame = pd.DataFrame([
                {"text": f"{label} example {i}", "label": label}
                for label in ("sadness", "joy", "love", "anger", "fear", "surprise")
                for i in range(20)
            ])
        else:
            frame = pd.DataFrame({"article": [f"{SUMMARY} Article {i}" for i in range(60)],
                                  "reference_summary": [SUMMARY] * 60})
        return SimpleNamespace(to_pandas=lambda: frame)

    def create(**kwargs):
        requests.append(kwargs)
        if kwargs["model"] == "fake-judge":
            assert kwargs["response_format"] == {"type": "json_object"}
            evidence = json.loads(kwargs["messages"][1]["content"])
            assert evidence["reference_summary"] == SUMMARY
            assert evidence["output"] == SUMMARY
            text = json.dumps({name: {"score": 5, "label": "excellent", "rationale": "Supported."}
                               for name in ("financial_figures", "conciseness", "faithfulness")})
        elif task == "emotion":
            text = kwargs["messages"][0]["content"].split("Text: ", 1)[1].split()[0]
        else:
            assert "reference_summary" not in kwargs["messages"][0]["content"]
            text = SUMMARY
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3))

    def client(**kwargs):
        assert kwargs == {"api_key": "offline-key", "max_retries": 0}
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    class FakeOptimizer:
        def __init__(self, **kwargs):
            assert kwargs["model_choice"] == "fake-meta"

        def optimize(self, *, dataset, output_column, feedback_columns):
            assert output_column == "output" and feedback_columns == ["feedback"]
            assert len(dataset) in (72, 36)
            return ("Classify. Text: {text}" if task == "emotion" else "Summarize: {article}")

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.setattr(cli, "OpenAI", client)
    monkeypatch.setattr(optimizer, "_default_optimizer_factory", FakeOptimizer)
    return requests


@pytest.mark.parametrize("task", ["emotion", "financial"])
def test_offline_run_builds_real_components_and_prints_summary(tmp_path, monkeypatch, capsys, task):
    requests = offline_boundaries(monkeypatch, task)
    assert cli.main(["run", "--config", str(config_path(tmp_path, task))]) == 0
    output = capsys.readouterr().out
    run_path, = (tmp_path / "runs" / task).iterdir()
    assert f"Run ID: {run_path.name}" in output
    assert str(run_path / "selected_prompt.txt") in output
    assert "Validation: 1 accepted, 0 rejected" in output
    assert str(run_path / "report.md") in output
    metrics = json.loads((run_path / "test_metrics.json").read_text())
    assert metrics["accuracy" if task == "emotion" else "overall_score"] == (1.0 if task == "emotion" else 100.0)
    assert requests


def test_experiment_failure_returns_nonzero_with_safe_corrective_action(tmp_path, monkeypatch, capsys):
    offline_boundaries(monkeypatch, "emotion", fail=True)
    assert cli.main(["run", "--config", str(config_path(tmp_path))]) != 0
    error = capsys.readouterr().err
    assert "Check" in error and "RuntimeError" in error
    assert "private remote" not in error and "secret" not in error


@pytest.mark.parametrize("task", ["emotion", "financial"])
def test_default_configs_can_build_the_task_without_network(task, tmp_path, monkeypatch):
    from prompt_learning.config import load_config

    monkeypatch.setattr(cli, "OpenAI", lambda **kwargs: SimpleNamespace())
    config = load_config(ROOT / "configs" / f"{task}.yaml")
    runner = cli.build_runner(config, api_key="offline-key")
    assert runner.baseline_prompt.text.count("{text}" if task == "emotion" else "{article}") == 1
    assert runner.gateway.max_retries == 3
    assert runner.gateway.cache_path == Path(".cache/prompt_learning.sqlite3")
    assert (config.split.train, config.split.validation, config.split.test) == (0.6, 0.2, 0.2)
    assert config.seed == 42 and config.failure_rate == 0.05
    assert config.dataset.samples_per_label == 20 and config.dataset.sample_count == 60
    assert config.phoenix.enabled is False
