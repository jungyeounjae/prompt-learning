"""Compatibility at the old command boundaries, with no external requests."""

import importlib.util
from pathlib import Path

import dotenv
import pytest

from prompt_learning import cli
from prompt_learning.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)


@pytest.fixture(params=["emotion", "financial"])
def legacy(request):
    filename = ("self_improving_emotion_classifier.py" if request.param == "emotion"
                else "self_improving_financial_summarizer.py")
    spec = importlib.util.spec_from_file_location(f"legacy_{request.param}", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return request.param, module


def test_help_deprecates_without_running_cli_or_constructing_clients(legacy, monkeypatch, capsys):
    task, module = legacy

    def forbidden(*args, **kwargs):
        pytest.fail("legacy --help must not execute experiments or construct clients")

    monkeypatch.setattr(cli, "main", forbidden)
    monkeypatch.setattr(cli, "OpenAI", forbidden)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0
    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower()
    assert f"configs/{task}.yaml" in captured.err
    assert "--iterations" in captured.out
    if task == "emotion":
        assert "per label" in captured.out


def test_omitted_legacy_flags_use_shared_defaults(legacy):
    task, module = legacy
    translate = getattr(module, f"{task}_legacy_args", None)
    assert callable(translate), "legacy entrypoints need an argument-to-config translator"
    assert translate([]) == {}


def test_emotion_argument_translation_rounds_total_up_to_balanced_labels():
    spec = importlib.util.spec_from_file_location("emotion_args", ROOT / "self_improving_emotion_classifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    translate = getattr(module, "emotion_legacy_args", None)
    assert callable(translate), "emotion options must translate to shared config overrides"
    assert translate(["--samples", "31", "--iterations", "2"]) == {
        "dataset": {"samples_per_label": 6}, "max_iterations": 2,
    }


def test_financial_argument_translation_preserves_nested_settings():
    spec = importlib.util.spec_from_file_location("financial_args", ROOT / "self_improving_financial_summarizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    translate = getattr(module, "financial_legacy_args", None)
    assert callable(translate), "financial options must translate to shared config overrides"
    assert translate([
        "--dataset", "offline-news", "--split", "validation", "--samples", "30",
        "--iterations", "2", "--summarizer-model", "inference",
        "--evaluator-model", "judge", "--meta-model", "meta", "--output-dir", "artifacts",
    ]) == {
        "dataset": {"name": "offline-news", "split": "validation", "sample_count": 30},
        "models": {"inference": "inference", "evaluator": "judge", "meta": "meta"},
        "max_iterations": 2, "artifact_root": "artifacts",
    }


@pytest.mark.parametrize("exit_code", [0, 2, 7])
def test_delegates_temporary_merged_config_and_returns_cli_status(
    legacy, monkeypatch, tmp_path, capsys, exit_code,
):
    task, module = legacy
    original = (ROOT / "configs" / f"{task}.yaml").read_bytes()
    monkeypatch.chdir(tmp_path)
    seen_paths = []

    def run(argv):
        assert argv[:2] == ["run", "--config"]
        path = Path(argv[2])
        seen_paths.append(path)
        config = load_config(path)
        assert config.task == task
        assert config.dataset.name == ("dair-ai/emotion" if task == "emotion"
                                       else "kdave/Indian_Financial_News")
        assert config.dataset.revision == "main"
        assert config.models.inference == ("gpt-4o-mini" if task == "emotion" else "custom-inference")
        assert config.models.evaluator == "gpt-4o"
        assert config.seed == 42 and config.max_iterations == 2
        assert config.evaluation
        if task == "emotion":
            assert config.dataset.samples_per_label == 6
        else:
            assert config.dataset.sample_count == 31
        return exit_code

    monkeypatch.setattr(cli, "main", run)
    argv = ["--samples", "31", "--iterations", "2"]
    if task == "financial":
        argv += ["--summarizer-model", "custom-inference"]
    assert module.main(argv) == exit_code
    assert len(seen_paths) == 1 and not seen_paths[0].exists()
    assert (ROOT / "configs" / f"{task}.yaml").read_bytes() == original
    assert "deprecated" in capsys.readouterr().err.lower()


def test_temporary_config_is_removed_when_delegate_raises(legacy, monkeypatch):
    _, module = legacy
    seen_paths = []

    def interrupted(argv):
        seen_paths.append(Path(argv[2]))
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "main", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.main([])
    assert len(seen_paths) == 1 and not seen_paths[0].exists()


def test_shared_cli_reports_missing_key_from_legacy_entrypoints(legacy, monkeypatch, tmp_path, capsys):
    _, module = legacy
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("missing key must fail before constructing clients")

    monkeypatch.setattr(cli, "OpenAI", forbidden)
    assert module.main([]) == 2
    error = capsys.readouterr().err
    assert "deprecated" in error.lower() and "export OPENAI_API_KEY" in error


@pytest.mark.parametrize("legacy,option,value,guidance", [
    ("emotion", "--target-accuracy", "0.95", "Validation"),
    ("emotion", "--output", "prompt.txt", "selected_prompt.txt"),
    ("financial", "--final-prompt", "prompt.txt", "selected_prompt.txt"),
], indirect=["legacy"])
def test_unmappable_flags_fail_with_migration_guidance(legacy, monkeypatch, capsys, option, value, guidance):
    _, module = legacy

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported options must fail before delegation")

    monkeypatch.setattr(cli, "main", forbidden)
    with pytest.raises(SystemExit) as caught:
        module.main([option, value])
    assert caught.value.code == 2
    assert guidance in capsys.readouterr().err


@pytest.mark.parametrize("option,value", [("--samples", "0"), ("--iterations", "0")])
def test_invalid_legacy_counts_fail_before_delegation(legacy, monkeypatch, capsys, option, value):
    _, module = legacy

    def forbidden(*args, **kwargs):
        pytest.fail("invalid counts must fail before delegation")

    monkeypatch.setattr(cli, "main", forbidden)
    with pytest.raises(SystemExit) as caught:
        module.main([option, value])
    assert caught.value.code == 2
    assert option in capsys.readouterr().err


def test_emotion_small_total_rejected_for_isolated_subsets(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("emotion_small", ROOT / "self_improving_emotion_classifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as caught:
        module.main(["--samples", "24"])
    assert caught.value.code == 2
    assert "5 per label" in capsys.readouterr().err
