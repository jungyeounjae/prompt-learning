from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from prompt_learning.config import ExperimentConfig, load_config
from prompt_learning.models import PromptCandidate


@pytest.fixture
def config_data():
    return {
        "task": "emotion",
        "dataset": {"name": "dair-ai/emotion", "revision": "main"},
        "models": {"inference": "fake", "evaluator": "fake", "meta": "fake"},
    }


def write_config(tmp_path, data):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_config_returns_typed_config(tmp_path, config_data):
    config = load_config(write_config(tmp_path, config_data))
    assert isinstance(config, ExperimentConfig)
    assert config.task == "emotion"
    assert config.dataset.name == "dair-ai/emotion"
    assert config.dataset.revision == "main"
    assert config.models.inference == "fake"
    assert config.seed == 42
    assert config.retry.count == 3
    assert config.failure_rate == 0.05
    assert config.phoenix.enabled is False
    assert (config.split.train, config.split.validation, config.split.test) == (0.6, 0.2, 0.2)
    assert config.dataset.samples_per_label == 20
    assert config.dataset.sample_count == 60
    assert isinstance(config.cache_path, Path)
    with pytest.raises(FrozenInstanceError):
        config.seed = 1


def test_split_ratios_must_sum_to_one(tmp_path, config_data):
    config_data["split"] = {"train": 0.7, "validation": 0.2, "test": 0.2}
    with pytest.raises(ValueError, match=r"split.*0.7"):
        load_config(write_config(tmp_path, config_data))


def test_rejects_unknown_task(tmp_path, config_data):
    config_data["task"] = "unknown"
    with pytest.raises(ValueError, match="task.*unknown"):
        load_config(write_config(tmp_path, config_data))


@pytest.mark.parametrize("key,value", [("sample_count", 0), ("sample_count", -1), ("samples_per_label", 0)])
def test_rejects_non_positive_sample_count(tmp_path, config_data, key, value):
    config_data["dataset"][key] = value
    with pytest.raises(ValueError, match=rf"dataset.{key}.*{value}"):
        load_config(write_config(tmp_path, config_data))


def test_explicit_overrides_and_financial_task(tmp_path, config_data):
    config_data.update(task="financial", seed=7, max_iterations=2, failure_rate=0.02,
                       retry={"count": 1}, phoenix={"enabled": True})
    config = load_config(write_config(tmp_path, config_data))
    assert config.task == "financial"
    assert config.seed == 7
    assert config.max_iterations == 2
    assert config.failure_rate == 0.02
    assert config.retry.count == 1
    assert config.phoenix.enabled is True


@pytest.mark.parametrize("key", ["task", "dataset", "models"])
def test_rejects_missing_required_sections(tmp_path, config_data, key):
    del config_data[key]
    with pytest.raises(ValueError, match=key):
        load_config(write_config(tmp_path, config_data))


@pytest.mark.parametrize("section,key,value", [
    ("dataset", "revision", ""), ("models", "inference", ""),
    ("split", "train", -0.6), ("split", "train", float("nan")),
    ("retry", "count", -1), ("retry", "count", True),
    ("phoenix", "enabled", "false"),
])
def test_rejects_invalid_nested_values(tmp_path, config_data, section, key, value):
    config_data.setdefault(section, {})[key] = value
    with pytest.raises(ValueError, match=rf"{section}.{key}.*"):
        load_config(write_config(tmp_path, config_data))


@pytest.mark.parametrize("key,value", [("failure_rate", -0.1), ("failure_rate", 1.1),
                                      ("max_iterations", 0), ("seed", True)])
def test_rejects_invalid_execution_values(tmp_path, config_data, key, value):
    config_data[key] = value
    with pytest.raises(ValueError, match=key):
        load_config(write_config(tmp_path, config_data))


@pytest.mark.parametrize("data", [None, [], "text", {"task": "emotion", "dataset": []}])
def test_rejects_non_mapping_sections(tmp_path, data):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, data))


def test_reports_invalid_yaml(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("task: [", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML"):
        load_config(path)


def test_prompt_candidate_hash_tracks_content():
    candidate = PromptCandidate(text="hello", version=1, source_iteration=0)
    assert candidate.hash == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert PromptCandidate(text="goodbye", version=1, source_iteration=0).hash != candidate.hash
