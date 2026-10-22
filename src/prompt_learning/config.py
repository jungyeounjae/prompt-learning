"""Load local experiment settings without constructing external clients."""

from dataclasses import dataclass, field
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    revision: str
    split: str = "train"
    samples_per_label: int = 20
    sample_count: int = 60


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2


@dataclass(frozen=True)
class ModelConfig:
    inference: str
    evaluator: str
    meta: str
    temperature: float = 0.0


@dataclass(frozen=True)
class RetryConfig:
    count: int = 3


@dataclass(frozen=True)
class PhoenixConfig:
    enabled: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    task: Literal["emotion", "financial"]
    dataset: DatasetConfig
    models: ModelConfig
    seed: int = 42
    split: SplitConfig = field(default_factory=SplitConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    failure_rate: float = 0.05
    phoenix: PhoenixConfig = field(default_factory=PhoenixConfig)
    max_iterations: int = 3
    cache_path: Path = Path(".cache/prompt_learning.sqlite3")
    artifact_root: Path = Path("runs")
    evaluation: Mapping[str, Any] = field(default_factory=dict)


def _invalid(key: str, value: Any, requirement: str) -> ValueError:
    return ValueError(f"{key}={value!r}: {requirement}")


def _mapping(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(key, value, "expected a mapping")
    return value


def _text(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(key, value, "expected a non-empty string")
    return value


def _integer(value: Any, key: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _invalid(key, value, f"expected an integer >= {minimum}")
    return value


def _number(value: Any, key: str, minimum: float, maximum: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not isfinite(value) or not minimum <= value <= maximum):
        raise _invalid(key, value, f"expected a finite number in [{minimum}, {maximum}]")
    return float(value)


def load_config(path: Path) -> ExperimentConfig:
    """Read YAML and report invalid keys/values before any external calls."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    data = _mapping(data, "config")
    task = data.get("task")
    if task not in ("emotion", "financial"):
        raise _invalid("task", task, "expected 'emotion' or 'financial'")
    dataset = _mapping(data.get("dataset"), "dataset")
    models = _mapping(data.get("models"), "models")
    split = _mapping(data.get("split", {}), "split")
    retry = _mapping(data.get("retry", {}), "retry")
    phoenix = _mapping(data.get("phoenix", {}), "phoenix")
    ratios = SplitConfig(**{
        key: _number(split.get(key, default), f"split.{key}", 0.0, 1.0)
        for key, default in (("train", 0.6), ("validation", 0.2), ("test", 0.2))
    })
    if not isclose(ratios.train + ratios.validation + ratios.test, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise _invalid("split", split, "ratios must sum to 1.0")
    enabled = phoenix.get("enabled", False)
    if type(enabled) is not bool:
        raise _invalid("phoenix.enabled", enabled, "expected a boolean")
    return ExperimentConfig(
        task=task,
        dataset=DatasetConfig(
            name=_text(dataset.get("name"), "dataset.name"),
            revision=_text(dataset.get("revision"), "dataset.revision"),
            split=_text(dataset.get("split", "train"), "dataset.split"),
            samples_per_label=_integer(dataset.get("samples_per_label", 20), "dataset.samples_per_label", 1),
            sample_count=_integer(dataset.get("sample_count", 60), "dataset.sample_count", 1),
        ),
        models=ModelConfig(
            inference=_text(models.get("inference"), "models.inference"),
            evaluator=_text(models.get("evaluator"), "models.evaluator"),
            meta=_text(models.get("meta"), "models.meta"),
            temperature=_number(models.get("temperature", 0.0), "models.temperature", 0.0, 2.0),
        ),
        seed=_integer(data.get("seed", 42), "seed"),
        split=ratios,
        retry=RetryConfig(count=_integer(retry.get("count", 3), "retry.count")),
        failure_rate=_number(data.get("failure_rate", 0.05), "failure_rate", 0.0, 1.0),
        phoenix=PhoenixConfig(enabled=enabled),
        max_iterations=_integer(data.get("max_iterations", 3), "max_iterations", 1),
        cache_path=Path(_text(data.get("cache_path", ".cache/prompt_learning.sqlite3"), "cache_path")),
        artifact_root=Path(_text(data.get("artifact_root", "runs"), "artifact_root")),
        evaluation=_mapping(data.get("evaluation", {}), "evaluation"),
    )
