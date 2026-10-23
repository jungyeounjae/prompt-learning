"""Shared values exchanged between task adapters and the experiment pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import pandas as pd

    from prompt_learning.config import ExperimentConfig


@dataclass(frozen=True)
class ModelRequest:
    task: str
    model: str
    messages: tuple[Mapping[str, Any], ...]
    temperature: float = 0.0
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool = False


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: Decimal
    output_per_million: Decimal


@dataclass(frozen=True)
class DatasetSplits:
    train: "pd.DataFrame"
    validation: "pd.DataFrame"
    test: "pd.DataFrame"
    exclusions: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptCandidate:
    text: str
    version: int
    source_iteration: int

    @property
    def hash(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MetricResult:
    aggregate: Mapping[str, Any]
    per_example: "pd.DataFrame"
    feedback: str = ""


@dataclass(frozen=True)
class SelectionDecision:
    accepted: bool
    reasons: tuple[str, ...] = ()
    regressions: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    task: str
    config: "ExperimentConfig"
    dataset_revision: str
    models: Mapping[str, str]
    seed: int
    started_at: datetime
    artifact_path: Path
    finished_at: datetime | None = None
    input_hashes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    dataset_exclusions: Mapping[str, int] = field(default_factory=dict)
