"""Shared interface for task-specific inference adapters."""

from typing import Protocol

import pandas as pd

from prompt_learning.gateway import ModelGateway
from prompt_learning.models import PromptCandidate


class TaskAdapter(Protocol):
    def run(
        self, frame: pd.DataFrame, prompt: PromptCandidate, gateway: ModelGateway,
        *, model: str, temperature: float = 0.0,
    ) -> pd.DataFrame: ...
