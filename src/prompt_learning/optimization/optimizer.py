"""Isolate Prompt Learning candidate generation from experiment evaluation."""

from collections.abc import Callable
from typing import Any

import pandas as pd

from prompt_learning.models import MetricResult, PromptCandidate


class InvalidCandidateError(ValueError):
    """A generated prompt cannot proceed to Validation."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _default_optimizer_factory(**kwargs: Any) -> Any:
    from optimizer_sdk.prompt_learning_optimizer import PromptLearningOptimizer

    return PromptLearningOptimizer(**kwargs)


class PromptOptimizer:
    """Generate one prompt candidate using only Train outputs and feedback."""

    def __init__(
        self, *, api_key: str | None = None, meta_model: str = "gpt-4o",
        optimizer_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.meta_model = meta_model
        self.optimizer_factory = optimizer_factory or _default_optimizer_factory

    def optimize(
        self, current: PromptCandidate, training_result: MetricResult, required_placeholder: str,
        *, training_inputs: pd.DataFrame,
    ) -> PromptCandidate:
        source_columns = {"{text}": "text", "{article}": "article"}
        try:
            source_column = source_columns[required_placeholder]
        except KeyError as error:
            raise ValueError(f"Unsupported required placeholder: {required_placeholder}") from error

        scored = training_result.per_example
        for name, frame in (("training inputs", training_inputs), ("training result", scored)):
            if ("example_id" not in frame or frame["example_id"].isna().any()
                    or frame["example_id"].duplicated().any()):
                raise ValueError(f"{name} must contain a non-null, unique example_id")
        if source_column not in training_inputs:
            raise ValueError(f"training inputs must contain {source_column}")
        missing_ids = set(scored["example_id"]) - set(training_inputs["example_id"])
        if missing_ids:
            raise ValueError("training inputs must contain every evaluated example")

        source_by_id = training_inputs.set_index("example_id")[source_column]
        dataset = pd.DataFrame({
            source_column: scored["example_id"].map(source_by_id).tolist(),
            "output": scored["output"].tolist(),
            "feedback": scored["feedback"].tolist(),
        })
        optimizer = self.optimizer_factory(
            prompt=current.text, model_choice=self.meta_model, openai_api_key=self.api_key,
        )
        generated = optimizer.optimize(
            dataset=dataset, output_column="output", feedback_columns=["feedback"],
        )
        text = generated.strip() if isinstance(generated, str) else ""
        if not text:
            raise InvalidCandidateError("empty_output")
        if required_placeholder not in text:
            raise InvalidCandidateError("missing_placeholder")
        return PromptCandidate(text, current.version + 1, current.source_iteration + 1)
