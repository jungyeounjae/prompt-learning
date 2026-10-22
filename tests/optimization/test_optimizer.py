"""The SDK boundary receives Train feedback and yields valid candidates."""

from hashlib import sha256

import pandas as pd
import pytest

from prompt_learning.models import MetricResult, PromptCandidate
from prompt_learning.optimization.optimizer import InvalidCandidateError, PromptOptimizer


class FakeSDK:
    def __init__(self, candidate: str):
        self.candidate = candidate
        self.dataset = None
        self.options = None

    def optimize(self, *, dataset, output_column, feedback_columns):
        self.dataset = dataset.copy(deep=True)
        self.options = (output_column, feedback_columns)
        return self.candidate


def train_result():
    return MetricResult(
        {"accuracy": 0.5, "private_validation_score": 0.99},
        pd.DataFrame({
            "example_id": ["train-1", "train-2"],
            "output": ["joy", "sadness"],
            "feedback": ["Correct", "Expected joy"],
            "expected_label": ["joy", "joy"],
            "split": ["train", "train"],
        }),
        "aggregate feedback must not be forwarded",
    )


def test_sdk_receives_only_train_outputs_and_per_example_feedback():
    sdk = FakeSDK("Classify carefully: {text}")
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return sdk

    current = PromptCandidate("Classify: {text}", version=4, source_iteration=2)
    result = PromptOptimizer(api_key="test-key", meta_model="meta", optimizer_factory=factory).optimize(
        current, train_result(), "{text}",
    )

    assert factory_calls == [{"prompt": current.text, "model_choice": "meta", "openai_api_key": "test-key"}]
    pd.testing.assert_frame_equal(sdk.dataset, pd.DataFrame({
        "output": ["joy", "sadness"], "feedback": ["Correct", "Expected joy"],
    }))
    assert sdk.options == ("output", ["feedback"])
    assert result.text == "Classify carefully: {text}"


def test_candidate_version_iteration_and_hash_follow_current_candidate():
    sdk = FakeSDK("Revised {text}")
    optimizer = PromptOptimizer(optimizer_factory=lambda **kwargs: sdk)
    current = PromptCandidate("Original {text}", version=7, source_iteration=3)

    first = optimizer.optimize(current, train_result(), "{text}")
    second = optimizer.optimize(first, train_result(), "{text}")

    assert (first.version, first.source_iteration) == (8, 4)
    assert (second.version, second.source_iteration) == (9, 5)
    assert first.hash == sha256(b"Revised {text}").hexdigest()


@pytest.mark.parametrize(("generated", "reason"), [
    ("   ", "empty_output"),
    ("No variable", "missing_placeholder"),
])
def test_invalid_candidate_is_typed_for_archiving_before_validation(generated, reason):
    optimizer = PromptOptimizer(optimizer_factory=lambda **kwargs: FakeSDK(generated))

    with pytest.raises(InvalidCandidateError) as caught:
        optimizer.optimize(PromptCandidate("Original {article}", 1, 0), train_result(), "{article}")

    assert caught.value.reason == reason
