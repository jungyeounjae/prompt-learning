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


def training_inputs(source_column="text"):
    return pd.DataFrame({
        "example_id": ["train-1", "train-2"],
        source_column: ["first source", "second source"],
        "expected_label": ["private-1", "private-2"],
        "reference_summary": ["private summary 1", "private summary 2"],
        "split": ["train", "train"],
    })


@pytest.mark.parametrize(("placeholder", "source_column"), [
    ("{text}", "text"),
    ("{article}", "article"),
])
def test_sdk_receives_required_source_aligned_by_example_id_and_no_private_fields(
    placeholder, source_column,
):
    sdk = FakeSDK(f"Classify carefully: {placeholder}")
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return sdk

    current = PromptCandidate("Classify: {text}", version=4, source_iteration=2)
    reversed_result = train_result()
    reversed_result = MetricResult(
        reversed_result.aggregate,
        reversed_result.per_example.iloc[::-1].reset_index(drop=True),
        reversed_result.feedback,
    )
    result = PromptOptimizer(api_key="test-key", meta_model="meta", optimizer_factory=factory).optimize(
        current, reversed_result, placeholder, training_inputs=training_inputs(source_column),
    )

    assert factory_calls == [{"prompt": current.text, "model_choice": "meta", "openai_api_key": "test-key"}]
    pd.testing.assert_frame_equal(sdk.dataset, pd.DataFrame({
        source_column: ["second source", "first source"],
        "output": ["sadness", "joy"],
        "feedback": ["Expected joy", "Correct"],
    }))
    assert sdk.options == ("output", ["feedback"])
    assert result.text == f"Classify carefully: {placeholder}"


def test_candidate_version_iteration_and_hash_follow_current_candidate():
    sdk = FakeSDK("Revised {text}")
    optimizer = PromptOptimizer(optimizer_factory=lambda **kwargs: sdk)
    current = PromptCandidate("Original {text}", version=7, source_iteration=3)

    first = optimizer.optimize(current, train_result(), "{text}", training_inputs=training_inputs())
    second = optimizer.optimize(first, train_result(), "{text}", training_inputs=training_inputs())

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
        optimizer.optimize(
            PromptCandidate("Original {article}", 1, 0), train_result(), "{article}",
            training_inputs=training_inputs("article"),
        )

    assert caught.value.reason == reason


def test_unsupported_placeholder_fails_before_sdk_call():
    def unexpected_factory(**kwargs):
        raise AssertionError("SDK must not be created")

    optimizer = PromptOptimizer(optimizer_factory=unexpected_factory)

    with pytest.raises(ValueError, match="Unsupported required placeholder"):
        optimizer.optimize(
            PromptCandidate("Original {unknown}", 1, 0), train_result(), "{unknown}",
            training_inputs=training_inputs(),
        )


def test_missing_training_identity_fails_before_sdk_call():
    def unexpected_factory(**kwargs):
        raise AssertionError("SDK must not be created")

    optimizer = PromptOptimizer(optimizer_factory=unexpected_factory)
    inputs = training_inputs().iloc[:1].copy()

    with pytest.raises(ValueError, match="every evaluated example"):
        optimizer.optimize(
            PromptCandidate("Original {text}", 1, 0), train_result(), "{text}",
            training_inputs=inputs,
        )


@pytest.mark.parametrize("duplicate_in", ["training_inputs", "training_result"])
def test_duplicate_training_identity_fails_before_sdk_call(duplicate_in):
    def unexpected_factory(**kwargs):
        raise AssertionError("SDK must not be created")

    optimizer = PromptOptimizer(optimizer_factory=unexpected_factory)
    inputs = training_inputs()
    result = train_result()
    if duplicate_in == "training_inputs":
        inputs.loc[1, "example_id"] = "train-1"
    else:
        result.per_example.loc[1, "example_id"] = "train-1"

    with pytest.raises(ValueError, match="unique example_id"):
        optimizer.optimize(
            PromptCandidate("Original {text}", 1, 0), result, "{text}",
            training_inputs=inputs,
        )
