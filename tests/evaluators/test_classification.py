"""Hand-checked classification scores and useful per-example feedback."""

import pandas as pd
import pytest


def examples():
    return pd.DataFrame({
        "example_id": ["s1", "s2", "j1", "j2", "l", "a", "f", "u"],
        "expected_label": ["sadness", "sadness", "joy", "joy", "love", "anger", "fear", "surprise"],
        "normalized_output": ["sadness", "joy", "joy", "invalid", "love", "anger", "fear", "surprise"],
        "raw_output": ["sadness", "joy", "joy", "disgust", "love", "anger", "fear", "surprise"],
        "input_tokens": [10] * 8,
        "output_tokens": [1] * 8,
        "latency_ms": [2.0] * 8,
        "error": [None, None, None, "invalid_output", None, None, None, None],
    }, index=[8, 9, 10, 11, 12, 13, 14, 15])


def test_metrics_count_invalid_as_wrong_without_seventh_target_class():
    from prompt_learning.evaluators.classification import evaluate_classification

    result = evaluate_classification(examples())
    assert result.aggregate["accuracy"] == 0.75
    assert result.aggregate["macro_f1"] == pytest.approx(31 / 36)
    assert result.aggregate["invalid_output_rate"] == 0.125
    assert list(result.aggregate["per_class"]) == ["sadness", "joy", "love", "anger", "fear", "surprise"]
    assert result.aggregate["per_class"]["sadness"] == {
        "precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "support": 2,
    }
    assert result.aggregate["per_class"]["joy"] == {
        "precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2,
    }
    assert result.aggregate["per_class"]["love"] == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1,
    }
    assert result.aggregate["confusion_matrix"] == {
        "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "matrix": [[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
    }


def test_feedback_explains_correct_confused_and_invalid_examples_without_mutation():
    from prompt_learning.evaluators.classification import evaluate_classification

    frame = examples()
    original = frame.copy(deep=True)
    result = evaluate_classification(frame)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(result.per_example[original.columns], original)
    assert result.per_example.is_correct.tolist() == [True, False, True, False, True, True, True, True]
    feedback = result.per_example.feedback.str.lower().tolist()
    assert "correct" in feedback[0] and "sadness" in feedback[0] and "preserve" in feedback[0]
    assert "sadness" in feedback[1] and "joy" in feedback[1] and "distinguish" in feedback[1]
    assert "joy" in feedback[3] and "exactly one" in feedback[3] and "label" in feedback[3]
    assert "s2" in result.feedback and "j2" in result.feedback
    assert "distinguish" in result.feedback.lower() and "exactly one" in result.feedback.lower()


def test_absent_classes_have_zero_scores_and_invalid_still_counts_as_miss():
    from prompt_learning.evaluators.classification import evaluate_classification

    frame = examples().iloc[[2, 3]]
    result = evaluate_classification(frame)
    assert result.aggregate["accuracy"] == 0.5
    assert result.aggregate["macro_f1"] == pytest.approx(1 / 9)
    assert result.aggregate["invalid_output_rate"] == 0.5
    assert result.aggregate["per_class"]["sadness"] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0,
    }


@pytest.mark.parametrize(("column", "value"), [
    ("expected_label", "invalid"), ("expected_label", "disgust"),
    ("normalized_output", "disgust"), ("normalized_output", None),
])
def test_evaluator_rejects_labels_outside_its_normalized_contract(column, value):
    from prompt_learning.evaluators.classification import evaluate_classification

    frame = examples()
    frame.loc[8, column] = value
    with pytest.raises(ValueError, match=column):
        evaluate_classification(frame)


def test_evaluator_rejects_empty_results_instead_of_misleading_zero_scores():
    from prompt_learning.evaluators.classification import evaluate_classification

    with pytest.raises(ValueError, match="empty"):
        evaluate_classification(examples().iloc[:0])


def test_evaluator_requires_normalized_predictions():
    from prompt_learning.evaluators.classification import evaluate_classification

    with pytest.raises(ValueError, match="normalized_output"):
        evaluate_classification(examples().drop(columns="normalized_output"))
