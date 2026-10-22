"""Canonical six-class metrics and per-example feedback."""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from prompt_learning.datasets.emotion import LABELS
from prompt_learning.models import MetricResult


def evaluate_classification(frame: pd.DataFrame) -> MetricResult:
    if frame.empty:
        raise ValueError("Cannot evaluate empty results")
    for column in ("example_id", "expected_label", "normalized_output"):
        if column not in frame:
            raise ValueError(f"Missing required column: {column}")
    if not frame.expected_label.isin(LABELS).all():
        raise ValueError("expected_label must be one of the canonical emotion labels")
    if not frame.normalized_output.isin((*LABELS, "invalid")).all():
        raise ValueError("normalized_output must be a canonical emotion label or invalid")

    expected = frame.expected_label.tolist()
    predicted = frame.normalized_output.tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=LABELS, zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }
    aggregate = {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": confusion_matrix(expected, predicted, labels=LABELS).tolist(),
        },
        "invalid_output_rate": predicted.count("invalid") / len(predicted),
    }

    per_example = frame.copy(deep=True)
    per_example["is_correct"] = per_example.expected_label.eq(per_example.normalized_output)
    feedback = []
    for row in per_example.itertuples(index=False):
        if row.is_correct:
            message = f"Correct {row.expected_label} classification; preserve this behavior."
        elif row.normalized_output == "invalid":
            message = f"Expected {row.expected_label}; return exactly one allowed label."
        else:
            message = (
                f"Expected {row.expected_label}, predicted {row.normalized_output}; "
                f"distinguish these emotions."
            )
        feedback.append(message)
    per_example["feedback"] = feedback
    summary = "\n".join(
        f"{example_id}: {message}"
        for example_id, message, correct in zip(
            per_example.example_id, feedback, per_example.is_correct,
        )
        if not correct
    )
    return MetricResult(aggregate, per_example, summary)
