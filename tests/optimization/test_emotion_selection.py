"""Emotion gate rules, inclusive boundaries, and deterministic tie handling."""

from copy import deepcopy

import pandas as pd
import pytest

from prompt_learning.models import MetricResult


def metrics(*, macro_f1=0.8, accuracy=0.8, invalid=0.1, recalls=None, rows=None):
    per_class = {
        label: {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 10}
        for label in ("sadness", "joy", "love", "anger", "fear", "surprise")
    }
    for label, value in (recalls or {}).items():
        per_class[label]["recall"] = value
    return MetricResult({
        "macro_f1": macro_f1, "accuracy": accuracy, "invalid_output_rate": invalid,
        "per_class": per_class,
    }, pd.DataFrame(rows) if rows else pd.DataFrame())


def test_gate_accepts_candidate_that_passes_every_guardrail():
    from prompt_learning.optimization.selection import select_emotion_candidate

    decision = select_emotion_candidate(metrics(), metrics(macro_f1=0.81), "short", "long prompt")
    assert decision.accepted


@pytest.mark.parametrize(("candidate", "reason"), [
    (metrics(macro_f1=0.799999999), "macro_f1_decreased"),
    (metrics(accuracy=0.789999999), "accuracy_drop_exceeds_0.01"),
    (metrics(invalid=0.100000001), "invalid_output_rate_increased"),
    (metrics(recalls={"sadness": 0.75}), "recall_drop_at_least_0.05:sadness"),
    (metrics(recalls={"surprise": 0.74}), "recall_drop_at_least_0.05:surprise"),
])
def test_gate_rejects_each_metric_regression(candidate, reason):
    from prompt_learning.optimization.selection import select_emotion_candidate

    decision = select_emotion_candidate(metrics(), candidate, "long prompt", "short")
    assert not decision.accepted
    assert decision.reasons == (reason,)


def test_gate_reports_all_failed_rules_including_multiple_recall_drops():
    from prompt_learning.optimization.selection import select_emotion_candidate

    decision = select_emotion_candidate(
        metrics(), metrics(macro_f1=0.7, accuracy=0.7, invalid=0.2,
                           recalls={"sadness": 0.75, "joy": 0.5}),
        "long prompt", "short",
    )
    assert not decision.accepted
    assert decision.reasons == (
        "macro_f1_decreased", "accuracy_drop_exceeds_0.01", "invalid_output_rate_increased",
        "recall_drop_at_least_0.05:sadness", "recall_drop_at_least_0.05:joy",
    )


@pytest.mark.parametrize("candidate", [metrics(accuracy=0.79), metrics(recalls={"sadness": 0.750000001})])
def test_gate_accepts_accuracy_boundary_and_recall_drop_below_boundary(candidate):
    from prompt_learning.optimization.selection import select_emotion_candidate

    assert select_emotion_candidate(metrics(), candidate, "baseline", "candidate").accepted


def test_gate_rejects_accuracy_drop_just_above_boundary():
    from prompt_learning.optimization.selection import select_emotion_candidate

    decision = select_emotion_candidate(
        metrics(), metrics(accuracy=0.789999999999), "baseline", "candidate",
    )
    assert decision.reasons == ("accuracy_drop_exceeds_0.01",)


def test_gate_accepts_recall_drop_just_below_boundary():
    from prompt_learning.optimization.selection import select_emotion_candidate

    decision = select_emotion_candidate(
        metrics(), metrics(recalls={"sadness": 0.7500000000001}), "baseline", "candidate",
    )
    assert decision.accepted


def test_gate_rejects_exact_five_point_recall_drop_from_real_evaluations():
    from prompt_learning.evaluators.classification import evaluate_classification
    from prompt_learning.optimization.selection import select_emotion_candidate

    rows = []
    for label in ("sadness", "joy", "love", "anger", "fear", "surprise"):
        for index in range(60):
            if label == "sadness":
                before = "sadness" if index < 38 else "joy"
                after = "sadness" if index < 35 else "joy"
            elif label == "joy":
                before = "joy" if index < 35 else "sadness"
                after = "joy" if index < 38 else "sadness"
            else:
                before = after = label
            rows.append((f"{label}-{index}", label, before, after))
    frame = pd.DataFrame(rows, columns=[
        "example_id", "expected_label", "baseline_output", "candidate_output",
    ])
    baseline = evaluate_classification(frame.rename(
        columns={"baseline_output": "normalized_output"},
    ).drop(columns="candidate_output"))
    candidate = evaluate_classification(frame.rename(
        columns={"candidate_output": "normalized_output"},
    ).drop(columns="baseline_output"))

    assert baseline.aggregate["per_class"]["sadness"]["recall"] == 38 / 60
    assert candidate.aggregate["per_class"]["sadness"]["recall"] == 35 / 60
    assert baseline.aggregate["accuracy"] == candidate.aggregate["accuracy"]
    assert baseline.aggregate["macro_f1"] == candidate.aggregate["macro_f1"]
    decision = select_emotion_candidate(baseline, candidate, "long prompt", "short")
    assert not decision.accepted
    assert "recall_drop_at_least_0.05:sadness" in decision.reasons


@pytest.mark.parametrize(("baseline_prompt", "candidate_prompt", "accepted"), [
    ("long prompt", "short", True), ("short", "long prompt", False),
    ("first", "other", False), ("same", "same", False),
])
def test_exact_metric_ties_choose_shorter_prompt_then_baseline(baseline_prompt, candidate_prompt, accepted):
    from prompt_learning.optimization.selection import select_emotion_candidate

    assert select_emotion_candidate(metrics(), metrics(), baseline_prompt, candidate_prompt).accepted is accepted


def test_precision_improvement_is_not_an_exact_metric_tie():
    from prompt_learning.optimization.selection import select_emotion_candidate

    aggregate = deepcopy(metrics().aggregate)
    aggregate["per_class"]["sadness"]["precision"] = 0.9
    candidate = MetricResult(aggregate, pd.DataFrame())
    assert select_emotion_candidate(metrics(), candidate, "short", "much longer prompt").accepted


def test_regressions_compare_example_ids_and_exclude_raw_text():
    from prompt_learning.optimization.selection import select_emotion_candidate

    baseline = metrics(rows=[
        {"example_id": "a", "expected_label": "joy", "normalized_output": "joy", "text": "private a"},
        {"example_id": "b", "expected_label": "fear", "normalized_output": "joy", "text": "private b"},
    ])
    candidate = metrics(macro_f1=0.7, rows=[
        {"example_id": "b", "expected_label": "fear", "normalized_output": "fear", "text": "private b"},
        {"example_id": "a", "expected_label": "joy", "normalized_output": "invalid", "text": "private a"},
    ])
    decision = select_emotion_candidate(baseline, candidate, "baseline", "candidate")
    assert decision.regressions == ({
        "example_id": "a", "expected_label": "joy", "baseline_output": "joy", "candidate_output": "invalid",
    },)
    assert "private" not in str(decision)
