"""Task-specific guardrails for prompt candidates."""

from decimal import Decimal
from fractions import Fraction

import pandas as pd

from prompt_learning.datasets.emotion import LABELS
from prompt_learning.models import MetricResult, SelectionDecision


def _count_based_recall(result: MetricResult, label: str) -> Fraction | None:
    """Recover the exact recall ratio when an evaluator confusion matrix exists."""
    matrix = result.aggregate.get("confusion_matrix")
    if matrix is None:
        return None
    position = matrix["labels"].index(label)
    support = result.aggregate["per_class"][label]["support"]
    return Fraction(matrix["matrix"][position][position], support) if support else Fraction(0)


def select_emotion_candidate(
    baseline: MetricResult, candidate: MetricResult,
    baseline_prompt: str, candidate_prompt: str,
) -> SelectionDecision:
    current = baseline.aggregate
    proposed = candidate.aggregate
    reasons = []
    if proposed["macro_f1"] < current["macro_f1"]:
        reasons.append("macro_f1_decreased")
    accuracy_drop = Decimal(str(current["accuracy"])) - Decimal(str(proposed["accuracy"]))
    if accuracy_drop > Decimal("0.01"):
        reasons.append("accuracy_drop_exceeds_0.01")
    if proposed["invalid_output_rate"] > current["invalid_output_rate"]:
        reasons.append("invalid_output_rate_increased")
    for label in LABELS:
        baseline_recall = _count_based_recall(baseline, label)
        candidate_recall = _count_based_recall(candidate, label)
        if baseline_recall is not None and candidate_recall is not None:
            recall_dropped = baseline_recall - candidate_recall >= Fraction(1, 20)
        else:
            recall_drop = (
                Decimal(str(current["per_class"][label]["recall"]))
                - Decimal(str(proposed["per_class"][label]["recall"]))
            )
            recall_dropped = recall_drop >= Decimal("0.05")
        if recall_dropped:
            reasons.append(f"recall_drop_at_least_0.05:{label}")

    regressions = []
    before = baseline.per_example
    after = candidate.per_example
    required = {"example_id", "expected_label", "normalized_output"}
    if required.issubset(before.columns) and required.issubset(after.columns):
        by_id = after.set_index("example_id")
        for row in before.itertuples(index=False):
            if row.example_id not in by_id.index or row.normalized_output != row.expected_label:
                continue
            new = by_id.loc[row.example_id]
            if new.normalized_output != row.expected_label:
                regressions.append({
                    "example_id": row.example_id,
                    "expected_label": row.expected_label,
                    "baseline_output": row.normalized_output,
                    "candidate_output": new.normalized_output,
                })
    tied = proposed == current
    accepted = not reasons and (not tied or len(candidate_prompt) < len(baseline_prompt))
    return SelectionDecision(accepted, tuple(reasons), tuple(regressions))


def _exact_summary_score(row) -> Fraction:
    """Recover exact score thirds from evaluator dimensions at the gate boundary."""
    if row["overall_score"] != 0 and all(
        f"{name}_score" in row for name in ("financial_figures", "conciseness", "faithfulness")
    ):
        scores = [row[f"{name}_score"] for name in ("financial_figures", "conciseness", "faithfulness")]
        applicable = [value for value in scores if not pd.isna(value)]
        if len(applicable) >= 2 and all(value == int(value) and 1 <= value <= 5 for value in applicable):
            return Fraction(sum(int(value) for value in applicable) * 20, len(applicable))
    return Fraction(str(row["overall_score"]))


def select_financial_candidate(
    baseline: MetricResult, candidate: MetricResult,
    baseline_prompt: str, candidate_prompt: str,
) -> SelectionDecision:
    """Reject aggregate regressions and excessive large drops on matched IDs."""
    current, proposed = baseline.aggregate, candidate.aggregate
    reasons = []
    for metric in ("overall_score", "mean_faithfulness"):
        if proposed[metric] < current[metric]:
            reasons.append(f"{metric}_decreased")
    if proposed["severe_hallucination_count"] > current["severe_hallucination_count"]:
        reasons.append("severe_hallucination_count_increased")

    before, after = baseline.per_example, candidate.per_example
    regressions = []
    matched, large_drops = 0, 0
    for frame in (before, after):
        if frame.empty:
            continue
        for column in ("example_id", "overall_score"):
            if column not in frame:
                raise ValueError(f"Missing required column: {column}")
        if frame.example_id.isna().any() or frame.example_id.duplicated().any():
            raise ValueError("example_id must be non-null and unique")
    if not before.empty and not after.empty:
        by_id = after.set_index("example_id")
        for row in before.itertuples(index=False):
            if row.example_id not in by_id.index:
                continue
            matched += 1
            new = by_id.loc[row.example_id]
            new_score = new["overall_score"]
            delta = _exact_summary_score(new) - _exact_summary_score(row._asdict())
            if delta < 0:
                regressions.append({
                    "example_id": row.example_id, "baseline_score": row.overall_score,
                    "candidate_score": new_score, "delta": float(delta),
                })
            if delta <= -20:
                large_drops += 1
    if matched and large_drops * 10 > matched:
        reasons.append("large_score_drop_rate_exceeds_0.10")

    tied = current == proposed
    accepted = not reasons and (not tied or len(candidate_prompt) < len(baseline_prompt))
    return SelectionDecision(accepted, tuple(reasons), tuple(regressions))
