"""Guardrails for emotion prompt candidates."""

from decimal import Decimal
from fractions import Fraction

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
