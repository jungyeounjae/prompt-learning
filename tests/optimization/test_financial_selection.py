"""Financial guardrails compare stable identities and inclusive thresholds."""

import pandas as pd
import pytest

from prompt_learning.models import MetricResult


def metrics(overall=80, faithfulness=4, severe=0, rows=None):
    return MetricResult({"overall_score": overall, "mean_faithfulness": faithfulness,
                         "severe_hallucination_count": severe}, pd.DataFrame(rows or []))


def test_gate_accepts_improvement_when_all_rules_pass():
    from prompt_learning.optimization.selection import select_financial_candidate

    assert select_financial_candidate(metrics(), metrics(81), "short", "longer prompt").accepted


@pytest.mark.parametrize(("candidate", "reason"), [
    (metrics(79.999999999), "overall_score_decreased"),
    (metrics(81, 3.999999999), "mean_faithfulness_decreased"),
    (metrics(81, severe=1), "severe_hallucination_count_increased"),
])
def test_each_regression_rejects_even_a_shorter_candidate(candidate, reason):
    from prompt_learning.optimization.selection import select_financial_candidate

    decision = select_financial_candidate(metrics(), candidate, "long baseline", "short")
    assert not decision.accepted and decision.reasons == (reason,)


def test_gate_collects_every_failed_guardrail():
    from prompt_learning.optimization.selection import select_financial_candidate

    baseline = metrics(rows=[{"example_id": "a", "overall_score": 90}])
    candidate = metrics(79, 3, 1, [{"example_id": "a", "overall_score": 70}])
    decision = select_financial_candidate(baseline, candidate, "baseline", "candidate")
    assert decision.reasons == ("overall_score_decreased", "mean_faithfulness_decreased",
                                "severe_hallucination_count_increased",
                                "large_score_drop_rate_exceeds_0.10")


@pytest.mark.parametrize(("count", "score", "accepted"), [(1, 60, True), (2, 60, False),
                                                           (2, 60.000000001, True)])
def test_exactly_ten_percent_is_allowed_and_twenty_point_drop_is_inclusive(count, score, accepted):
    from prompt_learning.optimization.selection import select_financial_candidate

    before = [{"example_id": str(i), "overall_score": 80} for i in range(10)]
    after = [{"example_id": str(i), "overall_score": score if i < count else 90} for i in reversed(range(10))]
    decision = select_financial_candidate(metrics(rows=before), metrics(81, rows=after), "base", "candidate")
    assert decision.accepted is accepted
    assert ("large_score_drop_rate_exceeds_0.10" in decision.reasons) is (not accepted)
    assert len(decision.regressions) == count


def test_regression_threshold_uses_only_matched_ids_and_reports_all_score_drops():
    from prompt_learning.optimization.selection import select_financial_candidate

    before = [{"example_id": str(i), "overall_score": 80} for i in range(11)]
    after = [{"example_id": str(i), "overall_score": 60 if i == 0 else 79} for i in range(9)]
    after.append({"example_id": "new", "overall_score": 100})
    decision = select_financial_candidate(metrics(rows=before), metrics(81, rows=after), "base", "candidate")
    assert not decision.accepted  # One large drop out of nine matched examples.
    assert len(decision.regressions) == 9
    assert decision.regressions[0] == {"example_id": "0", "baseline_score": 80,
                                       "candidate_score": 60, "delta": -20}
    assert decision.regressions[1]["delta"] == -1


@pytest.mark.parametrize(("baseline", "candidate", "accepted"), [
    ("long baseline", "short", True), ("short", "long candidate", False),
    ("first", "other", False), ("same", "same", False),
])
def test_metric_ties_prefer_shorter_prompt_then_baseline(baseline, candidate, accepted):
    from prompt_learning.optimization.selection import select_financial_candidate

    assert select_financial_candidate(metrics(), metrics(), baseline, candidate).accepted is accepted


def test_regression_records_do_not_include_source_or_reference():
    from prompt_learning.optimization.selection import select_financial_candidate

    rows = [{"example_id": "a", "overall_score": 80, "article": "private source",
             "reference_summary": "private reference"}]
    decision = select_financial_candidate(metrics(rows=rows), metrics(81, rows=[
        {**rows[0], "overall_score": 79}]), "baseline", "candidate")
    assert "private" not in str(decision)


def test_gate_rejects_duplicate_example_ids_instead_of_ambiguous_alignment():
    from prompt_learning.optimization.selection import select_financial_candidate

    rows = [{"example_id": "a", "overall_score": 80}] * 2
    with pytest.raises(ValueError, match="unique"):
        select_financial_candidate(metrics(rows=rows), metrics(81, rows=rows), "base", "candidate")


def test_real_evaluations_reject_mathematically_exact_twenty_point_drop():
    import json

    from prompt_learning.evaluators.summarization import evaluate_summaries
    from prompt_learning.optimization.selection import select_financial_candidate

    data = pd.DataFrame([{"example_id": name, "article": "Company announced its results.",
                          "reference_summary": "A reference.", "output": "x" * 40}
                         for name in ("a", "b")])

    def judge(scores):
        outcomes = iter(scores)

        def evaluate(**kwargs):
            financial, concise, faithful = next(outcomes)
            return json.dumps({
                name: {"score": score, "label": "rated", "rationale": "Source-based assessment."}
                for name, score in zip(("financial_figures", "conciseness", "faithfulness"),
                                       (financial, concise, faithful))
            })
        return evaluate

    baseline = evaluate_summaries(data, judge([(5, 3, 5), (1, 2, 5)]))
    candidate = evaluate_summaries(data, judge([(2, 3, 5), (5, 5, 5)]))
    decision = select_financial_candidate(baseline, candidate, "base", "candidate")
    assert candidate.aggregate["overall_score"] > baseline.aggregate["overall_score"]
    assert not decision.accepted
    assert decision.reasons == ("large_score_drop_rate_exceeds_0.10",)
    assert decision.regressions[0]["delta"] == -20


@pytest.mark.parametrize("reverse_baseline", [False, True])
@pytest.mark.parametrize(("baseline_prompt", "candidate_prompt", "accepted"), [
    ("long baseline prompt", "short", True), ("short", "long candidate prompt", False),
])
def test_reordered_real_evaluations_are_exact_metric_ties(
    reverse_baseline, baseline_prompt, candidate_prompt, accepted,
):
    import json

    from prompt_learning.evaluators.summarization import evaluate_summaries
    from prompt_learning.optimization.selection import select_financial_candidate

    scores = [(1, 1, 1), (1, 1, 2), (5, 4, 5), (5, 5, 5), (5, 5, 5)]
    data = pd.DataFrame([{"example_id": str(i), "article": str(i),
                          "reference_summary": "Reference", "output": "x" * 40}
                         for i in range(5)])

    def evaluator(*, article, **kwargs):
        return json.dumps({
            name: {"score": score, "label": "rated", "rationale": "Source-based assessment."}
            for name, score in zip(("financial_figures", "conciseness", "faithfulness"), scores[int(article)])
        })

    normal = evaluate_summaries(data, evaluator)
    reversed_result = evaluate_summaries(data.iloc[::-1], evaluator)
    baseline, candidate = (reversed_result, normal) if reverse_baseline else (normal, reversed_result)
    assert baseline.aggregate["overall_score"] != candidate.aggregate["overall_score"]
    decision = select_financial_candidate(baseline, candidate, baseline_prompt, candidate_prompt)
    assert decision.accepted is accepted
    assert decision.reasons == () and decision.regressions == ()


def test_faithfulness_comparison_uses_exact_underlying_scores():
    from prompt_learning.optimization.selection import select_financial_candidate

    row = {"example_id": "a", "overall_score": 80, "financial_figures_score": 4,
           "conciseness_score": 4, "faithfulness_score": 4}
    baseline = metrics(faithfulness=4.000000000000001, rows=[row])
    candidate = metrics(faithfulness=4, rows=[row])
    decision = select_financial_candidate(baseline, candidate, "long baseline", "short")
    assert decision.accepted and decision.reasons == ()


@pytest.mark.parametrize(("candidate_row", "reason"), [
    ({"financial_figures_score": 3}, "overall_score_decreased"),
    ({"financial_figures_score": 5, "faithfulness_score": 3}, "mean_faithfulness_decreased"),
])
def test_exact_comparisons_reject_real_decreases_even_if_aggregate_floats_are_equal(candidate_row, reason):
    from prompt_learning.optimization.selection import select_financial_candidate

    before = {"example_id": "a", "overall_score": 80, "financial_figures_score": 4,
              "conciseness_score": 4, "faithfulness_score": 4}
    after = {**before, **candidate_row}
    decision = select_financial_candidate(metrics(rows=[before]), metrics(rows=[after]), "long baseline", "short")
    assert not decision.accepted and reason in decision.reasons


def test_exact_aggregate_preserves_zero_for_deterministic_failure():
    from prompt_learning.optimization.selection import select_financial_candidate

    row = {"example_id": "a", "overall_score": 100, "financial_figures_score": 5,
           "conciseness_score": 5, "faithfulness_score": 5}
    failed = {**row, "overall_score": 0, "evaluation_failed": True,
              "deterministic_errors": ("summary_length_out_of_range",)}
    decision = select_financial_candidate(metrics(100, 5, rows=[row]),
                                         metrics(0, 5, rows=[failed]), "long baseline", "short")
    assert not decision.accepted and "overall_score_decreased" in decision.reasons
    assert decision.regressions[0]["delta"] == -100
