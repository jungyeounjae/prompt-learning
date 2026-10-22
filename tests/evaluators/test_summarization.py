"""Structured scores and deterministic financial-summary checks."""

import json

import pandas as pd
import pytest


def payload(financial=3, concise=4, faithful=5):
    return {
        "financial_figures": {"score": financial, "label": "not_applicable" if financial is None else "good",
                              "rationale": "Preserves material figures."},
        "conciseness": {"score": concise, "label": "good", "rationale": "Focused summary."},
        "faithfulness": {"score": faithful, "label": "good", "rationale": "Supported by source."},
    }


def frame(output="A focused summary of the company and its latest results.", article="Company news."):
    return pd.DataFrame([{"example_id": "a", "article": article,
                          "reference_summary": "private reference", "output": output}])


@pytest.mark.parametrize(("financial", "expected"), [(3, 80), (None, 90)])
def test_parser_scores_mean_of_applicable_dimensions(financial, expected):
    from prompt_learning.evaluators.summarization import parse_summary_evaluation

    result = parse_summary_evaluation(json.dumps(payload(financial)))
    assert result.financial_figures.score == financial
    assert result.conciseness.score == 4 and result.faithfulness.score == 5
    assert result.overall_score == expected
    assert result.faithfulness.label == "good"
    assert result.faithfulness.rationale == "Supported by source."


@pytest.mark.parametrize(("dimension", "field", "value"), [
    ("conciseness", "score", 0), ("faithfulness", "score", 6),
    ("faithfulness", "score", True), ("conciseness", "score", 4.0),
    ("faithfulness", "score", "5"), ("faithfulness", "score", None),
    ("financial_figures", "score", None), ("financial_figures", "label", "not_applicable"),
    ("conciseness", "rationale", ""), ("faithfulness", "label", " "),
])
def test_parser_rejects_invalid_range_types_and_na_mismatches(dimension, field, value):
    from prompt_learning.evaluators.summarization import parse_summary_evaluation

    data = payload()
    data[dimension][field] = value
    with pytest.raises(ValueError):
        parse_summary_evaluation(json.dumps(data))


@pytest.mark.parametrize("raw", ["private invalid JSON", "[]", "null", "{}"])
def test_parser_rejects_invalid_schema_without_echoing_input(raw):
    from prompt_learning.evaluators.summarization import parse_summary_evaluation

    with pytest.raises(ValueError) as error:
        parse_summary_evaluation(raw)
    assert "private" not in str(error.value)


def test_parser_requires_rationale():
    from prompt_learning.evaluators.summarization import parse_summary_evaluation

    data = payload()
    del data["financial_figures"]["rationale"]
    with pytest.raises(ValueError, match="rationale"):
        parse_summary_evaluation(json.dumps(data))


def test_evaluator_keeps_inputs_and_aggregates_na_without_mutation():
    from prompt_learning.evaluators.summarization import evaluate_summaries

    data = pd.concat([frame(), frame()], ignore_index=True)
    data["example_id"] = ["a", "b"]
    original = data.copy(deep=True)
    outcomes = iter([json.dumps(payload()), json.dumps(payload(None))])

    def evaluator(*, article, reference_summary, output):
        assert article == "Company news." and reference_summary == "private reference"
        assert output == "A focused summary of the company and its latest results."
        return next(outcomes)

    result = evaluate_summaries(data, evaluator)
    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_frame_equal(result.per_example[original.columns], original)
    assert result.per_example.overall_score.tolist() == [80, 90]
    assert result.aggregate["overall_score"] == 85
    assert result.aggregate["mean_faithfulness"] == 5
    assert result.aggregate["financial_figures_not_applicable_count"] == 1
    assert result.aggregate["severe_hallucination_count"] == 0
    assert result.aggregate["failure_count"] == 0
    assert "Preserves material figures." in result.feedback


@pytest.mark.parametrize(("length", "valid"), [(39, False), (40, True), (800, True), (801, False)])
def test_length_check_includes_40_and_800(length, valid):
    from prompt_learning.evaluators.summarization import evaluate_summaries

    result = evaluate_summaries(frame("x" * length), lambda **_: json.dumps(payload()))
    assert ("summary_length_out_of_range" not in result.per_example.iloc[0].deterministic_errors) is valid
    assert result.per_example.iloc[0].overall_score == (80 if valid else 0)
    assert result.aggregate["failure_count"] == (0 if valid else 1)
    assert result.per_example.iloc[0].conciseness_score == 4


@pytest.mark.parametrize(("article", "claim", "unsupported"), [
    ("Revenue reached $100 million and rose 5%.", "$100 million and 5%", False),
    ("Revenue reached $100 million and rose 5%.", "$200 million and 5%", True),
    ("Revenue rose 5%.", "15%", True),
    ("Revenue reached $100 million.", "$100 billion", True),
    ("Profit was EUR 10 million.", "EUR 10 million", False),
    ("Profit was EUR 10 million.", "USD 10 million", True),
    ("Revenue rose 5 percent.", "5%", False),
    ("Revenue reached $1,000.", "$1000", False),
    ("Revenue reached $100.", "100%", True),
    ("Profit was Rs. 10 crore.", "Rs. 20 crore", True),
    ("Profit was US$ 10 million.", "US$ 20 million", True),
    ("Revenue rose 5 pct.", "15 pct", True),
    ("Revenue reached USD 1 million.", "$1,000,000", False),
    ("Profit was EUR 10 million.", "10 million euros", False),
])
def test_unsupported_currency_or_percentage_is_severe(article, claim, unsupported):
    from prompt_learning.evaluators.summarization import evaluate_summaries

    result = evaluate_summaries(frame(f"The company reported {claim} in its latest results.", article),
                                lambda **_: json.dumps(payload()))
    row = result.per_example.iloc[0]
    assert bool(row.severe_hallucination) is unsupported
    assert ("unsupported_financial_expression" in row.deterministic_errors) is unsupported


def test_faithfulness_one_is_severe_without_unsupported_expressions():
    from prompt_learning.evaluators.summarization import evaluate_summaries

    result = evaluate_summaries(frame(), lambda **_: json.dumps(payload(faithful=1)))
    assert result.aggregate["severe_hallucination_count"] == 1


@pytest.mark.parametrize("outcome", ["private malformed data", ConnectionError("secret-key private article")])
def test_failed_evaluation_rows_are_retained_and_penalized_with_safe_errors(outcome):
    from prompt_learning.evaluators.summarization import evaluate_summaries

    def evaluator(**kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    result = evaluate_summaries(frame(), evaluator)
    assert len(result.per_example) == 1
    assert result.aggregate["failure_count"] == 1
    assert result.aggregate["overall_score"] == 0
    assert "secret" not in result.feedback and "private" not in result.feedback
    assert "private" not in result.per_example.iloc[0].error


def test_empty_summary_is_failed_without_calling_judge():
    from prompt_learning.evaluators.summarization import evaluate_summaries

    result = evaluate_summaries(frame("  "), lambda **_: pytest.fail("empty summary reached judge"))
    assert "empty_summary" in result.per_example.iloc[0].deterministic_errors
    assert result.aggregate["failure_count"] == 1 and result.aggregate["overall_score"] == 0


def test_upstream_failure_is_preserved_and_not_sent_to_judge():
    from prompt_learning.evaluators.summarization import evaluate_summaries

    data = frame()
    data["error"] = "ConnectionError"
    result = evaluate_summaries(data, lambda **_: pytest.fail("failed generation reached judge"))
    assert result.per_example.iloc[0].error == "ConnectionError"
    assert result.aggregate["failure_count"] == 1


def test_evaluator_rejects_empty_frames_and_missing_inputs():
    from prompt_learning.evaluators.summarization import evaluate_summaries

    with pytest.raises(ValueError, match="empty"):
        evaluate_summaries(frame().iloc[:0], lambda **_: "")
    with pytest.raises(ValueError, match="reference_summary"):
        evaluate_summaries(frame().drop(columns="reference_summary"), lambda **_: "")
