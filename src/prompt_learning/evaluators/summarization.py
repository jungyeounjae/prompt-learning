"""Typed financial-summary scores, deterministic checks, and row-safe evaluation.

Evaluator callables receive article, reference_summary, and output keywords and
return JSON with financial_figures, conciseness, and faithfulness objects. Each
object requires score, label, and rationale. Only financial_figures may have a
null score, paired with the exact label not_applicable.
"""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
import json
import re

import openai
import pandas as pd

from prompt_learning.models import MetricResult


@dataclass(frozen=True)
class DimensionEvaluation:
    score: int
    label: str
    rationale: str


@dataclass(frozen=True)
class FinancialFiguresEvaluation:
    score: int | None
    label: str
    rationale: str


@dataclass(frozen=True)
class SummaryEvaluation:
    financial_figures: FinancialFiguresEvaluation
    conciseness: DimensionEvaluation
    faithfulness: DimensionEvaluation

    @property
    def overall_score(self) -> float:
        scores = [self.conciseness.score, self.faithfulness.score]
        if self.financial_figures.score is not None:
            scores.append(self.financial_figures.score)
        return sum(scores) / len(scores) * 20


_DIMENSIONS = ("financial_figures", "conciseness", "faithfulness")


def parse_summary_evaluation(raw: str) -> SummaryEvaluation:
    """Reject malformed schema without including evaluator text in errors."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise ValueError("Evaluator response must be valid JSON") from None
    if not isinstance(payload, dict) or set(payload) != set(_DIMENSIONS):
        raise ValueError("Evaluator response requires exactly three score dimensions")
    dimensions = []
    for name in _DIMENSIONS:
        value = payload[name]
        if not isinstance(value, dict) or set(value) != {"score", "label", "rationale"}:
            raise ValueError(f"{name} requires score, label, and rationale")
        for field in ("label", "rationale"):
            if not isinstance(value[field], str) or not value[field].strip():
                raise ValueError(f"{name}.{field} must be nonempty text")
        score, label = value["score"], value["label"]
        if name == "financial_figures" and label == "not_applicable":
            if score is not None:
                raise ValueError("financial_figures not_applicable requires a null score")
        elif type(score) is not int or not 1 <= score <= 5 or label == "not_applicable":
            raise ValueError(f"{name}.score must be an integer from 1 to 5")
        cls = FinancialFiguresEvaluation if name == "financial_figures" else DimensionEvaluation
        dimensions.append(cls(score, label, value["rationale"].strip()))
    return SummaryEvaluation(*dimensions)


_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_CURRENCY = r"(?:US\$|A\$|USD|EUR|GBP|INR|JPY|CAD|AUD|CNY|CHF|HKD|SGD|Rs\.?|rupees?|dollars?|euros?|pounds?|yen|[$€£¥₹])"
_SCALE = r"(?:thousand|million|billion|trillion|crore|lakh|bn|mn|k|m|b|t|cr)"
# Consume the whole attached suffix, even if its scale is unknown. Without
# this, e.g. $100widgets would silently compare equal to a supported $100.
_MONEY_AMOUNT = rf"(?P<number>{_NUMBER})(?:(?:\s*-\s*|\s*)(?P<scale>{_SCALE})(?!\w)|(?P<suffix>[^\W\d]\w*))?"
_PREFIX_MONEY = re.compile(
    rf"(?<!\w)(?P<currency>{_CURRENCY})\s*{_MONEY_AMOUNT}(?!\w|[.,]\d)",
    re.IGNORECASE,
)
_SUFFIX_MONEY = re.compile(
    rf"(?<![\w.,]){_MONEY_AMOUNT}\s*(?P<currency>{_CURRENCY})(?!\w)",
    re.IGNORECASE,
)
_PERCENT = re.compile(rf"(?<![\w.,])(?P<number>{_NUMBER})\s*(?:%|percent\b|per\s+cent\b|pct\b)", re.IGNORECASE)
_HALF_PERCENT = re.compile(r"\bhalf\s+(?:a\s+)?(?:percent|per\s+cent|pct)\b", re.IGNORECASE)
_PAIRED_PERCENT = re.compile(
    rf"(?<![\w.,])(?P<first>{_NUMBER})\s+"
    r"(?:growth|return|increase|decrease|rise|fall|decline)\s+"
    rf"(?:versus|vs\.?|compared\s+(?:with|to))\s+(?P<second>{_NUMBER})\s*"
    r"(?:%|percent\b|per\s+cent\b|pct\b)",
    re.IGNORECASE,
)
_MULTIPLIERS = {"thousand": 1000, "million": 10**6, "billion": 10**9,
                "trillion": 10**12, "crore": 10**7, "lakh": 10**5, "bn": 10**9, "mn": 10**6,
                "k": 1000, "m": 10**6, "b": 10**9, "t": 10**12, "cr": 10**7}
_CURRENCY_ALIASES = {"$": "usd", "us$": "usd", "a$": "aud", "dollar": "usd", "dollars": "usd",
                     "€": "eur", "euro": "eur", "euros": "eur",
                     "£": "gbp", "pound": "gbp", "pounds": "gbp",
                     "₹": "inr", "rs": "inr", "rs.": "inr", "rupee": "inr", "rupees": "inr",
                     "¥": "jpy", "yen": "jpy"}


def _financial_expressions(text: str) -> set[tuple[str, Decimal, str]]:
    expressions = set()
    for pattern in (_PREFIX_MONEY, _SUFFIX_MONEY):
        for match in pattern.finditer(text):
            currency = match["currency"].casefold()
            currency = _CURRENCY_ALIASES.get(currency, currency)
            amount = Decimal(match["number"].replace(",", ""))
            scale = (match["scale"] or "").casefold()
            suffix = (match.groupdict().get("suffix") or "").casefold()
            expressions.add((currency, amount * _MULTIPLIERS.get(scale, 1), suffix))
    for match in _PERCENT.finditer(text):
        amount = abs(Decimal(match["number"].replace(",", "")))
        expressions.add(("percent", amount, ""))
    if _HALF_PERCENT.search(text):
        expressions.add(("percent", Decimal("0.5"), ""))
    for match in _PAIRED_PERCENT.finditer(text):
        amount = abs(Decimal(match["first"].replace(",", "")))
        expressions.add(("percent", amount, ""))
    return expressions


def evaluate_summaries(frame: pd.DataFrame, evaluator: Callable[..., str]) -> MetricResult:
    """Evaluate all rows; failed rows score zero instead of disappearing.

    Deterministic-invalid rows also score zero; parsed dimension scores remain
    available for diagnosis. Missing judge scores use conservative 1s for
    faithfulness/conciseness and an unavailable financial score (not N/A).
    """
    if frame.empty:
        raise ValueError("Cannot evaluate empty results")
    for column in ("example_id", "article", "reference_summary", "output"):
        if column not in frame:
            raise ValueError(f"Missing required column: {column}")
    rows = []
    for source in frame.to_dict("records"):
        record = dict(source)
        output = source["output"]
        errors = []
        empty = not isinstance(output, str) or not output.strip()
        if empty:
            errors.append("empty_summary")
        elif not 40 <= len(output) <= 800:
            errors.append("summary_length_out_of_range")
        unsupported = not empty and bool(
            _financial_expressions(output) - _financial_expressions(source["article"])
        )
        if unsupported:
            errors.append("unsupported_financial_expression")
        upstream_error = source.get("error")
        failed = upstream_error is not None and not pd.isna(upstream_error)
        record["error"] = None
        if failed:
            record["error"] = (upstream_error if isinstance(upstream_error, str)
                               and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*Error", upstream_error)
                               else "generation_failed")
        evaluation = None
        if not failed and not empty:
            try:
                evaluation = parse_summary_evaluation(evaluator(
                    article=source["article"], reference_summary=source["reference_summary"], output=output,
                ))
            except (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError):
                raise
            except Exception as error:
                record["error"] = type(error).__name__
        elif empty and not failed:
            record["error"] = "empty_summary"
        for name in _DIMENSIONS:
            dimension = getattr(evaluation, name) if evaluation else None
            record[f"{name}_score"] = (dimension.score if dimension else
                                        None if name == "financial_figures" else 1)
            record[f"{name}_label"] = dimension.label if dimension else "evaluation_failed"
            record[f"{name}_rationale"] = dimension.rationale if dimension else "No valid evaluation available."
        record["deterministic_errors"] = tuple(errors)
        record["severe_hallucination"] = record["faithfulness_score"] == 1 or unsupported
        record["evaluation_failed"] = bool(record["error"] or errors)
        record["overall_score"] = evaluation.overall_score if evaluation and not errors else 0.0
        record["feedback"] = ("; ".join(errors + ([record["error"]] if record["error"] else []))
                              if record["evaluation_failed"] else " ".join(
                                  f"{name}: {getattr(evaluation, name).rationale}" for name in _DIMENSIONS))
        rows.append(record)
    per_example = pd.DataFrame.from_records(rows, index=frame.index)
    financial_mean = per_example.financial_figures_score.mean()
    aggregate = {
        "overall_score": float(per_example.overall_score.mean()),
        "mean_financial_figures": None if pd.isna(financial_mean) else float(financial_mean),
        "mean_conciseness": float(per_example.conciseness_score.mean()),
        "mean_faithfulness": float(per_example.faithfulness_score.mean()),
        "financial_figures_not_applicable_count": int(per_example.financial_figures_label.eq("not_applicable").sum()),
        "severe_hallucination_count": int(per_example.severe_hallucination.sum()),
        "failure_count": int(per_example.evaluation_failed.sum()),
    }
    feedback = "\n".join(f"{row.example_id}: {row.feedback}" for row in per_example.itertuples(index=False))
    return MetricResult(aggregate, per_example, feedback)
