"""Financial inference keeps reference data out of model requests."""

import pandas as pd
import pytest

from prompt_learning.gateway import ModelGateway
from prompt_learning.models import ModelResponse, PromptCandidate


def test_render_requires_article_and_preserves_literal_braces():
    from prompt_learning.tasks.financial_news import render_financial_prompt

    with pytest.raises(ValueError, match=r"\{article\}"):
        render_financial_prompt("Summarize", "news")
    assert render_financial_prompt('JSON {"summary": "?"}: {article}', '{article} {literal}') == (
        'JSON {"summary": "?"}: {article} {literal}'
    )


def test_inference_preserves_evaluator_inputs_usage_and_identity(tmp_path):
    from prompt_learning.tasks.financial_news import run_financial_inference

    def client(request):
        assert request.task == "financial" and request.model == "offline"
        assert request.temperature == 0.2
        assert request.messages == ({"role": "user", "content": "Summarize: news {literal}"},)
        return ModelResponse("A useful summary.", 12, 4, 1)

    frame = pd.DataFrame([{"example_id": "a", "article": "news {literal}",
                           "reference_summary": "private reference"}], index=[7])
    original = frame.copy(deep=True)
    result = run_financial_inference(frame, PromptCandidate("Summarize: {article}", 1, 0),
                                    ModelGateway(client, tmp_path / "cache"),
                                    model="offline", temperature=0.2)
    pd.testing.assert_frame_equal(frame, original)
    row = result.iloc[0]
    assert (row.example_id, row.article, row.reference_summary) == (
        "a", "news {literal}", "private reference")
    assert row.output == row.raw_output == "A useful summary."
    assert (row.input_tokens, row.output_tokens) == (12, 4)
    assert row.latency_ms >= 0 and not row.cached and row.error is None


@pytest.mark.parametrize("outcome", [ModelResponse(" \n", 2, 1, 1),
                                     ConnectionError("secret-key private article")])
def test_empty_and_failed_generations_keep_rows_and_redact_errors(tmp_path, outcome):
    from prompt_learning.tasks.financial_news import run_financial_inference

    def client(request):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    frame = pd.DataFrame([{"example_id": "a", "article": "news", "reference_summary": "ref"}])
    gateway = ModelGateway(client, tmp_path / "cache", max_retries=0)
    result = run_financial_inference(frame, PromptCandidate("{article}", 1, 0), gateway,
                                    model="offline")
    assert len(result) == 1 and result.iloc[0].output == ""
    assert result.iloc[0].error in ("InvalidModelOutput", "ConnectionError")
    assert "secret" not in str(result.iloc[0].error)


def test_invalid_inputs_fail_before_inference(tmp_path):
    from prompt_learning.tasks.financial_news import run_financial_inference

    gateway = ModelGateway(lambda _: pytest.fail("invalid input reached model"), tmp_path / "cache")
    frame = pd.DataFrame(columns=["example_id", "article", "reference_summary"])
    with pytest.raises(ValueError, match="article"):
        run_financial_inference(frame, PromptCandidate("missing", 1, 0), gateway, model="offline")
    with pytest.raises(ValueError, match="reference_summary"):
        run_financial_inference(frame.drop(columns="reference_summary"),
                                PromptCandidate("{article}", 1, 0), gateway, model="offline")
