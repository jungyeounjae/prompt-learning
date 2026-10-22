"""Emotion inference contracts, exercised through an offline model gateway."""

import pandas as pd
import pytest
import httpx
import openai

from prompt_learning.gateway import ModelGateway
from prompt_learning.models import ModelResponse, PromptCandidate


def test_render_preserves_literal_braces_in_template_and_text():
    from prompt_learning.tasks.emotion import render_emotion_prompt

    assert render_emotion_prompt('JSON {"label": "?"}: {text}', '{text} {joy}') == (
        'JSON {"label": "?"}: {text} {joy}'
    )


def test_render_requires_text_placeholder():
    from prompt_learning.tasks.emotion import render_emotion_prompt

    with pytest.raises(ValueError, match=r"\{text\}"):
        render_emotion_prompt("Classify this", "example")


@pytest.mark.parametrize(("raw", "expected"), [
    (" SADNESS ", "sadness"), ('"joy".', "joy"),
    ("The emotion is love.", "love"), ("anger!", "anger"),
    ("I infer fear", "fear"), ("surprise", "surprise"),
    ("joy joy", "joy"), ("joy or sadness", "invalid"),
    ("disgust", "invalid"), ("enjoy", "invalid"), ("", "invalid"),
])
def test_normalization_resolves_one_distinct_allowed_label(raw, expected):
    from prompt_learning.tasks.emotion import normalize_emotion_output

    assert normalize_emotion_output(raw) == expected


def test_inference_preserves_identity_outputs_and_model_usage(tmp_path):
    from prompt_learning.tasks.emotion import run_emotion_inference

    def client(request):
        assert request.task == "emotion" and request.model == "offline-model"
        assert request.temperature == 0.25
        assert request.messages == ({"role": "user", "content": "Classify: text {literal}"},)
        return ModelResponse("It is joy.", 17, 4, 2.0)

    frame = pd.DataFrame([{
        "example_id": "id-1", "text": "text {literal}", "expected_label": "joy",
    }], index=[8])
    original = frame.copy(deep=True)
    result = run_emotion_inference(
        frame, PromptCandidate("Classify: {text}", 1, 0),
        ModelGateway(client, tmp_path / "cache.sqlite"), model="offline-model", temperature=0.25,
    )
    pd.testing.assert_frame_equal(frame, original)
    row = result.iloc[0]
    assert row.example_id == "id-1" and row.expected_label == "joy"
    assert row.raw_output == "It is joy." and row.normalized_output == "joy"
    assert row.output == row.normalized_output
    assert (row.input_tokens, row.output_tokens) == (17, 4)
    assert row.latency_ms >= 0 and not row.cached and row.error is None


def test_inference_keeps_invalid_output_and_failed_rows_without_error_secrets(tmp_path):
    from prompt_learning.tasks.emotion import run_emotion_inference

    outcomes = iter([ModelResponse("joy or sadness", 9, 3, 2),
                     ConnectionError("secret-api-key private user text")])

    def client(request):
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return result

    frame = pd.DataFrame([
        {"example_id": "a", "text": "one", "expected_label": "joy"},
        {"example_id": "b", "text": "two", "expected_label": "fear"},
    ])
    result = run_emotion_inference(
        frame, PromptCandidate("{text}", 1, 0),
        ModelGateway(client, tmp_path / "cache.sqlite", max_retries=0), model="offline",
    )
    assert result.example_id.tolist() == ["a", "b"]
    assert result.normalized_output.tolist() == ["invalid", "invalid"]
    assert result.output.tolist() == result.normalized_output.tolist()
    assert result.raw_output.tolist() == ["joy or sadness", ""]
    assert result.error.tolist() == ["invalid_output", "ConnectionError"]
    assert result.iloc[0].input_tokens == 9 and result.iloc[0].output_tokens == 3
    assert pd.isna(result.iloc[1].input_tokens) and pd.isna(result.iloc[1].output_tokens)
    assert result.latency_ms.ge(0).all()
    assert "secret-api-key" not in result.to_json()


def test_inference_rejects_missing_placeholder_before_any_call(tmp_path):
    from prompt_learning.tasks.emotion import run_emotion_inference

    def client(request):
        pytest.fail("invalid prompt must fail before inference")

    with pytest.raises(ValueError, match=r"\{text\}"):
        run_emotion_inference(
            pd.DataFrame(columns=["example_id", "text", "expected_label"]),
            PromptCandidate("missing", 1, 0),
            ModelGateway(client, tmp_path / "cache.sqlite"), model="offline",
        )


def test_inference_rejects_missing_input_columns(tmp_path):
    from prompt_learning.tasks.emotion import run_emotion_inference

    with pytest.raises(ValueError, match="example_id"):
        run_emotion_inference(
            pd.DataFrame({"text": ["one"], "expected_label": ["joy"]}),
            PromptCandidate("{text}", 1, 0),
            ModelGateway(lambda _: pytest.fail("invalid input"), tmp_path / "cache.sqlite"),
            model="offline",
        )


@pytest.mark.parametrize("error_type", [openai.AuthenticationError, openai.BadRequestError])
def test_fatal_api_configuration_errors_stop_inference_immediately(tmp_path, error_type):
    from prompt_learning.tasks.emotion import run_emotion_inference

    calls = []
    error = error_type("invalid configuration", response=httpx.Response(
        401 if error_type is openai.AuthenticationError else 400,
        request=httpx.Request("POST", "https://example.invalid")), body=None)

    def client(request):
        calls.append(request)
        raise error

    frame = pd.DataFrame([
        {"example_id": "a", "text": "one", "expected_label": "joy"},
        {"example_id": "b", "text": "two", "expected_label": "fear"},
    ])
    with pytest.raises(error_type):
        run_emotion_inference(
            frame, PromptCandidate("{text}", 1, 0),
            ModelGateway(client, tmp_path / "cache.sqlite"), model="offline",
        )
    assert len(calls) == 1
