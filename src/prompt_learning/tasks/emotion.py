"""Render and score emotion model responses without external API setup."""

import re
import time

import openai
import pandas as pd

from prompt_learning.datasets.emotion import LABELS
from prompt_learning.gateway import ModelGateway
from prompt_learning.models import ModelRequest, PromptCandidate


_LABEL_PATTERN = re.compile(r"\b(?:" + "|".join(LABELS) + r")\b", re.IGNORECASE)


def render_emotion_prompt(template: str, text: str) -> str:
    if "{text}" not in template:
        raise ValueError("Emotion prompt requires a {text} placeholder")
    return template.replace("{text}", text)


def normalize_emotion_output(raw: str) -> str:
    found = {match.group().lower() for match in _LABEL_PATTERN.finditer(raw)}
    return next(iter(found)) if len(found) == 1 else "invalid"


def run_emotion_inference(
    frame: pd.DataFrame, prompt: PromptCandidate, gateway: ModelGateway,
    *, model: str, temperature: float = 0.0,
) -> pd.DataFrame:
    """Keep every example, including ambiguous responses and recoverable errors."""
    required = ("example_id", "text", "expected_label")
    for column in required:
        if column not in frame:
            raise ValueError(f"Missing required column: {column}")
    if "{text}" not in prompt.text:
        raise ValueError("Emotion prompt requires a {text} placeholder")

    rows = []
    for source in frame.itertuples(index=False):
        started = time.perf_counter()
        record = {
            "example_id": source.example_id,
            "expected_label": source.expected_label,
            "raw_output": "",
            "normalized_output": "invalid",
            "output": "invalid",
            "input_tokens": None,
            "output_tokens": None,
            "latency_ms": 0.0,
            "cached": False,
            "error": None,
        }
        request = ModelRequest(
            task="emotion", model=model,
            messages=({"role": "user", "content": render_emotion_prompt(prompt.text, source.text)},),
            temperature=temperature,
        )
        try:
            # Task normalization needs the raw response and token counts even when
            # the output is ambiguous; the gateway handles calls, retries and cache.
            response = gateway.complete(request, validator=lambda _: True)
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError):
            raise
        except Exception as error:
            record["error"] = type(error).__name__
        else:
            normalized = normalize_emotion_output(response.text)
            record.update(
                raw_output=response.text,
                normalized_output=normalized,
                output=normalized,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                cached=response.cached,
                error="invalid_output" if normalized == "invalid" else None,
            )
        if record["latency_ms"] == 0.0:
            record["latency_ms"] = (time.perf_counter() - started) * 1000
        rows.append(record)
    return pd.DataFrame.from_records(rows, columns=(
        "example_id", "expected_label", "raw_output", "normalized_output", "output",
        "input_tokens", "output_tokens", "latency_ms", "cached", "error",
    ))
