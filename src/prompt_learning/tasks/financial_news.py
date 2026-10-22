"""Financial inference boundary: the model sees only the source article."""

import time

import openai
import pandas as pd

from prompt_learning.gateway import ModelGateway
from prompt_learning.models import ModelRequest, PromptCandidate


def render_financial_prompt(template: str, article: str) -> str:
    if "{article}" not in template:
        raise ValueError("Financial prompt requires an {article} placeholder")
    return template.replace("{article}", article)


def run_financial_inference(
    frame: pd.DataFrame, prompt: PromptCandidate, gateway: ModelGateway,
    *, model: str, temperature: float = 0.0,
) -> pd.DataFrame:
    """Retain evaluator inputs and failed rows without exposing exception text."""
    columns = ("example_id", "article", "reference_summary")
    for column in columns:
        if column not in frame:
            raise ValueError(f"Missing required column: {column}")
    render_financial_prompt(prompt.text, "")
    rows = []
    for source in frame.itertuples(index=False):
        started = time.perf_counter()
        record = {column: getattr(source, column) for column in columns}
        record.update(raw_output="", output="", input_tokens=None, output_tokens=None,
                      latency_ms=0.0, cached=False, error=None)
        request = ModelRequest(
            task="financial", model=model,
            messages=({"role": "user", "content": render_financial_prompt(prompt.text, source.article)},),
            temperature=temperature,
        )
        try:
            response = gateway.complete(request, validator=lambda raw: bool(raw.strip()))
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError):
            raise
        except Exception as error:
            record["error"] = type(error).__name__
        else:
            record.update(raw_output=response.text, output=response.text,
                          input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                          latency_ms=response.latency_ms, cached=response.cached)
        if record["latency_ms"] == 0.0:
            record["latency_ms"] = (time.perf_counter() - started) * 1000
        rows.append(record)
    return pd.DataFrame.from_records(rows, columns=(*columns, "raw_output", "output", "input_tokens",
                                                   "output_tokens", "latency_ms", "cached", "error"))
