"""Validated model calls with bounded retries and a request-private SQLite cache.

The injected callable owns vendor authentication and response normalization. It
accepts a ModelRequest and returns a ModelResponse; no client is created here.
"""

from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import time

import openai

from prompt_learning.models import ModelPricing, ModelRequest, ModelResponse


class InvalidModelOutput(ValueError):
    """A completed model call failed the task's output validator."""


def estimate_cost_usd(
    response: ModelResponse, pricing: ModelPricing | None,
) -> Decimal | None:
    """Estimate incurred token cost; absent pricing is unavailable, cache is free."""
    if pricing is None:
        return None
    if response.cached:
        return Decimal("0")
    return (Decimal(response.input_tokens) * pricing.input_per_million
            + Decimal(response.output_tokens) * pricing.output_per_million) / Decimal(1_000_000)


def _request_key(request: ModelRequest) -> str:
    payload = {
        "task": request.task, "model": request.model,
        "messages": [dict(message) for message in request.messages],
        "temperature": request.temperature, "parameters": dict(request.parameters),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _transient(error: Exception) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError, openai.APIConnectionError)):
        return True
    return isinstance(error, openai.APIStatusError) and (
        error.status_code == 429 or 500 <= error.status_code < 600
    )


class ModelGateway:
    def __init__(
        self,
        client: Callable[[ModelRequest], ModelResponse],
        cache_path: Path,
        *,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("max_retries must be an integer between 0 and 3")
        self.client = client
        self.cache_path = Path(cache_path)
        self.max_retries = max_retries
        self.sleep = sleep
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.cache_path)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")

    def complete(
        self, request: ModelRequest, validator: Callable[[str], bool],
    ) -> ModelResponse:
        started = time.perf_counter()
        key = _request_key(request)
        with closing(sqlite3.connect(self.cache_path)) as connection, connection:
            row = connection.execute(
                "SELECT text, input_tokens, output_tokens FROM responses WHERE key = ?", (key,),
            ).fetchone()
            if row is not None:
                if validator(row[0]):
                    return ModelResponse(*row, latency_ms=(time.perf_counter() - started) * 1000,
                                         cached=True)
                connection.execute("DELETE FROM responses WHERE key = ?", (key,))

        for attempt in range(self.max_retries + 1):
            try:
                result = self.client(request)
            except Exception as error:
                if attempt == self.max_retries or not _transient(error):
                    raise
                self.sleep(float(2 ** attempt))
                continue
            if validator(result.text):
                break
            if attempt == self.max_retries:
                raise InvalidModelOutput("Model output failed validation")
        with closing(sqlite3.connect(self.cache_path)) as connection, connection:
            connection.execute(
                """INSERT OR REPLACE INTO responses (key, text, input_tokens, output_tokens)
                   VALUES (?, ?, ?, ?)""",
                (key, result.text, result.input_tokens, result.output_tokens),
            )
        return replace(result, latency_ms=(time.perf_counter() - started) * 1000, cached=False)
