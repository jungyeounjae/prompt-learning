"""Offline gateway contracts: retry bounds, validated cache and configured costs."""

from dataclasses import replace
from decimal import Decimal
import sqlite3

import httpx
import openai
import pytest

from prompt_learning.config import load_config
from prompt_learning.gateway import ModelGateway, InvalidModelOutput, estimate_cost_usd
from prompt_learning.models import ModelRequest, ModelResponse


def request(**changes):
    values = dict(task="emotion", model="fake", messages=(
        {"role": "system", "content": "private full prompt"},
        {"role": "user", "content": "private raw article"},
    ), temperature=0.0)
    return ModelRequest(**(values | changes))


def response(text="joy"):
    return ModelResponse(text=text, input_tokens=120, output_tokens=8, latency_ms=12.0)


class ScriptedClient:
    def __init__(self, *outcomes):
        self.outcomes = iter(outcomes)
        self.requests = []

    def __call__(self, item):
        self.requests.append(item)
        result = next(self.outcomes)
        if isinstance(result, Exception):
            raise result
        return result


def status_error(code):
    types = {400: openai.BadRequestError, 401: openai.AuthenticationError,
             429: openai.RateLimitError, 503: openai.InternalServerError}
    return types[code]("failure", response=httpx.Response(
        code, request=httpx.Request("POST", "https://example.invalid")), body=None)


@pytest.mark.parametrize("error", [ConnectionError("offline"),
    openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid")),
    status_error(429), status_error(503)])
def test_gateway_retries_transient_error_three_times(tmp_path, error):
    client = ScriptedClient(error, error, error, response())
    delays = []
    gateway = ModelGateway(client, tmp_path / "cache.sqlite", sleep=delays.append)
    result = gateway.complete(request(), lambda text: text == "joy")
    assert result.text == "joy"
    assert len(client.requests) == 4
    assert delays == [1.0, 2.0, 4.0]


def test_gateway_stops_after_three_retries(tmp_path):
    client = ScriptedClient(*[ConnectionError("offline") for _ in range(5)])
    gateway = ModelGateway(client, tmp_path / "cache.sqlite", sleep=lambda _: None)
    with pytest.raises(ConnectionError):
        gateway.complete(request(), lambda _: True)
    assert len(client.requests) == 4


@pytest.mark.parametrize("code", [401, 400])
def test_gateway_does_not_retry_authentication_error(tmp_path, code):
    error = status_error(code)
    client = ScriptedClient(error, response())
    gateway = ModelGateway(client, tmp_path / "cache.sqlite", sleep=lambda _: pytest.fail("retried"))
    with pytest.raises(type(error)):
        gateway.complete(request(), lambda _: True)
    assert len(client.requests) == 1


def test_gateway_caches_by_hashed_request(tmp_path):
    path = tmp_path / "nested" / "cache.sqlite"
    client = ScriptedClient(response())
    first = ModelGateway(client, path).complete(request(), lambda _: True)
    second = ModelGateway(client, path).complete(request(), lambda _: True)
    assert first.cached is False
    assert second.cached is True
    assert (second.text, second.input_tokens, second.output_tokens) == ("joy", 120, 8)
    assert len(client.requests) == 1
    with sqlite3.connect(path) as connection:
        key, = connection.execute("SELECT key FROM responses").fetchone()
    assert len(key) == 64 and all(char in "0123456789abcdef" for char in key)


@pytest.mark.parametrize("changes", [
    {"task": "financial"}, {"model": "other"}, {"temperature": 0.5},
    {"messages": ({"role": "user", "content": "other article"},)},
    {"parameters": {"max_tokens": 8}},
])
def test_cache_identity_includes_all_request_inputs(tmp_path, changes):
    client = ScriptedClient(response(), response("changed"))
    gateway = ModelGateway(client, tmp_path / "cache.sqlite")
    gateway.complete(request(), lambda _: True)
    assert gateway.complete(request(**changes), lambda _: True).text == "changed"
    assert len(client.requests) == 2


def test_cache_does_not_store_api_key_or_raw_input_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-api-key")
    path = tmp_path / "cache.sqlite"
    ModelGateway(ScriptedClient(response()), path).complete(request(), lambda _: True)
    with sqlite3.connect(path) as connection:
        dump = "\n".join(connection.iterdump())
        columns = {row[1] for row in connection.execute("PRAGMA table_info(responses)")}
    assert columns == {"key", "text", "input_tokens", "output_tokens", "created_at"}
    for secret in ("secret-api-key", "private raw article", "private full prompt"):
        assert secret not in dump
        assert secret.encode() not in path.read_bytes()


def test_gateway_discards_invalid_cached_value(tmp_path):
    path = tmp_path / "cache.sqlite"
    ModelGateway(ScriptedClient(response("bad")), path).complete(request(), lambda _: True)

    def live(item):
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT count(*) FROM responses").fetchone()[0] == 0
        return response()

    result = ModelGateway(live, path).complete(request(), lambda text: text == "joy")
    assert result.text == "joy" and result.cached is False


def test_invalid_live_output_is_retried_and_only_valid_replacement_is_cached(tmp_path):
    path = tmp_path / "cache.sqlite"
    client = ScriptedClient(response("bad"), response())
    gateway = ModelGateway(client, path, sleep=lambda _: pytest.fail("format retry slept"))
    result = gateway.complete(request(), lambda text: text == "joy")
    assert result.text == "joy" and not result.cached
    assert gateway.complete(request(), lambda text: text == "joy").cached
    assert len(client.requests) == 2


def test_gateway_stops_after_invalid_output_retry_budget(tmp_path):
    client = ScriptedClient(*[response("bad") for _ in range(5)])
    gateway = ModelGateway(client, tmp_path / "cache.sqlite", sleep=lambda _: None)
    with pytest.raises(InvalidModelOutput):
        gateway.complete(request(), lambda text: text == "joy")
    assert len(client.requests) == 4


@pytest.mark.parametrize("count", [-1, 4, True])
def test_gateway_rejects_retry_settings_outside_policy(tmp_path, count):
    with pytest.raises(ValueError, match="max_retries"):
        ModelGateway(ScriptedClient(), tmp_path / "cache.sqlite", max_retries=count)


def test_gateway_supports_disabling_retries(tmp_path):
    client = ScriptedClient(ConnectionError("offline"), response())
    gateway = ModelGateway(client, tmp_path / "cache.sqlite", max_retries=0)
    with pytest.raises(ConnectionError):
        gateway.complete(request(), lambda _: True)
    assert len(client.requests) == 1


def test_unexpected_errors_are_not_retried(tmp_path):
    client = ScriptedClient(RuntimeError("application failure"), response())
    gateway = ModelGateway(client, tmp_path / "cache.sqlite")
    with pytest.raises(RuntimeError):
        gateway.complete(request(), lambda _: True)
    assert len(client.requests) == 1


def test_hash_ignores_mapping_key_order(tmp_path):
    client = ScriptedClient(response())
    gateway = ModelGateway(client, tmp_path / "cache.sqlite")
    gateway.complete(request(parameters={"a": 1, "b": 2}), lambda _: True)
    result = gateway.complete(request(parameters={"b": 2, "a": 1}), lambda _: True)
    assert result.cached and len(client.requests) == 1


def pricing_config(tmp_path, rates="input_per_million: '0.125'\n    output_per_million: '0.875'"):
    path = tmp_path / "config.yaml"
    path.write_text("task: emotion\ndataset: {name: fake, revision: pinned}\n"
                    "models: {inference: fake, evaluator: fake, meta: fake}\n"
                    "pricing:\n  fake:\n    " + rates + "\n", encoding="utf-8")
    return load_config(path)


def test_cost_uses_separate_input_and_output_rates(tmp_path):
    config = pricing_config(tmp_path)
    pricing = config.pricing["fake"]
    assert pricing.input_per_million == Decimal("0.125")
    assert pricing.output_per_million == Decimal("0.875")
    assert estimate_cost_usd(response(), pricing) == Decimal("0.000022")


def test_missing_pricing_is_unavailable(tmp_path):
    config = pricing_config(tmp_path)
    assert estimate_cost_usd(response(), config.pricing.get("unknown")) is None


def test_cached_responses_incur_no_model_cost(tmp_path):
    pricing = pricing_config(tmp_path).pricing["fake"]
    assert estimate_cost_usd(replace(response(), cached=True), pricing) == Decimal("0")


@pytest.mark.parametrize("rate", ["-1", "NaN", "Infinity", "true", "oops"])
def test_config_rejects_invalid_model_prices(tmp_path, rate):
    with pytest.raises(ValueError, match="pricing.fake.input_per_million"):
        pricing_config(tmp_path, f"input_per_million: {rate}\n    output_per_million: '1'")
