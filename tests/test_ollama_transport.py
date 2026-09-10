"""Tests for the Ollama transport using an injected fake opener (no network)."""

from __future__ import annotations

import json
import urllib.error

import pytest

from cognilateral_trust.bench.runners.base import RunConfig
from cognilateral_trust.bench.runners.ollama import (
    OllamaError,
    OllamaTransport,
    OllamaUnavailableError,
    list_models,
    model_present,
    ollama_available,
    ollama_version,
)


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._b = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._b

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *a) -> bool:
        return False


def make_opener(routes: dict[str, dict], *, captured: list | None = None, raise_exc: Exception | None = None):
    def opener(request, timeout=None):
        if captured is not None:
            captured.append(request)
        if raise_exc is not None:
            raise raise_exc
        url = request.full_url
        for key, payload in routes.items():
            if key in url:
                return _Resp(payload)
        raise AssertionError(f"no route for {url}")

    return opener


class TestGenerate:
    def test_returns_response_text(self) -> None:
        opener = make_opener({"/api/generate": {"response": "Answer: Paris\nConfidence: 0.9"}})
        t = OllamaTransport(RunConfig(model="qwen3:8b"), opener=opener)
        assert t.generate("hi") == "Answer: Paris\nConfidence: 0.9"

    def test_sends_deterministic_options(self) -> None:
        captured: list = []
        opener = make_opener({"/api/generate": {"response": "x"}}, captured=captured)
        t = OllamaTransport(RunConfig(model="qwen3:8b", temperature=0.0, seed=7, max_tokens=128), opener=opener)
        t.generate("hello prompt")
        body = json.loads(captured[0].data.decode("utf-8"))
        assert body["model"] == "qwen3:8b"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0.0
        assert body["options"]["seed"] == 7
        assert body["options"]["num_predict"] == 128
        assert body["prompt"] == "hello prompt"

    def test_unreachable_raises_unavailable(self) -> None:
        opener = make_opener({}, raise_exc=urllib.error.URLError("connection refused"))
        t = OllamaTransport(RunConfig(model="m"), opener=opener)
        with pytest.raises(OllamaUnavailableError):
            t.generate("x")

    def test_http_error_raises_ollama_error(self) -> None:
        err = urllib.error.HTTPError("http://h/api/generate", 500, "boom", {}, None)
        opener = make_opener({}, raise_exc=err)
        t = OllamaTransport(RunConfig(model="m"), opener=opener)
        with pytest.raises(OllamaError):
            t.generate("x")


class TestDiagnostics:
    def test_version(self) -> None:
        opener = make_opener({"/api/version": {"version": "0.3.14"}})
        assert ollama_version("http://h", opener=opener) == "0.3.14"

    def test_list_models(self) -> None:
        opener = make_opener({"/api/tags": {"models": [{"name": "qwen3:8b"}, {"name": "llama3:8b"}]}})
        assert list_models("http://h", opener=opener) == ["qwen3:8b", "llama3:8b"]

    def test_model_present_exact_and_base(self) -> None:
        opener = make_opener({"/api/tags": {"models": [{"name": "qwen3:8b"}]}})
        assert model_present("qwen3:8b", "http://h", opener=opener) is True
        assert model_present("qwen3", "http://h", opener=opener) is True
        assert model_present("mistral", "http://h", opener=opener) is False

    def test_available_true_and_false(self) -> None:
        ok = make_opener({"/api/version": {"version": "0.3.0"}})
        assert ollama_available("http://h", opener=ok) is True
        down = make_opener({}, raise_exc=urllib.error.URLError("refused"))
        assert ollama_available("http://h", opener=down) is False
