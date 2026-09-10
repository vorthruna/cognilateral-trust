"""Ollama transport for TrustBench — evaluate a local model, zero dependencies.

Talks to a local Ollama server over its HTTP API using only the Python standard
library (``urllib``), so the package keeps its zero-dependency promise. The HTTP
opener is injectable, which lets the whole transport be unit-tested without a running
Ollama instance.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from cognilateral_trust.bench.runners.base import RunConfig

__all__ = [
    "DEFAULT_HOST",
    "OllamaError",
    "OllamaUnavailableError",
    "OllamaTransport",
    "ollama_version",
    "list_models",
    "ollama_available",
    "model_present",
]

DEFAULT_HOST = "http://localhost:11434"

# An opener mirrors urllib.request.urlopen(req, timeout=...). Injectable for testing.
Opener = Callable[..., Any]


class OllamaError(RuntimeError):
    """Ollama returned an error or an unexpected response."""


class OllamaUnavailableError(OllamaError):
    """The Ollama server could not be reached."""


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
    opener: Opener | None = None,
) -> dict[str, Any]:
    """Issue a JSON request and return the parsed body. Raises OllamaError variants."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # server reachable, returned an error code
        raise OllamaError(f"Ollama HTTP {exc.code} for {url}: {exc.reason}") from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise OllamaUnavailableError(f"Cannot reach Ollama at {url}: {reason}") from exc

    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Invalid JSON from {url}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise OllamaError(f"Unexpected response shape from {url}: {type(parsed).__name__}")
    return parsed


class OllamaTransport:
    """ModelTransport backed by a local Ollama server.

    Sends deterministic generation requests (temperature/seed from ``config``) to
    ``{host}/api/generate`` and returns the model's text.
    """

    def __init__(self, config: RunConfig, *, opener: Opener | None = None) -> None:
        self.config = config
        self._opener = opener

    def generate(self, prompt: str) -> str:
        url = f"{self.config.host.rstrip('/')}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "num_predict": self.config.max_tokens,
            },
        }
        data = _request(
            url,
            method="POST",
            payload=payload,
            timeout=self.config.timeout_s,
            opener=self._opener,
        )
        response = data.get("response", "")
        if not isinstance(response, str):
            raise OllamaError("Ollama response field was not text")
        return response


def ollama_version(host: str = DEFAULT_HOST, *, timeout: float = 5.0, opener: Opener | None = None) -> str:
    """Return the Ollama server version string (raises if unreachable)."""
    data = _request(f"{host.rstrip('/')}/api/version", timeout=timeout, opener=opener)
    return str(data.get("version", ""))


def list_models(host: str = DEFAULT_HOST, *, timeout: float = 5.0, opener: Opener | None = None) -> list[str]:
    """List locally available model tags (raises if unreachable)."""
    data = _request(f"{host.rstrip('/')}/api/tags", timeout=timeout, opener=opener)
    models = data.get("models") or []
    return [str(m.get("name", "")) for m in models if isinstance(m, dict)]


def ollama_available(host: str = DEFAULT_HOST, *, timeout: float = 5.0, opener: Opener | None = None) -> bool:
    """True if the Ollama server responds to a version probe."""
    try:
        ollama_version(host, timeout=timeout, opener=opener)
        return True
    except OllamaError:
        return False


def model_present(
    model: str, host: str = DEFAULT_HOST, *, timeout: float = 5.0, opener: Opener | None = None
) -> bool:
    """True if ``model`` (exact tag or base name) is available locally."""
    names = list_models(host, timeout=timeout, opener=opener)
    base = model.split(":")[0]
    return any(name == model or name.split(":")[0] == base for name in names)
