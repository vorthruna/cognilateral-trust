"""Live integration test — runs only if a real Ollama server is reachable.

Skipped by default (CI and machines without Ollama). To exercise it, set
TRUSTBENCH_OLLAMA_MODEL to an installed model (e.g. 'qwen3:8b') and have
`ollama serve` running.
"""

from __future__ import annotations

import os

import pytest

from cognilateral_trust.bench.runners.base import RunConfig
from cognilateral_trust.bench.runners.ollama import OllamaTransport, model_present, ollama_available

_HOST = os.environ.get("TRUSTBENCH_OLLAMA_HOST", "http://localhost:11434")
_MODEL = os.environ.get("TRUSTBENCH_OLLAMA_MODEL")

_reason = "Ollama not reachable or TRUSTBENCH_OLLAMA_MODEL not set"
_enabled = bool(_MODEL) and ollama_available(_HOST) and (_MODEL is None or model_present(_MODEL, _HOST))


@pytest.mark.skipif(not _enabled, reason=_reason)
def test_real_ollama_smoke() -> None:
    from cognilateral_trust.bench.lab import run_lab

    config = RunConfig(model=_MODEL, host=_HOST, num_samples=1, max_tokens=256)
    # Evaluate just a couple of scenarios by limiting through the transport is not
    # supported; run the full set but assert structure only.
    manifest = run_lab(config, transport=OllamaTransport(config))
    assert manifest["scores"]["mock"] is False
    assert manifest["provenance"]["scenarios_evaluated"] > 0
    assert 0.0 <= manifest["metrics"]["band_calibration"]["overall"]["in_band_rate"] <= 1.0
