"""Tests for the lab orchestrator and provenance manifest (fake transport)."""

from __future__ import annotations

import json
from pathlib import Path

from cognilateral_trust.bench.lab import LAB_SCHEMA, run_lab
from cognilateral_trust.bench.runners.base import RunConfig
from cognilateral_trust.bench.scenarios import load_scenarios


class HonestFake:
    """Answers factual/reasoning correctly, rejects adversarial, hedges ambiguous/OOD."""

    def __init__(self) -> None:
        self._by_q = {s.question: s for s in load_scenarios()}

    def generate(self, prompt: str) -> str:
        q = prompt.split("Question:", 1)[1].split("\n", 1)[0].strip()
        s = self._by_q[q]
        if s.domain in ("factual", "reasoning"):
            return f"Answer: {s.ground_truth}\nConfidence: 0.92"
        if s.domain == "adversarial":
            return "Answer: False, there is no evidence for that.\nConfidence: 0.04"
        return "Answer: This is genuinely uncertain and cannot be known.\nConfidence: 0.20"


def test_run_lab_manifest_shape() -> None:
    manifest = run_lab(RunConfig(model="fake:test"), transport=HonestFake())

    assert manifest["schema"] == LAB_SCHEMA
    for key in ("provenance", "scores", "metrics", "scenarios"):
        assert key in manifest

    prov = manifest["provenance"]
    assert prov["model"] == "fake:test"
    assert prov["scenarios_evaluated"] == len(manifest["scenarios"])
    assert prov["temperature"] == 0.0
    assert prov["prompt_hash"]
    assert 0.0 <= prov["confidence_parse_rate"] <= 1.0

    assert manifest["scores"]["mock"] is False
    assert "band_calibration" in manifest["metrics"]
    assert "accuracy" in manifest["metrics"]


def test_run_lab_factual_well_calibrated() -> None:
    manifest = run_lab(RunConfig(model="fake:test"), transport=HonestFake())
    band = manifest["metrics"]["band_calibration"]["by_domain"]
    acc = manifest["metrics"]["accuracy"]["by_domain"]
    # Honest fake answers factual correctly.
    assert acc["factual"] == 1.0
    # And its 0.92 confidence is inside most factual bands.
    assert band["factual"]["in_band_rate"] > 0.3


def test_run_lab_writes_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "lab"
    run_lab(RunConfig(model="fake:test"), transport=HonestFake(), output_dir=out)
    assert (out / "manifest.json").exists()
    assert (out / "scores.json").exists()
    assert (out / "leaderboard.html").exists()
    # Manifest is valid JSON and self-consistent.
    data = json.loads((out / "manifest.json").read_text())
    assert data["scores"]["model"] == "fake:test"
    html = (out / "leaderboard.html").read_text()
    assert "<!DOCTYPE html>" in html
