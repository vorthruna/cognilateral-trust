"""Tests for band-calibration and accuracy metrics."""

from __future__ import annotations

from cognilateral_trust.bench.metrics import accuracy_report, band_report, band_stats, metrics_report
from cognilateral_trust.bench.runners.base import ScenarioOutcome


def _outcome(domain: str, confidence: float, correctness: float, low: float, high: float) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=f"{domain}_x",
        domain=domain,
        question="q",
        expected_confidence_low=low,
        expected_confidence_high=high,
        confidence=confidence,
        correctness=correctness,
        samples=(),
    )


class TestBandStats:
    def test_in_band(self) -> None:
        st = band_stats([_outcome("factual", 0.95, 1.0, 0.9, 1.0)])
        assert st.in_band == 1 and st.in_band_rate == 1.0
        assert st.overconfident == 0 and st.underconfident == 0
        assert st.mean_signed_deviation == 0.0

    def test_overconfident(self) -> None:
        # band [0.0, 0.1], stated 0.25 => overconfident by 0.15
        st = band_stats([_outcome("out_of_distribution", 0.25, 1.0, 0.0, 0.1)])
        assert st.overconfident == 1 and st.in_band == 0
        assert abs(st.mean_signed_deviation - 0.15) < 1e-9

    def test_underconfident(self) -> None:
        # band [0.9, 1.0], stated 0.5 => underconfident by -0.4
        st = band_stats([_outcome("factual", 0.5, 1.0, 0.9, 1.0)])
        assert st.underconfident == 1
        assert abs(st.mean_signed_deviation - (-0.4)) < 1e-9

    def test_empty(self) -> None:
        st = band_stats([])
        assert st.count == 0 and st.in_band_rate == 0.0


class TestReports:
    def test_band_report_overall_and_by_domain(self) -> None:
        outs = [
            _outcome("factual", 0.95, 1.0, 0.9, 1.0),       # in band
            _outcome("out_of_distribution", 0.25, 1.0, 0.0, 0.1),  # over
        ]
        rep = band_report(outs)
        assert rep["overall"]["count"] == 2
        assert rep["by_domain"]["factual"]["in_band_rate"] == 1.0
        assert rep["by_domain"]["out_of_distribution"]["overconfident"] == 1

    def test_accuracy_report(self) -> None:
        outs = [
            _outcome("factual", 0.9, 1.0, 0.9, 1.0),
            _outcome("factual", 0.9, 0.0, 0.9, 1.0),
        ]
        rep = accuracy_report(outs)
        assert rep["overall"] == 0.5
        assert rep["by_domain"]["factual"] == 0.5

    def test_metrics_report_has_both_blocks(self) -> None:
        rep = metrics_report([_outcome("factual", 0.95, 1.0, 0.9, 1.0)])
        assert "band_calibration" in rep and "accuracy" in rep
