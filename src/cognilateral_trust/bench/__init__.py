"""TrustBench — epistemic honesty benchmark for AI agents."""

from __future__ import annotations

from cognilateral_trust.bench.fingerprint import (
    CalibrationFingerprint,
    FingerprintSpoke,
    fingerprint_to_dict,
    generate_fingerprint,
)
from cognilateral_trust.bench.leaderboard import generate_leaderboard
from cognilateral_trust.bench.scenarios import (
    DOMAINS,
    BenchScenario,
    load_scenarios,
)
from cognilateral_trust.bench.scoring import (
    BenchResult,
    BenchScore,
    DomainScore,
    calibration_score,
    expected_calibration_error,
    score_results,
)

__all__ = [
    # Scenarios
    "BenchScenario",
    "DOMAINS",
    "load_scenarios",
    # Scoring
    "BenchResult",
    "DomainScore",
    "BenchScore",
    "expected_calibration_error",
    "calibration_score",
    "score_results",
    # Fingerprint
    "FingerprintSpoke",
    "CalibrationFingerprint",
    "generate_fingerprint",
    "fingerprint_to_dict",
    # Leaderboard
    "generate_leaderboard",
]
