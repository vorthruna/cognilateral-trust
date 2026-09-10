"""Model runners for TrustBench — turn a model into a (confidence, correctness) signal."""

from __future__ import annotations

from cognilateral_trust.bench.runners.base import (
    ModelTransport,
    RunConfig,
    SampleOutcome,
    ScenarioOutcome,
    evaluate_scenario,
    make_scenario_runner,
    parse_response,
)

__all__ = [
    "ModelTransport",
    "RunConfig",
    "SampleOutcome",
    "ScenarioOutcome",
    "evaluate_scenario",
    "make_scenario_runner",
    "parse_response",
]
