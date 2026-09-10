"""Framework-agnostic trust evaluation — single entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cognilateral_trust.accountability import (
    AccountabilityRecord,
    create_accountability_record,
)
from cognilateral_trust.core import (
    ConfidenceTier,
    evaluate_tier_routing,
)


@dataclass(frozen=True)
class TrustEvaluation:
    """Complete trust evaluation result for a single decision point."""

    confidence: float
    tier: ConfidenceTier
    route: str
    requires_warrant_check: bool
    requires_sovereignty_gate: bool
    should_proceed: bool
    accountability_record: AccountabilityRecord | None


def evaluate_trust(
    confidence: float,
    *,
    is_reversible: bool = True,
    touches_external: bool = False,
    context: dict[str, Any] | None = None,
) -> TrustEvaluation:
    """Evaluate trust for a decision point.

    Maps confidence to a tier, evaluates routing, and determines whether
    the action should proceed or be escalated.

    Usage:
        result = evaluate_trust(0.7)
        if result.should_proceed:
            perform_action()
        else:
            escalate(result.accountability_record.reasons)
    """
    if not isinstance(confidence, (int, float)):
        raise TypeError(f"confidence must be a number, got {type(confidence).__name__}")
    confidence = max(0.0, min(1.0, float(confidence)))
    tier_val = min(9, max(0, int(confidence * 10)))
    tier = ConfidenceTier(tier_val)
    routing = evaluate_tier_routing(tier_val)

    should_proceed = True
    verdict = "ACT"
    reasons: tuple[str, ...] = ()

    if routing.requires_sovereignty_gate:
        # Note: the sovereignty gate only engages at tier >= 7, i.e. confidence >= 0.7.
        # A low-confidence escalation check here would be dead code (confidence can
        # never be < 0.5 inside this branch). Low-confidence actions are routed to the
        # "basic"/"warrant_check" bands by route_by_tier(), not to the sovereignty gate.
        if not is_reversible:
            should_proceed = False
            verdict = "ESCALATE"
            reasons = ("irreversible action at sovereignty-grade tier",)
        elif touches_external:
            should_proceed = False
            verdict = "ESCALATE"
            reasons = ("external system impact at sovereignty-grade tier",)

    record = create_accountability_record(
        verdict=verdict,
        reasons=reasons,
        context=context or {},
        confidence=confidence,
        confidence_tier=tier_val,
    )

    return TrustEvaluation(
        confidence=confidence,
        tier=tier,
        route=routing.route,
        requires_warrant_check=routing.requires_warrant_check,
        requires_sovereignty_gate=routing.requires_sovereignty_gate,
        should_proceed=should_proceed,
        accountability_record=record,
    )
