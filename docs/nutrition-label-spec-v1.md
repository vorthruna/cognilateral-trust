# AI Trust Nutrition Label — Specification v1.0

**Status:** Draft
**Date:** 2026-03-27
**Authors:** Cognilateral maintainers
**License:** CC-BY-4.0

## Purpose

A standard disclosure format for AI-generated content that has been trust-evaluated. The label tells the person downstream what confidence the system had, whether it was calibrated, and what the routing decision was.

## Format

### JSON

```json
{
  "ai_trust_label": {
    "version": "1.0",
    "evaluated": true,
    "confidence": 0.70,
    "tier": "C7",
    "route": "sovereignty_gate",
    "verdict": "ACT",
    "calibration_accuracy": 0.625,
    "calibration_method": "brier",
    "timestamp": "2026-03-27T12:00:00Z"
  }
}
```

### HTTP Header

```
X-AI-Trust-Label: evaluated=true; confidence=0.70; tier=C7; verdict=ACT; calibration=0.625
```

### Human-Readable

```
Trust evaluated. Confidence: 0.70 (C7). Calibration: 62.5%. Verdict: ACT.
```

### Not Evaluated

```json
{
  "ai_trust_label": {
    "version": "1.0",
    "evaluated": false
  }
}
```

Human-readable: `Not trust-evaluated. No confidence assessment was performed on this response.`

## Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Spec version ("1.0") |
| `evaluated` | boolean | Yes | Whether trust evaluation was performed |
| `confidence` | float | If evaluated | Agent confidence (0.0-1.0) |
| `tier` | string | If evaluated | Epistemic tier (C0-C9) |
| `route` | string | If evaluated | Routing decision: basic, warrant_check, sovereignty_gate |
| `verdict` | string | If evaluated | ACT or ESCALATE |
| `calibration_accuracy` | float | No | Historical calibration accuracy (0.0-1.0). Null if unknown. |
| `calibration_method` | string | No | How calibration was measured (e.g., "brier", "ece") |
| `timestamp` | string | If evaluated | ISO 8601 UTC timestamp of evaluation |

## Principles

1. **The person downstream deserves to know.** The label exists for the affected person, not the developer.
2. **Absence is information.** `evaluated: false` is a valid and important disclosure.
3. **Calibration is optional but encouraged.** Systems without outcome data should omit the field, not fabricate it.
4. **The label cannot lie.** If the system says 0.70 confidence, it must have evaluated at 0.70. Inflating or deflating for display purposes violates the spec.
5. **Stricter is always allowed.** A consent profile can cause ESCALATE where the base evaluation said ACT. The label reflects the final verdict, not the base verdict.

## Implementation

### Python (cognilateral-trust)

```python
from cognilateral_trust import nutrition_label

label = nutrition_label(0.70, calibration_accuracy=0.625)
print(label.disclosure)
# Trust evaluated. Confidence: 0.70 (C7). Calibration: 62.5%. Verdict: ACT.
```

## Adoption

Attach the label to any AI-generated content:
- API response body (`disclosure` field)
- HTTP response header (`X-AI-Trust-Label`)
- UI footer or tooltip
- Document metadata

## Versioning

Breaking changes increment the major version. New optional fields increment the minor version. This spec follows semver.
