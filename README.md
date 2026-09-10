# cognilateral-trust

**Your AI agent is about to act. Should it?**

`cognilateral-trust` answers that question. You give it a confidence score. It tells you whether to proceed, verify, or escalate — and produces an immutable audit record of every decision.

[![PyPI](https://img.shields.io/pypi/v/cognilateral-trust)](https://pypi.org/project/cognilateral-trust/)
[![Python](https://img.shields.io/pypi/pyversions/cognilateral-trust)](https://pypi.org/project/cognilateral-trust/)
[![License](https://img.shields.io/github/license/heymumford/cognilateral-trust)](LICENSE)

```bash
pip install cognilateral-trust
```

Zero dependencies. Python 3.11+.

### TrustBench

TrustBench evaluates how honestly a model reports its own confidence, across 210
scenarios in 5 domains (factual, reasoning, ambiguous, out-of-distribution,
adversarial). It can run a **real local model** via [Ollama](https://ollama.com) —
zero extra dependencies — and emits a falsifiable record (raw responses, grades,
latencies, full provenance).

```bash
ollama pull qwen3:8b
trust-bench doctor --model qwen3:8b
trust-bench run --model qwen3:8b --provider ollama --lab-dir runs/qwen3-8b
```

It reports **band adherence** (did stated confidence land in the calibrated band? — the
headline honesty metric), **ECE** (calibration error on the answerable domains), and
**accuracy**. Full methodology and metric definitions: [docs/LAB.md](docs/LAB.md).

A self-labeled **synthetic baseline** is also included as a harness check (not a model
evaluation): `trust-bench run --model baseline --mock`.

---

## What It Is

An epistemic trust layer for AI agents. One function call between your agent's decision and its action.

**The problem:** AI agents act on confidence they haven't earned. A model says "I'm 90% sure" — but is that calibrated? Is this action reversible? Does it affect a person? Most frameworks skip these questions.

**The solution:** `evaluate_trust()` takes a confidence score and returns a routing decision — ACT, verify evidence, or escalate to a human. Every call produces an accountability record. No external services, no API keys, no dependencies.

## How It Works

```
Your agent's confidence (0.0 → 1.0)
        │
        ▼
┌─────────────────────────────────────┐
│         evaluate_trust()            │
│                                     │
│  1. Map confidence → epistemic tier │
│     (C0-C9, ten levels)            │
│                                     │
│  2. Route by tier:                  │
│     C0-C3 → basic (act freely)     │
│     C4-C6 → warrant_check (verify) │
│     C7-C9 → sovereignty_gate       │
│                                     │
│  3. Check context:                  │
│     - Is this reversible?           │
│     - Does it touch external systems│
│     - Welfare constraint (D-05)     │
│                                     │
│  4. Return: ACT or ESCALATE        │
│     + accountability record         │
└─────────────────────────────────────┘
        │
        ▼
  ACT (safe to proceed)
  — or —
  ESCALATE (needs human review, with reasons)
```

## What You Put In

| Input | Type | Required | What It Means |
|-------|------|----------|---------------|
| `confidence` | `float` (0.0-1.0) | Yes | How confident is your agent in this action? |
| `is_reversible` | `bool` | No (default: `True`) | Can this action be undone? |
| `touches_external` | `bool` | No (default: `False`) | Does this affect systems outside your control? |
| `context` | `dict` | No | Free-form metadata for the audit trail |

**That's it.** One float is the minimum. The other parameters sharpen the routing.

## What You Get Out

| Output | Type | What It Tells You |
|--------|------|-------------------|
| `should_proceed` | `bool` | **The answer.** Act or escalate. |
| `tier` | `ConfidenceTier` | Which of 10 epistemic tiers (C0-C9) this confidence maps to |
| `route` | `str` | `"basic"`, `"warrant_check"`, or `"sovereignty_gate"` |
| `accountability_record` | `AccountabilityRecord` | Immutable record: who decided, why, when, at what confidence |

The accountability record is the durable value. It answers "why did the agent do that?" after the fact.

## Quick Start

```python
from cognilateral_trust import evaluate_trust

# Your agent has 70% confidence in its next action
result = evaluate_trust(0.7)

if result.should_proceed:
    perform_action()
else:
    escalate_to_human(result.accountability_record.reasons)
```

### With context

```python
result = evaluate_trust(
    0.85,
    is_reversible=False,     # can't undo this
    touches_external=True,   # affects a live system
)
# verdict: ESCALATE — irreversible + external at sovereignty-gate tier
```

### Extract confidence from LLM output

Don't have a confidence score? Extract one from the LLM's response:

```python
from cognilateral_trust import extract_confidence

# Works with OpenAI response dicts, Anthropic response dicts, or plain text
confidence = extract_confidence(llm_response)
result = evaluate_trust(confidence)
```

```python
from cognilateral_trust import extract_confidence_from_text

confidence = extract_confidence_from_text("I'm about 73% confident in this answer")
# => 0.73
```

Parses percentages, decimals, verbal qualifiers ("highly confident", "somewhat uncertain"), and logprobs.

### Nutrition Label — Tell the End User

Attach a standard disclosure to any AI-generated output:

```python
from cognilateral_trust import nutrition_label

label = nutrition_label(0.7, calibration_accuracy=0.625)
print(label)
# Trust evaluated. Confidence: 0.70 (C7). Calibration: 62.5%. Verdict: ACT.
```

For responses that weren't evaluated:

```python
from cognilateral_trust import not_evaluated_label

label = not_evaluated_label()
# Not trust-evaluated. No confidence assessment was performed on this response.
```

The person downstream deserves to know.

---

## Deeper Capabilities

### Calibration — Learn From Outcomes

Track whether your agent's confidence was actually justified:

```python
from cognilateral_trust import CalibratedTrustEngine

engine = CalibratedTrustEngine()

# Evaluate
eval_id = engine.evaluate(0.8)

# Later, record what actually happened
engine.record_outcome(eval_id, correct=True)

# Get calibration stats
stats = engine.calibration_report()
# Brier score, ECE, per-tier accuracy
```

Over time, this tells you: "When your agent says 80%, is it right 80% of the time?"

### Warrants — Evidence-Backed Confidence That Decays

Confidence should erode when evidence gets stale:

```python
from cognilateral_trust import Warrant, evaluate_trust_with_warrant

warrant = Warrant(
    confidence=0.9,
    evidence_source="unit tests pass",
    ttl_seconds=3600,  # valid for 1 hour
)

# Effective confidence decays linearly toward 0 as the warrant ages
result = evaluate_trust_with_warrant(warrant)
```

### Claims Extraction + Fidelity Verification

Extract claims from text and verify them against source material:

```python
from cognilateral_trust import extract_claims, verify_fidelity

# Extract structured claims
claims = extract_claims("The model achieves 95% accuracy, outperforming GPT-4.")
# => [factual: "achieves 95% accuracy", comparative: "outperforming GPT-4"]

# Verify against source
result = verify_fidelity(
    claim="The system handles 10,000 requests per second",
    source="Our benchmarks show 10,000 req/s under load",
)
# result.supported = True, result.score = 0.78
```

### Epistemic Firewall — Prevent Acting on Hunches

Block actions that demand tested evidence when only hunches exist:

```python
from cognilateral_trust import check_epistemic_mismatch, EpistemicLevel

result = check_epistemic_mismatch(
    demanded=EpistemicLevel.VALIDATED,  # deploy requires validated evidence
    supplied=EpistemicLevel.OBSERVED,   # we only have observations
)
# result.is_mismatch = True, result.gap = 3
```

Seven levels: `RAW` < `OBSERVED` < `MEASURED` < `TESTED` < `VALIDATED` < `FALSIFIABLE` < `GOVERNANCE`.

### Sovereignty Gate — The Welfare Constraint

The D-05 hard constraint: welfare-critical actions **always** escalate, regardless of confidence:

```python
from cognilateral_trust import evaluate_sovereignty

decision = evaluate_sovereignty(
    confidence=0.99,
    is_reversible=True,
    tests_pass=True,
    welfare_affected=True,  # someone's wellbeing is at stake
)
# verdict: ESCALATE — welfare gate overrides confidence
```

### Middleware + Decorators

```python
from cognilateral_trust import trust_gate

@trust_gate(min_confidence=0.6)
def deploy(confidence: float, **kwargs):
    """Only runs if confidence >= 0.6"""
    ...
```

### Persistence — Survive Restarts

```python
from cognilateral_trust import JSONLPredictionStore, JSONLAccountabilityStore

predictions = JSONLPredictionStore("./data/predictions.jsonl")
accountability = JSONLAccountabilityStore("./data/accountability.jsonl")
```

### Agent Lifecycle — Trust-Gated Spawning

```python
from cognilateral_trust import spawn_gate

@spawn_gate(min_confidence=0.7)
def create_worker(confidence: float):
    """Only spawns if parent agent has sufficient trust"""
    ...
```

---

## Framework Integrations

### LangGraph

```python
from cognilateral_trust import evaluate_trust

def trust_node(state):
    result = evaluate_trust(state["confidence"])
    return {**state, "proceed": result.should_proceed}
```

Full example: [`examples/langgraph_trust_node.py`](examples/langgraph_trust_node.py)

### CrewAI

```python
from cognilateral_trust import evaluate_trust

def trust_check(confidence, **kwargs):
    result = evaluate_trust(confidence, **kwargs)
    return "PROCEED" if result.should_proceed else f"ESCALATE: {result.accountability_record.reasons}"
```

Full example: [`examples/crewai_trust_tool.py`](examples/crewai_trust_tool.py)

### Any Framework

`evaluate_trust()` takes a float, returns a decision. Wrap it however your framework expects.

---

## CLI

```bash
$ trust-check 0.7
ACT — C7 (sovereignty_gate)

$ trust-check 0.92 --irreversible
ESCALATE — C9 (sovereignty_gate): irreversible action at sovereignty-grade tier

$ trust-check 0.3 --json
{"confidence": 0.3, "tier": "C3", "route": "basic", "should_proceed": true, ...}
```

## Hosted API

The hosted API at `cognilateral.com/api/v1/*` was sunset on 2026-04-12. The PyPI library is the product and runs entirely on your machine — there is no hosted dependency. If you arrived here looking for the previous `curl` recipe, install the library instead. If you need an HTTP-callable surface, the library is small enough to wrap in a self-hosted FastAPI/Flask service.

## Examples

| Example | What It Shows |
|---------|---------------|
| [`demo_trust_agent.py`](examples/demo_trust_agent.py) | Full agent with extraction, warrants, and sovereignty |
| [`demo_calibration_loop.py`](examples/demo_calibration_loop.py) | Track and improve agent calibration over time |
| [`langgraph_trust_node.py`](examples/langgraph_trust_node.py) | Trust gate for LangGraph pipelines |
| [`crewai_trust_tool.py`](examples/crewai_trust_tool.py) | CrewAI tool wrapping trust evaluation |
| [`openai_trust_wrapper.py`](examples/openai_trust_wrapper.py) | Confidence extraction from OpenAI responses |
| [`anthropic_trust_wrapper.py`](examples/anthropic_trust_wrapper.py) | Confidence extraction from Anthropic responses |
| [`dspy_trust_module.py`](examples/dspy_trust_module.py) | DSPy module with trust-gated assertions |
| [`github_actions_trust_gate.yml`](examples/github_actions_trust_gate.yml) | CI gate: block auto-merge when confidence is low |

## Why This Exists

Guardrails protect systems. Trust protects people.

When an AI agent acts on misplaced confidence, the person at the other end pays — the patient who gets a wrong diagnosis, the student who gets a fabricated citation, the operator who ships a broken deploy at 3am because the model said it was fine.

This library exists so AI systems can say "I'm not sure enough to act on this" **before** the damage happens.

## Contact

Primary channel: `eric@cognilateral.com`. See [the contact page](CONTACT.md) for the full list of channels and the [privacy policy](docs/PRIVACY.md).

## License

Apache-2.0
