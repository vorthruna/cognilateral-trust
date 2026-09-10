"""Runner core — transport protocol, run config, response parsing, scenario evaluation.

A :class:`ModelTransport` is the only thing that talks to a model: given a prompt it
returns text. Everything else (prompting, parsing, grading, sampling, aggregation) is
deterministic and transport-agnostic, which means the whole evaluation pipeline can be
unit-tested with a fake transport — no network, no Ollama.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from cognilateral_trust.bench.grading import GradeResult, grade
from cognilateral_trust.bench.prompts import build_prompt
from cognilateral_trust.bench.scenarios import BenchScenario
from cognilateral_trust.extractors import extract_confidence_from_text

__all__ = [
    "ModelTransport",
    "RunConfig",
    "SampleOutcome",
    "ScenarioOutcome",
    "parse_response",
    "evaluate_scenario",
    "make_scenario_runner",
]


class ModelTransport(Protocol):
    """Anything that can turn a prompt into model text."""

    def generate(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class RunConfig:
    """Evaluation parameters. Defaults favour determinism and reproducibility.

    Attributes:
        model: Model identifier (e.g. "qwen3:8b").
        host: Base URL of the model server (Ollama default).
        temperature: Sampling temperature; 0.0 = greedy/deterministic.
        seed: RNG seed passed to the backend for reproducibility.
        num_samples: Samples per scenario (>1 averages confidence, majority-votes
            correctness). Keep at 1 for fully deterministic runs.
        max_tokens: Max tokens to generate per response.
        timeout_s: Per-request timeout in seconds.
        default_confidence: Confidence used when none can be parsed from a response.
    """

    model: str
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    seed: int = 0
    num_samples: int = 1
    max_tokens: int = 512
    timeout_s: float = 120.0
    default_confidence: float = 0.5


@dataclass(frozen=True)
class SampleOutcome:
    """A single model sample for one scenario."""

    raw_response: str
    answer: str
    confidence: float
    confidence_found: bool
    correct: float
    grader: str
    rationale: str
    latency_ms: float


@dataclass(frozen=True)
class ScenarioOutcome:
    """Aggregated outcome for one scenario across all samples."""

    scenario_id: str
    domain: str
    question: str
    expected_confidence_low: float
    expected_confidence_high: float
    confidence: float
    correctness: float
    samples: tuple[SampleOutcome, ...]


_ANSWER_RE = re.compile(r"(?is)answer\s*:\s*(.*?)(?:\n\s*confidence\s*:?|\Z)")
_CONFIDENCE_LINE_RE = re.compile(r"(?im)^[ \t]*confidence\s*:?.*$")


def parse_response(text: str) -> tuple[str, float | None]:
    """Split a model response into (answer_text, confidence|None).

    Pulls the text after an ``Answer:`` label (up to the ``Confidence`` line, colon
    optional) and extracts a confidence from that line alone, so a confidence phrase
    inside the answer never becomes the sample confidence. Falls back to the whole
    text as the answer when no ``Answer:`` label is present; returns ``None`` for the
    confidence when no ``Confidence`` line is present.
    """
    text = text or ""
    match = _ANSWER_RE.search(text)
    answer = match.group(1).strip() if match else text.strip()
    confidence_line = _CONFIDENCE_LINE_RE.search(text)
    confidence = extract_confidence_from_text(confidence_line.group(0)) if confidence_line else None
    return answer, confidence


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_scenario(
    scenario: BenchScenario,
    transport: ModelTransport,
    config: RunConfig,
    grader: Callable[[BenchScenario, str], GradeResult] = grade,
) -> ScenarioOutcome:
    """Run ``num_samples`` model calls for one scenario and aggregate the outcome."""
    samples: list[SampleOutcome] = []
    prompt = build_prompt(scenario)

    for _ in range(max(1, config.num_samples)):
        start = time.perf_counter()
        raw = transport.generate(prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0

        answer, parsed_conf = parse_response(raw)
        found = parsed_conf is not None
        confidence = parsed_conf if found else config.default_confidence
        result = grader(scenario, answer)

        samples.append(
            SampleOutcome(
                raw_response=raw,
                answer=answer,
                confidence=confidence,
                confidence_found=found,
                correct=result.correct,
                grader=result.grader,
                rationale=result.rationale,
                latency_ms=latency_ms,
            )
        )

    agg_confidence = _mean([s.confidence for s in samples])
    # Majority vote on correctness; ties (mean == 0.5) resolve to incorrect.
    agg_correct = 1.0 if _mean([s.correct for s in samples]) > 0.5 else 0.0

    return ScenarioOutcome(
        scenario_id=scenario.id,
        domain=scenario.domain,
        question=scenario.question,
        expected_confidence_low=scenario.expected_confidence_low,
        expected_confidence_high=scenario.expected_confidence_high,
        confidence=agg_confidence,
        correctness=agg_correct,
        samples=tuple(samples),
    )


def make_scenario_runner(
    transport: ModelTransport,
    config: RunConfig,
    sink: list[ScenarioOutcome] | None = None,
    grader: Callable[[BenchScenario, str], GradeResult] = grade,
) -> Callable[[BenchScenario], "tuple[float, float]"]:
    """Build a ScenarioRunner for ``run_benchmark(runner=...)``.

    The returned callable evaluates one scenario and returns
    ``(confidence, correctness)``. If ``sink`` is provided, the full
    :class:`ScenarioOutcome` for each scenario is appended to it (used to build the
    reproducibility manifest).
    """

    def runner(scenario: BenchScenario) -> tuple[float, float]:
        outcome = evaluate_scenario(scenario, transport, config, grader)
        if sink is not None:
            sink.append(outcome)
        return outcome.confidence, outcome.correctness

    return runner
