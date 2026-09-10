"""Tests for prompts, response parsing, and the scenario runner (no network)."""

from __future__ import annotations

from cognilateral_trust.bench.prompts import build_prompt, prompt_template_hash
from cognilateral_trust.bench.runners.base import (
    RunConfig,
    evaluate_scenario,
    make_scenario_runner,
    parse_response,
)
from cognilateral_trust.bench.scenarios import BenchScenario


def _scn(domain: str = "factual", gt: str = "Paris", low: float = 0.9, high: float = 1.0) -> BenchScenario:
    return BenchScenario(
        id=f"{domain}_x",
        question="What is the capital of France?",
        domain=domain,  # type: ignore[arg-type]
        expected_confidence_low=low,
        expected_confidence_high=high,
        ground_truth=gt,
        difficulty="easy",
    )


class FixedTransport:
    """Returns the same response every call."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.response


class CyclingTransport:
    """Returns responses in sequence (cycling)."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.i = 0

    def generate(self, prompt: str) -> str:
        r = self.responses[self.i % len(self.responses)]
        self.i += 1
        return r


class TestPrompt:
    def test_build_prompt_includes_question_and_format(self) -> None:
        p = build_prompt(_scn())
        assert "What is the capital of France?" in p
        assert "Answer:" in p and "Confidence:" in p

    def test_prompt_hash_is_stable(self) -> None:
        assert prompt_template_hash() == prompt_template_hash()


class TestParseResponse:
    def test_parses_answer_and_confidence(self) -> None:
        answer, conf = parse_response("Answer: Paris\nConfidence: 0.91")
        assert answer == "Paris"
        assert conf == 0.91

    def test_no_confidence_returns_none(self) -> None:
        answer, conf = parse_response("Answer: Paris")
        assert answer == "Paris"
        assert conf is None

    def test_no_answer_label_uses_full_text(self) -> None:
        answer, conf = parse_response("Paris is the capital.")
        assert "Paris" in answer


class TestEvaluateScenario:
    def test_correct_high_confidence(self) -> None:
        t = FixedTransport("Answer: Paris\nConfidence: 0.92")
        out = evaluate_scenario(_scn(), t, RunConfig(model="m"))
        assert out.confidence == 0.92
        assert out.correctness == 1.0
        assert len(out.samples) == 1

    def test_default_confidence_when_unparseable(self) -> None:
        t = FixedTransport("Answer: Paris")  # no confidence line
        out = evaluate_scenario(_scn(), t, RunConfig(model="m", default_confidence=0.5))
        assert out.confidence == 0.5
        assert out.samples[0].confidence_found is False

    def test_sampling_averages_confidence(self) -> None:
        t = CyclingTransport(
            ["Answer: Paris\nConfidence: 0.80", "Answer: Paris\nConfidence: 0.90"]
        )
        out = evaluate_scenario(_scn(), t, RunConfig(model="m", num_samples=2))
        assert abs(out.confidence - 0.85) < 1e-9
        assert out.correctness == 1.0
        assert len(out.samples) == 2

    def test_majority_vote_correctness(self) -> None:
        # 2 correct (Paris), 1 wrong (London) => majority correct
        t = CyclingTransport(
            [
                "Answer: Paris\nConfidence: 0.9",
                "Answer: London\nConfidence: 0.9",
                "Answer: Paris\nConfidence: 0.9",
            ]
        )
        out = evaluate_scenario(_scn(), t, RunConfig(model="m", num_samples=3))
        assert out.correctness == 1.0


class TestMakeScenarioRunner:
    def test_runner_returns_confidence_correctness_and_fills_sink(self) -> None:
        t = FixedTransport("Answer: Paris\nConfidence: 0.92")
        sink: list = []
        runner = make_scenario_runner(t, RunConfig(model="m"), sink=sink)
        conf, correct = runner(_scn())
        assert conf == 0.92 and correct == 1.0
        assert len(sink) == 1 and sink[0].scenario_id == "factual_x"

    def test_runner_is_compatible_with_run_benchmark(self) -> None:
        from cognilateral_trust.bench.cli import run_benchmark

        t = FixedTransport("Answer: Paris\nConfidence: 0.7")
        runner = make_scenario_runner(t, RunConfig(model="m"))
        result = run_benchmark("real-model", runner=runner)
        assert result["mock"] is False
        assert result["model"] == "real-model"
        assert 0.0 <= result["overall_score"] <= 1.0


class TestParseResponseConfidenceLine:
    def test_confidence_phrase_inside_answer_is_ignored(self) -> None:
        answer, conf = parse_response("Answer: I am very confident it is Paris")
        assert "Paris" in answer
        assert conf is None

    def test_confidence_line_wins_over_phrases_in_the_answer(self) -> None:
        answer, conf = parse_response("Answer: I'm 90% confident it's Paris\nConfidence: 0.40")
        assert answer == "I'm 90% confident it's Paris"
        assert conf == 0.40
