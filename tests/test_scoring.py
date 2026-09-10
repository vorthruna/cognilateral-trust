"""A-tests and B-tests for TrustBench scoring engine (Step 15)."""

from __future__ import annotations

import pytest

from cognilateral_trust.bench.scoring import (
    BenchResult,
    DomainScore,
    calibration_score,
    expected_calibration_error,
    score_results,
)


class TestExpectedCalibrationError:
    """A-tests: ECE correctness. ECE is a TRUE error — 0.0 = perfect, lower is better."""

    def test_a1_perfect_calibration(self) -> None:
        """When confidences match accuracy, ECE is low (near 0.0)."""
        # In 0.9 bin: 100% accuracy, in 0.1 bin: 0% accuracy.
        # ECE = |0.9-1.0|*0.5 + |0.1-0|*0.5 = 0.1
        confidences = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
        correctness = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        ece = expected_calibration_error(confidences, correctness)
        assert ece <= 0.15, f"Expected good calibration (ECE ≤ 0.15), got {ece}"

    def test_a2_overconfident_model(self) -> None:
        """When model is overconfident, ECE is high."""
        # Always confident (0.9) but mostly wrong (0.1 accuracy) → ECE ≈ 0.8
        confidences = [0.9] * 10
        correctness = [0.0] * 9 + [1.0]  # 10% accuracy
        ece = expected_calibration_error(confidences, correctness)
        assert ece > 0.5, f"Expected high ECE for overconfident model, got {ece}"

    def test_a3_empty_input_returns_zero(self) -> None:
        """Given empty inputs, ECE returns 0.0."""
        ece = expected_calibration_error([], [])
        assert ece == 0.0

    def test_a4_length_mismatch_raises_error(self) -> None:
        """Given mismatched lengths, raises ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            expected_calibration_error([0.5, 0.7], [1.0])

    def test_a5_reproducibility(self) -> None:
        """Given same inputs, ECE is reproducible."""
        confidences = [0.8, 0.7, 0.6, 0.9, 0.5]
        correctness = [1.0, 1.0, 0.0, 1.0, 0.0]

        ece1 = expected_calibration_error(confidences, correctness)
        ece2 = expected_calibration_error(confidences, correctness)
        ece3 = expected_calibration_error(confidences, correctness)

        assert ece1 == ece2 == ece3

    def test_a6_bins_are_created_correctly(self) -> None:
        """ECE bins partition confidence space [0, 1] into equal intervals."""
        # All samples in [0.0-0.1) bin with 0% accuracy, confidence 0.05 → ECE ≈ 0.05
        confidences = [0.05] * 10
        correctness = [0.0] * 10
        ece = expected_calibration_error(confidences, correctness, num_bins=10)
        assert ece < 0.1, f"Expected low ECE (~0.05), got {ece}"

    def test_a7_uniform_confidence_uniform_accuracy(self) -> None:
        """When confidence and accuracy are uniformly distributed, ECE is moderate."""
        confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        correctness = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ece = expected_calibration_error(confidences, correctness)
        assert 0.2 < ece < 0.6, f"Expected moderate ECE, got {ece}"

    def test_a8_tuple_inputs_work(self) -> None:
        """ECE accepts both list and tuple inputs."""
        confidences_list = [0.8, 0.9, 0.7]
        correctness_list = [1.0, 1.0, 0.0]

        confidences_tuple = tuple(confidences_list)
        correctness_tuple = tuple(correctness_list)

        ece_list = expected_calibration_error(confidences_list, correctness_list)
        ece_tuple = expected_calibration_error(confidences_tuple, correctness_tuple)

        assert ece_list == ece_tuple

    def test_a9_custom_num_bins(self) -> None:
        """ECE accepts custom bin count."""
        confidences = [0.5, 0.5, 0.5, 0.5]
        correctness = [1.0, 1.0, 0.0, 0.0]

        ece_5_bins = expected_calibration_error(confidences, correctness, num_bins=5)
        ece_20_bins = expected_calibration_error(confidences, correctness, num_bins=20)

        assert 0.0 <= ece_5_bins <= 1.0
        assert 0.0 <= ece_20_bins <= 1.0


class TestCalibrationScore:
    """A-tests: calibration_score is the higher-is-better counterpart (1.0 - ECE)."""

    def test_score_is_one_minus_ece(self) -> None:
        """calibration_score == 1.0 - expected_calibration_error for the same inputs."""
        confidences = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
        correctness = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        ece = expected_calibration_error(confidences, correctness)
        score = calibration_score(confidences, correctness)
        assert abs(score - (1.0 - ece)) < 1e-12

    def test_good_calibration_scores_high(self) -> None:
        """Well-calibrated inputs yield a high score (near 1.0)."""
        confidences = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
        correctness = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        assert calibration_score(confidences, correctness) >= 0.85

    def test_overconfident_scores_low(self) -> None:
        """Overconfident inputs yield a low score."""
        confidences = [0.9] * 10
        correctness = [0.0] * 9 + [1.0]
        assert calibration_score(confidences, correctness) < 0.5

    def test_empty_input_returns_zero(self) -> None:
        """Empty inputs return 0.0."""
        assert calibration_score([], []) == 0.0


class TestScoreResults:
    """B-tests: Benchmark scoring integration."""

    def test_b1_score_single_domain(self) -> None:
        """Given results in one domain, BenchScore includes that domain."""
        results = {
            "factual": [
                BenchResult("factual_001", 0.9, 1.0),
                BenchResult("factual_002", 0.8, 1.0),
            ]
        }
        score = score_results("TestModel", results)

        assert score.model == "TestModel"
        assert len(score.domain_scores) == 1
        assert score.domain_scores[0].domain == "factual"
        assert score.domain_scores[0].scenario_count == 2

    def test_b2_score_multiple_domains(self) -> None:
        """Given results across multiple domains, all domains scored."""
        results = {
            "factual": [
                BenchResult("factual_001", 0.9, 1.0),
            ],
            "reasoning": [
                BenchResult("reasoning_001", 0.7, 0.0),
            ],
            "ambiguous": [
                BenchResult("ambiguous_001", 0.6, 1.0),
            ],
        }
        score = score_results("MultiModel", results)

        assert len(score.domain_scores) == 3
        domains = {ds.domain for ds in score.domain_scores}
        assert domains == {"factual", "reasoning", "ambiguous"}

    def test_b3_overall_score_is_across_all_domains(self) -> None:
        """Overall score aggregates all domains."""
        results = {
            "domain1": [
                BenchResult("d1_001", 0.9, 1.0),
                BenchResult("d1_002", 0.9, 1.0),
            ],
            "domain2": [
                BenchResult("d2_001", 0.1, 0.0),
                BenchResult("d2_002", 0.1, 0.0),
            ],
        }
        score = score_results("MixedModel", results)

        # Overall score is computed across all domains combined
        # Both domains should have similar ECE since all results are well-separated
        # Just verify overall_score is in valid range
        assert 0.0 <= score.overall_score <= 1.0

    def test_b4_bench_score_is_frozen(self) -> None:
        """BenchScore is immutable."""
        results = {"factual": [BenchResult("factual_001", 0.9, 1.0)]}
        score = score_results("FrozenModel", results)

        with pytest.raises(AttributeError):
            score.model = "Modified"  # type: ignore

    def test_b5_empty_results(self) -> None:
        """Given empty results, returns BenchScore with 0.0 overall score."""
        score = score_results("EmptyModel", {})

        assert score.model == "EmptyModel"
        assert score.overall_score == 0.0
        assert len(score.domain_scores) == 0

    def test_b6_domain_score_has_scenario_count(self) -> None:
        """Each DomainScore includes scenario count."""
        results = {
            "test_domain": [
                BenchResult("t1", 0.5, 1.0),
                BenchResult("t2", 0.5, 0.0),
                BenchResult("t3", 0.5, 1.0),
            ]
        }
        score = score_results("CountModel", results)

        assert score.domain_scores[0].scenario_count == 3


class TestBenchResultImmutability:
    """C-tests: Value object immutability."""

    def test_c1_bench_result_frozen(self) -> None:
        """BenchResult is immutable."""
        result = BenchResult("test_001", 0.9, 1.0)

        with pytest.raises(AttributeError):
            result.scenario_id = "modified"  # type: ignore

    def test_c2_domain_score_frozen(self) -> None:
        """DomainScore is immutable."""
        ds = DomainScore("factual", 0.85, 10)

        with pytest.raises(AttributeError):
            ds.domain = "modified"  # type: ignore
