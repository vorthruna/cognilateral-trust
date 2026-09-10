"""A-tests and B-tests for TrustBench CLI (Step 17).

The benchmark refuses to fabricate a model evaluation: callers must either pass a
real ``runner`` (Python API) or explicitly opt into the synthetic baseline via
``mock=True`` / ``--mock``. Mock results are self-labeled (``"mock": true`` and a
``"mock:"`` model-name prefix) so they can never be mistaken for a real run.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cognilateral_trust.bench.cli import main, run_benchmark, generate_fingerprint_from_file


class TestCLIArgumentParsing:
    """A-tests: CLI argument parsing."""

    def test_a1_run_subcommand_requires_model(self) -> None:
        """bench run without --model shows usage."""
        with pytest.raises(SystemExit):
            main(["run"])

    def test_a2_run_with_model_and_mock_succeeds(self, capsys: pytest.CaptureFixture) -> None:
        """bench run --model <name> --mock succeeds and outputs labeled JSON."""
        exit_code = main(["run", "--model", "TestModel", "--mock"])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["model"] == "mock:TestModel"
        assert output["mock"] is True

    def test_a2b_run_without_mock_or_runner_fails_loudly(self, capsys: pytest.CaptureFixture) -> None:
        """bench run without --mock (and no real runner) refuses and returns non-zero."""
        exit_code = main(["run", "--model", "TestModel"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "runner" in captured.err.lower() or "mock" in captured.err.lower()

    def test_a3_run_with_output_writes_file(self) -> None:
        """bench run --model <name> --mock --output <path> writes JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            exit_code = main(["run", "--model", "FileModel", "--mock", "--output", str(output_path)])

            assert exit_code == 0
            assert output_path.exists()
            with open(output_path) as f:
                result = json.load(f)
            assert result["model"] == "mock:FileModel"
            assert result["mock"] is True

    def test_a4_fingerprint_subcommand_requires_results(self) -> None:
        """bench fingerprint without --results shows usage."""
        with pytest.raises(SystemExit):
            main(["fingerprint"])

    def test_a5_fingerprint_with_results_succeeds(self, capsys: pytest.CaptureFixture) -> None:
        """bench fingerprint --results <path> succeeds and outputs JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "bench_results.json"
            run_benchmark("FPModel", results_path, mock=True)

            exit_code = main(["fingerprint", "--results", str(results_path)])

            assert exit_code == 0
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["model"] == "mock:FPModel"
            assert "spokes" in output

    def test_a6_no_command_shows_help(self, capsys: pytest.CaptureFixture) -> None:
        """main() with no subcommand shows help and returns non-zero (invalid usage)."""
        exit_code = main([])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "TrustBench" in captured.out or "usage" in captured.out.lower()


class TestRunBenchmark:
    """B-tests: Benchmark execution (synthetic baseline)."""

    def test_b1_run_benchmark_returns_dict(self) -> None:
        """run_benchmark returns dict with model, mock flag, and scores."""
        result = run_benchmark("TestModel", mock=True)

        assert isinstance(result, dict)
        assert "model" in result
        assert "mock" in result and result["mock"] is True
        assert "overall_score" in result
        assert "domain_scores" in result

    def test_b1b_run_benchmark_refuses_without_mock_or_runner(self) -> None:
        """run_benchmark raises rather than fabricating a model evaluation."""
        with pytest.raises(ValueError, match="runner"):
            run_benchmark("TestModel")

    def test_b1c_run_benchmark_with_real_runner_is_not_mock(self) -> None:
        """A provided runner produces a real (non-mock) result with an unprefixed name."""
        # Deterministic fake runner: confidence 0.8, always correct.
        result = run_benchmark("RealModel", runner=lambda scenario: (0.8, 1.0))

        assert result["mock"] is False
        assert result["model"] == "RealModel"
        # confidence 0.8 vs accuracy 1.0 => ECE ~0.2 per domain
        for ds in result["domain_scores"]:
            assert 0.0 <= ds["calibration_error"] <= 1.0

    def test_b2_run_benchmark_model_name_in_output(self) -> None:
        """Output model name reflects input (mock-prefixed for the baseline)."""
        result = run_benchmark("CustomModelName", mock=True)

        assert result["model"] == "mock:CustomModelName"

    def test_b3_run_benchmark_scores_are_valid(self) -> None:
        """Output scores are in valid range [0.0, 1.0]."""
        result = run_benchmark("ValidModel", mock=True)

        assert 0.0 <= result["overall_score"] <= 1.0
        for ds in result["domain_scores"]:
            assert 0.0 <= ds["calibration_error"] <= 1.0

    def test_b4_run_benchmark_has_all_domains(self) -> None:
        """Output includes all 5 domains."""
        result = run_benchmark("AllDomainsModel", mock=True)

        domains = {ds["domain"] for ds in result["domain_scores"]}
        assert "factual" in domains
        assert "reasoning" in domains
        assert "ambiguous" in domains
        assert "out_of_distribution" in domains
        assert "adversarial" in domains

    def test_b5_run_benchmark_creates_parent_dirs(self) -> None:
        """Output path creation makes parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "results.json"
            run_benchmark("NestedModel", nested_path, mock=True)

            assert nested_path.exists()


class TestFingerprintFromFile:
    """C-tests: Loading and generating fingerprints."""

    def test_c1_fingerprint_from_file_returns_dict(self) -> None:
        """generate_fingerprint_from_file returns dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.json"
            run_benchmark("FPModel", results_path, mock=True)

            result = generate_fingerprint_from_file(results_path)

            assert isinstance(result, dict)
            assert "model" in result
            assert "spokes" in result

    def test_c2_fingerprint_from_file_missing_file(self) -> None:
        """generate_fingerprint_from_file raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            generate_fingerprint_from_file("/nonexistent/path/results.json")

    def test_c3_fingerprint_matches_model(self) -> None:
        """Fingerprint model name matches results model name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.json"
            run_benchmark("MatchModel", results_path, mock=True)

            fp = generate_fingerprint_from_file(results_path)

            assert fp["model"] == "mock:MatchModel"

    def test_c4_fingerprint_spokes_match_domains(self) -> None:
        """Fingerprint spokes include all domains from results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.json"
            run_benchmark("DomainsModel", results_path, mock=True)

            fp = generate_fingerprint_from_file(results_path)
            fp_domains = {s["domain"] for s in fp["spokes"]}

            assert "factual" in fp_domains
            assert "reasoning" in fp_domains


class TestCLIErrorHandling:
    """D-tests: Error handling."""

    def test_d1_cli_handles_invalid_results_path(self, capsys: pytest.CaptureFixture) -> None:
        """CLI handles missing results file gracefully."""
        exit_code = main(["fingerprint", "--results", "/nonexistent/file.json"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err or "not found" in captured.err.lower()

    def test_d2_cli_returns_zero_on_success(self) -> None:
        """CLI returns exit code 0 on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            exit_code = main(["run", "--model", "SuccessModel", "--mock", "--output", str(output_path)])

            assert exit_code == 0
