"""TrustBench CLI — evaluate model calibration on epistemic honesty scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from cognilateral_trust.bench.fingerprint import generate_fingerprint, fingerprint_to_dict
from cognilateral_trust.bench.scenarios import BenchScenario, load_scenarios
from cognilateral_trust.bench.scoring import BenchResult, BenchScore, DomainScore, score_results

__all__ = ["main", "run_benchmark", "ScenarioRunner"]

# A ScenarioRunner evaluates one scenario and returns (confidence, correctness),
# where confidence is the model's stated confidence [0.0, 1.0] and correctness is
# 1.0 if the model's answer was correct, else 0.0. This is the injection point for
# evaluating a *real* model — the shipped CLI provides only a synthetic baseline.
ScenarioRunner = Callable[[BenchScenario], "tuple[float, float]"]

_MOCK_CONFIDENCE = 0.75


def mock_runner(scenario: BenchScenario) -> tuple[float, float]:
    """Synthetic baseline runner — NOT a real model evaluation.

    Always reports a fixed confidence and derives "correctness" from whether that
    fixed confidence falls inside the scenario's expected band. Useful only as a
    plumbing/regression fixture; it tells you nothing about any actual model.
    """
    confidence = _MOCK_CONFIDENCE
    correctness = (
        1.0
        if (scenario.expected_confidence_low <= confidence <= scenario.expected_confidence_high)
        else 0.0
    )
    return confidence, correctness


def run_benchmark(
    model_name: str,
    output_path: str | Path | None = None,
    *,
    runner: ScenarioRunner | None = None,
    mock: bool = False,
) -> dict:
    """Run the TrustBench calibration benchmark.

    To evaluate a real model, pass ``runner`` — a callable that takes a
    ``BenchScenario`` and returns ``(confidence, correctness)``. To run the
    synthetic baseline (no real model involved), pass ``mock=True``; the result is
    clearly labeled with ``"mock": true`` and a ``"mock:"`` model-name prefix so it
    can never be mistaken for a real evaluation.

    Args:
        model_name: Name of the model being evaluated
        output_path: Optional path to write results JSON
        runner: Real per-scenario evaluator returning (confidence, correctness)
        mock: If True (and no runner given), use the synthetic baseline runner

    Returns:
        Dict with benchmark score, a ``mock`` flag, and per-domain calibration error

    Raises:
        ValueError: If neither ``runner`` nor ``mock=True`` is provided. The
            benchmark refuses to fabricate a model evaluation.
    """
    if runner is None and not mock:
        raise ValueError(
            "No model runner provided. Pass runner=<callable returning "
            "(confidence, correctness)> to evaluate a real model, or set mock=True "
            "to run the synthetic baseline (clearly labeled, NOT a real evaluation)."
        )

    is_mock = runner is None
    active_runner: ScenarioRunner = mock_runner if is_mock else runner  # type: ignore[assignment]

    scenarios = load_scenarios()

    # Group scenarios by domain
    by_domain: dict[str, list[BenchScenario]] = {}
    for scenario in scenarios:
        by_domain.setdefault(scenario.domain, []).append(scenario)

    results_by_domain: dict[str, list[BenchResult]] = {}
    for domain, domain_scenarios in by_domain.items():
        results = []
        for scenario in domain_scenarios:
            model_confidence, correctness = active_runner(scenario)
            results.append(
                BenchResult(
                    scenario_id=scenario.id,
                    model_confidence=model_confidence,
                    correctness=correctness,
                )
            )
        results_by_domain[domain] = results

    # Score the results
    reported_model = f"mock:{model_name}" if is_mock else model_name
    bench_score = score_results(reported_model, results_by_domain)

    result = {
        "model": bench_score.model,
        "mock": is_mock,
        "overall_score": bench_score.overall_score,
        "domain_scores": [
            {
                "domain": ds.domain,
                "calibration_error": ds.calibration_error,
                "scenario_count": ds.scenario_count,
            }
            for ds in bench_score.domain_scores
        ],
    }

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    return result


def generate_fingerprint_from_file(results_path: str | Path) -> dict:
    """Generate fingerprint from saved benchmark results.

    Args:
        results_path: Path to results JSON file

    Returns:
        Dict with fingerprint (model, spokes)
    """
    results_file = Path(results_path)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_file, "r") as f:
        results_dict = json.load(f)

    # Reconstruct BenchScore from dict
    model = results_dict["model"]
    overall_score = results_dict["overall_score"]
    domain_scores = [
        DomainScore(
            domain=ds["domain"],
            calibration_error=ds["calibration_error"],
            scenario_count=ds["scenario_count"],
        )
        for ds in results_dict["domain_scores"]
    ]

    bench_score = BenchScore(
        model=model,
        overall_score=overall_score,
        domain_scores=tuple(domain_scores),
    )

    fp = generate_fingerprint(bench_score)
    return fingerprint_to_dict(fp)


def main(args: list[str] | None = None) -> int:
    """CLI entry point for TrustBench.

    Subcommands:
      run --model <name> --provider ollama [--host --samples --seed ...]  (real eval)
      run --model <name> --mock                                           (synthetic baseline)
      doctor [--host --model]                                             (check Ollama)
      fingerprint --results <path>

    Args:
        args: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 = success)
    """
    parser = argparse.ArgumentParser(description="TrustBench — evaluate model epistemic honesty")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run benchmark on a model")
    run_parser.add_argument("--model", required=True, help="Model name or identifier")
    run_parser.add_argument(
        "--output",
        default=None,
        help="Output path for results/manifest JSON (optional)",
    )
    run_parser.add_argument(
        "--provider",
        choices=["mock", "ollama"],
        default=None,
        help="Evaluation backend: 'ollama' for a real local model, 'mock' for the synthetic baseline.",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Alias for --provider mock. Synthetic baseline, labeled mock — NOT a real evaluation.",
    )
    # Ollama / lab options
    run_parser.add_argument("--host", default="http://localhost:11434", help="Ollama host (provider=ollama)")
    run_parser.add_argument("--samples", type=int, default=1, help="Samples per scenario (provider=ollama)")
    run_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = deterministic)")
    run_parser.add_argument("--seed", type=int, default=0, help="Backend RNG seed for reproducibility")
    run_parser.add_argument("--max-tokens", type=int, default=512, dest="max_tokens", help="Max tokens per response")
    run_parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    run_parser.add_argument(
        "--lab-dir",
        default=None,
        help="Directory to write the full lab record (manifest.json, scores.json, leaderboard.html)",
    )

    # Subcommand: doctor
    doctor_parser = subparsers.add_parser("doctor", help="Check Ollama connectivity and model availability")
    doctor_parser.add_argument("--host", default="http://localhost:11434", help="Ollama host to probe")
    doctor_parser.add_argument("--model", default=None, help="Optionally verify a specific model is installed")

    # Subcommand: fingerprint
    fp_parser = subparsers.add_parser(
        "fingerprint",
        help="Generate calibration fingerprint from results",
    )
    fp_parser.add_argument(
        "--results",
        required=True,
        help="Path to benchmark results JSON",
    )

    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 1

    try:
        if parsed.command == "run":
            if parsed.provider == "ollama":
                return _run_ollama(parsed)
            mock = parsed.mock or parsed.provider == "mock"
            result = run_benchmark(parsed.model, parsed.output, mock=mock)
            print(json.dumps(result, indent=2))
            return 0

        elif parsed.command == "doctor":
            return _doctor(parsed.host, parsed.model)

        elif parsed.command == "fingerprint":
            fp_result = generate_fingerprint_from_file(parsed.results)
            print(json.dumps(fp_result, indent=2))
            return 0

        else:
            parser.print_help()
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _run_ollama(parsed: argparse.Namespace) -> int:
    """Run a real evaluation against a local Ollama model via the lab orchestrator."""
    from cognilateral_trust.bench.lab import run_lab
    from cognilateral_trust.bench.runners.base import RunConfig
    from cognilateral_trust.bench.runners.ollama import OllamaTransport, ollama_available

    config = RunConfig(
        model=parsed.model,
        host=parsed.host,
        temperature=parsed.temperature,
        seed=parsed.seed,
        num_samples=parsed.samples,
        max_tokens=parsed.max_tokens,
        timeout_s=parsed.timeout,
    )
    if not ollama_available(config.host):
        print(
            f"Ollama not reachable at {config.host}. Run `trust-bench doctor` for help.",
            file=sys.stderr,
        )
        return 2

    transport = OllamaTransport(config)
    manifest = run_lab(config, transport=transport, output_dir=parsed.lab_dir)

    if parsed.output:
        Path(parsed.output).write_text(json.dumps(manifest, indent=2))

    summary = {
        "model": manifest["scores"]["model"],
        "ece_overall_score": manifest["scores"]["overall_score"],
        "band_in_band_rate": manifest["metrics"]["band_calibration"]["overall"]["in_band_rate"],
        "band_mean_signed_deviation": manifest["metrics"]["band_calibration"]["overall"]["mean_signed_deviation"],
        "accuracy_overall": manifest["metrics"]["accuracy"]["overall"],
        "scenarios": manifest["provenance"]["scenarios_evaluated"],
    }
    print(json.dumps(summary, indent=2))
    return 0


def _doctor(host: str, model: str | None) -> int:
    """Diagnose Ollama connectivity and (optionally) a model's availability."""
    from cognilateral_trust.bench.runners.ollama import (
        list_models,
        model_present,
        ollama_available,
        ollama_version,
    )

    if not ollama_available(host):
        print(f"✗ Ollama not reachable at {host}")
        print("  Install from https://ollama.com, then run `ollama serve`.")
        return 1

    version = ollama_version(host)
    models = list_models(host)
    print(f"✓ Ollama reachable at {host} (version {version or 'unknown'})")
    if models:
        print(f"  Models installed: {', '.join(models)}")
    else:
        print("  No models installed — pull one, e.g. `ollama pull qwen3:8b`.")

    if model:
        present = model_present(model, host)
        if present:
            print(f"  ✓ model '{model}' is available")
        else:
            print(f"  ✗ model '{model}' not found — run `ollama pull {model}`")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
