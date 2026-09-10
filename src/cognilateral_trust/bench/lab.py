"""TrustBench Lab — orchestrate a real model evaluation into a falsifiable record.

``run_lab`` drives a :class:`ModelTransport` over every scenario, scores calibration
with the existing pipeline, and writes a **provenance manifest**: model, parameters,
prompt version/hash, library version, git commit, server version, UTC timestamps, and
the full per-scenario / per-sample trace (raw responses, parsed answers, confidences,
grades, latencies). Anyone can re-run with the same config and compare, or inspect the
raw responses to falsify a score. That is what makes it a lab rather than a number.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from cognilateral_trust.bench.cli import run_benchmark
from cognilateral_trust.bench.grading import GradeResult, grade
from cognilateral_trust.bench.leaderboard import generate_leaderboard
from cognilateral_trust.bench.metrics import metrics_report
from cognilateral_trust.bench.prompts import PROMPT_VERSION, prompt_template_hash
from cognilateral_trust.bench.runners.base import (
    ModelTransport,
    RunConfig,
    ScenarioOutcome,
    make_scenario_runner,
)
from cognilateral_trust.bench.scoring import BenchScore, DomainScore

__all__ = ["run_lab", "build_manifest", "LAB_SCHEMA"]

LAB_SCHEMA = "trustbench-lab/v1"


def _library_version() -> str:
    try:
        from cognilateral_trust import __version__  # type: ignore

        return str(__version__)
    except Exception:
        return "unknown"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _server_version(transport: ModelTransport, config: RunConfig) -> str:
    """Best-effort backend version (Ollama). Never raises."""
    try:
        from cognilateral_trust.bench.runners.ollama import OllamaTransport, ollama_version

        if isinstance(transport, OllamaTransport):
            return ollama_version(config.host, opener=transport._opener)
    except Exception:
        pass
    return "n/a"


def _sample_to_dict(sample) -> dict:
    return {
        "answer": sample.answer,
        "confidence": sample.confidence,
        "confidence_found": sample.confidence_found,
        "correct": sample.correct,
        "grader": sample.grader,
        "rationale": sample.rationale,
        "latency_ms": round(sample.latency_ms, 1),
        "raw_response": sample.raw_response,
    }


def _scenario_to_dict(outcome: ScenarioOutcome) -> dict:
    latencies = [s.latency_ms for s in outcome.samples]
    return {
        "scenario_id": outcome.scenario_id,
        "domain": outcome.domain,
        "question": outcome.question,
        "expected_confidence_low": outcome.expected_confidence_low,
        "expected_confidence_high": outcome.expected_confidence_high,
        "confidence": round(outcome.confidence, 4),
        "correctness": outcome.correctness,
        "num_samples": len(outcome.samples),
        "latency_ms_mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "samples": [_sample_to_dict(s) for s in outcome.samples],
    }


def build_manifest(
    config: RunConfig,
    result: dict,
    outcomes: list[ScenarioOutcome],
    *,
    started_at: str,
    finished_at: str,
    duration_s: float,
    grader_name: str,
    server_version: str,
) -> dict:
    """Assemble the full provenance manifest from a completed run."""
    confidence_found = sum(1 for o in outcomes for s in o.samples if s.confidence_found)
    total_samples = sum(len(o.samples) for o in outcomes)
    return {
        "schema": LAB_SCHEMA,
        "provenance": {
            "model": config.model,
            "host": config.host,
            "server_version": server_version,
            "temperature": config.temperature,
            "seed": config.seed,
            "num_samples": config.num_samples,
            "max_tokens": config.max_tokens,
            "default_confidence": config.default_confidence,
            "grader": grader_name,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_template_hash(),
            "library_version": _library_version(),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_s": round(duration_s, 2),
            "scenarios_evaluated": len(outcomes),
            "confidence_parsed": confidence_found,
            "confidence_parse_rate": round(confidence_found / total_samples, 3) if total_samples else 0.0,
        },
        "scores": {
            "model": result["model"],
            "mock": result["mock"],
            "overall_score": result["overall_score"],
            "domain_scores": result["domain_scores"],
        },
        "metrics": metrics_report(outcomes),
        "scenarios": [_scenario_to_dict(o) for o in outcomes],
    }


def run_lab(
    config: RunConfig,
    *,
    transport: ModelTransport | None = None,
    grader: Callable[..., GradeResult] = grade,
    output_dir: str | Path | None = None,
    write_leaderboard: bool = True,
) -> dict:
    """Evaluate a model end-to-end and return the provenance manifest.

    Args:
        config: Evaluation parameters (model, host, sampling, determinism).
        transport: Model transport. Defaults to an Ollama transport on ``config.host``.
        grader: Correctness grader (default: deterministic domain-aware grader).
        output_dir: If set, write ``manifest.json``, ``scores.json`` and
            ``leaderboard.html`` here.
        write_leaderboard: Whether to render the leaderboard HTML.

    Returns:
        The provenance manifest dict.
    """
    if transport is None:
        from cognilateral_trust.bench.runners.ollama import OllamaTransport

        transport = OllamaTransport(config)

    grader_name = getattr(grader, "__name__", grader.__class__.__name__)

    outcomes: list[ScenarioOutcome] = []
    runner = make_scenario_runner(transport, config, sink=outcomes, grader=grader)

    started = datetime.now(timezone.utc)
    result = run_benchmark(config.model, runner=runner)
    finished = datetime.now(timezone.utc)

    manifest = build_manifest(
        config,
        result,
        outcomes,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        duration_s=(finished - started).total_seconds(),
        grader_name=grader_name,
        server_version=_server_version(transport, config),
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (out / "scores.json").write_text(json.dumps(manifest["scores"], indent=2))
        if write_leaderboard:
            bench_score = BenchScore(
                model=result["model"],
                overall_score=result["overall_score"],
                domain_scores=tuple(
                    DomainScore(ds["domain"], ds["calibration_error"], ds["scenario_count"])
                    for ds in result["domain_scores"]
                ),
            )
            (out / "leaderboard.html").write_text(generate_leaderboard([bench_score]))

    return manifest
