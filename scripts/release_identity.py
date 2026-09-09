#!/usr/bin/env python3
"""Resolve a release package identity only after explicit operator approval.

This local instrument reports identity evidence. It never edits project metadata,
builds artifacts, publishes, or treats a candidate as selected without approval.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DISTRIBUTION = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*\Z")
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9.-]+)?\Z")


@dataclass(frozen=True)
class PackageIdentity:
    """A distribution name and version as a single release identity."""

    distribution: str
    version: str

    @property
    def normalized(self) -> tuple[str, str]:
        distribution = re.sub(r"[-_.]+", "-", self.distribution).lower()
        return distribution, self.version.lower()

    def as_dict(self) -> dict[str, str]:
        return {"distribution": self.distribution, "version": self.version}


def parse_identity(value: str) -> PackageIdentity:
    """Parse the explicit ``distribution==version`` spelling used by the CLI."""
    if value.count("==") != 1:
        raise ValueError("identity must use exactly distribution==version")
    distribution, version = (part.strip() for part in value.split("=="))
    if not _DISTRIBUTION.fullmatch(distribution):
        raise ValueError("identity distribution is invalid")
    if not _VERSION.fullmatch(version):
        raise ValueError("identity version is invalid")
    return PackageIdentity(distribution=distribution, version=version)


def _git(repo: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Git identity evidence could not be read") from error
    return completed.stdout


def _observed_identity(repo: Path) -> PackageIdentity:
    try:
        project = tomllib.loads(_git(repo, "show", "HEAD:pyproject.toml"))
    except (ValueError, tomllib.TOMLDecodeError) as error:
        raise ValueError("committed pyproject.toml is unavailable or invalid") from error
    metadata = project.get("project")
    if not isinstance(metadata, dict):
        raise ValueError("committed pyproject.toml has no project table")
    try:
        return parse_identity(f"{metadata['name']}=={metadata['version']}")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("committed project identity is invalid") from error


def _result(
    *,
    status: str,
    observed: PackageIdentity | None,
    candidates: list[PackageIdentity],
    approved: PackageIdentity | None = None,
    selected: PackageIdentity | None = None,
    reason: str,
) -> dict[str, Any]:
    return {
        "approved": approved.as_dict() if approved is not None else None,
        "candidates": [candidate.as_dict() for candidate in candidates],
        "observed": observed.as_dict() if observed is not None else None,
        "reason": reason,
        "selected": selected.as_dict() if selected is not None else None,
        "status": status,
    }


def resolve_identity(
    repo: Path,
    candidate_values: list[str],
    *,
    approved: str | None = None,
) -> dict[str, Any]:
    """Return a fail-closed local decision without changing repository state."""
    try:
        candidates = sorted(
            (parse_identity(value) for value in candidate_values),
            key=lambda identity: identity.normalized,
        )
    except ValueError as error:
        return _result(
            status="INVALID_CANDIDATE",
            observed=None,
            candidates=[],
            reason=str(error),
        )

    if len({candidate.normalized for candidate in candidates}) != len(candidates):
        return _result(
            status="AMBIGUOUS_CANDIDATES",
            observed=None,
            candidates=candidates,
            reason="candidate identities collide after distribution-name normalization",
        )

    try:
        observed = _observed_identity(repo)
    except ValueError as error:
        return _result(
            status="SOURCE_METADATA_INVALID",
            observed=None,
            candidates=candidates,
            reason=str(error),
        )

    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all").strip():
        return _result(
            status="DIRTY_WORKTREE",
            observed=observed,
            candidates=candidates,
            reason="release identity requires a clean worktree, including untracked files",
        )

    if approved is None:
        return _result(
            status="APPROVAL_REQUIRED",
            observed=observed,
            candidates=candidates,
            reason="operator approval is required before selecting a package identity",
        )

    try:
        approved_identity = parse_identity(approved)
    except ValueError as error:
        return _result(
            status="INVALID_APPROVAL",
            observed=observed,
            candidates=candidates,
            reason=str(error),
        )

    if approved_identity.normalized not in {candidate.normalized for candidate in candidates}:
        return _result(
            status="APPROVAL_NOT_A_CANDIDATE",
            observed=observed,
            candidates=candidates,
            approved=approved_identity,
            reason="approved identity was not one of the supplied candidates",
        )

    if approved_identity.normalized != observed.normalized:
        return _result(
            status="IDENTITY_MISMATCH",
            observed=observed,
            candidates=candidates,
            approved=approved_identity,
            reason="approved identity does not match committed project metadata",
        )

    return _result(
        status="READY",
        observed=observed,
        candidates=candidates,
        approved=approved_identity,
        selected=observed,
        reason="explicit approval matches the clean committed project identity",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--approved")
    arguments = parser.parse_args(argv)
    try:
        report = resolve_identity(arguments.repo, arguments.candidate, approved=arguments.approved)
    except (OSError, ValueError) as error:
        report = _result(status="IDENTITY_EVIDENCE_UNAVAILABLE", observed=None, candidates=[], reason=str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
