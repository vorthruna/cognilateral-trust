#!/usr/bin/env python3
"""Resolve a release package identity only after explicit operator approval.

This local instrument reports identity evidence. It never edits project metadata,
builds artifacts, publishes, or treats a candidate as selected without approval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DISTRIBUTION = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*\Z")

# PEP 440 Appendix B grammar, kept on the standard library so the fail-closed JSON
# report survives in an environment with no third-party packages installed.
_VERSION = re.compile(
    r"""
    ^\s*v?
    (?:(?P<epoch>[0-9]+)!)?
    (?P<release>[0-9]+(?:\.[0-9]+)*)
    (?P<pre>[-_.]?(?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)[-_.]?(?P<pre_n>[0-9]+)?)?
    (?P<post>(?:-(?P<post_n1>[0-9]+))|(?:[-_.]?(?P<post_l>post|rev|r)[-_.]?(?P<post_n2>[0-9]+)?))?
    (?P<dev>[-_.]?(?P<dev_l>dev)[-_.]?(?P<dev_n>[0-9]+)?)?
    (?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_PRE_LABELS = {"alpha": "a", "a": "a", "beta": "b", "b": "b", "c": "rc", "rc": "rc", "pre": "rc", "preview": "rc"}


def canonical_version(value: str) -> str:
    """Return the PEP 440 canonical spelling, so equivalent spellings compare equal.

    ``1.0``, ``1.0.0`` and ``v1.0`` share one key; so do ``1.0alpha1`` and ``1.0a1``.
    Raises ``ValueError`` for anything that is not a valid PEP 440 version.
    """
    match = _VERSION.match(value)
    if match is None:
        raise ValueError("identity version is invalid")
    parts: list[str] = []
    epoch = int(match.group("epoch") or 0)
    if epoch:
        parts.append(f"{epoch}!")
    release = [int(segment) for segment in match.group("release").split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    parts.append(".".join(str(segment) for segment in release))
    if match.group("pre") is not None:
        parts.append(f"{_PRE_LABELS[match.group('pre_l').lower()]}{int(match.group('pre_n') or 0)}")
    if match.group("post") is not None:
        parts.append(f".post{int(match.group('post_n1') or match.group('post_n2') or 0)}")
    if match.group("dev") is not None:
        parts.append(f".dev{int(match.group('dev_n') or 0)}")
    local = match.group("local")
    if local:
        segments = [segment.lower() for segment in re.split(r"[-_.]", local)]
        parts.append("+" + ".".join(str(int(s)) if s.isdigit() else s for s in segments))
    return "".join(parts)


@dataclass(frozen=True)
class PackageIdentity:
    """A distribution name and version as a single release identity."""

    distribution: str
    version: str

    @property
    def normalized(self) -> tuple[str, str]:
        distribution = re.sub(r"[-_.]+", "-", self.distribution).lower()
        return distribution, canonical_version(self.version)

    def as_dict(self) -> dict[str, str]:
        return {"distribution": self.distribution, "version": self.version}


def parse_identity(value: str) -> PackageIdentity:
    """Parse the explicit ``distribution==version`` spelling used by the CLI."""
    if value.count("==") != 1:
        raise ValueError("identity must use exactly distribution==version")
    distribution, version = (part.strip() for part in value.split("=="))
    if not _DISTRIBUTION.fullmatch(distribution):
        raise ValueError("identity distribution is invalid")
    canonical_version(version)
    return PackageIdentity(distribution=distribution, version=version)


def _git(repo: Path, *arguments: str) -> str:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
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
    name = metadata.get("name")
    version = metadata.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError("committed project name and version must be strings")
    try:
        return parse_identity(f"{name}=={version}")
    except ValueError as error:
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

    try:
        dirty = _git(repo, "status", "--porcelain=v1", "--untracked-files=all").strip()
    except ValueError as error:
        return _result(
            status="IDENTITY_EVIDENCE_UNAVAILABLE",
            observed=observed,
            candidates=candidates,
            reason=str(error),
        )

    if dirty:
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
