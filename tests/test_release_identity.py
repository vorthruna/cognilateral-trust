"""Fail-closed tests for local release package identity decisions."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release_identity.py"


def _load_instrument() -> Any:
    spec = importlib.util.spec_from_file_location("release_identity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


identity = _load_instrument()


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True, text=True)


def _commit_project_metadata(repo: Path, *, name: str, version: str) -> None:
    (repo / "pyproject.toml").write_text(
        f"""[project]
name = {name}
version = {version}
""",
        encoding="utf-8",
    )
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-qm", "update metadata")


@pytest.fixture
def clean_project(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    (repo / "src" / "demo_package").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        """[project]
name = "cognilateral-trust"
version = "1.4.0"
""",
        encoding="utf-8",
    )
    (repo / "src" / "demo_package" / "__init__.py").write_text("\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "release-identity@example.invalid")
    _git(repo, "config", "user.name", "Release Identity Fixture")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo


def test_competing_candidates_remain_unresolved_without_explicit_approval(clean_project: Path) -> None:
    result = identity.resolve_identity(
        clean_project,
        [
            "leverage-finder==4.0.0a0",
            "cognilateral-trust==1.4.0",
            "cognilateral-trust==2.0.0",
        ],
    )

    assert result["status"] == "APPROVAL_REQUIRED"
    assert result["observed"] == {"distribution": "cognilateral-trust", "version": "1.4.0"}
    assert result["selected"] is None
    assert result["candidates"] == [
        {"distribution": "cognilateral-trust", "version": "1.4.0"},
        {"distribution": "cognilateral-trust", "version": "2.0.0"},
        {"distribution": "leverage-finder", "version": "4.0.0a0"},
    ]


def test_explicit_approval_cannot_override_committed_project_identity(clean_project: Path) -> None:
    result = identity.resolve_identity(
        clean_project,
        ["leverage-finder==4.0.0a0", "cognilateral-trust==2.0.0"],
        approved="leverage-finder==4.0.0a0",
    )

    assert result["status"] == "IDENTITY_MISMATCH"
    assert result["selected"] is None
    assert result["approved"] == {"distribution": "leverage-finder", "version": "4.0.0a0"}


def test_exact_approved_committed_identity_is_the_only_ready_state(clean_project: Path) -> None:
    result = identity.resolve_identity(
        clean_project,
        ["leverage-finder==4.0.0a0", "cognilateral-trust==1.4.0"],
        approved="cognilateral-trust==1.4.0",
    )

    assert result["status"] == "READY"
    assert result["selected"] == {"distribution": "cognilateral-trust", "version": "1.4.0"}


def test_normalized_duplicate_candidates_are_ambiguous(clean_project: Path) -> None:
    result = identity.resolve_identity(
        clean_project,
        ["cognilateral-trust==1.4.0", "cognilateral_trust==1.4.0"],
    )

    assert result["status"] == "AMBIGUOUS_CANDIDATES"
    assert result["selected"] is None


def test_dirty_worktree_blocks_even_an_exact_approval(clean_project: Path) -> None:
    (clean_project / "unreviewed.txt").write_text("unreviewed\n", encoding="utf-8")

    result = identity.resolve_identity(
        clean_project,
        ["cognilateral-trust==1.4.0"],
        approved="cognilateral-trust==1.4.0",
    )

    assert result["status"] == "DIRTY_WORKTREE"
    assert result["selected"] is None


def test_cli_reports_blocked_identity_as_json_and_nonzero_exit(clean_project: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(clean_project),
            "--candidate",
            "leverage-finder==4.0.0a0",
            "--candidate",
            "cognilateral-trust==2.0.0",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["status"] == "APPROVAL_REQUIRED"
    assert report["selected"] is None
    assert completed.stderr == ""


@pytest.mark.parametrize(
    "version",
    ["1", "1!2.3", "1.2.3rc1", "1.2.dev1", "1.2+local"],
)
def test_parse_identity_accepts_pep_440_versions(version: str) -> None:
    parsed = identity.parse_identity(f"cognilateral-trust=={version}")

    assert parsed.version == version


@pytest.mark.parametrize("version", ["", "1..2", "1.2+local..part", "not-a-version", "1.2+bad!"])
def test_parse_identity_rejects_invalid_pep_440_versions(version: str) -> None:
    with pytest.raises(ValueError, match="identity version is invalid"):
        identity.parse_identity(f"cognilateral-trust=={version}")


def test_invalid_candidate_has_structured_blocked_result(clean_project: Path) -> None:
    result = identity.resolve_identity(clean_project, ["cognilateral-trust==not-a-version"])

    assert result == {
        "approved": None,
        "candidates": [],
        "observed": None,
        "reason": "identity version is invalid",
        "selected": None,
        "status": "INVALID_CANDIDATE",
    }


@pytest.mark.parametrize(
    ("name", "version"),
    [("42", '"1.4.0"'), ('"cognilateral-trust"', "1")],
)
def test_non_string_committed_metadata_is_invalid(
    clean_project: Path,
    name: str,
    version: str,
) -> None:
    _commit_project_metadata(clean_project, name=name, version=version)

    result = identity.resolve_identity(clean_project, ["cognilateral-trust==1.4.0"])

    assert result["status"] == "SOURCE_METADATA_INVALID"
    assert result["observed"] is None
    assert result["reason"] == "committed project name and version must be strings"


def test_invalid_committed_metadata_has_structured_blocked_result(clean_project: Path) -> None:
    (clean_project / "pyproject.toml").write_text("not valid toml =", encoding="utf-8")
    _git(clean_project, "add", "pyproject.toml")
    _git(clean_project, "commit", "-qm", "break metadata")

    result = identity.resolve_identity(clean_project, ["cognilateral-trust==1.4.0"])

    assert result["status"] == "SOURCE_METADATA_INVALID"
    assert result["observed"] is None


def test_invalid_approval_has_structured_blocked_result(clean_project: Path) -> None:
    result = identity.resolve_identity(
        clean_project,
        ["cognilateral-trust==1.4.0"],
        approved="cognilateral-trust==not-a-version",
    )

    assert result["status"] == "INVALID_APPROVAL"
    assert result["approved"] is None
    assert result["selected"] is None


def test_approval_outside_candidates_has_structured_blocked_result(clean_project: Path) -> None:
    result = identity.resolve_identity(
        clean_project,
        ["cognilateral-trust==1.4.0"],
        approved="other-package==1.0",
    )

    assert result["status"] == "APPROVAL_NOT_A_CANDIDATE"
    assert result["approved"] == {"distribution": "other-package", "version": "1.0"}
    assert result["selected"] is None


def test_git_status_failure_returns_structured_unavailable_result(
    clean_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_git = identity._git

    def fail_status(repo: Path, *arguments: str) -> str:
        if arguments[:1] == ("status",):
            raise ValueError("status inspection failed")
        return original_git(repo, *arguments)

    monkeypatch.setattr(identity, "_git", fail_status)

    result = identity.resolve_identity(
        clean_project,
        ["cognilateral-trust==1.4.0"],
        approved="cognilateral-trust==1.4.0",
    )

    assert result["status"] == "IDENTITY_EVIDENCE_UNAVAILABLE"
    assert result["observed"] == {"distribution": "cognilateral-trust", "version": "1.4.0"}
    assert result["candidates"] == [{"distribution": "cognilateral-trust", "version": "1.4.0"}]
    assert result["selected"] is None
    assert result["reason"] == "status inspection failed"


def test_git_fixture_ignores_ambient_git_dir(clean_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_DIR", str(ROOT / ".git"))

    result = identity.resolve_identity(clean_project, ["cognilateral-trust==1.4.0"])

    assert result["status"] == "APPROVAL_REQUIRED"
    assert result["observed"] == {"distribution": "cognilateral-trust", "version": "1.4.0"}
