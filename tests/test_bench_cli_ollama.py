"""CLI tests for `doctor` and `run --provider ollama`, with urlopen monkeypatched."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest


from cognilateral_trust.bench.cli import main


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._b = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._b

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *a) -> bool:
        return False


def _install_fake_ollama(monkeypatch, *, reachable: bool = True, models=("qwen3:8b",)) -> None:
    def fake_urlopen(request, timeout=None):
        if not reachable:
            raise urllib.error.URLError("connection refused")
        url = request.full_url
        if "/api/version" in url:
            return _Resp({"version": "0.3.14"})
        if "/api/tags" in url:
            return _Resp({"models": [{"name": m} for m in models]})
        if "/api/generate" in url:
            return _Resp({"response": "Answer: Paris\nConfidence: 0.70"})
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


class TestDoctor:
    def test_doctor_reachable(self, monkeypatch, capsys) -> None:
        _install_fake_ollama(monkeypatch)
        code = main(["doctor", "--host", "http://h", "--model", "qwen3:8b"])
        assert code == 0
        out = capsys.readouterr().out
        assert "reachable" in out and "qwen3:8b" in out

    def test_doctor_unreachable(self, monkeypatch, capsys) -> None:
        _install_fake_ollama(monkeypatch, reachable=False)
        code = main(["doctor", "--host", "http://h"])
        assert code == 1
        assert "not reachable" in capsys.readouterr().out

    def test_doctor_missing_model(self, monkeypatch, capsys) -> None:
        _install_fake_ollama(monkeypatch, models=("llama3:8b",))
        code = main(["doctor", "--model", "qwen3:8b"])
        assert code == 1
        assert "not found" in capsys.readouterr().out


class TestRunOllama:
    def test_run_ollama_writes_manifest_and_prints_summary(self, monkeypatch, capsys, tmp_path: Path) -> None:
        _install_fake_ollama(monkeypatch)
        lab_dir = tmp_path / "lab"
        code = main(["run", "--model", "qwen3:8b", "--provider", "ollama", "--lab-dir", str(lab_dir)])
        assert code == 0

        summary = json.loads(capsys.readouterr().out)
        assert summary["model"] == "qwen3:8b"
        assert "band_in_band_rate" in summary
        assert "ece_overall_score" in summary
        assert (lab_dir / "manifest.json").exists()

    def test_run_ollama_unreachable_returns_2(self, monkeypatch, capsys) -> None:
        _install_fake_ollama(monkeypatch, reachable=False)
        code = main(["run", "--model", "qwen3:8b", "--provider", "ollama"])
        assert code == 2
        assert "not reachable" in capsys.readouterr().err


class TestSampleCountValidation:
    def test_zero_samples_is_rejected_before_any_run(self, monkeypatch, capsys) -> None:
        _install_fake_ollama(monkeypatch)
        with pytest.raises(SystemExit) as exit_info:
            main(["run", "--model", "qwen3:8b", "--provider", "ollama", "--samples", "0"])
        assert exit_info.value.code == 2
        assert "at least 1" in capsys.readouterr().err


class TestDoctorTagMatching:
    def test_same_base_different_tag_is_not_present(self, monkeypatch, capsys) -> None:
        _install_fake_ollama(monkeypatch, models=("qwen3:14b",))
        code = main(["doctor", "--host", "http://h", "--model", "qwen3:8b"])
        assert code == 1
        assert "not found" in capsys.readouterr().out

    def test_bare_base_name_matches_any_installed_tag(self, monkeypatch, capsys) -> None:
        _install_fake_ollama(monkeypatch, models=("qwen3:14b",))
        code = main(["doctor", "--host", "http://h", "--model", "qwen3"])
        assert code == 0
