from __future__ import annotations

import types
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from vidscope.cli import app
from vidscope.doctor import DoctorReport, run_doctor, setup_pot_provider
from vidscope.settings import Settings

runner = CliRunner()


def test_run_doctor_healthy_mock(tmp_path: Path) -> None:
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.touch(mode=0o755)
    ffprobe = tmp_path / "ffprobe"
    ffprobe.touch(mode=0o755)
    tess = tmp_path / "tesseract"
    tess.touch(mode=0o755)
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape cookies\n")

    settings = Settings(
        ffmpeg_bin=ffmpeg,
        ffprobe_bin=ffprobe,
        tesseract_bin=tess,
        cookies_file=cookies,
    )

    def fake_runner(cmd: list[str]) -> Any:
        return types.SimpleNamespace(returncode=0, stdout="mock version 1.0.0")

    report = run_doctor(settings=settings, runner=fake_runner)
    assert isinstance(report, DoctorReport)
    assert report.ok is True
    check_names = [c.name for c in report.checks]
    assert "FFmpeg" in check_names
    assert "FFprobe" in check_names
    assert "Tesseract OCR" in check_names
    assert "Cookies File" in check_names


def test_run_doctor_missing_critical_binaries(monkeypatch: Any) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    settings = Settings(ffmpeg_bin=None, ffprobe_bin=None)

    report = run_doctor(settings=settings)
    assert report.ok is False
    ffmpeg_check = next(c for c in report.checks if c.name == "FFmpeg")
    assert ffmpeg_check.status == "missing"
    assert ffmpeg_check.recommendation is not None


def test_run_doctor_missing_optional_binaries(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.touch(mode=0o755)
    ffprobe = tmp_path / "ffprobe"
    ffprobe.touch(mode=0o755)

    settings = Settings(
        ffmpeg_bin=ffmpeg,
        ffprobe_bin=ffprobe,
        tesseract_bin=None,
    )

    def fake_which(name: str) -> str | None:
        if name in ("ffmpeg", "ffprobe"):
            return str(tmp_path / name)
        return None

    monkeypatch.setattr("shutil.which", fake_which)

    def fake_runner(cmd: list[str]) -> Any:
        return types.SimpleNamespace(returncode=0, stdout="1.0.0")

    report = run_doctor(settings=settings, runner=fake_runner)
    assert report.ok is True
    tess_check = next(c for c in report.checks if c.name == "Tesseract OCR")
    assert tess_check.status == "optional_missing"
    assert tess_check.recommendation is not None


def test_setup_pot_provider_missing_prerequisites(monkeypatch: Any) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = setup_pot_provider()
    assert result["ok"] is False
    assert "Node.js" in result["error"]


def test_setup_pot_provider_already_built(tmp_path: Path, monkeypatch: Any) -> None:
    dest = tmp_path / "pot-provider"
    script = dest / "server" / "build" / "generate_once.js"
    script.parent.mkdir(parents=True)
    script.touch(mode=0o755)

    monkeypatch.setattr("shutil.which", lambda name: f"/mock/{name}")

    def fake_runner(cmd: list[str]) -> Any:
        return types.SimpleNamespace(returncode=0, stdout="2.0.0")

    result = setup_pot_provider(target_dir=dest, runner=fake_runner, force=False)
    assert result["ok"] is True
    assert result["already_built"] is True
    assert result["version"] == "2.0.0"


def test_setup_pot_provider_success_flow(tmp_path: Path, monkeypatch: Any) -> None:
    dest = tmp_path / "pot-provider"

    monkeypatch.setattr("shutil.which", lambda name: f"/mock/{name}")

    calls: list[list[str]] = []

    def fake_runner(cmd: list[str]) -> Any:
        calls.append(cmd)
        if "clone" in cmd:
            (dest / "server" / "build").mkdir(parents=True, exist_ok=True)
            (dest / "server" / "build" / "generate_once.js").touch(mode=0o755)
        return types.SimpleNamespace(returncode=0, stdout="2.0.0")

    result = setup_pot_provider(target_dir=dest, runner=fake_runner)
    assert result["ok"] is True
    assert result["already_built"] is False
    assert len(calls) >= 3


def test_cli_doctor_command(monkeypatch: Any) -> None:
    monkeypatch.setattr("shutil.which", lambda name: f"/mock/{name}")
    monkeypatch.setattr(
        "vidscope.doctor._run_cmd",
        lambda cmd, **kw: (0, "mock version 1.0.0"),
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Vidscope System Diagnostics" in result.stdout
    assert "FFmpeg" in result.stdout


def test_cli_doctor_command_unhealthy(monkeypatch: Any) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "Vidscope System Diagnostics" in result.stdout
    assert "FFmpeg" in result.stdout


def test_cli_doctor_json_command(monkeypatch: Any) -> None:
    import json

    monkeypatch.setattr("shutil.which", lambda name: f"/mock/{name}")
    monkeypatch.setattr(
        "vidscope.doctor._run_cmd",
        lambda cmd, **kw: (0, "mock version 1.0.0"),
    )
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "ok" in payload
    assert payload["ok"] is True
    assert "checks" in payload


def test_cli_doctor_json_command_unhealthy(monkeypatch: Any) -> None:
    import json

    monkeypatch.setattr("shutil.which", lambda name: None)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "ok" in payload
    assert payload["ok"] is False
    assert "checks" in payload


def test_cli_setup_pot_command(tmp_path: Path, monkeypatch: Any) -> None:
    dest = tmp_path / "pot"
    script = dest / "server" / "build" / "generate_once.js"
    script.parent.mkdir(parents=True)
    script.touch(mode=0o755)

    monkeypatch.setattr("shutil.which", lambda name: f"/mock/{name}")
    monkeypatch.setattr(
        "vidscope.doctor._run_cmd",
        lambda cmd, **kw: (0, "2.0.0"),
    )

    result = runner.invoke(app, ["setup-pot", "--path", str(dest)])
    assert result.exit_code == 0
    assert "already built" in result.stdout or "Successfully built" in result.stdout
