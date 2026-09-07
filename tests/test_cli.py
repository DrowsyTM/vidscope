from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def _construct(model: type[Any], **values: Any) -> Any:
    """Construct a contract model without coupling adapter tests to defaults."""
    fields = getattr(model, "model_fields", {})
    selected = {
        name: value for name, value in values.items() if not fields or name in fields
    }
    return model.model_construct(**selected)


def _request(source: Path, output: Path) -> Any:
    from video_analyzer.contracts import AnalyzeVideoRequest, TimeRange

    return AnalyzeVideoRequest(
        source=str(source),
        time_range=TimeRange(start_seconds=0, end_seconds=30),
        tasks={"metadata"},
        output_directory=output,
    )


def _success_result() -> Any:
    from video_analyzer.contracts import AnalysisResult, AnalysisSummary, ArtifactRef

    artifact_uri = "video-analyzer://runs/run-cli/artifacts/transcript"
    artifact = _construct(
        ArtifactRef,
        artifact_id="transcript",
        uri=artifact_uri,
        media_type="application/jsonl",
        byte_size=32,
        sha256="a" * 64,
    )
    summary = _construct(
        AnalysisSummary,
        source="file:///tmp/clip.mp4",
        duration_seconds=30.0,
        task_count=1,
    )
    return _construct(
        AnalysisResult,
        ok=True,
        status="completed",
        summary=summary,
        stages=[],
        warnings=[],
        manifest_uri="video-analyzer://runs/run-cli/manifest",
        artifacts=[artifact],
        artifact_refs=[artifact],
    )


def _error(code: str = "INTERNAL_STAGE_FAILED", stage: str = "transcribe") -> Any:
    from video_analyzer.contracts import AnalysisError

    return _construct(
        AnalysisError,
        ok=False,
        status="failed",
        code=code,
        stage=stage,
        message="deterministic test failure",
        retryable=False,
        diagnostics={"detail": "fixture"},
        artifact_refs=[],
        manifest_uri="video-analyzer://runs/run-cli/manifest",
    )


def _failure(error: Any) -> BaseException:
    from video_analyzer.core import VideoAnalyzerFailure

    return VideoAnalyzerFailure(error)


def _one_json_line(stdout: str) -> tuple[str, dict[str, Any]]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"expected one JSON envelope, got {lines!r}"
    line = lines[0]
    payload = json.loads(line)
    assert isinstance(payload, dict)
    assert line == line.strip()
    return line, payload


def test_cli_success_emits_one_compact_artifact_only_envelope_and_shared_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_analyzer.cli import app

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"deterministic fixture")
    output = tmp_path / "results"
    output.mkdir()
    expected = _success_result()
    seen: list[Any] = []

    def fake_analyze(request: Any, *, context: Any = None) -> Any:
        seen.append((request, context))
        return expected

    monkeypatch.setattr("video_analyzer.cli.analyze_video", fake_analyze)
    result = runner.invoke(
        app,
        [
            "analyze-video",
            "--source",
            str(source),
            "--out",
            str(output),
            "--start-seconds",
            "5",
            "--end-seconds",
            "20",
            "--task",
            "metadata",
            "--task",
            "frames",
            "--frame-timestamp",
            "7.5",
            "--language",
            "en",
        ],
    )

    assert result.exit_code == 0, result.stdout or result.stderr
    line, payload = _one_json_line(result.stdout)
    assert line == json.dumps(payload, separators=(",", ":"))
    assert payload["ok"] is True
    assert payload["status"] == "completed"
    assert payload["manifest_uri"].startswith("video-analyzer://")
    encoded = result.stdout.lower()
    assert "deterministic transcript text" not in encoded
    assert "base64" not in encoded
    assert all(
        "video-analyzer://" in artifact["uri"] for artifact in payload["artifacts"]
    )

    assert len(seen) == 1
    request, context = seen[0]
    assert context is None
    assert Path(str(request.source)).resolve() == source.resolve()
    assert Path(request.output_directory) == output
    assert request.time_range.start_seconds == 5
    assert request.time_range.end_seconds == 20
    assert request.tasks == {"metadata", "frames"}
    assert request.frame_timestamps_seconds == (7.5,)
    assert request.language == "en"


def test_cli_terminal_failure_emits_typed_error_envelope_and_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_analyzer.cli import app

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"deterministic fixture")
    output = tmp_path / "results"
    output.mkdir()
    error = _error()
    calls: list[Any] = []

    def fake_analyze(request: Any, *, context: Any = None) -> Any:
        calls.append(request)
        raise _failure(error)

    monkeypatch.setattr("video_analyzer.cli.analyze_video", fake_analyze)
    result = runner.invoke(
        app,
        [
            "analyze-video",
            "--source",
            str(source),
            "--out",
            str(output),
            "--task",
            "transcript",
        ],
    )

    assert result.exit_code != 0
    _, payload = _one_json_line(result.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["code"] == "INTERNAL_STAGE_FAILED"
    assert payload["stage"] == "transcribe"
    assert payload["manifest_uri"].startswith("video-analyzer://")
    assert len(calls) == 1
    assert "traceback" not in result.stdout.lower()


def test_cli_request_validation_is_typed_and_does_not_call_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_analyzer.cli import app

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"deterministic fixture")
    output = tmp_path / "results"
    output.mkdir()
    calls: list[Any] = []

    def fake_analyze(request: Any, *, context: Any = None) -> Any:
        calls.append(request)
        return _success_result()

    monkeypatch.setattr("video_analyzer.cli.analyze_video", fake_analyze)
    result = runner.invoke(
        app,
        [
            "analyze-video",
            "--source",
            str(source),
            "--out",
            str(output),
            "--start-seconds",
            "20",
            "--end-seconds",
            "5",
        ],
    )

    assert result.exit_code != 0
    _, payload = _one_json_line(result.stdout)
    assert payload["ok"] is False
    assert payload["code"] == "INVALID_REQUEST"
    assert calls == []


def test_cli_mcp_invokes_mcp_main(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_analyzer.cli import app

    called = False

    def fake_main() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("video_analyzer.mcp.main", fake_main)
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 0
    assert called is True


def test_cli_logging_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_analyzer.cli import app

    seen_levels: list[int] = []

    def fake_configure(level: int = logging.INFO) -> None:
        seen_levels.append(level)

    monkeypatch.setattr("video_analyzer.logging.configure_logging", fake_configure)
    monkeypatch.setattr("video_analyzer.mcp.main", lambda: None)
    runner.invoke(app, ["--verbose", "mcp"])
    runner.invoke(app, ["--quiet", "mcp"])
    assert seen_levels == [logging.DEBUG, logging.WARNING]
