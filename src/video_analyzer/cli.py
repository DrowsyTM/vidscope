from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from .contracts import AnalysisError, AnalyzeVideoRequest, ErrorCode, TimeRange
from .core import VideoAnalyzerFailure, analyze_video

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _root() -> None:
    """Video analysis command group."""
    return

def _emit(value: object) -> None:
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _invalid_request(error: BaseException) -> AnalysisError:
    return AnalysisError(
        code=ErrorCode.INVALID_REQUEST,
        stage="validate_source",
        message=str(error)[:2_048] or "request validation failed",
        retryable=False,
    )


@app.command("analyze-video")
def analyze_video_command(
    source: Annotated[str, typer.Option("--source", help="Local path or HTTPS video URL.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")],
    start_seconds: Annotated[float, typer.Option("--start-seconds")] = 0.0,
    end_seconds: Annotated[float, typer.Option("--end-seconds")] = 180.0,
    task: Annotated[list[str] | None, typer.Option("--task", help="Task; repeat for multiple tasks.")] = None,
    frame_timestamp: Annotated[list[float] | None, typer.Option("--frame-timestamp", help="Explicit frame timestamp; repeatable.")] = None,
    language: Annotated[str, typer.Option("--language")] = "en",
    caption_preference: Annotated[str, typer.Option("--caption-preference")] = "manual_then_automatic_then_asr",
    asr_enabled: Annotated[bool, typer.Option("--asr-enabled/--no-asr")] = True,
    max_frames: Annotated[int, typer.Option("--max-frames")] = 6,
    max_frame_width: Annotated[int, typer.Option("--max-frame-width")] = 1_280,
    max_download_bytes: Annotated[int, typer.Option("--max-download-bytes")] = 268_435_456,
    max_output_bytes: Annotated[int, typer.Option("--max-output-bytes")] = 67_108_864,
    timeout_seconds: Annotated[int, typer.Option("--timeout-seconds")] = 600,
    request_id: Annotated[str | None, typer.Option("--request-id")] = None,
) -> None:
    try:
        values: dict[str, Any] = {
            "source": source,
            "time_range": TimeRange(start_seconds=start_seconds, end_seconds=end_seconds),
            "tasks": set(task or ()) if task else {"metadata", "transcript"},
            "output_directory": out.expanduser().resolve(strict=False),
            "language": language,
            "caption_preference": caption_preference,
            "asr_enabled": asr_enabled,
            "frame_timestamps_seconds": tuple(frame_timestamp or ()) if frame_timestamp else None,
            "max_frames": max_frames,
            "max_frame_width": max_frame_width,
            "max_download_bytes": max_download_bytes,
            "max_output_bytes": max_output_bytes,
            "timeout_seconds": timeout_seconds,
            "request_id": request_id,
        }
        request = AnalyzeVideoRequest(**values)
    except (ValidationError, ValueError, TypeError) as exc:
        _emit(_invalid_request(exc))
        raise typer.Exit(code=2) from exc
    try:
        result = analyze_video(request)
    except VideoAnalyzerFailure as exc:
        _emit(exc.error)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        _emit(
            AnalysisError(
                code=ErrorCode.INTERNAL_STAGE_FAILED,
                stage="orchestration",
                message=str(exc)[:2_048] or "analysis failed",
                retryable=False,
            )
        )
        raise typer.Exit(code=1) from exc
    _emit(result)


if __name__ == "__main__":
    app()


__all__ = ["app"]
