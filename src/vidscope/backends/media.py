from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..contracts import AnalysisError, ErrorCode
from ..settings import get_settings


def _error(code: str, message: str, *, stage: str = "media") -> AnalysisError:
    return AnalysisError(
        code=ErrorCode(code), stage=stage, message=message, retryable=False
    )


class MediaBackendFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, stage: str = "media") -> None:
        self.error = _error(code, message, stage=stage)
        self.analysis_error = self.error
        self.code = self.error.code
        super().__init__(message)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    local_path = getattr(value, "local_path", None)
    if local_path is not None:
        return Path(local_path)
    raise MediaBackendFailure("ARTIFACT_NOT_FOUND", "media input has no local path")


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _work_directory(
    request: Any = None,
    store: Any = None,
    output_directory: Path | None = None,
) -> Path:
    if output_directory is not None:
        directory = Path(output_directory)
    elif store is not None:
        directory = Path(
            getattr(
                store, "staging_directory", getattr(store, "run_directory", Path.cwd())
            )
        )
    else:
        request_output = _field(request, "output_directory", None)
        request_id = _field(request, "request_id", "run")
        directory = Path(request_output or Path.cwd()) / str(request_id) / "staging"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


DEFAULT_MEDIA_TIMEOUT: float = 300.0


def _timeout(request: Any) -> float:
    value = _field(request, "timeout_seconds", None)
    try:
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass
    return DEFAULT_MEDIA_TIMEOUT


def _run_command(
    runner: Callable[..., Any] | None,
    command: Sequence[str],
    *,
    timeout: float | None = None,
) -> Any:
    if runner is None:
        return subprocess.run(
            list(command),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    for kwargs in (
        {"capture_output": True, "check": False, "text": True, "timeout": timeout},
        {"capture_output": True, "check": False, "timeout": timeout},
        {},
    ):
        try:
            return runner(list(command), **kwargs)
        except TypeError:
            continue
    return runner(list(command))


def _result(result: Any) -> tuple[int, str, str]:
    if isinstance(result, Mapping):
        return (
            int(result.get("returncode", result.get("code", 0))),
            _text(result.get("stdout")),
            _text(result.get("stderr")),
        )
    return (
        int(getattr(result, "returncode", 0)),
        _text(getattr(result, "stdout", "")),
        _text(getattr(result, "stderr", "")),
    )


def _settings_value(settings: Any, name: str, default: str) -> str:
    if settings is not None:
        value = _field(settings, name, None)
        if value:
            return str(value)
    value = _field(get_settings(), name, None)
    return str(value) if value else default


def _publish(store: Any, path: Path, *, name: str, media_type: str) -> Any:
    if store is None:
        return path
    publish = getattr(store, "publish_file", None)
    if callable(publish):
        return publish(path, name=name, media_type=media_type)
    return path


class FFmpegBackend:
    def __init__(
        self,
        runner: Callable[..., Any] | None = None,
        ffmpeg_bin: str | Path | None = None,
        ffprobe_bin: str | Path | None = None,
        settings: Any = None,
    ) -> None:
        self.runner = runner
        self.settings = settings
        self.ffmpeg_bin = (
            str(ffmpeg_bin)
            if ffmpeg_bin
            else _settings_value(settings, "ffmpeg_bin", "ffmpeg")
        )
        self.ffprobe_bin = (
            str(ffprobe_bin)
            if ffprobe_bin
            else _settings_value(settings, "ffprobe_bin", "ffprobe")
        )

    def probe(self, media: Any, request: Any = None, **_: Any) -> dict[str, Any]:
        source = _path(media)
        if not source.is_file():
            raise MediaBackendFailure(
                "ARTIFACT_NOT_FOUND", "media input does not exist", stage="probe"
            )
        command = [
            self.ffprobe_bin,
            "-protocol_whitelist",
            "file,pipe,crypto,data",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(source),
        ]
        try:
            result = _run_command(self.runner, command, timeout=_timeout(request))
        except FileNotFoundError as exc:
            raise MediaBackendFailure(
                "TOOL_UNAVAILABLE", "ffprobe is unavailable", stage="probe"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaBackendFailure(
                "TIMEOUT", "ffprobe timed out", stage="probe"
            ) from exc
        code, stdout, stderr = _result(result)
        if code:
            raise MediaBackendFailure(
                "MEDIA_DECODE_FAILED", f"ffprobe failed: {stderr[:512]}", stage="probe"
            )
        try:
            value = json.loads(stdout)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MediaBackendFailure(
                "MEDIA_DECODE_FAILED", "ffprobe returned malformed JSON", stage="probe"
            ) from exc
        if not isinstance(value, dict):
            raise MediaBackendFailure(
                "MEDIA_DECODE_FAILED",
                "ffprobe returned an invalid document",
                stage="probe",
            )
        return value

    def extract_audio(
        self,
        media: Any,
        request: Any = None,
        store: Any = None,
        *,
        output_directory: Path | None = None,
    ) -> Any:
        if (
            isinstance(request, (str, Path))
            and store is None
            and output_directory is None
        ):
            output_directory, request = Path(request), None
        source = _path(media)
        work = _work_directory(request, store, output_directory)
        output = work / "audio.wav"
        command = [
            self.ffmpeg_bin,
            "-nostdin",
            "-protocol_whitelist",
            "file,pipe,crypto,data",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(source),
            "-vn",
            "-map",
            "0:a:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
        self._run_media(command, output, request, "extract_audio")
        return _publish(store, output, name="audio.wav", media_type="audio/wav")

    def extract_frames(
        self,
        media: Any,
        request: Any = None,
        store: Any = None,
        *,
        timestamps_seconds: Sequence[float] | None = None,
        max_frame_width: int | None = None,
        output_directory: Path | None = None,
    ) -> list[Any]:
        if (
            isinstance(request, (str, Path))
            and store is None
            and output_directory is None
        ):
            output_directory, request = Path(request), None
        source = _path(media)
        width = int(max_frame_width or _field(request, "max_frame_width", 1280))
        max_frames = int(_field(request, "max_frames", 6))
        timestamps = timestamps_seconds or _field(
            request, "frame_timestamps_seconds", None
        )
        time_range = _field(request, "time_range", None)
        window_start = float(_field(time_range, "start_seconds", 0.0))
        if timestamps is None:
            end = float(_field(time_range, "end_seconds", 180.0))
            timestamps = tuple(
                window_start + (end - window_start) * i / (max_frames + 1)
                for i in range(1, max_frames + 1)
            )
        timestamps = tuple(float(value) for value in timestamps)
        if len(timestamps) == 0 or len(timestamps) > min(max_frames, 12):
            raise MediaBackendFailure(
                "FRAME_LIMIT_EXCEEDED",
                "requested frame count is outside bounds",
                stage="extract_frames",
            )
        work = _work_directory(request, store, output_directory) / "frames"
        work.mkdir(parents=True, exist_ok=True)
        refs: list[Any] = []
        for index, timestamp in enumerate(timestamps, start=1):
            if not math.isfinite(timestamp) or timestamp < 0:
                raise MediaBackendFailure(
                    "FRAME_LIMIT_EXCEEDED",
                    "frame timestamp is invalid",
                    stage="extract_frames",
                )
            seek_offset = (
                max(0.0, timestamp - window_start)
                if timestamp >= window_start
                else timestamp
            )
            output = work / f"frame-{index:04d}.jpg"
            command = [
                self.ffmpeg_bin,
                "-nostdin",
                "-protocol_whitelist",
                "file,pipe,crypto,data",
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-ss",
                f"{seek_offset:.6f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                f"scale='min({width},iw)':-2",
                str(output),
            ]
            self._run_media(command, output, request, "extract_frames")
            refs.append(
                _publish(
                    store,
                    output,
                    name=f"frames/frame-{index:04d}.jpg",
                    media_type="image/jpeg",
                )
            )
        return refs

    def _run_media(
        self, command: list[str], output: Path, request: Any, stage: str
    ) -> None:
        try:
            result = _run_command(self.runner, command, timeout=_timeout(request))
        except FileNotFoundError as exc:
            raise MediaBackendFailure(
                "TOOL_UNAVAILABLE", "ffmpeg is unavailable", stage=stage
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaBackendFailure(
                "TIMEOUT", "ffmpeg timed out", stage=stage
            ) from exc
        code, _, stderr = _result(result)
        if code:
            raise MediaBackendFailure(
                "MEDIA_DECODE_FAILED", f"ffmpeg failed: {stderr[:512]}", stage=stage
            )
        if not output.is_file() or output.stat().st_size <= 0:
            raise MediaBackendFailure(
                "MEDIA_DECODE_FAILED", "ffmpeg produced an empty output", stage=stage
            )


__all__ = ["FFmpegBackend", "MediaBackendFailure"]
