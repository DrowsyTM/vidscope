from __future__ import annotations

# Provider boundaries intentionally normalize arbitrary third-party failures.
# ruff: noqa: BLE001, S110, S112
"""Source inspection, caption resolution, and bounded media acquisition.

This module deliberately keeps third-party imports inside the operations that use
those providers.  Importing :mod:`video_analyzer.backends.source` therefore does
not load yt-dlp, the YouTube transcript client, or any media/model runtime.
"""

import contextlib
import html
import json
import math
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..contracts import AnalysisError, ErrorCode

MAX_DIAGNOSTIC_CHARS = 4_000
MAX_CAPTION_SEGMENTS = 10_000
MAX_CAPTION_TEXT_CHARS = 32_000
MAX_METADATA_ITEMS = 256
MAX_METADATA_DEPTH = 4
MAX_CAPTION_BYTES = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Small normalized source models


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _text(value: Any, *, limit: int = MAX_DIAGNOSTIC_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    result = str(value)
    return result if len(result) <= limit else result[:limit] + "…"


def _bounded(value: Any, *, depth: int = 0) -> Any:
    """Return JSON-friendly metadata without retaining unbounded provider data."""

    if depth >= MAX_METADATA_DEPTH:
        if isinstance(value, (str, bytes)):
            return _text(value)
        return _text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return _text(value)
    if isinstance(value, str):
        return _text(value, limit=MAX_CAPTION_TEXT_CHARS)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_METADATA_ITEMS:
                break
            key_text = _text(key, limit=128)
            if key_text.lower() in {"comments", "comment", "entries", "playlist_entries"}:
                continue
            result[key_text] = _bounded(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_bounded(item, depth=depth + 1) for item in list(value)[:MAX_METADATA_ITEMS]]
    return _text(value)


def _segment_value(segment: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in segment and segment[name] is not None:
            return segment[name]
    return None


def _normalize_segment(segment: Any) -> dict[str, Any] | None:
    if not isinstance(segment, Mapping):
        # Provider transcript snippets commonly expose attributes instead.
        segment = {
            key: getattr(segment, key)
            for key in ("start", "duration", "end", "text", "words")
            if hasattr(segment, key)
        }
    start = _finite_float(_segment_value(segment, "start_seconds", "start", "offset", "begin", "tStartMs"))
    if start is None:
        return None
    if "tStartMs" in segment and "start_seconds" not in segment and "start" not in segment:
        start /= 1000.0
    end = _finite_float(_segment_value(segment, "end_seconds", "end", "to"))
    if end is None:
        duration = _finite_float(_segment_value(segment, "duration", "duration_seconds", "dDurationMs"))
        if duration is not None:
            if "dDurationMs" in segment and "duration" not in segment and "duration_seconds" not in segment:
                duration /= 1000.0
            end = start + max(0.0, duration)
    if end is None:
        end = start
    if end < start:
        start, end = end, start
    text = _segment_value(segment, "text", "caption", "value")
    if text is None and isinstance(segment.get("segs"), Sequence):
        text = "".join(_text(item.get("utf8", "")) for item in segment["segs"] if isinstance(item, Mapping))
    normalized: dict[str, Any] = {
        "start_seconds": max(0.0, start),
        "end_seconds": max(max(0.0, start), end),
        "text": _text(text, limit=MAX_CAPTION_TEXT_CHARS),
    }
    words = segment.get("words")
    if isinstance(words, Sequence) and not isinstance(words, (str, bytes)):
        normalized["words"] = _bounded(list(words))
    return normalized


def _normalize_segments(segments: Any) -> list[dict[str, Any]]:
    if isinstance(segments, Mapping):
        # A JSON3 transcript uses an events array.
        events = segments.get("events")
        segments = events if isinstance(events, Sequence) else [segments]
    if not isinstance(segments, Iterable) or isinstance(segments, (str, bytes)):
        return []
    result: list[dict[str, Any]] = []
    for item in segments:
        normalized = _normalize_segment(item)
        if normalized is not None:
            result.append(normalized)
        if len(result) >= MAX_CAPTION_SEGMENTS:
            break
    return result


@dataclass(slots=True)
class CaptionTrack:
    """A normalized manual or automatically generated caption track."""

    kind: Literal["manual", "automatic"] | str
    language: str
    provider: str
    source_url: str | None
    segments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = _text(self.kind, limit=32).lower() or "manual"
        self.language = _text(self.language, limit=64)
        self.provider = _text(self.provider, limit=128)
        self.source_url = None if self.source_url is None else _text(self.source_url, limit=2_048)
        self.segments = _normalize_segments(self.segments)
        self.metadata = _bounded(self.metadata)
        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def model_dump(self, *, mode: str = "python", **_: Any) -> dict[str, Any]:
        value = {
            "kind": self.kind,
            "language": self.language,
            "provider": self.provider,
            "source_url": self.source_url,
            "segments": _bounded(self.segments),
            "metadata": _bounded(self.metadata),
        }
        return value


@dataclass(slots=True)
class SourceInspection:
    """Provider-neutral metadata discovered without downloading media."""

    source: str
    is_url: bool
    duration_seconds: float | None
    caption_tracks: list[CaptionTrack] = field(default_factory=list)
    formats: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    streams: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source = _text(self.source, limit=4_096)
        self.is_url = bool(self.is_url)
        self.duration_seconds = _finite_float(self.duration_seconds)
        tracks: list[CaptionTrack] = []
        for track in self.caption_tracks or []:
            if isinstance(track, CaptionTrack):
                tracks.append(track)
            elif isinstance(track, Mapping):
                try:
                    tracks.append(CaptionTrack(**dict(track)))
                except (TypeError, ValueError):
                    continue
        self.caption_tracks = tracks[:MAX_METADATA_ITEMS]
        self.formats = [dict(_bounded(item)) for item in (self.formats or []) if isinstance(item, Mapping)][:
            MAX_METADATA_ITEMS
        ]
        self.streams = [dict(_bounded(item)) for item in (self.streams or []) if isinstance(item, Mapping)][:
            MAX_METADATA_ITEMS
        ]
        self.metadata = _bounded(self.metadata)
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        if self.streams and "streams" not in self.metadata:
            self.metadata["streams"] = _bounded(self.streams)

    def model_dump(self, *, mode: str = "python", **_: Any) -> dict[str, Any]:
        return {
            "source": self.source,
            "is_url": self.is_url,
            "duration_seconds": self.duration_seconds,
            "caption_tracks": [track.model_dump(mode=mode) for track in self.caption_tracks],
            "formats": _bounded(self.formats),
            "metadata": _bounded(self.metadata),
            "streams": _bounded(self.streams),
        }


# ---------------------------------------------------------------------------
# Typed backend errors


def _make_analysis_error(
    code: str,
    message: str,
    *,
    stage: str,
    retryable: bool = False,
    diagnostics: Sequence[str] = (),
) -> AnalysisError:
    error_code: ErrorCode = ErrorCode.INTERNAL_STAGE_FAILED
    try:
        error_code = ErrorCode(code)
    except ValueError:
        error_code = ErrorCode.INTERNAL_STAGE_FAILED
    return AnalysisError(
        code=error_code,
        status="failed",
        stage=stage,
        message=_text(message),
        retryable=bool(retryable),
        diagnostics={"details": [_text(item) for item in diagnostics][:8]},
    )


class SourceBackendFailure(RuntimeError):
    """Stable, serializable failure raised by a source backend."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str = "inspect_source",
        retryable: bool = False,
        diagnostics: Sequence[str] = (),
        cause: BaseException | None = None,
    ) -> None:
        self.code = code
        self.stage = stage
        self.retryable = retryable
        self.diagnostics = tuple(_text(item) for item in diagnostics)[:8]
        self.analysis_error = _make_analysis_error(
            code,
            message,
            stage=stage,
            retryable=retryable,
            diagnostics=self.diagnostics,
        )
        self.error = self.analysis_error
        self.failure = self.analysis_error
        self.cause = cause
        super().__init__(_text(message))


def _failure(
    code: str,
    message: str,
    *,
    stage: str,
    diagnostics: Sequence[str] = (),
    cause: BaseException | None = None,
) -> SourceBackendFailure:
    return SourceBackendFailure(code, message, stage=stage, diagnostics=diagnostics, cause=cause)


# ---------------------------------------------------------------------------
# Generic provider/process helpers


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _settings_value(settings: Any, name: str, default: Any = None) -> Any:
    if settings is not None:
        if isinstance(settings, Mapping):
            value = settings.get(name, default)
            if value is not None:
                return value
        else:
            value = getattr(settings, name, default)
            if value is not None:
                return value
    env_name = {
        "ffprobe_bin": "VIDEO_ANALYZER_FFPROBE_BIN",
        "ffmpeg_bin": "VIDEO_ANALYZER_FFMPEG_BIN",
        "allowed_input_root": "VIDEO_ANALYZER_ALLOWED_INPUT_ROOT",
        "allowed_output_root": "VIDEO_ANALYZER_ALLOWED_OUTPUT_ROOT",
    }.get(name)
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]
    try:
        from ..settings import get_settings  # type: ignore

        loaded = get_settings()
        value = getattr(loaded, name, default)
        if value is not None:
            return value
    except Exception:
        pass
    return default


def _command_output(result: Any) -> tuple[int, bytes, bytes]:
    if isinstance(result, Mapping):
        code = result.get("returncode", result.get("code", 0))
        stdout = result.get("stdout", result.get("output", b""))
        stderr = result.get("stderr", b"")
    elif isinstance(result, tuple) and len(result) >= 2:
        stdout, stderr = result[0], result[1]
        code = result[2] if len(result) > 2 else 0
    else:
        code = getattr(result, "returncode", 0)
        stdout = getattr(result, "stdout", b"")
        stderr = getattr(result, "stderr", b"")
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        code_int = 0
    if isinstance(stdout, str):
        stdout = stdout.encode("utf-8", errors="replace")
    if isinstance(stderr, str):
        stderr = stderr.encode("utf-8", errors="replace")
    return code_int, bytes(stdout or b""), bytes(stderr or b"")


def _run_command(
    runner: Callable[..., Any] | None,
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> Any:
    if runner is None:
        return subprocess.run(
            list(command),
            capture_output=True,
            check=False,
            env=dict(env) if env is not None else None,
            cwd=str(cwd) if cwd is not None else None,
            timeout=timeout,
        )
    # Constructor-injected runners in tests range from subprocess.run-like
    # callables to a simple ``runner(command)`` function.  Prefer the richer
    # call, then progressively remove optional kwargs.
    attempts: tuple[dict[str, Any], ...] = (
        {"capture_output": True, "check": False, "env": env, "cwd": str(cwd) if cwd else None, "timeout": timeout},
        {"env": env, "cwd": str(cwd) if cwd else None, "timeout": timeout},
        {},
    )
    for kwargs in attempts:
        try:
            return runner(list(command), **kwargs)
        except TypeError:
            continue
    return runner(list(command))

def _provider_call(provider: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    target = getattr(provider, method, None)
    if target is None and callable(provider):
        target = provider
    if target is None:
        raise AttributeError(f"provider has no {method}")
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        (args, kwargs),
        (args, {key: value for key, value in kwargs.items() if key not in {"options", "download", "progress_hooks"}}),
        (args, {}),
    ]
    for call_args, call_kwargs in attempts:
        try:
            return target(*call_args, **call_kwargs)
        except TypeError:
            continue
    return target(*args)


@contextlib.contextmanager
def _yt_env() -> Iterator[None]:
    previous = os.environ.get("YTDLP_IGNORE_CONFIG")
    os.environ["YTDLP_IGNORE_CONFIG"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("YTDLP_IGNORE_CONFIG", None)
        else:
            os.environ["YTDLP_IGNORE_CONFIG"] = previous

def _source_path(source: str) -> Path | None:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme.lower() == "file":
        if parsed.username or parsed.password or parsed.fragment:
            raise _failure(
                "SOURCE_NOT_ALLOWED",
                "file URI credentials and fragments are not allowed",
                stage="validate_source",
            )
        if parsed.netloc not in ("", "localhost"):
            raise _failure("SOURCE_NOT_ALLOWED", "file URI host is not allowed", stage="validate_source")
        return Path(urllib.parse.unquote(parsed.path))
    if parsed.scheme:
        return None
    return Path(source).expanduser()


def _source_is_url(source: str) -> bool:
    parsed = urllib.parse.urlparse(source)
    return parsed.scheme.lower() == "https"


def _validate_local_path(path: Path, settings: Any = None) -> Path:
    if not path.is_absolute():
        raise _failure("SOURCE_NOT_ALLOWED", "local source must be an absolute path", stage="validate_source")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise _failure("SOURCE_NOT_FOUND", "local source does not exist", stage="validate_source", cause=exc) from exc
    except OSError as exc:
        raise _failure("SOURCE_NOT_FOUND", "local source cannot be resolved", stage="validate_source", cause=exc) from exc
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        raise _failure("SOURCE_NOT_FOUND", "local source is not a readable file", stage="validate_source")
    allowed = _settings_value(settings, "allowed_input_root")
    if allowed:
        try:
            root = Path(allowed).expanduser().resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            raise _failure("SOURCE_NOT_ALLOWED", "local source is outside the allowed input root", stage="validate_source")
    return resolved


def _duration_from_probe(data: Mapping[str, Any]) -> float | None:
    format_data = data.get("format")
    duration = _finite_float(_mapping_value(format_data, "duration")) if format_data else None
    if duration is not None:
        return max(0.0, duration)
    for stream in data.get("streams", []) if isinstance(data.get("streams"), Sequence) else []:
        duration = _finite_float(_mapping_value(stream, "duration"))
        if duration is not None:
            return max(0.0, duration)
    return None


def _stream_language(stream: Mapping[str, Any]) -> str:
    tags = stream.get("tags")
    if isinstance(tags, Mapping):
        for key in ("language", "lang", "LANGUAGE"):
            value = tags.get(key)
            if value:
                return _text(value, limit=64)
    return "und"


# ---------------------------------------------------------------------------
# Source inspection


class SourceInspector:
    """Inspect local media with ffprobe or URL metadata with yt-dlp."""

    def __init__(
        self,
        runner: Callable[..., Any] | None = None,
        yt_dlp_provider: Any = None,
        settings: Any = None,
        ffprobe_bin: str | Path | None = None,
        provider: Any = None,
    ) -> None:
        self.runner = runner
        self.yt_dlp_provider = yt_dlp_provider if yt_dlp_provider is not None else provider
        self.settings = settings
        self.ffprobe_bin = ffprobe_bin

    def inspect(self, request: Any) -> SourceInspection:
        source = _text(_mapping_value(request, "source", ""), limit=4_096)
        if _source_is_url(source):
            return self._inspect_url(source)
        parsed = urllib.parse.urlparse(source)
        if parsed.scheme and parsed.scheme.lower() != "file":
            raise _failure("URL_SCHEME_NOT_ALLOWED", "source URL scheme is not allowed", stage="validate_source")
        path = _validate_local_path(_source_path(source) or Path(source), self.settings)
        return self._inspect_local(source, path)

    def _inspect_local(self, source: str, path: Path) -> SourceInspection:
        ffprobe = self.ffprobe_bin or _settings_value(self.settings, "ffprobe_bin", "ffprobe") or "ffprobe"
        command = [
            str(ffprobe),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = _run_command(self.runner, command)
        except FileNotFoundError as exc:
            raise _failure("TOOL_UNAVAILABLE", "ffprobe is not available", stage="inspect_source", cause=exc) from exc
        except subprocess.TimeoutExpired as exc:
            raise _failure("TIMEOUT", "ffprobe inspection timed out", stage="inspect_source", cause=exc) from exc
        except OSError as exc:
            raise _failure("TOOL_UNAVAILABLE", "ffprobe could not be started", stage="inspect_source", cause=exc) from exc
        if isinstance(result, Mapping) and ("streams" in result or "format" in result) and "stdout" not in result:
            returncode, stdout, stderr = 0, b"", b""
            probe: Any = result
        elif isinstance(result, (str, bytes)):
            returncode, stdout, stderr = 0, (result.encode("utf-8") if isinstance(result, str) else result), b""
            probe = None
        else:
            returncode, stdout, stderr = _command_output(result)
            probe = None
        if returncode:
            raise _failure(
                "MEDIA_DECODE_FAILED",
                "ffprobe could not inspect the local source",
                stage="inspect_source",
                diagnostics=[_text(stderr)],
            )
        if probe is None:
            try:
                probe = json.loads(stdout.decode("utf-8", errors="replace"))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise _failure("MEDIA_DECODE_FAILED", "ffprobe returned malformed JSON", stage="inspect_source", cause=exc) from exc
        if not isinstance(probe, Mapping):
            raise _failure("MEDIA_DECODE_FAILED", "ffprobe returned an invalid document", stage="inspect_source")
        streams = [dict(_bounded(item)) for item in probe.get("streams", []) if isinstance(item, Mapping)]
        tracks: list[CaptionTrack] = []
        for index, stream in enumerate(streams):
            if str(stream.get("codec_type", "")).lower() != "subtitle":
                continue
            language = _stream_language(stream)
            tracks.append(
                CaptionTrack(
                    kind="manual",
                    language=language,
                    provider="ffprobe",
                    source_url=source,
                    segments=[],
                    metadata={"stream_index": index, "codec_name": stream.get("codec_name")},
                )
            )
        suffix = path.suffix.lower().lstrip(".") or "bin"
        size = path.stat().st_size
        formats = [{"format_id": "local", "ext": suffix, "protocol": "file", "filesize": size}]
        format_data = probe.get("format")
        metadata = {
            "source_kind": "local",
            "path": str(path),
            "format": _bounded(format_data if isinstance(format_data, Mapping) else {}),
            "streams": _bounded(streams),
            "probe": _bounded(probe),
        }
        return SourceInspection(
            source=source,
            is_url=False,
            duration_seconds=_duration_from_probe(probe),
            caption_tracks=tracks,
            formats=formats,
            metadata=metadata,
            streams=streams,
        )

    def _inspect_url(self, source: str) -> SourceInspection:
        parsed = urllib.parse.urlparse(source)
        if parsed.username or parsed.password or parsed.fragment:
            raise _failure("SOURCE_NOT_ALLOWED", "URL credentials and fragments are not allowed", stage="validate_source")
        options: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "skip_download": True,
            "noplaylist": True,
            "ignoreconfig": True,
            "postprocessors": [],
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "getcomments": False,
        }
        try:
            with _yt_env():
                info = self._extract_info(source, options)
        except SourceBackendFailure:
            raise
        except (ImportError, ModuleNotFoundError) as exc:
            raise _failure("TOOL_UNAVAILABLE", "yt-dlp is not available", stage="inspect_source", cause=exc) from exc
        except Exception as exc:
            raise _failure(
                "INTERNAL_STAGE_FAILED",
                "URL metadata inspection failed",
                stage="inspect_source",
                diagnostics=[_text(exc)],
                cause=exc,
            ) from exc
        if not isinstance(info, Mapping):
            raise _failure("INTERNAL_STAGE_FAILED", "yt-dlp returned invalid metadata", stage="inspect_source")
        if info.get("_type") == "playlist" or info.get("entries"):
            raise _failure("SOURCE_NOT_ALLOWED", "playlist extraction is disabled", stage="inspect_source")
        streams = [dict(_bounded(item)) for item in info.get("formats", []) if isinstance(item, Mapping)]
        tracks = self._caption_tracks_from_info(source, info)
        metadata = {
            key: _bounded(value)
            for key, value in info.items()
            if key not in {"formats", "subtitles", "automatic_captions", "requested_subtitles", "entries", "comments"}
        }
        metadata["source_kind"] = "url"
        metadata["webpage_url"] = _text(info.get("webpage_url") or source, limit=4_096)
        duration = _finite_float(info.get("duration"))
        formats = [dict(_bounded(item)) for item in info.get("formats", []) if isinstance(item, Mapping)]
        return SourceInspection(
            source=source,
            is_url=True,
            duration_seconds=duration,
            caption_tracks=tracks,
            formats=formats,
            metadata=metadata,
            streams=streams,
        )

    def _extract_info(self, source: str, options: Mapping[str, Any]) -> Any:
        provider = self.yt_dlp_provider
        if provider is not None:
            return _provider_call(provider, "extract_info", source, download=False, options=dict(options))
        import yt_dlp  # type: ignore  # lazy optional provider import

        with yt_dlp.YoutubeDL(dict(options)) as ydl:
            return ydl.extract_info(source, download=False)

    @staticmethod
    def _caption_tracks_from_info(source: str, info: Mapping[str, Any]) -> list[CaptionTrack]:
        tracks: list[CaptionTrack] = []
        for kind, key in (("manual", "subtitles"), ("automatic", "automatic_captions")):
            captions = info.get(key)
            if not isinstance(captions, Mapping):
                continue
            for language, entries in list(captions.items())[:MAX_METADATA_ITEMS]:
                candidates = entries if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)) else [entries]
                candidate = next((item for item in candidates if isinstance(item, Mapping)), {})
                url = candidate.get("url") if isinstance(candidate, Mapping) else None
                tracks.append(
                    CaptionTrack(
                        kind=kind,
                        language=_text(language, limit=64),
                        provider="yt-dlp",
                        source_url=_text(url, limit=4_096) if url else source,
                        segments=[],
                        metadata={
                            "ext": candidate.get("ext") if isinstance(candidate, Mapping) else None,
                            "name": candidate.get("name") if isinstance(candidate, Mapping) else None,
                            "formats": _bounded(candidates),
                        },
                    )
                )
        return tracks


# ---------------------------------------------------------------------------
# Captions


def _language_matches(language: str, requested: str) -> tuple[bool, bool]:
    language = language.lower().replace("_", "-")
    requested = requested.lower().replace("_", "-")
    return language == requested, language.split("-", 1)[0] == requested.split("-", 1)[0]


def _choose_track(tracks: Sequence[CaptionTrack], language: str) -> CaptionTrack | None:
    ranked: list[tuple[int, int, CaptionTrack]] = []
    requested = language.lower().replace("_", "-")
    requested_base = requested.split("-", 1)[0]
    for index, track in enumerate(tracks):
        available = track.language.lower().replace("_", "-")
        available_base = available.split("-", 1)[0]
        if available != requested and available_base != requested_base:
            continue
        kind_rank = 0 if track.kind == "manual" else 1
        language_rank = 0 if available == requested else 1
        ranked.append((kind_rank * 10 + language_rank, index, track))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def _parse_timestamp(value: str) -> float | None:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        return float(value)
    except ValueError:
        return None


def _strip_caption_markup(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()

def _parse_caption_payload(payload: Any, *, extension: str = "") -> list[dict[str, Any]]:
    if isinstance(payload, (Mapping, list)):
        if isinstance(payload, Mapping) and isinstance(payload.get("events"), Sequence):
            return _normalize_segments(payload)
        if isinstance(payload, list):
            return _normalize_segments(payload)
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    if not isinstance(payload, str):
        return []
    text = payload.lstrip("\ufeff")
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            return _parse_caption_payload(json.loads(stripped), extension=extension)
        except (ValueError, TypeError):
            pass
    if extension.lower().lstrip(".") in {"ttml", "xml", "dfxp"} or "<tt" in text[:512].lower():
        xml_rows: list[dict[str, Any]] = []
        for xml_match in re.finditer(r"<p\b([^>]*)>(.*?)</p>", text, flags=re.IGNORECASE | re.DOTALL):
            attrs, xml_body = xml_match.groups()
            begin_match = re.search(r"(?:begin|start)=['\"]([^'\"]+)", attrs, flags=re.IGNORECASE)
            end_match = re.search(r"(?:end)=['\"]([^'\"]+)", attrs, flags=re.IGNORECASE)
            start = _parse_timestamp(begin_match.group(1)) if begin_match else None
            end = _parse_timestamp(end_match.group(1)) if end_match else None
            if start is not None:
                xml_rows.append(_normalize_segment({"start": start, "end": end, "text": _strip_caption_markup(xml_body)}) or {})
        return [item for item in xml_rows if item]
    lines = text.splitlines()
    result: list[dict[str, Any]] = []
    index = 0
    timestamp_re = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})")
    while index < len(lines):
        timestamp_match = timestamp_re.search(lines[index])
        if timestamp_match is None:
            index += 1
            continue
        start = _parse_timestamp(timestamp_match.group(1))
        end = _parse_timestamp(timestamp_match.group(2))
        index += 1
        body: list[str] = []
        while index < len(lines) and lines[index].strip():
            if not timestamp_re.search(lines[index]):
                body.append(lines[index].strip())
            index += 1
        if start is not None and end is not None:
            item = _normalize_segment({"start": start, "end": end, "text": _strip_caption_markup(" ".join(body))})
            if item is not None:
                result.append(item)
    return result[:MAX_CAPTION_SEGMENTS]


def _transcript_video_id(source: str) -> str | None:
    parsed = urllib.parse.urlparse(source)
    host = parsed.netloc.lower().split(":", 1)[0].rstrip(".")
    is_youtube_host = host == "youtube.com" or host.endswith(".youtube.com")
    if host in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/").split("/", 1)[0] or None
    if is_youtube_host or host == "youtube-nocookie.com" or host.endswith(".youtube-nocookie.com"):
        query_id = urllib.parse.parse_qs(parsed.query).get("v")
        if query_id:
            return query_id[0]
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
            return parts[1]
    return None


def _segments_from_transcript(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return _normalize_segments(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _normalize_segments(value)
    snippets = getattr(value, "snippets", None)
    if snippets is not None:
        return _normalize_segments(snippets)
    return _normalize_segments(value if isinstance(value, Iterable) else [])


class CaptionResolver:
    """Resolve a requested language using native captions before local ASR."""
    def __init__(
        self,
        transcript_provider: Any = None,
        youtube_provider: Any = None,
        yt_dlp_provider: Any = None,
        runner: Callable[..., Any] | None = None,
        settings: Any = None,
        max_caption_bytes: int = MAX_CAPTION_BYTES,
        provider: Any = None,
        downloader: Any = None,
    ) -> None:
        self.transcript_provider = transcript_provider or youtube_provider or provider
        self.yt_dlp_provider = yt_dlp_provider if yt_dlp_provider is not None else downloader
        self.runner = runner
        self.settings = settings
        self.max_caption_bytes = max(1, min(int(max_caption_bytes), MAX_CAPTION_BYTES))
        if isinstance(self.transcript_provider, type):
            with contextlib.suppress(Exception):
                self.transcript_provider = self.transcript_provider()

    def resolve(self, inspection: SourceInspection, request: Any) -> CaptionTrack | None:
        requested_language = _text(_mapping_value(request, "language", "en"), limit=64) or "en"
        video_id = _transcript_video_id(inspection.source) if inspection.is_url else None
        if video_id:
            track = self._resolve_youtube(video_id, inspection.source, requested_language)
            if track is not None:
                return track
        selected = _choose_track(inspection.caption_tracks, requested_language)
        if selected is None:
            return None
        if selected.segments:
            return selected
        payload = self._fetch_caption(selected)
        if payload is not None:
            segments = _parse_caption_payload(payload, extension=_caption_extension(selected))
            if segments:
                return CaptionTrack(
                    kind=selected.kind,
                    language=selected.language,
                    provider=selected.provider,
                    source_url=selected.source_url,
                    segments=segments,
                    metadata=selected.metadata,
                )
        local_segments = self._extract_embedded_caption(selected, inspection)
        if local_segments:
            return CaptionTrack(
                kind=selected.kind,
                language=selected.language,
                provider=selected.provider,
                source_url=selected.source_url,
                segments=local_segments,
                metadata=selected.metadata,
            )
        return None

    def _resolve_youtube(self, video_id: str, source: str, language: str) -> CaptionTrack | None:
        provider = self.transcript_provider
        if provider is None:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

                provider = YouTubeTranscriptApi()
            except (ImportError, ModuleNotFoundError):
                return None
            except Exception:
                provider = None
        if provider is None:
            return None
        try:
            listing = self._transcript_listing(provider, video_id, language)
        except Exception:
            return None
        candidates: list[tuple[int, Any, str, str]] = []
        if isinstance(listing, Mapping):
            iterable: Iterable[Any] = [
                {"language_code": key, "is_generated": bool(value.get("is_generated")) if isinstance(value, Mapping) else False, "data": value}
                for key, value in listing.items()
            ]
        elif isinstance(listing, Iterable) and not isinstance(listing, (str, bytes)):
            iterable = listing
        else:
            iterable = [listing]
        for index, item in enumerate(iterable):
            item_language = _text(
                _mapping_value(item, "language_code", _mapping_value(item, "language", language)),
                limit=64,
            )
            exact, base = _language_matches(item_language, language)
            if not (exact or base):
                continue
            generated = bool(
                _mapping_value(
                    item,
                    "is_generated",
                    _mapping_value(
                        item,
                        "is_generated_caption",
                        _mapping_value(item, "generated", False),
                    ),
                )
            )
            kind = "automatic" if generated else "manual"
            rank = (0 if exact else 1) * 10 + (1 if generated else 0)
            candidates.append((rank * 10 + index, item, item_language, kind))
        candidates.sort(key=lambda item: item[0])
        for _, item, item_language, kind in candidates:
            try:
                data = _mapping_value(item, "data")
                if data is None:
                    fetch = getattr(item, "fetch", None)
                    data = fetch() if callable(fetch) else item
                segments = _segments_from_transcript(data)
            except Exception:
                continue
            if segments:
                return CaptionTrack(
                    kind=kind,
                    language=item_language,
                    provider="youtube-transcript-api",
                    source_url=source,
                    segments=segments,
                    metadata={"video_id": video_id},
                )
        return None

    @staticmethod
    def _transcript_listing(provider: Any, video_id: str, language: str) -> Any:
        listing_method = getattr(provider, "list", None) or getattr(provider, "list_transcripts", None)
        if callable(listing_method):
            return listing_method(video_id)
        get_method = getattr(provider, "get_transcript", None)
        if callable(get_method):
            return [{"language_code": language, "is_generated": False, "data": get_method(video_id, languages=[language])}]
        if callable(provider):
            return provider(video_id, language=language)
        return []

    def _fetch_caption(self, track: CaptionTrack) -> Any:
        if not track.source_url:
            return None
        provider = self.yt_dlp_provider
        if provider is not None:
            for method in ("fetch_caption", "download_caption", "read_caption"):
                target = getattr(provider, method, None)
                if callable(target):
                    try:
                        return target(track.source_url)
                    except Exception:
                        return None
        request = urllib.request.Request(track.source_url, headers={"User-Agent": "video-analyzer/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read(self.max_caption_bytes + 1)
        except Exception:
            return None
        if len(payload) > self.max_caption_bytes:
            return None
        return payload

    def _extract_embedded_caption(self, track: CaptionTrack, inspection: SourceInspection) -> list[dict[str, Any]]:
        stream_index = _mapping_value(track.metadata, "stream_index")
        if stream_index is None or inspection.is_url:
            return []
        path = _source_path(inspection.source)
        if path is None:
            return []
        ffmpeg = _settings_value(self.settings, "ffmpeg_bin", "ffmpeg") or "ffmpeg"
        command = [str(ffmpeg), "-v", "error", "-i", str(path), "-map", f"0:{stream_index}", "-f", "webvtt", "pipe:1"]
        try:
            result = _run_command(self.runner, command)
        except Exception:
            return []
        returncode, stdout, _ = _command_output(result)
        if returncode:
            return []
        return _parse_caption_payload(stdout, extension="vtt")


def _caption_extension(track: CaptionTrack) -> str:
    value = _mapping_value(track.metadata, "ext", "")
    if value:
        return _text(value, limit=16)
    if track.source_url:
        return Path(urllib.parse.urlparse(track.source_url).path).suffix
    return ""


# ---------------------------------------------------------------------------
# Bounded media acquisition


def _request_window(request: Any) -> tuple[float, float]:
    value = _mapping_value(request, "time_range")
    start = _finite_float(_mapping_value(value, "start_seconds", 0.0), 0.0) or 0.0
    end = _finite_float(_mapping_value(value, "end_seconds", 180.0), 180.0) or 180.0
    if end <= start:
        raise _failure("INVALID_REQUEST", "media window must have positive duration", stage="acquire_media")
    return max(0.0, start), max(start, end)


def _request_limit(request: Any, name: str, default: int) -> int:
    value = _mapping_value(request, name, default)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _store_stage_directory(store: Any, request: Any) -> Path:
    candidates = ("staging_directory", "staging_dir", "run_directory", "run_dir", "root", "directory")
    for name in candidates:
        value = getattr(store, name, None) if store is not None else None
        if callable(value):
            try:
                value = value()
            except TypeError:
                value = None
        if value:
            path = Path(value)
            if name not in {"staging_directory", "staging_dir"}:
                path = path / "staging"
            path.mkdir(parents=True, exist_ok=True)
            return path
    output = _mapping_value(request, "output_directory")
    request_id = _mapping_value(request, "request_id") or "run"
    base = Path(output) if output else Path(tempfile.mkdtemp(prefix="video-analyzer-output-"))
    path = base / str(request_id) / "staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _media_type(path: Path) -> str:
    return {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".wav": "audio/wav",
    }.get(path.suffix.lower(), "application/octet-stream")


def _format_choice(
    formats: Sequence[Mapping[str, Any]],
    max_bytes: int,
    *,
    need_video: bool = False,
    need_audio: bool = False,
    window_seconds: float | None = None,
) -> str:
    candidates: list[tuple[int, int, str]] = []
    for item in formats:
        format_id = item.get("format_id")
        if not format_id:
            continue
        size = _finite_float(item.get("filesize")) or _finite_float(item.get("filesize_approx"))
        if size is None and window_seconds is not None:
            bitrate = _finite_float(item.get("tbr"))
            if bitrate is not None:
                size = bitrate * 1_000 * window_seconds / 8 * 1.25
        if size is not None and size > max_bytes:
            continue
        video = str(item.get("vcodec", "")).lower() not in {"", "none"}
        audio = str(item.get("acodec", "")).lower() not in {"", "none"}
        if need_video and not video:
            continue
        if need_audio and not audio:
            continue
        quality = int(_finite_float(item.get("height")) or 0) * 1_000_000 + int(
            _finite_float(item.get("tbr")) or 0
        )
        stream_rank = 0 if video and audio else 1
        candidates.append((stream_rank, -quality, _text(format_id, limit=128)))
    if not candidates:
        raise _failure(
            "MEDIA_DECODE_FAILED",
            "no inspected media format satisfies the requested streams and download limit",
            stage="acquire_media",
        )
    candidates.sort()
    return candidates[0][2]


def _artifact_from_store(store: Any, path: Path, *, metadata: Mapping[str, Any]) -> Any:
    if store is None:
        return path
    methods = ("add_artifact", "register_artifact", "create_artifact", "persist_artifact", "add_file")
    for method_name in methods:
        method = getattr(store, method_name, None)
        if not callable(method):
            continue
        attempts: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
            ((path,), {"media_type": _media_type(path), "metadata": dict(metadata)}),
            ((path,), {"kind": "media", "metadata": dict(metadata)}),
            ((path,), {}),
        )
        for args, kwargs in attempts:
            try:
                result = method(*args, **kwargs)
            except TypeError:
                continue
            if result is not None:
                return result
    return path


class MediaAcquirer:
    def __init__(
        self,
        runner: Callable[..., Any] | None = None,
        downloader: Any = None,
        settings: Any = None,
        progress_hook: Callable[[Mapping[str, Any]], Any] | None = None,
        ffmpeg_bin: str | Path | None = None,
        provider: Any = None,
    ) -> None:
        self.runner = runner
        self.downloader = downloader if downloader is not None else provider
        self.settings = settings
        self.progress_hook = progress_hook
        self.ffmpeg_bin = ffmpeg_bin

    def acquire_window(self, inspection: SourceInspection, request: Any, store: Any = None) -> Any:
        start, end = _request_window(request)
        max_bytes = min(
            _request_limit(request, "max_download_bytes", 268_435_456),
            _request_limit(request, "max_output_bytes", 67_108_864),
        )
        staging = _store_stage_directory(store, request)
        if inspection.is_url or _source_is_url(inspection.source):
            return self._acquire_url(inspection, request, store, staging, start, end, max_bytes)
        path = _validate_local_path(_source_path(inspection.source) or Path(inspection.source), self.settings)
        return self._acquire_local(path, request, store, staging, start, end, max_bytes)

    def _acquire_local(
        self,
        source: Path,
        request: Any,
        store: Any,
        staging: Path,
        start: float,
        end: float,
        max_bytes: int,
    ) -> Any:
        timeout_seconds = _request_limit(request, "timeout_seconds", 600)
        ffmpeg = self.ffmpeg_bin or _settings_value(self.settings, "ffmpeg_bin", "ffmpeg") or "ffmpeg"
        output = staging / "media-window.mkv"
        command = [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-i",
            str(source),
            "-t",
            f"{end - start:.6f}",
            "-map",
            "0",
            "-c",
            "copy",
            str(output),
        ]
        try:
            result = _run_command(
                self.runner,
                command,
                cwd=staging,
                timeout=float(timeout_seconds),
            )
        except FileNotFoundError as exc:
            raise _failure("TOOL_UNAVAILABLE", "ffmpeg is not available", stage="acquire_media", cause=exc) from exc
        except subprocess.TimeoutExpired as exc:
            raise _failure("TIMEOUT", "local media extraction timed out", stage="acquire_media", cause=exc) from exc
        except OSError as exc:
            raise _failure("TOOL_UNAVAILABLE", "ffmpeg could not be started", stage="acquire_media", cause=exc) from exc
        returncode, _, stderr = _command_output(result)
        if returncode:
            raise _failure(
                "MEDIA_DECODE_FAILED",
                "ffmpeg could not extract the local media window",
                stage="acquire_media",
                diagnostics=[_text(stderr)],
            )
        if not output.exists() or _format_size(output) <= 0:
            raise _failure("MEDIA_DECODE_FAILED", "ffmpeg produced no media output", stage="acquire_media")
        if _format_size(output) > max_bytes:
            with contextlib.suppress(OSError):
                output.unlink()
            raise _failure(
                "DOWNLOAD_LIMIT_EXCEEDED",
                "local media window exceeds max_download_bytes",
                stage="acquire_media",
            )
        return _artifact_from_store(
            store,
            output,
            metadata={"source": str(source), "start_seconds": start, "end_seconds": end},
        )

    def _acquire_url(
        self,
        inspection: SourceInspection,
        request: Any,
        store: Any,
        staging: Path,
        start: float,
        end: float,
        max_bytes: int,
    ) -> Any:
        raw_tasks = _mapping_value(request, "tasks", ())
        task_names = {
            _text(getattr(task, "value", task)).lower()
            for task in (raw_tasks if isinstance(raw_tasks, Iterable) else (raw_tasks,))
        }
        has_captions = bool(_mapping_value(inspection, "caption_tracks", ()))
        format_id = _format_choice(
            inspection.formats,
            max_bytes,
            need_video=bool({"frames", "ocr"} & task_names),
            need_audio=bool({"vad"} & task_names) or ("transcript" in task_names and not has_captions),
            window_seconds=end - start,
        )
        output_template = str(staging / "media.%(ext)s")
        downloaded: list[Path] = []

        timeout_seconds = _request_limit(request, "timeout_seconds", 600)
        deadline = time.monotonic() + float(timeout_seconds)

        def hook(status: Mapping[str, Any]) -> None:
            if time.monotonic() > deadline:
                raise _failure("TIMEOUT", "URL media acquisition timed out", stage="acquire_media")
            if self.progress_hook is not None:
                self.progress_hook(status)
            downloaded_bytes = _finite_float(status.get("downloaded_bytes"), 0.0) or 0.0
            if downloaded_bytes > max_bytes:
                raise _failure("DOWNLOAD_LIMIT_EXCEEDED", "download exceeded max_download_bytes", stage="acquire_media")
            filename = status.get("filename")
            if filename:
                candidate = Path(str(filename))
                if _format_size(candidate) > max_bytes:
                    raise _failure("DOWNLOAD_LIMIT_EXCEEDED", "download exceeded max_download_bytes", stage="acquire_media")

        options: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "ignoreconfig": True,
            "postprocessors": [],
            "format": format_id,
            "outtmpl": output_template,
            "download_sections": [f"*{start:.6f}-{end:.6f}"],
            "max_filesize": max_bytes,
            "progress_hooks": [hook],
            "writesubtitles": False,
            "writeautomaticsub": False,
            "getcomments": False,
            "socket_timeout": timeout_seconds,
        }
        # yt-dlp's Python API uses ``download_ranges`` for section-bounded
        # downloads.  Keep the textual option as a compatibility hint for
        # injected downloaders that inspect options without importing yt-dlp.
        try:
            from yt_dlp.utils import download_range_func  # type: ignore

            options["download_ranges"] = download_range_func(None, [(start, end)])
        except (ImportError, AttributeError, TypeError):
            pass
        source = inspection.source
        try:
            with _yt_env():
                self._download(source, options)
        except SourceBackendFailure:
            raise
        except FileNotFoundError as exc:
            raise _failure("TOOL_UNAVAILABLE", "yt-dlp is not available", stage="acquire_media", cause=exc) from exc
        except Exception as exc:
            raise _failure("MEDIA_DECODE_FAILED", "URL media acquisition failed", stage="acquire_media", diagnostics=[_text(exc)], cause=exc) from exc
        for candidate in staging.iterdir():
            if candidate.is_file() and not candidate.name.endswith((".part", ".ytdl")):
                downloaded.append(candidate)
        if not downloaded:
            raise _failure("MEDIA_DECODE_FAILED", "yt-dlp produced no media output", stage="acquire_media")
        downloaded.sort(key=lambda item: (_format_size(item) <= 0, -_format_size(item), item.name))
        output = downloaded[0]
        size = _format_size(output)
        if size <= 0:
            raise _failure("MEDIA_DECODE_FAILED", "downloaded media output is empty", stage="acquire_media")
        if size > max_bytes:
            with contextlib.suppress(OSError):
                output.unlink()
            raise _failure("DOWNLOAD_LIMIT_EXCEEDED", "download exceeded max_download_bytes", stage="acquire_media")
        return _artifact_from_store(store, output, metadata={"source": source, "format_id": format_id, "start_seconds": start, "end_seconds": end})

    def _download(self, source: str, options: Mapping[str, Any]) -> Any:
        provider = self.downloader
        if provider is None:
            import yt_dlp  # type: ignore  # lazy optional provider import

            with yt_dlp.YoutubeDL(dict(options)) as ydl:
                return ydl.download([source])
        target = getattr(provider, "download", None)
        if target is None and callable(provider):
            target = provider
        if target is None:
            raise TypeError("downloader has no download method")
        attempts: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
            ((source,), {"options": dict(options), "progress_hook": options.get("progress_hooks", [None])[0]}),
            ((source,), {"options": dict(options)}),
            (([source],), dict(options)),
            ((source,), {}),
        )
        for args, kwargs in attempts:
            try:
                return target(*args, **kwargs)
            except TypeError:
                continue
        return target(source)


__all__ = [
    "CaptionResolver",
    "CaptionTrack",
    "MediaAcquirer",
    "SourceBackendFailure",
    "SourceInspection",
    "SourceInspector",
]
