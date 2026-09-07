"""Deterministic preflight planning for vidscope.

The planner deliberately knows nothing about backend implementations.  It turns a
validated request and source inspection into a bounded, ordered DAG, and rejects
missing local capabilities before any media stage can run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from inspect import signature
from math import isfinite
from typing import Any, Final, cast

from .contracts import AnalysisError, AnalysisPlan, AnalyzeVideoRequest, StageRecord

_TASKS: Final[frozenset[str]] = frozenset(
    {"metadata", "transcript", "vad", "frames", "ocr"}
)
_CAPABILITY_FIELDS: Final[tuple[str, ...]] = (
    "ffprobe",
    "ffmpeg",
    "asr",
    "vad",
    "tesseract",
    "captions",
)


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Capabilities available to the current local process.

    The planner receives this explicit snapshot rather than probing binaries or
    loading models.  Defaults are intentionally unavailable: callers must make
    local capability discovery explicit before planning media work.
    """

    ffprobe: bool = False
    ffmpeg: bool = False
    asr: bool = False
    vad: bool = False
    tesseract: bool = False
    captions: bool = False

    def as_mapping(self) -> dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in _CAPABILITY_FIELDS}

    # ``model_dump`` makes this small local value convenient in plan serializers
    # without coupling it to Pydantic.
    def model_dump(self, *, mode: str = "python") -> dict[str, bool]:
        del mode
        return self.as_mapping()


class PlanningFailure(RuntimeError):
    """Typed preflight failure carrying the shared :class:`AnalysisError`.

    ``core`` can translate this into its public ``VideoAnalyzerFailure`` without
    importing ``core`` here (which would introduce an import cycle).
    """

    def __init__(self, error: AnalysisError) -> None:
        self.error = error
        # Some adapters use this name while translating failures.
        self.analysis_error = error
        message = _field(error, "message", "planning failed")
        super().__init__(str(message))

    @property
    def code(self) -> Any:
        return _field(self.error, "code", None)


# ---------------------------------------------------------------------------
# Small model/mapping helpers


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _model_field_names(model_type: type[Any]) -> set[str] | None:
    fields = getattr(model_type, "model_fields", None)
    if isinstance(fields, Mapping):
        return {str(name) for name in fields}
    try:
        return {
            name
            for name, parameter in signature(model_type).parameters.items()
            if name != "self"
            and parameter.kind
            in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        }
    except (TypeError, ValueError):
        return None


def _construct_model(model_type: type[Any], values: Mapping[str, Any]) -> Any:
    """Construct a contract model while tolerating optional schema extensions.

    The canonical contracts intentionally permit a few extra planning metadata
    fields.  Filtering against ``model_fields`` lets this planner work with a
    slightly older contract during an incremental package build.  The
    ``model_construct`` fallback is only used after normal validation rejects an
    otherwise bounded internal payload; it keeps planner failures from becoming
    import-time compatibility failures.
    """

    names = _model_field_names(model_type)
    filtered = (
        dict(values)
        if names is None
        else {k: v for k, v in values.items() if k in names}
    )
    try:
        return model_type(**filtered)
    except Exception:
        construct = getattr(model_type, "model_construct", None)
        if callable(construct):
            return construct(**filtered)
        raise


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    enum_value = getattr(value, "value", value)
    return str(enum_value)


def _task_names(request: AnalyzeVideoRequest) -> tuple[str, ...]:
    raw_tasks = _field(request, "tasks", ())
    if isinstance(raw_tasks, (str, bytes)):
        raw_tasks = (raw_tasks,)
    try:
        values = {_text(task).lower() for task in raw_tasks}
    except TypeError:
        values = {_text(raw_tasks).lower()}
    unknown = values - _TASKS
    if unknown:
        joined = ", ".join(sorted(unknown))
        _raise_failure("INVALID_REQUEST", f"unsupported analysis task(s): {joined}")
    # A fixed order avoids set iteration affecting either stages or plan JSON.
    return tuple(
        name
        for name in ("metadata", "transcript", "vad", "frames", "ocr")
        if name in values
    )


def _time_bounds(request: AnalyzeVideoRequest) -> tuple[float, float]:
    time_range = _field(request, "time_range", None)
    try:
        start = float(_field(time_range, "start_seconds", 0.0))
        end = float(_field(time_range, "end_seconds", 180.0))
    except (TypeError, ValueError):
        _raise_failure("INVALID_REQUEST", "time_range bounds must be numeric")
    if (
        not (isfinite(start) and isfinite(end))
        or start < 0
        or end <= start
        or end - start > 180.0
    ):
        _raise_failure(
            "INVALID_REQUEST",
            "time_range must be finite, non-negative, increasing, and at most 180 seconds",
        )
    return start, end


def _effective_limits(request: AnalyzeVideoRequest) -> dict[str, int]:
    names = (
        "max_frames",
        "max_frame_width",
        "max_download_bytes",
        "max_output_bytes",
        "timeout_seconds",
    )
    limits: dict[str, int] = {}
    for name in names:
        try:
            value = int(_field(request, name))
        except (TypeError, ValueError):
            _raise_failure("INVALID_REQUEST", f"{name} must be an integer")
        if value <= 0:
            _raise_failure("INVALID_REQUEST", f"{name} must be positive")
        limits[name] = value
    # The request model owns hard-cap validation.  Keep this guard for callers
    # using an equivalent duck-typed request, so stage settings remain bounded.
    hard_caps = {
        "max_frames": 12,
        "max_frame_width": 1920,
        "max_download_bytes": 268_435_456,
        "max_output_bytes": 67_108_864,
        "timeout_seconds": 600,
    }
    for name, cap in hard_caps.items():
        if limits[name] > cap:
            _raise_failure("INVALID_REQUEST", f"{name} exceeds its hard cap")
    return limits


def _source_tracks(inspection: object) -> tuple[object, ...]:
    for name in ("caption_tracks", "captions", "tracks"):
        value = _field(inspection, name, None)
        if value is not None:
            if isinstance(value, Mapping):
                value = tuple(value.values())
            try:
                return tuple(value)
            except TypeError:
                return (value,)
    return ()


def _normal_language(value: object) -> str:
    return _text(value).strip().replace("_", "-").lower()


def _language_matches(track_language: object, requested: object) -> bool:
    available = _normal_language(track_language)
    wanted = _normal_language(requested)
    if not available or not wanted:
        return False
    if available == wanted:
        return True
    # A locale fallback (``en-US`` to ``en``) is deterministic and still stays
    # within the requested language family; exact matches are considered first.
    return available.split("-", 1)[0] == wanted.split("-", 1)[0]


def _select_caption(inspection: object, request: AnalyzeVideoRequest) -> object | None:
    requested_language = _field(request, "language", "en")
    wanted_language = _normal_language(requested_language)
    candidates: list[tuple[int, int, int, object]] = []
    for index, track in enumerate(_source_tracks(inspection)):
        track_language = _normal_language(_field(track, "language", ""))
        if not track_language or not _language_matches(track_language, wanted_language):
            continue
        kind = _normal_language(_field(track, "kind", ""))
        if kind not in {"manual", "automatic"}:
            continue
        kind_rank = 0 if kind == "manual" else 1
        language_rank = 0 if track_language == wanted_language else 1
        candidates.append((kind_rank, language_rank, index, track))
    candidates.sort(key=lambda item: item[:3])
    return candidates[0][3] if candidates else None


def _track_provenance(track: object) -> dict[str, Any]:
    kind = _text(_field(track, "kind", ""), "unknown").lower()
    provider = _text(_field(track, "provider", ""), "unknown")[:128]
    language = _text(_field(track, "language", ""), "")[:32]
    source_url = _field(track, "source_url", None)
    if source_url is not None:
        source_url = _text(source_url)[:512]
    segments = _field(track, "segments", ())
    try:
        segment_count = min(len(segments), 100_000)
    except TypeError:
        segment_count = 0
    return {
        "route": "captions",
        "kind": kind,
        "provider": provider,
        "language": language,
        "source_url": source_url,
        "segment_count": segment_count,
    }


def _uniform_timestamps(
    request: AnalyzeVideoRequest, start: float, end: float
) -> tuple[float, ...]:
    provided = _field(request, "frame_timestamps_seconds", None)
    max_frames = int(_field(request, "max_frames", 1))
    if provided is not None:
        try:
            values = tuple(float(value) for value in provided)
        except (TypeError, ValueError):
            _raise_failure(
                "INVALID_REQUEST", "frame_timestamps_seconds must be numeric"
            )
        if len(values) > max_frames:
            _raise_failure(
                "INVALID_REQUEST",
                "frame timestamp count cannot exceed max_frames",
            )
        if any(not isfinite(value) or value < start or value > end for value in values):
            _raise_failure(
                "INVALID_REQUEST",
                "frame timestamps must be finite and inside time_range",
            )
        return values
    count = min(max_frames, 12)
    duration = end - start
    return tuple(
        start + duration * index / (count + 1) for index in range(1, count + 1)
    )


def _sanitize(value: object, *, depth: int = 0) -> Any:
    """Convert settings/provenance to bounded JSON-compatible values."""

    if depth > 3:
        return _text(value)[:128]
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str):
            return value[:512]
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in list(value.items())[:32]:
            result[_text(key)[:128]] = _sanitize(child, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize(child, depth=depth + 1) for child in list(value)[:32]]
    return _text(value)[:128]


def _stage(
    name: str,
    dependencies: Iterable[str],
    settings: Mapping[str, Any] | None = None,
) -> StageRecord:
    payload: dict[str, Any] = {
        "name": name,
        "status": "planned",
        "dependencies": list(dependencies),
        "start_timestamp": None,
        "end_timestamp": None,
        "elapsed_ms": None,
        "settings": _sanitize(dict(settings or {})),
        "input_artifact_ids": [],
        "output_artifact_ids": [],
        "warnings": [],
        "error": None,
    }
    return cast(StageRecord, _construct_model(StageRecord, payload))


def _source_identity(
    request: AnalyzeVideoRequest, inspection: object
) -> dict[str, Any]:
    duration = _field(inspection, "duration_seconds", None)
    try:
        duration_value = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_value = None
    if duration_value is not None and not isfinite(duration_value):
        duration_value = None
    formats = _field(inspection, "formats", ())
    try:
        format_count = min(len(formats), 100_000)
    except TypeError:
        format_count = 0
    tracks = _source_tracks(inspection)
    return {
        "source": _text(_field(request, "source", ""))[:1024],
        "is_url": bool(_field(inspection, "is_url", False)),
        "duration_seconds": duration_value,
        "format_count": format_count,
        "caption_track_count": min(len(tracks), 100_000),
    }


def _normalized_request(
    request: AnalyzeVideoRequest,
    tasks: tuple[str, ...],
    limits: Mapping[str, int],
    timestamps: tuple[float, ...],
    start: float,
    end: float,
) -> dict[str, Any]:
    return {
        "source": _text(_field(request, "source", ""))[:1024],
        "time_range": {"start_seconds": start, "end_seconds": end},
        "tasks": list(tasks),
        "language": _text(_field(request, "language", "en"))[:32],
        "caption_preference": _text(
            _field(request, "caption_preference", "manual_then_automatic_then_asr")
        )[:64],
        "asr_enabled": bool(_field(request, "asr_enabled", True)),
        "frame_timestamps_seconds": list(timestamps),
        **dict(limits),
    }


# ---------------------------------------------------------------------------
# Capability/error handling


def _make_error(code: str, message: str, *, stage: str = "planning") -> AnalysisError:
    payload = {
        "ok": False,
        "status": "failed",
        "code": code,
        "stage": stage,
        "message": message[:512],
        "retryable": False,
    }
    return cast(AnalysisError, _construct_model(AnalysisError, payload))


def _raise_failure(code: str, message: str, *, stage: str = "planning") -> None:
    raise PlanningFailure(_make_error(code, message, stage=stage))


def _require_media_capabilities(capabilities: Capabilities) -> None:
    if not bool(capabilities.ffprobe) or not bool(capabilities.ffmpeg):
        _raise_failure(
            "TOOL_UNAVAILABLE",
            "local FFprobe and FFmpeg capabilities are required for media work",
        )


# ---------------------------------------------------------------------------
# Public planner


def build_execution_plan(
    request: AnalyzeVideoRequest,
    inspection: object,
    capabilities: Capabilities,
) -> AnalysisPlan:
    """Build the bounded local execution DAG for one inspected request."""

    tasks = _task_names(request)
    limits = _effective_limits(request)
    start, end = _time_bounds(request)

    wants_metadata = "metadata" in tasks
    wants_transcript = "transcript" in tasks
    wants_vad = "vad" in tasks
    wants_frames = "frames" in tasks
    wants_ocr = "ocr" in tasks
    wants_visual = wants_frames or wants_ocr
    timestamps = _uniform_timestamps(request, start, end) if wants_visual else ()

    caption_track = _select_caption(inspection, request) if wants_transcript else None
    has_captions = caption_track is not None and bool(capabilities.captions)
    asr_fallback = wants_transcript and not has_captions
    asr_enabled = bool(_field(request, "asr_enabled", True))

    # Preflight requested task capabilities before constructing any media stage.
    # Check the most specific user-visible capability first when several are
    # unavailable (e.g. OCR and FFmpeg both missing).
    if wants_ocr and not bool(capabilities.tesseract):
        _raise_failure(
            "OCR_UNAVAILABLE",
            "local Tesseract capability is unavailable (ensure tesseract is installed in PATH)",
        )
    if asr_fallback:
        if not asr_enabled:
            _raise_failure(
                "CAPTIONS_UNAVAILABLE",
                "no accepted caption track is available and ASR fallback is disabled",
            )
        if not bool(capabilities.asr):
            _raise_failure(
                "ASR_MODEL_UNAVAILABLE",
                "local ASR capability is unavailable for transcript fallback (install with: pip install 'vidscope[asr]')",
            )
        if not bool(capabilities.vad):
            _raise_failure(
                "TOOL_UNAVAILABLE",
                "local VAD capability is unavailable for transcript fallback (install with: pip install 'vidscope[asr]')",
            )
    if wants_vad and not bool(capabilities.vad):
        _raise_failure(
            "TOOL_UNAVAILABLE",
            "local VAD capability is unavailable (install with: pip install 'vidscope[vad]')",
        )
    if wants_visual or wants_vad or asr_fallback:
        _require_media_capabilities(capabilities)

    # Fixed insertion order is part of the persisted plan contract.  Never
    # iterate the caller's task set while adding stages.
    stage_records: list[StageRecord] = []
    stage_names: set[str] = set()

    def add_stage(
        name: str,
        dependencies: Iterable[str],
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        if name in stage_names:
            return
        stage_records.append(_stage(name, dependencies, settings))
        stage_names.add(name)

    add_stage("validate_source", ())
    add_stage(
        "inspect_source",
        ("validate_source",),
        {
            "source_type": "url"
            if bool(_field(inspection, "is_url", False))
            else "local",
            "caption_track_count": len(_source_tracks(inspection)),
        },
    )
    add_stage(
        "persist_plan", ("inspect_source",), {"format": "json", "cloud_policy": "deny"}
    )

    if wants_metadata:
        add_stage("metadata", ("persist_plan",), {"route": "source_inspection"})
    if wants_transcript and has_captions:
        add_stage(
            "captions",
            ("persist_plan",),
            _track_provenance(caption_track),
        )

    needs_audio = wants_vad or asr_fallback
    needs_media = needs_audio or wants_visual
    if needs_media:
        add_stage(
            "acquire_media",
            ("persist_plan",),
            {
                "time_range": {"start_seconds": start, "end_seconds": end},
                "max_download_bytes": limits["max_download_bytes"],
                "timeout_seconds": limits["timeout_seconds"],
            },
        )
        add_stage("probe", ("acquire_media",), {"format": "json"})

    if needs_audio:
        add_stage(
            "extract_audio",
            ("probe",),
            {
                "sample_rate_hz": 16_000,
                "channels": 1,
                "codec": "pcm_s16le",
                "max_output_bytes": limits["max_output_bytes"],
            },
        )
        if wants_vad or asr_fallback:
            add_stage(
                "vad",
                ("extract_audio",),
                {"route": "silero-vad", "sample_rate_hz": 16_000},
            )
        if asr_fallback:
            add_stage(
                "transcribe",
                ("extract_audio",),
                {
                    "route": "faster-whisper",
                    "model": "tiny.en",
                    "device": "cpu",
                    "compute_type": "int8",
                    "language": _text(_field(request, "language", "en"))[:32],
                    "beam_size": 5,
                    "word_timestamps": True,
                    "vad_filter": True,
                },
            )

    if wants_visual:
        add_stage(
            "extract_frames",
            ("probe",),
            {
                "route": "ffmpeg",
                "timestamps_seconds": list(timestamps),
                "max_frames": limits["max_frames"],
                "max_frame_width": limits["max_frame_width"],
                "max_output_bytes": limits["max_output_bytes"],
            },
        )
        if wants_ocr:
            add_stage(
                "ocr",
                ("extract_frames",),
                {
                    "route": "tesseract",
                    "language": _text(_field(request, "language", "en"))[:32],
                    "max_output_bytes": limits["max_output_bytes"],
                },
            )

    routes: dict[str, Any] = {}
    if wants_metadata:
        routes["metadata"] = {"route": "source_inspection"}
    if wants_transcript:
        if has_captions:
            routes["transcript"] = _track_provenance(caption_track)
        else:
            routes["transcript"] = {
                "route": "asr",
                "provider": "faster-whisper",
                "model": "tiny.en",
                "device": "cpu",
                "compute_type": "int8",
                "language": _text(_field(request, "language", "en"))[:32],
            }
    if needs_media:
        routes["media"] = {"route": "local", "provider": "ffmpeg"}
        routes["probe"] = {"route": "ffprobe"}
    if needs_audio:
        routes["audio"] = {"route": "ffmpeg", "sample_rate_hz": 16_000}
    if wants_vad or asr_fallback:
        routes["vad"] = {"route": "silero-vad", "device": "cpu"}
    if wants_visual:
        routes["frames"] = {"route": "ffmpeg", "max_frames": limits["max_frames"]}
    if wants_ocr:
        routes["ocr"] = {
            "route": "tesseract",
            "language": _text(_field(request, "language", "en"))[:32],
        }

    normalized_request = _normalized_request(
        request,
        tasks,
        limits,
        timestamps,
        start,
        end,
    )
    diagnostics = {
        "source_duration_seconds": _field(inspection, "duration_seconds", None),
        "caption_route": routes.get("transcript", {}).get("kind")
        if isinstance(routes.get("transcript"), Mapping)
        else None,
    }
    plan_payload: dict[str, Any] = {
        "source": _text(_field(request, "source", ""))[:1024],
        "source_identity": _source_identity(request, inspection),
        "request": normalized_request,
        "normalized_request": normalized_request,
        "effective_limits": limits,
        "capabilities": capabilities.as_mapping(),
        "routes": routes,
        "stages": stage_records,
        "frame_timestamps_seconds": timestamps,
        "cloud_policy": "deny",
        "diagnostics": _sanitize(diagnostics),
        "warnings": [],
    }
    return cast(AnalysisPlan, _construct_model(AnalysisPlan, plan_payload))


__all__ = ["Capabilities", "PlanningFailure", "build_execution_plan"]
