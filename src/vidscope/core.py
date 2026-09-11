from __future__ import annotations

import functools
import importlib.util
import os
import shutil
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .artifacts import ArtifactStore, ArtifactStoreFailure
from .backends.asr import FasterWhisperBackend
from .backends.media import FFmpegBackend
from .backends.ocr import TesseractBackend
from .backends.source import CaptionResolver, MediaAcquirer, SourceInspector
from .backends.vad import SileroVadBackend
from .contracts import (
    AnalysisError,
    AnalysisMetrics,
    AnalysisResult,
    AnalysisSummary,
    AnalyzeVideoRequest,
    ArtifactRef,
    ErrorCode,
    StageRecord,
)
from .logging import logger
from .planner import Capabilities, PlanningFailure, build_execution_plan
from .settings import get_settings
from .telemetry import (
    compute_download_throughput_mbps,
    compute_ocr_fps,
    compute_rtf,
    get_peak_rss_mb,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _model_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_model_dump(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _task_name(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _error_code(value: Any) -> ErrorCode:
    if isinstance(value, ErrorCode):
        return value
    try:
        return ErrorCode(str(value))
    except ValueError:
        return ErrorCode.INTERNAL_STAGE_FAILED


def _error_from_exception(exc: BaseException, stage: str) -> AnalysisError:
    candidate = getattr(exc, "error", getattr(exc, "analysis_error", None))
    if isinstance(candidate, AnalysisError):
        if candidate.stage == "unknown":
            return candidate.model_copy(update={"stage": stage})
        return candidate
    code = _error_code(
        getattr(candidate, "code", getattr(exc, "code", "INTERNAL_STAGE_FAILED"))
    )
    return AnalysisError(
        code=code,
        stage=stage,
        message=str(exc)[:2_048] or "analysis stage failed",
        retryable=False,
    )


def _path_from_ref(store: ArtifactStore, value: Any) -> Path:
    if isinstance(value, (str, Path)):
        return Path(value)
    return store.resolve_artifact_path(value)


def _is_ref(value: Any) -> bool:
    if isinstance(value, ArtifactRef):
        return True
    return isinstance(value, Mapping) and {"artifact_id", "uri", "sha256"} <= set(value)


class VideoAnalyzerFailure(RuntimeError):
    def __init__(
        self, error: AnalysisError, *, manifest_uri: str | None = None
    ) -> None:
        self.error = error
        self.analysis_error = error
        self.failure = error
        self.manifest_uri = manifest_uri or error.manifest_uri
        super().__init__(error.message)


@dataclass
class AnalysisContext:
    source_inspector: Any = None
    caption_resolver: Any = None
    media_acquirer: Any = None
    media_backend: Any = None
    asr_backend: Any = None
    vad_backend: Any = None
    ocr_backend: Any = None
    cancel_event: threading.Event | None = None
    deadline: float | None = None
    max_workers: int = 2
    progress_callback: Any = None
    _provided: set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = (
            "source_inspector",
            "caption_resolver",
            "media_acquirer",
            "media_backend",
            "asr_backend",
            "vad_backend",
            "ocr_backend",
        )
        self._provided = {name for name in names if getattr(self, name) is not None}
        if self.source_inspector is None:
            self.source_inspector = SourceInspector()
        if self.caption_resolver is None:
            self.caption_resolver = CaptionResolver()
        if self.media_acquirer is None:
            self.media_acquirer = MediaAcquirer()
        if self.media_backend is None:
            self.media_backend = FFmpegBackend()
        if self.asr_backend is None:
            self.asr_backend = FasterWhisperBackend()
        if self.vad_backend is None:
            self.vad_backend = SileroVadBackend()
        if self.ocr_backend is None:
            self.ocr_backend = TesseractBackend()


@contextmanager
def _url_policy(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    previous = os.environ.get("YTDLP_IGNORE_CONFIG")
    os.environ["YTDLP_IGNORE_CONFIG"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("YTDLP_IGNORE_CONFIG", None)
        else:
            os.environ["YTDLP_IGNORE_CONFIG"] = previous


def _available_binary(configured: Any, name: str) -> bool:
    if configured:
        return (
            Path(str(configured)).is_file() or shutil.which(str(configured)) is not None
        )
    return shutil.which(name) is not None


@functools.cache
def _module_importable(name: str) -> bool:
    # import_module (not find_spec): a present-but-broken install must report unavailable.
    # Broad catch: transitive native init may raise OSError/RuntimeError, not just ImportError.
    # Cached: the import cost is paid once per process; restarts pick up environment changes.
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def _capabilities(context: AnalysisContext, inspection: Any) -> Capabilities:
    settings = get_settings()
    media_injected = "media_backend" in context._provided
    asr_injected = "asr_backend" in context._provided
    vad_injected = "vad_backend" in context._provided
    ocr_injected = "ocr_backend" in context._provided
    return Capabilities(
        ffprobe=media_injected or _available_binary(settings.ffprobe_bin, "ffprobe"),
        ffmpeg=media_injected or _available_binary(settings.ffmpeg_bin, "ffmpeg"),
        asr=asr_injected or _module_importable("faster_whisper"),
        vad=vad_injected or _module_importable("silero_vad"),
        tesseract=ocr_injected
        or _available_binary(settings.tesseract_bin, "tesseract"),
        captions=True,
    )


def _base_stage_records() -> list[dict[str, Any]]:
    return [
        {
            "name": "validate_source",
            "status": "planned",
            "dependencies": [],
            "settings": {},
            "warnings": [],
            "input_artifact_ids": [],
            "output_artifact_ids": [],
            "error": None,
        },
        {
            "name": "inspect_source",
            "status": "planned",
            "dependencies": ["validate_source"],
            "settings": {},
            "warnings": [],
            "input_artifact_ids": [],
            "output_artifact_ids": [],
            "error": None,
        },
        {
            "name": "persist_plan",
            "status": "planned",
            "dependencies": ["inspect_source"],
            "settings": {},
            "warnings": [],
            "input_artifact_ids": [],
            "output_artifact_ids": [],
            "error": None,
        },
    ]


def _stage_records(store: ArtifactStore) -> list[dict[str, Any]]:
    values = store.manifest.get("stages", [])
    return list(values.values()) if isinstance(values, Mapping) else list(values)


def _refs(store: ArtifactStore) -> list[ArtifactRef]:
    result: list[ArtifactRef] = []
    seen: set[str] = set()
    for value in store.manifest.get("artifacts", {}).values():
        if not isinstance(value, Mapping):
            continue
        try:
            ref = ArtifactRef(**value)
        except (TypeError, ValueError):
            continue
        if ref.artifact_id not in seen:
            result.append(ref)
            seen.add(ref.artifact_id)
    return result


def _publish_value(
    store: ArtifactStore, value: Any, *, name: str, media_type: str
) -> ArtifactRef:
    if _is_ref(value):
        return value if isinstance(value, ArtifactRef) else ArtifactRef(**dict(value))
    if isinstance(value, (bytes, bytearray)):
        return store.write_bytes(bytes(value), name=name, media_type=media_type)
    return store.publish_file(Path(value), name=name, media_type=media_type)


def _segments(value: Any) -> list[dict[str, Any]]:
    raw = _field(value, "segments", value if isinstance(value, list) else [])
    rows: list[dict[str, Any]] = []
    for item in raw or []:
        dumped = _model_dump(item)
        if isinstance(dumped, Mapping):
            row = dict(dumped)
            if "start_seconds" not in row and "start" in row:
                row["start_seconds"] = row.pop("start")
            if "end_seconds" not in row and "end" in row:
                row["end_seconds"] = row.pop("end")
            if str(row.get("text", "")).strip():
                rows.append(row)
    return rows


def _rows(value: Any, key: str) -> list[dict[str, Any]]:
    raw = _field(value, key, [])
    result: list[dict[str, Any]] = []
    for item in raw or []:
        dumped = _model_dump(item)
        if isinstance(dumped, Mapping):
            result.append(dict(dumped))
    return result


def _vtt(rows: list[dict[str, Any]]) -> str:
    def stamp(seconds: float) -> str:
        total = max(0.0, float(seconds))
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        remainder = total % 60
        return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"

    lines = ["WEBVTT", ""]
    for row in rows:
        lines.extend(
            [
                f"{stamp(row.get('start_seconds', row.get('start', 0)))} --> {stamp(row.get('end_seconds', row.get('end', 0)))}",
                str(row.get("text", "")),
                "",
            ]
        )
    return "\n".join(lines)


def _requested_tasks(request: AnalyzeVideoRequest) -> set[str]:
    return {_task_name(value) for value in request.tasks}


def _summary(
    request: AnalyzeVideoRequest,
    inspection: Any,
    completed: set[str],
    failed: set[str],
    caption: Any,
) -> AnalysisSummary:
    return AnalysisSummary(
        source=request.source,
        duration_seconds=_field(inspection, "duration_seconds", None),
        task_count=len(completed),
        requested_task_count=len(request.tasks),
        completed_task_count=len(completed),
        failed_task_count=len(failed),
        requested_tasks=sorted(request.tasks, key=_task_name),
        completed_tasks=sorted(completed),
        failed_tasks=sorted(failed),
        caption_kind=_field(caption, "kind", None),
        caption_provider=_field(caption, "provider", None),
        metadata={"cloud_policy": "deny"},
    )


def _failure_with_manifest(
    error: AnalysisError, store: ArtifactStore
) -> VideoAnalyzerFailure:
    updated = error.model_copy(
        update={"manifest_uri": store.manifest_uri, "artifact_refs": _refs(store)}
    )
    return VideoAnalyzerFailure(updated, manifest_uri=store.manifest_uri)


def _mark_interrupt(
    store: ArtifactStore, code: str, status: Literal["cancelled", "timed_out"]
) -> VideoAnalyzerFailure:
    records = _stage_records(store)
    pending = next(
        (record for record in records if record.get("status") == "planned"), None
    )
    if pending is None and records:
        pending = records[-1]
    interrupt_status: Literal["cancelled", "timed_out"] = status
    if pending is not None:
        pending["status"] = interrupt_status
        pending["start_timestamp"] = pending.get("start_timestamp") or _now()
        pending["end_timestamp"] = _now()
        pending["elapsed_ms"] = 0.0
        pending_error = AnalysisError(
            code=_error_code(code),
            status=interrupt_status,
            stage=pending.get("name", "unknown"),
            message=code,
            retryable=True,
        )
        pending["error"] = _model_dump(pending_error)
        store.update_stage(
            str(pending.get("name", "unknown")),
            status=interrupt_status,
            start_timestamp=pending["start_timestamp"],
            end_timestamp=pending["end_timestamp"],
            elapsed_ms=0.0,
            error=pending_error,
        )
    return _failure_with_manifest(
        AnalysisError(
            code=_error_code(code),
            status=interrupt_status,
            stage=pending.get("name", "unknown") if pending else "orchestration",
            message=code,
            retryable=True,
        ),
        store,
    )


def analyze_video(
    request: AnalyzeVideoRequest,
    *,
    context: AnalysisContext | None = None,
) -> AnalysisResult:
    context = context or AnalysisContext()
    store = ArtifactStore(
        request.output_directory,
        request.request_id,
        request.max_output_bytes,
    ).create()
    url_source = request.is_url
    with _url_policy(url_source):
        try:
            return _run_analysis(request, context, store)
        except VideoAnalyzerFailure:
            raise
        except ArtifactStoreFailure as exc:
            raise _failure_with_manifest(exc.error, store) from exc
        except Exception as exc:
            error = _error_from_exception(exc, "orchestration")
            raise _failure_with_manifest(error, store) from exc


def _run_analysis(
    request: AnalyzeVideoRequest, context: AnalysisContext, store: ArtifactStore
) -> AnalysisResult:
    overall_started = time.monotonic()
    inspection: Any = None
    caption: Any = None
    try:
        inspection = context.source_inspector.inspect(request)
        if "transcript" in _requested_tasks(request):
            caption = context.caption_resolver.resolve(inspection, request)
        plan = build_execution_plan(
            request, inspection, _capabilities(context, inspection)
        )
    except PlanningFailure as exc:
        store.initialize_manifest(stages=_base_stage_records())
        raise _failure_with_manifest(exc.error, store) from exc
    except Exception as exc:
        store.initialize_manifest(stages=_base_stage_records())
        raise _failure_with_manifest(
            _error_from_exception(exc, "inspect_source"), store
        ) from exc

    store.write_plan(plan)
    store.initialize_manifest(stages=getattr(plan, "stages", []))

    deadline = (
        context.deadline
        if context.deadline is not None
        else time.monotonic() + request.timeout_seconds
    )
    stage_records = _stage_records(store)
    by_name = {str(record.get("name")): record for record in stage_records}
    outputs: dict[str, list[ArtifactRef]] = {}
    values: dict[str, Any] = {}
    completed_tasks: set[str] = set()
    failed_tasks: set[str] = set()
    requested = _requested_tasks(request)
    stage_to_task = {
        "metadata": "metadata",
        "captions": "transcript",
        "transcribe": "transcript",
        "vad": "vad",
        "extract_frames": "frames",
        "ocr": "ocr",
    }

    def check_limits() -> None:
        if context.cancel_event is not None and context.cancel_event.is_set():
            raise _mark_interrupt(store, "CANCELLED", "cancelled")
        if time.monotonic() > deadline:
            raise _mark_interrupt(store, "TIMEOUT", "timed_out")

    def mark_running(record: dict[str, Any]) -> float:
        started = time.monotonic()
        record["status"] = "running"
        record["start_timestamp"] = _now()
        store.update_stage(
            str(record["name"]),
            status="running",
            start_timestamp=record["start_timestamp"],
        )
        return started

    def mark_done(
        record: dict[str, Any], started: float, refs: list[ArtifactRef] | None = None
    ) -> None:
        record["status"] = "completed"
        record["end_timestamp"] = _now()
        record["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        record["output_artifact_ids"] = [ref.artifact_id for ref in refs or []]
        store.update_stage(
            str(record["name"]),
            status="completed",
            end_timestamp=record["end_timestamp"],
            elapsed_ms=record["elapsed_ms"],
            output_artifact_ids=record["output_artifact_ids"],
        )

    def mark_failed(
        record: dict[str, Any], started: float, error: AnalysisError
    ) -> None:
        record["status"] = "failed"
        record["end_timestamp"] = _now()
        record["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        record["error"] = _model_dump(error)
        store.update_stage(
            str(record["name"]),
            status="failed",
            end_timestamp=record["end_timestamp"],
            elapsed_ms=record["elapsed_ms"],
            error=error,
        )

    for stage_idx, record in enumerate(stage_records, start=1):
        name = str(record.get("name"))
        if context.progress_callback is not None:
            with suppress(Exception):
                context.progress_callback(
                    name,
                    float(stage_idx),
                    float(len(stage_records)),
                    f"Executing stage {name}",
                )
        logger.info(
            "Executing stage %s (%d/%d) for request %s",
            name,
            stage_idx,
            len(stage_records),
            request.request_id,
        )
        dependencies = set(record.get("dependencies", []))
        if any(
            by_name.get(dep, {}).get("status")
            in {"failed", "skipped", "cancelled", "timed_out"}
            for dep in dependencies
        ):
            record["status"] = "skipped"
            record["start_timestamp"] = record["start_timestamp"] or _now()
            record["end_timestamp"] = _now()
            record["elapsed_ms"] = 0.0
            store.update_stage(
                name,
                status="skipped",
                start_timestamp=record["start_timestamp"],
                end_timestamp=record["end_timestamp"],
                elapsed_ms=0.0,
            )
            continue
        check_limits()
        started = mark_running(record)
        refs: list[ArtifactRef] = []
        try:
            if name in {"validate_source", "inspect_source", "persist_plan"}:
                pass
            elif name == "metadata":
                metadata = {
                    "source": request.source,
                    "duration_seconds": _field(inspection, "duration_seconds", None),
                    "metadata": _field(inspection, "metadata", {}),
                }
                refs.append(store.write_json(metadata, name="metadata.json"))
            elif name == "captions":
                rows = _segments(caption)
                if not rows:
                    raise RuntimeError("caption transcript output was empty")
                start = float(request.time_range.start_seconds)
                end = float(request.time_range.end_seconds)
                bounded_rows = [
                    row
                    for row in rows
                    if float(row.get("end_seconds", row.get("end", 0.0))) > start
                    and float(row.get("start_seconds", row.get("start", 0.0))) < end
                ]
                if bounded_rows:
                    rows = bounded_rows
                refs.append(
                    store.write_jsonl(
                        rows,
                        name="transcript.jsonl",
                        metadata={"caption_kind": _field(caption, "kind", None)},
                    )
                )
                refs.append(
                    store.write_text(
                        _vtt(rows), name="captions.vtt", media_type="text/vtt"
                    )
                )
            elif name == "acquire_media":
                acquired = context.media_acquirer.acquire_window(
                    inspection, request, store
                )
                ref = _publish_value(
                    store, acquired, name="media-window.mkv", media_type="video/mp4"
                )
                refs.append(ref)
                values["media"] = store.resolve_artifact_path(ref)
            elif name == "probe":
                probe = context.media_backend.probe(values["media"], request)
                refs.append(store.write_json(probe, name="probe.json"))
                values["probe"] = probe
            elif name == "extract_audio":
                audio = context.media_backend.extract_audio(
                    values["media"], request, store
                )
                ref = _publish_value(
                    store, audio, name="audio.wav", media_type="audio/wav"
                )
                refs.append(ref)
                values["audio"] = store.resolve_artifact_path(ref)
            elif name == "vad":
                vad = context.vad_backend.detect(values["audio"], request)
                intervals = _rows(vad, "intervals")
                refs.append(
                    store.write_json(
                        {
                            "intervals": intervals,
                            "metadata": _field(vad, "metadata", {}),
                        },
                        name="vad.json",
                    )
                )
            elif name == "transcribe":
                transcript = context.asr_backend.transcribe(values["audio"], request)
                rows = _segments(transcript)
                if not rows:
                    raise RuntimeError("transcript artifact was empty")
                offset = float(request.time_range.start_seconds)
                if offset > 0:
                    for row in rows:
                        if "start_seconds" in row:
                            row["start_seconds"] = round(
                                float(row["start_seconds"]) + offset, 3
                            )
                        elif "start" in row:
                            row["start_seconds"] = round(
                                float(row.pop("start")) + offset, 3
                            )
                        if "end_seconds" in row:
                            row["end_seconds"] = round(
                                float(row["end_seconds"]) + offset, 3
                            )
                        elif "end" in row:
                            row["end_seconds"] = round(
                                float(row.pop("end")) + offset, 3
                            )
                        for word in row.get("words", []) or []:
                            if isinstance(word, dict):
                                if "start_seconds" in word:
                                    word["start_seconds"] = round(
                                        float(word["start_seconds"]) + offset, 3
                                    )
                                elif "start" in word:
                                    word["start_seconds"] = round(
                                        float(word.pop("start")) + offset, 3
                                    )
                                if "end_seconds" in word:
                                    word["end_seconds"] = round(
                                        float(word["end_seconds"]) + offset, 3
                                    )
                                elif "end" in word:
                                    word["end_seconds"] = round(
                                        float(word.pop("end")) + offset, 3
                                    )
                refs.append(
                    store.write_jsonl(
                        rows,
                        name="transcript.jsonl",
                        metadata={
                            "provider": _field(
                                _field(transcript, "metadata", {}),
                                "provider",
                                "faster-whisper",
                            )
                        },
                    )
                )
            elif name == "extract_frames":
                frames = context.media_backend.extract_frames(
                    values["media"], request, store
                )
                if not frames:
                    raise RuntimeError("frame output was empty")
                for index, frame in enumerate(frames, start=1):
                    refs.append(
                        _publish_value(
                            store,
                            frame,
                            name=f"frames/frame-{index:04d}.jpg",
                            media_type="image/jpeg",
                        )
                    )
                values["frames"] = refs.copy()
            elif name == "ocr":
                frame_refs = values.get("frames", [])
                records: list[dict[str, Any]] = []
                for frame_ref in frame_refs:
                    frame_path = store.resolve_artifact_path(frame_ref)
                    ocr = context.ocr_backend.recognize(
                        frame_path,
                        "eng" if request.language == "en" else request.language,
                        request,
                    )
                    rows = _rows(ocr, "rows")
                    if not rows:
                        raise RuntimeError("OCR output was empty")
                    records.extend(
                        {"frame_artifact_id": frame_ref.artifact_id, **row}
                        for row in rows
                    )
                if not records:
                    raise RuntimeError("OCR output was empty")
                refs.append(store.write_jsonl(records, name="ocr.jsonl"))
            else:
                raise RuntimeError(f"unsupported execution stage: {name}")
            outputs[name] = refs
            if name in stage_to_task:
                completed_tasks.add(stage_to_task[name])
            mark_done(record, started, refs)
        except (VideoAnalyzerFailure, ArtifactStoreFailure) as exc:
            error = exc.error
            if isinstance(exc, ArtifactStoreFailure) and error.stage in {
                "persist_artifact",
                "artifact_store",
            }:
                error = error.model_copy(update={"stage": name})
            mark_failed(record, started, error)
            task = stage_to_task.get(name)
            if task:
                failed_tasks.add(task)
            if task == "transcript" or (
                task in {"metadata", "vad", "frames", "ocr"}
                and not (completed_tasks - {"metadata"})
            ):
                raise _failure_with_manifest(error, store) from exc
        except Exception as exc:
            error = _error_from_exception(exc, name)
            mark_failed(record, started, error)
            task = stage_to_task.get(name)
            if task:
                failed_tasks.add(task)
            if task == "transcript" or (
                task in {"metadata", "vad", "frames", "ocr"}
                and not (completed_tasks - {"metadata"})
            ):
                raise _failure_with_manifest(error, store) from exc

    # A completed caption/transcript stage is the transcript task's success marker.
    if "transcript" in requested and "transcript" not in completed_tasks:
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="transcript",
            message="requested transcript did not complete",
            retryable=False,
        )
        raise _failure_with_manifest(error, store)
    if "metadata" in requested and "metadata" not in completed_tasks:
        failed_tasks.add("metadata")
    if "frames" in requested and "frames" not in completed_tasks:
        failed_tasks.add("frames")
    if "ocr" in requested and "ocr" not in completed_tasks:
        failed_tasks.add("ocr")
    if "vad" in requested and "vad" not in completed_tasks:
        failed_tasks.add("vad")

    status: Literal["completed", "partial"] = "partial" if failed_tasks else "completed"

    total_elapsed_ms = round((time.monotonic() - overall_started) * 1000.0, 2)
    stage_elapsed: dict[str, float] = {}
    for record in _stage_records(store):
        stage_name = str(record.get("name", ""))
        stage_ms = float(record.get("elapsed_ms") or 0.0)
        if stage_name:
            stage_elapsed[stage_name] = stage_ms

    window_sec = max(
        0.001, float(request.time_range.end_seconds - request.time_range.start_seconds)
    )
    rtf: float | None = None
    if "transcribe" in stage_elapsed and stage_elapsed["transcribe"] > 0:
        rtf = compute_rtf(stage_elapsed["transcribe"] / 1000.0, window_sec)
    elif "captions" in stage_elapsed and stage_elapsed["captions"] > 0:
        rtf = compute_rtf(stage_elapsed["captions"] / 1000.0, window_sec)

    ocr_fps: float | None = None
    if "ocr" in stage_elapsed and "frames" in values:
        ocr_fps = compute_ocr_fps(
            len(values.get("frames", [])), stage_elapsed["ocr"] / 1000.0
        )

    throughput_mbps: float | None = None
    if "acquire_media" in stage_elapsed and "media" in values:
        try:
            acquired_path = Path(values["media"])
            if acquired_path.is_file():
                throughput_mbps = compute_download_throughput_mbps(
                    acquired_path.stat().st_size,
                    stage_elapsed["acquire_media"] / 1000.0,
                )
        except OSError:
            pass

    metrics = AnalysisMetrics(
        total_elapsed_ms=total_elapsed_ms,
        peak_rss_mb=get_peak_rss_mb(),
        rtf=rtf,
        ocr_fps=ocr_fps,
        download_throughput_mbps=throughput_mbps,
        stage_elapsed_ms=stage_elapsed,
    )
    store.set_metrics(metrics.model_dump())

    result = AnalysisResult(
        status=status,
        summary=_summary(request, inspection, completed_tasks, failed_tasks, caption),
        stages=[StageRecord(**record) for record in _stage_records(store)],
        warnings=[f"stage failed: {name}" for name in sorted(failed_tasks)],
        metrics=metrics,
        manifest_uri=store.manifest_uri,
        artifacts=_refs(store),
    )
    return result


__all__ = ["AnalysisContext", "VideoAnalyzerFailure", "analyze_video"]
