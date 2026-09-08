from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools.base import ToolResult
from fastmcp.utilities.types import Image
from pydantic import ValidationError

from .artifacts import ArtifactStore, ArtifactStoreFailure, read_artifact_resource
from .backends.ocr import TesseractBackend
from .backends.source import CaptionResolver, SourceBackendFailure, SourceInspector
from .contracts import (
    AnalysisError,
    AnalysisResult,
    AnalyzeVideoRequest,
    ErrorCode,
    TimeRange,
)
from .core import AnalysisContext, VideoAnalyzerFailure
from .core import analyze_video as core_analyze_video
from .jobs import format_timestamp, global_job_manager
from .logging import configure_logging
from .settings import get_settings

logger = logging.getLogger("vidscope.mcp")

mcp = FastMCP("vidscope")


def _error_payload(error: AnalysisError) -> dict[str, Any]:
    return error.model_dump(mode="json")


@mcp.tool(
    name="analyze_video",
    annotations={"readOnlyHint": False, "idempotentHint": False},
)
def analyze_video(
    request: AnalyzeVideoRequest,
    ctx: Any = None,
) -> AnalysisResult | ToolResult:
    """Analyze one bounded local or HTTPS video through the shared core API."""

    context: AnalysisContext | None = None
    if ctx is not None:

        def progress(stage: str, progress: float, total: float, message: str) -> None:
            try:
                if hasattr(ctx, "info"):
                    ctx.info(f"[{stage}] {message}")
            except Exception:
                pass

        context = AnalysisContext(progress_callback=progress)

    try:
        return core_analyze_video(request, context=context)
    except VideoAnalyzerFailure as exc:
        payload = _error_payload(exc.error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except (ValidationError, ValueError, TypeError) as exc:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="validate_source",
            message=str(exc)[:2_048] or "request validation failed",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except Exception as exc:  # noqa: BLE001 - adapter boundary must return a typed envelope
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="orchestration",
            message=str(exc)[:2_048] or "analysis failed",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)


@mcp.tool(
    name="get_video_info",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_video_info(source: str) -> dict[str, Any] | ToolResult:
    """Fast preflight metadata discovery without downloading media.

    Returns video title, duration, available subtitle/caption languages,
    and native chapters/sections if present.
    """
    try:
        inspector = SourceInspector()
        inspection = inspector.inspect({"source": source})

        languages: set[str] = set()
        for track in inspection.caption_tracks:
            if track.language:
                languages.add(track.language)
        for stream in inspection.streams:
            lang = (
                stream.get("tags", {}).get("language")
                if isinstance(stream.get("tags"), dict)
                else None
            )
            if lang:
                languages.add(str(lang))

        chapters: list[dict[str, Any]] = []
        raw_chapters = inspection.metadata.get("chapters")
        if isinstance(raw_chapters, list):
            for ch in raw_chapters:
                if isinstance(ch, dict):
                    start = ch.get("start_time")
                    end = ch.get("end_time")
                    title = str(ch.get("title") or "")
                    if start is not None and end is not None:
                        chapters.append(
                            {
                                "title": title,
                                "start_seconds": round(float(start), 2),
                                "end_seconds": round(float(end), 2),
                                "formatted_start": format_timestamp(float(start)),
                                "formatted_end": format_timestamp(float(end)),
                            }
                        )

        title = str(
            inspection.metadata.get("title")
            or (Path(source).name if not inspection.is_url else source)
        )

        return {
            "source": inspection.source,
            "title": title,
            "duration_seconds": inspection.duration_seconds,
            "formatted_duration": (
                format_timestamp(inspection.duration_seconds)
                if inspection.duration_seconds is not None
                else None
            ),
            "has_captions": len(inspection.caption_tracks) > 0,
            "languages": sorted(languages),
            "chapters": chapters,
        }
    except VideoAnalyzerFailure as exc:
        payload = _error_payload(exc.error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except SourceBackendFailure as exc:
        payload = _error_payload(exc.error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except (ValidationError, ValueError, TypeError) as exc:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="validate_source",
            message=str(exc)[:2_048] or "request validation failed",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except Exception as exc:
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="inspect_source",
            message=str(exc)[:2_048] or "metadata inspection failed",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)


@mcp.tool(
    name="search_video",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def search_video(
    source: str,
    query: str,
    language: str = "en",
) -> dict[str, Any] | ToolResult:
    """Fast grep across video transcript/captions returning matching snippets and exact timestamps."""
    cleaned_query = query.strip()
    if not cleaned_query:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="validate_source",
            message="search query cannot be empty",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    try:
        inspector = SourceInspector()
        inspection = inspector.inspect({"source": source})
        resolver = CaptionResolver()
        track = resolver.resolve(inspection, {"language": language})

        if track is None or not track.segments:
            return {
                "source": source,
                "query": query,
                "language": language,
                "matches_count": 0,
                "matches": [],
                "message": "No captions available for this source to search.",
            }

        q = cleaned_query.lower()
        matches: list[dict[str, Any]] = []
        for seg in track.segments:
            text = str(seg.get("text") or "")
            if q in text.lower():
                matches.append(
                    {
                        "start_seconds": round(float(seg["start_seconds"]), 2),
                        "end_seconds": round(float(seg["end_seconds"]), 2),
                        "formatted_time": format_timestamp(float(seg["start_seconds"])),
                        "snippet": text,
                    }
                )

        return {
            "source": source,
            "query": query,
            "language": track.language,
            "matches_count": len(matches),
            "matches": matches,
        }
    except (VideoAnalyzerFailure, SourceBackendFailure) as exc:
        payload = _error_payload(exc.error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except Exception as exc:
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="search_video",
            message=str(exc)[:2_048] or "search failed",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)


@mcp.tool(
    name="get_video_transcript",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_video_transcript(
    source: str,
    start_seconds: float = 0.0,
    end_seconds: float = 180.0,
    language: str = "en",
) -> dict[str, Any] | ToolResult:
    """Fetch compact, strictly window-bounded transcript records without downloading media."""
    if start_seconds < 0 or end_seconds < start_seconds:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="validate_source",
            message="invalid time range: start_seconds must be >= 0 and <= end_seconds",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    try:
        inspector = SourceInspector()
        inspection = inspector.inspect({"source": source})
        resolver = CaptionResolver()
        track = resolver.resolve(inspection, {"language": language})

        if track is None or not track.segments:
            return {
                "source": source,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "language": language,
                "segment_count": 0,
                "segments": [],
                "full_text": "",
                "message": "No captions available for this source.",
            }

        filtered_segments: list[dict[str, Any]] = []
        text_snippets: list[str] = []
        for seg in track.segments:
            seg_start = float(seg["start_seconds"])
            seg_end = float(seg["end_seconds"])
            if seg_end >= start_seconds and seg_start <= end_seconds:
                clean_text = str(seg.get("text") or "").strip()
                if clean_text:
                    filtered_segments.append(
                        {
                            "start_seconds": round(seg_start, 2),
                            "end_seconds": round(seg_end, 2),
                            "formatted_time": format_timestamp(seg_start),
                            "text": clean_text,
                        }
                    )
                    text_snippets.append(clean_text)

        return {
            "source": source,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "language": track.language,
            "segment_count": len(filtered_segments),
            "segments": filtered_segments,
            "full_text": " ".join(text_snippets),
        }
    except (VideoAnalyzerFailure, SourceBackendFailure) as exc:
        payload = _error_payload(exc.error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except Exception as exc:
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="get_transcript",
            message=str(exc)[:2_048] or "failed to retrieve transcript",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)


@mcp.tool(
    name="view_frame",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def view_frame(
    source: str | None = None,
    timestamp_seconds: float | None = None,
    job_id: str | None = None,
    frame_id: str | None = None,
    max_width: int = 1280,
) -> ToolResult:
    """View a video frame as a native MCP Image content block with OCR metadata.

    Can retrieve a previously extracted frame via frame_id (and optional job_id),
    or extract a frame directly given source and timestamp_seconds.
    """
    if frame_id is not None:
        frame_data = global_job_manager.get_frame(frame_id, job_id=job_id)
        if frame_data is None:
            error = AnalysisError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                stage="view_frame",
                message=f"Frame '{frame_id}' not found or expired.",
                retryable=False,
            )
            payload = _error_payload(error)
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )
        frame_path = Path(frame_data["path"])
        if not frame_path.is_file():
            error = AnalysisError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                stage="view_frame",
                message=f"Frame file for '{frame_id}' is no longer on disk.",
                retryable=False,
            )
            payload = _error_payload(error)
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )
        meta = {
            "frame_id": frame_id,
            "timestamp_seconds": frame_data.get("timestamp_seconds"),
            "formatted_time": format_timestamp(frame_data.get("timestamp_seconds")),
            "ocr_text": frame_data.get("ocr_text"),
        }
        return ToolResult(
            content=[
                meta,
                Image(path=frame_path, format="jpeg"),
            ]
        )

    if source is not None and timestamp_seconds is not None:
        if timestamp_seconds < 0:
            error = AnalysisError(
                code=ErrorCode.INVALID_REQUEST,
                stage="view_frame",
                message="timestamp_seconds must be >= 0",
                retryable=False,
            )
            payload = _error_payload(error)
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )

        try:
            with tempfile.TemporaryDirectory(prefix="vidscope_view_frame_") as temp_dir:
                output_dir = Path(temp_dir)
                request = AnalyzeVideoRequest(
                    source=source,
                    time_range=TimeRange(
                        start_seconds=timestamp_seconds,
                        end_seconds=timestamp_seconds + 1.0,
                    ),
                    tasks={"frames"},
                    max_frames=1,
                    max_frame_width=max_width,
                    output_directory=output_dir,
                )
                result = core_analyze_video(request)
                if not result.ok or not result.artifact_refs:
                    error = AnalysisError(
                        code=ErrorCode.MEDIA_DECODE_FAILED,
                        stage="view_frame",
                        message="Failed to extract frame at requested timestamp",
                        retryable=False,
                    )
                    payload = _error_payload(error)
                    return ToolResult(
                        content=payload, structured_content=payload, is_error=True
                    )

                frame_files = list(output_dir.glob("runs/*/frames/frame-*.jpg"))
                if not frame_files:
                    error = AnalysisError(
                        code=ErrorCode.MEDIA_DECODE_FAILED,
                        stage="view_frame",
                        message="Frame file not found after extraction",
                        retryable=False,
                    )
                    payload = _error_payload(error)
                    return ToolResult(
                        content=payload, structured_content=payload, is_error=True
                    )

                extracted_frame = frame_files[0]
                frame_bytes = extracted_frame.read_bytes()

                ocr_text: str | None = None
                try:
                    ocr_backend = TesseractBackend()
                    ocr_res = ocr_backend.recognize(extracted_frame)
                    words = [
                        str(r.get("text") or "") for r in ocr_res.rows if r.get("text")
                    ]
                    ocr_text = " ".join(words).strip() or None
                except Exception:
                    ocr_text = None

                new_frame_id = f"frame_{uuid.uuid4().hex[:8]}_{int(timestamp_seconds)}"
                cache_dir = Path(tempfile.gettempdir()) / "vidscope_frames"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached_path = cache_dir / f"{new_frame_id}.jpg"
                cached_path.write_bytes(frame_bytes)

                frame_meta = {
                    "path": cached_path,
                    "timestamp_seconds": timestamp_seconds,
                    "ocr_text": ocr_text,
                }
                global_job_manager.register_frame(new_frame_id, frame_meta)

                meta = {
                    "frame_id": new_frame_id,
                    "source": source,
                    "timestamp_seconds": timestamp_seconds,
                    "formatted_time": format_timestamp(timestamp_seconds),
                    "ocr_text": ocr_text,
                }
                return ToolResult(
                    content=[
                        meta,
                        Image(data=frame_bytes, format="jpeg"),
                    ]
                )
        except VideoAnalyzerFailure as exc:
            payload = _error_payload(exc.error)
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )
        except Exception as exc:
            error = AnalysisError(
                code=ErrorCode.INTERNAL_STAGE_FAILED,
                stage="view_frame",
                message=str(exc)[:2_048] or "failed to view frame",
                retryable=False,
            )
            payload = _error_payload(error)
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )

    error = AnalysisError(
        code=ErrorCode.INVALID_REQUEST,
        stage="view_frame",
        message="Must provide either frame_id or (source, timestamp_seconds)",
        retryable=False,
    )
    payload = _error_payload(error)
    return ToolResult(content=payload, structured_content=payload, is_error=True)


@mcp.tool(
    name="get_video_timeline",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_video_timeline(
    source: str,
    start_seconds: float = 0.0,
    end_seconds: float = 180.0,
    max_keyframes: int = 4,
) -> dict[str, Any] | ToolResult:
    """Synchronously extract a unified timeline fusing dialogue and keyframes with OCR metadata.

    Window duration must be <= 180 seconds. For longer videos, use start_video_analysis.
    """
    if start_seconds < 0 or end_seconds <= start_seconds:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="validate_source",
            message="invalid time range: start_seconds must be >= 0 and < end_seconds",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if end_seconds - start_seconds > 180.0:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="validate_source",
            message=(
                f"Requested window duration ({end_seconds - start_seconds:.1f}s) exceeds the 180s synchronous limit. "
                "For longer videos, use start_video_analysis to run asynchronously without timeouts."
            ),
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    try:
        cache_dir = Path(tempfile.gettempdir()) / "vidscope_frames"
        cache_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="vidscope_timeline_") as temp_dir:
            output_dir = Path(temp_dir)
            request = AnalyzeVideoRequest(
                source=source,
                time_range=TimeRange(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                ),
                tasks={"metadata", "transcript", "frames"},
                max_frames=max_keyframes,
                max_frame_width=1280,
                output_directory=output_dir,
            )
            core_analyze_video(request)

            transcript_segments: list[dict[str, Any]] = []
            transcript_files = list(output_dir.glob("runs/*/transcript.jsonl"))
            if transcript_files:
                for line in (
                    transcript_files[0].read_text(encoding="utf-8").splitlines()
                ):
                    if line.strip():
                        try:
                            seg = json.loads(line)
                            transcript_segments.append(
                                {
                                    "start_seconds": round(
                                        float(seg["start_seconds"]), 2
                                    ),
                                    "end_seconds": round(float(seg["end_seconds"]), 2),
                                    "formatted_time": format_timestamp(
                                        float(seg["start_seconds"])
                                    ),
                                    "text": str(seg.get("text") or "").strip(),
                                }
                            )
                        except Exception:
                            continue

            frame_files = sorted(output_dir.glob("runs/*/frames/frame-*.jpg"))
            keyframes: list[dict[str, Any]] = []

            n_frames = len(frame_files)
            timestamps = [
                start_seconds + (end_seconds - start_seconds) * i / (n_frames + 1)
                for i in range(1, n_frames + 1)
            ]

            for index, frame_path in enumerate(frame_files):
                ts = timestamps[index] if index < len(timestamps) else start_seconds
                frame_id = f"frame_{uuid.uuid4().hex[:8]}_{int(ts)}"

                ocr_text = None
                try:
                    ocr_backend = TesseractBackend()
                    ocr_res = ocr_backend.recognize(frame_path)
                    words = [
                        str(r.get("text") or "") for r in ocr_res.rows if r.get("text")
                    ]
                    ocr_text = " ".join(words).strip() or None
                except Exception:
                    ocr_text = None

                cached_frame = cache_dir / f"{frame_id}.jpg"
                shutil.copy2(frame_path, cached_frame)
                global_job_manager.register_frame(
                    frame_id,
                    {
                        "path": cached_frame,
                        "timestamp_seconds": ts,
                        "ocr_text": ocr_text,
                    },
                )

                dialogue = [
                    s["text"]
                    for s in transcript_segments
                    if abs(s["start_seconds"] - ts) <= 15.0
                ]

                keyframes.append(
                    {
                        "frame_id": frame_id,
                        "timestamp_seconds": round(ts, 2),
                        "formatted_time": format_timestamp(ts),
                        "ocr_text": ocr_text,
                        "surrounding_dialogue": (
                            " ".join(dialogue) if dialogue else None
                        ),
                    }
                )

            full_transcript = " ".join(
                s["text"] for s in transcript_segments if s["text"]
            )

            return {
                "source": source,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": round(end_seconds - start_seconds, 2),
                "keyframes_count": len(keyframes),
                "keyframes": keyframes,
                "transcript_summary": full_transcript[:4_000],
                "transcript_segments": transcript_segments,
                "hint": "Call view_frame(frame_id='<frame_id>') to view any keyframe image directly in conversation context.",
            }
    except VideoAnalyzerFailure as exc:
        payload = _error_payload(exc.error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except Exception as exc:
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="get_timeline",
            message=str(exc)[:2_048] or "failed to generate timeline",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)


def _process_job_chunks(
    job_id: str,
    source: str,
    chunks: list[tuple[float, float]],
) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "vidscope_frames"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        for index, (start_sec, end_sec) in enumerate(chunks, start=1):
            with tempfile.TemporaryDirectory(
                prefix=f"vidscope_job_{job_id}_{index}_"
            ) as temp_dir:
                output_dir = Path(temp_dir)
                request = AnalyzeVideoRequest(
                    source=source,
                    time_range=TimeRange(
                        start_seconds=start_sec,
                        end_seconds=end_sec,
                    ),
                    tasks={"metadata", "transcript", "frames"},
                    max_frames=4,
                    max_frame_width=1280,
                    output_directory=output_dir,
                )
                core_analyze_video(request)

                transcript_segments: list[dict[str, Any]] = []
                transcript_files = list(output_dir.glob("runs/*/transcript.jsonl"))
                if transcript_files:
                    for line in (
                        transcript_files[0].read_text(encoding="utf-8").splitlines()
                    ):
                        if line.strip():
                            try:
                                seg = json.loads(line)
                                transcript_segments.append(
                                    {
                                        "start_seconds": round(
                                            float(seg["start_seconds"]), 2
                                        ),
                                        "end_seconds": round(
                                            float(seg["end_seconds"]), 2
                                        ),
                                        "formatted_time": format_timestamp(
                                            float(seg["start_seconds"])
                                        ),
                                        "text": str(seg.get("text") or "").strip(),
                                    }
                                )
                            except Exception:
                                continue

                frame_files = sorted(output_dir.glob("runs/*/frames/frame-*.jpg"))
                keyframes: list[dict[str, Any]] = []
                n_frames = len(frame_files)
                timestamps = [
                    start_sec + (end_sec - start_sec) * i / (n_frames + 1)
                    for i in range(1, n_frames + 1)
                ]

                for f_idx, frame_path in enumerate(frame_files):
                    ts = timestamps[f_idx] if f_idx < len(timestamps) else start_sec
                    frame_id = f"frame_{uuid.uuid4().hex[:8]}_{int(ts)}"

                    ocr_text = None
                    try:
                        ocr_backend = TesseractBackend()
                        ocr_res = ocr_backend.recognize(frame_path)
                        words = [
                            str(r.get("text") or "")
                            for r in ocr_res.rows
                            if r.get("text")
                        ]
                        ocr_text = " ".join(words).strip() or None
                    except Exception:
                        ocr_text = None

                    cached_frame = cache_dir / f"{frame_id}.jpg"
                    shutil.copy2(frame_path, cached_frame)
                    global_job_manager.register_frame(
                        frame_id,
                        {
                            "path": cached_frame,
                            "timestamp_seconds": ts,
                            "ocr_text": ocr_text,
                        },
                        job_id=job_id,
                    )

                    dialogue = [
                        s["text"]
                        for s in transcript_segments
                        if abs(s["start_seconds"] - ts) <= 15.0
                    ]

                    keyframes.append(
                        {
                            "frame_id": frame_id,
                            "timestamp_seconds": round(ts, 2),
                            "formatted_time": format_timestamp(ts),
                            "ocr_text": ocr_text,
                            "surrounding_dialogue": (
                                " ".join(dialogue) if dialogue else None
                            ),
                        }
                    )

                full_transcript = " ".join(
                    s["text"] for s in transcript_segments if s["text"]
                )

                section = {
                    "chunk_index": index,
                    "start_seconds": start_sec,
                    "end_seconds": end_sec,
                    "formatted_range": (
                        f"{format_timestamp(start_sec)} - {format_timestamp(end_sec)}"
                    ),
                    "keyframes": keyframes,
                    "transcript_summary": full_transcript[:4_000],
                    "transcript_segments": transcript_segments,
                }
                global_job_manager.update_job_progress(job_id, section, index)

        global_job_manager.complete_job(job_id)
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        global_job_manager.fail_job(job_id, str(exc))


@mcp.tool(
    name="start_video_analysis",
    annotations={"readOnlyHint": False, "idempotentHint": False},
)
def start_video_analysis(
    source: str,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    chunk_duration_seconds: float = 180.0,
) -> dict[str, Any] | ToolResult:
    """Start an asynchronous multi-chunk video analysis job.

    Returns immediately with a job_id. Call get_job_status(job_id) to monitor progress
    and stream partial sections as each chunk completes.
    """
    if start_seconds < 0:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="validate_source",
            message="start_seconds must be >= 0",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if chunk_duration_seconds <= 0 or chunk_duration_seconds > 180.0:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="validate_source",
            message="chunk_duration_seconds must be > 0 and <= 180.0",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    try:
        resolved_end = end_seconds
        if resolved_end is None:
            inspector = SourceInspector()
            inspection = inspector.inspect({"source": source})
            resolved_end = inspection.duration_seconds

        if resolved_end is None or resolved_end <= start_seconds:
            resolved_end = start_seconds + chunk_duration_seconds

        chunk_ranges: list[tuple[float, float]] = []
        curr = start_seconds
        while curr < resolved_end:
            nxt = min(curr + chunk_duration_seconds, resolved_end)
            chunk_ranges.append((curr, nxt))
            curr = nxt

        job = global_job_manager.create_job(source, chunk_ranges)

        thread = threading.Thread(
            target=_process_job_chunks,
            args=(job.job_id, source, chunk_ranges),
            daemon=True,
            name=f"vidscope-worker-{job.job_id}",
        )
        thread.start()

        return {
            "job_id": job.job_id,
            "source": source,
            "status": "processing",
            "total_chunks": len(chunk_ranges),
            "chunks": job.chunks,
            "message": (
                "Analysis started in the background. "
                "Call get_job_status(job_id) to check progress and view partial sections."
            ),
        }
    except (VideoAnalyzerFailure, SourceBackendFailure) as exc:
        payload = _error_payload(exc.error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except Exception as exc:
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="start_video_analysis",
            message=str(exc)[:2_048] or "failed to start analysis",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)


@mcp.tool(
    name="get_job_status",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_job_status(job_id: str) -> dict[str, Any] | ToolResult:
    """Check the status of an asynchronous video analysis job and retrieve completed sections."""
    job = global_job_manager.get_job(job_id)
    if job is None:
        error = AnalysisError(
            code=ErrorCode.ARTIFACT_NOT_FOUND,
            stage="get_job_status",
            message=f"Job '{job_id}' was not found or has expired.",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    return job.to_dict()


@mcp.resource(
    "vidscope://runs/{run_id}/artifacts/{artifact_id}{?page,offset,limit}",
    name="read_artifact",
)
def _read_artifact_resource(
    run_id: str,
    artifact_id: str,
    page: int | None = None,
    offset: int = 0,
    limit: int = 200,
) -> str | bytes:
    uri = f"vidscope://runs/{run_id}/artifacts/{artifact_id}"
    result = read_artifact_resource(uri, page=page, offset=offset, limit=limit)
    if isinstance(result, ArtifactStoreFailure):
        return json.dumps(result.error.model_dump(mode="json"))
    return result


@mcp.resource("vidscope://runs/{run_id}/manifest", name="read_manifest")
def _read_manifest_resource(run_id: str) -> str:
    settings = get_settings()
    root = settings.allowed_output_root or Path.cwd()
    try:
        _, manifest = ArtifactStore._load_resource_manifest(root, run_id)
        return json.dumps(manifest, indent=2)
    except ArtifactStoreFailure as exc:
        return json.dumps(exc.error.model_dump(mode="json"))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("vidscope://runs/{run_id}/plan", name="read_plan")
def _read_plan_resource(run_id: str) -> str:
    settings = get_settings()
    root = settings.allowed_output_root or Path.cwd()
    try:
        run_dir, _ = ArtifactStore._load_resource_manifest(root, run_id)
        plan_path = run_dir / "plan.json"
        if not plan_path.is_file():
            return json.dumps({"error": "plan not found"})
        return plan_path.read_text(encoding="utf-8")
    except ArtifactStoreFailure as exc:
        return json.dumps(exc.error.model_dump(mode="json"))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def main() -> None:
    configure_logging()
    mcp.run()


__all__ = [
    "analyze_video",
    "get_job_status",
    "get_video_info",
    "get_video_timeline",
    "get_video_transcript",
    "main",
    "mcp",
    "read_artifact_resource",
    "search_video",
    "start_video_analysis",
    "view_frame",
]
