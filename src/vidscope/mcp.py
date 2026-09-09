from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ValidationError as FastMCPValidationError
from fastmcp.server.middleware.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult
from fastmcp.utilities.types import Image
from pydantic import Field
from pydantic import ValidationError as PydanticValidationError

from .artifacts import ArtifactStore, ArtifactStoreFailure, read_artifact_resource
from .backends.ocr import TesseractBackend
from .backends.source import (
    SourceBackendFailure,
    SourceInspector,
)
from .contracts import (
    AnalysisError,
    AnalysisTask,
    AnalyzeVideoRequest,
    ErrorCode,
    TimeRange,
)
from .core import VideoAnalyzerFailure
from .core import analyze_video as core_analyze_video
from .jobs import (
    format_timestamp,
    global_job_manager,
    reconcile_transcript_segments,
)
from .logging import configure_logging
from .settings import get_settings

logger = logging.getLogger("vidscope.mcp")

OVERLAP_BUFFER_SECONDS: float = 2.0

mcp = FastMCP("vidscope")


class ErrorNormalizationMiddleware(Middleware):
    """Normalize FastMCP and Pydantic argument validation errors into Vidscope ToolResult envelopes."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ) -> Any:
        try:
            return await call_next(context)
        except (FastMCPValidationError, PydanticValidationError) as exc:
            tool_name = (
                getattr(context.message, "name", "unknown")
                if hasattr(context, "message")
                else "unknown"
            )
            cause = getattr(exc, "__cause__", None)
            if isinstance(cause, PydanticValidationError):
                err_msgs = [
                    f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                    for err in cause.errors(include_url=False)
                ]
                msg = "; ".join(err_msgs)
            elif isinstance(exc, PydanticValidationError):
                err_msgs = [
                    f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                    for err in exc.errors(include_url=False)
                ]
                msg = "; ".join(err_msgs)
            else:
                msg = str(exc).splitlines()[0]

            error = AnalysisError(
                code=ErrorCode.INVALID_REQUEST,
                stage=tool_name,
                message=msg[:2_048] or "argument validation failed",
                retryable=False,
            )
            payload = _error_payload(error)
            return ToolResult(
                content=payload,
                structured_content=payload,
                is_error=True,
            )


mcp.add_middleware(ErrorNormalizationMiddleware())


def _get_host_capabilities() -> dict[str, bool]:
    has_tesseract = bool(shutil.which("tesseract"))
    has_faster_whisper = False
    has_silero_vad = False
    try:
        import faster_whisper  # noqa: F401

        has_faster_whisper = True
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        import silero_vad  # noqa: F401

        has_silero_vad = True
    except (ImportError, ModuleNotFoundError):
        pass
    return {
        "local_asr_available": has_faster_whisper and has_silero_vad,
        "local_ocr_available": has_tesseract,
    }


def _error_payload(error: AnalysisError) -> dict[str, Any]:
    return error.model_dump(mode="json")


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

        capabilities = _get_host_capabilities()
        caption_tracks_listed = [
            {
                "language": track.language,
                "kind": track.kind,
                "provider": track.provider,
            }
            for track in inspection.caption_tracks
        ]

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
            "caption_tracks_listed": caption_tracks_listed,
            "capabilities": capabilities,
            "chapters": chapters,
        }
    except VideoAnalyzerFailure as exc:
        payload = _error_payload(exc.error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except SourceBackendFailure as exc:
        payload = _error_payload(exc.error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except (PydanticValidationError, ValueError, TypeError) as exc:
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


def _process_job_chunks(
    job_id: str,
    source: str,
    start_seconds: float,
    end_seconds: float | None,
    chunk_duration_seconds: float,
) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "vidscope_frames"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolved_end = end_seconds
        video_duration: float | None = None
        try:
            inspector = SourceInspector()
            inspection = inspector.inspect({"source": source})
            video_duration = inspection.duration_seconds
        except Exception as exc:
            logger.warning("Job %s duration inspection failed: %s", job_id, exc)

        if resolved_end is None:
            resolved_end = video_duration

        if resolved_end is None or resolved_end <= start_seconds:
            resolved_end = start_seconds + chunk_duration_seconds

        chunk_ranges: list[tuple[float, float]] = []
        curr = start_seconds
        while curr < resolved_end:
            nxt = min(curr + chunk_duration_seconds, resolved_end)
            chunk_ranges.append((curr, nxt))
            curr = nxt

        global_job_manager.set_job_chunks(
            job_id,
            chunk_ranges,
            video_duration_seconds=video_duration,
        )

        for index, (start_sec, end_sec) in enumerate(chunk_ranges, start=1):
            req_end = (
                min(end_sec + OVERLAP_BUFFER_SECONDS, resolved_end)
                if index < len(chunk_ranges)
                else end_sec
            )
            with tempfile.TemporaryDirectory(
                prefix=f"vidscope_job_{job_id}_{index}_"
            ) as temp_dir:
                output_dir = Path(temp_dir)

                # Attempt analysis with transcript and frames.
                # If captions fail (e.g. 429 on YouTube captions) and local ASR is unavailable,
                # fall back to visual-only analysis so keyframes and OCR are still produced.
                tasks_to_try = [
                    {
                        AnalysisTask.METADATA,
                        AnalysisTask.TRANSCRIPT,
                        AnalysisTask.FRAMES,
                    },
                    {
                        AnalysisTask.METADATA,
                        AnalysisTask.FRAMES,
                    },
                ]

                res_ok = False
                transcript_error: str | None = None
                for tasks in tasks_to_try:
                    try:
                        request = AnalyzeVideoRequest(
                            source=source,
                            time_range=TimeRange(
                                start_seconds=start_sec,
                                end_seconds=req_end,
                            ),
                            tasks=tasks,
                            max_frames=4,
                            max_frame_width=1280,
                            output_directory=output_dir,
                        )
                        core_analyze_video(request)
                        res_ok = True
                        break
                    except VideoAnalyzerFailure as exc:
                        if tasks == tasks_to_try[0]:
                            transcript_error = str(exc)
                            logger.warning(
                                "Job %s chunk %d: transcript extraction failed (%s); falling back to visual-only",
                                job_id,
                                index,
                                exc,
                            )
                            continue
                        raise

                if not res_ok:
                    raise RuntimeError(f"Chunk {index} analysis failed")

                transcript_segments: list[dict[str, Any]] = []
                transcript_files = sorted(
                    f
                    for f in output_dir.rglob("transcript.jsonl")
                    if "staging" not in f.parts
                )
                if transcript_files:
                    for line in (
                        transcript_files[0].read_text(encoding="utf-8").splitlines()
                    ):
                        if line.strip():
                            try:
                                seg = json.loads(line)
                                seg_dict: dict[str, Any] = {
                                    "start_seconds": round(
                                        float(seg["start_seconds"]), 2
                                    ),
                                    "end_seconds": round(float(seg["end_seconds"]), 2),
                                    "formatted_time": format_timestamp(
                                        float(seg["start_seconds"])
                                    ),
                                    "text": str(seg.get("text") or "").strip(),
                                }
                                if "words" in seg and seg["words"]:
                                    seg_dict["words"] = seg["words"]
                                transcript_segments.append(seg_dict)
                            except Exception:
                                continue

                # Reconcile transcript segments against previously accumulated transcript to eliminate
                # duplicate overlap tokens at chunk boundaries before building keyframe dialogue and chunk summary
                job = global_job_manager.get_job(job_id)
                prev_transcript = job.full_transcript if job else []
                if prev_transcript and transcript_segments:
                    combined = reconcile_transcript_segments(
                        prev_transcript, transcript_segments
                    )
                    transcript_segments = combined[len(prev_transcript) :]

                frame_files = sorted(
                    f
                    for f in output_dir.rglob("frame-*.jpg")
                    if "staging" not in f.parts
                )
                keyframes: list[dict[str, Any]] = []
                n_frames = len(frame_files)
                timestamps = [
                    start_sec + (end_sec - start_sec) * i / (n_frames + 1)
                    for i in range(1, n_frames + 1)
                ]

                has_tesseract = bool(shutil.which("tesseract"))
                for f_idx, frame_path in enumerate(frame_files):
                    ts = timestamps[f_idx] if f_idx < len(timestamps) else start_sec
                    frame_id = f"frame_{uuid.uuid4().hex[:8]}_{int(ts)}"

                    ocr_text = None
                    if has_tesseract:
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
                            "ocr_status": (
                                "detected"
                                if ocr_text
                                else (
                                    "no_text_detected"
                                    if has_tesseract
                                    else "tesseract_unavailable"
                                )
                            ),
                            "surrounding_dialogue": (
                                " ".join(dialogue) if dialogue else None
                            ),
                        }
                    )

                full_transcript = " ".join(
                    s["text"] for s in transcript_segments if s["text"]
                )
                if full_transcript:
                    summary_text = full_transcript[:4_000]
                    transcript_status = "completed"
                else:
                    detected_texts = [
                        f"[{kf['formatted_time']}] {kf['ocr_text']}"
                        for kf in keyframes
                        if kf.get("ocr_text")
                    ]
                    if detected_texts:
                        ocr_summary = "On-screen text detected: " + "; ".join(
                            detected_texts[:5]
                        )
                    elif has_tesseract:
                        ocr_summary = "OCR analyzed (no on-screen text detected)."
                    else:
                        ocr_summary = (
                            "OCR unavailable (tesseract binary not installed on host)."
                        )
                    summary_text = (
                        f"Keyframe extraction only ({len(keyframes)} frames, "
                        f"{format_timestamp(start_sec)}-{format_timestamp(end_sec)}). {ocr_summary}"
                    )
                    if transcript_error:
                        transcript_status = "failed"
                        summary_text += (
                            f" (Transcript extraction failed: {transcript_error})"
                        )
                    else:
                        transcript_status = "no_speech_detected"
                        summary_text += " (No speech detected in audio window)."

                section: dict[str, Any] = {
                    "chunk_index": index,
                    "mode": "speech_and_visual" if full_transcript else "visual_only",
                    "transcript_status": transcript_status,
                    "start_seconds": round(start_sec, 2),
                    "end_seconds": round(end_sec, 2),
                    "formatted_range": (
                        f"{format_timestamp(start_sec)} - {format_timestamp(end_sec)}"
                    ),
                    "summary": summary_text,
                    "keyframes": keyframes,
                }
                if transcript_error:
                    section["transcript_error"] = transcript_error
                global_job_manager.update_job_progress(
                    job_id,
                    section,
                    index,
                    transcript_segments=transcript_segments,
                )

        global_job_manager.complete_job(job_id)
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        global_job_manager.fail_job(job_id, str(exc))


@mcp.tool(
    name="analyze_video",
    annotations={"readOnlyHint": False, "idempotentHint": False},
)
def analyze_video(
    source: Annotated[
        str,
        Field(description="URL or local path of the video to analyze"),
    ],
    start_seconds: Annotated[
        float,
        Field(ge=0.0, description="Start offset in seconds (must be >= 0)"),
    ] = 0.0,
    end_seconds: Annotated[
        float | None,
        Field(description="Optional end offset in seconds"),
    ] = None,
    chunk_duration_seconds: Annotated[
        float,
        Field(
            gt=0.0,
            le=180.0,
            description="Duration of each analysis window in seconds (0 < duration <= 180.0)",
        ),
    ] = 180.0,
    sync_timeout_seconds: Annotated[
        float,
        Field(
            ge=0.0,
            description="Maximum seconds to wait synchronously before transitioning to background job",
        ),
    ] = 5.0,
) -> dict[str, Any] | ToolResult:
    """Analyze a video and extract a token-efficient visual & speech timeline.

    Runs synchronously for up to sync_timeout_seconds (~5s). If the analysis finishes
    within this window, returns the complete timeline immediately. For longer videos,
    seamlessly transitions to background processing and returns a job_id with estimated
    time to completion.
    """
    if start_seconds < 0:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="analyze_video",
            message="start_seconds must be >= 0",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if chunk_duration_seconds <= 0 or chunk_duration_seconds > 180.0:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="analyze_video",
            message="chunk_duration_seconds must be > 0 and <= 180.0",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    try:
        job_key = f"{source}_{start_seconds}_{end_seconds}_{chunk_duration_seconds}"
        existing_job = global_job_manager.find_job_by_key(job_key)
        if existing_job is not None and existing_job.status in (
            "processing",
            "completed",
        ):
            job = existing_job
        else:
            initial_chunks: list[tuple[float, float]] = []
            if end_seconds is not None and end_seconds > start_seconds:
                curr = start_seconds
                while curr < end_seconds:
                    nxt = min(curr + chunk_duration_seconds, end_seconds)
                    initial_chunks.append((curr, nxt))
                    curr = nxt
            job = global_job_manager.create_job(source, initial_chunks, job_key=job_key)

            thread = threading.Thread(
                target=_process_job_chunks,
                args=(
                    job.job_id,
                    source,
                    start_seconds,
                    end_seconds,
                    chunk_duration_seconds,
                ),
                daemon=True,
                name=f"vidscope-worker-{job.job_id}",
            )
            thread.start()

        # Wait synchronously up to sync_timeout_seconds for fast completion
        job.completed_event.wait(timeout=max(0.1, sync_timeout_seconds))

        if job.status == "completed":
            resolved_end = (
                job.chunks[-1]["end_seconds"]
                if job.chunks
                else (end_seconds or (start_seconds + chunk_duration_seconds))
            )
            return {
                "status": "completed",
                "job_id": job.job_id,
                "source": source,
                "start_seconds": start_seconds,
                "end_seconds": resolved_end,
                "coverage": job.coverage(),
                "total_chunks": job.total_chunks,
                "completed_chunks": job.completed_chunks,
                "next_since_chunk": len(job.available_sections),
                "has_more": False,
                "timeline": job.available_sections,
                "hint": "Use search_video(job_id=...) to search transcript or view_frame(frame_id=...) to inspect frames.",
            }

        if job.status == "failed":
            error = AnalysisError(
                code=ErrorCode.INTERNAL_STAGE_FAILED,
                stage="analyze_video",
                message=job.error or "analysis failed",
                retryable=False,
            )
            payload = _error_payload(error)
            payload["job_id"] = job.job_id
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )

        processing_end: float | None = (
            job.chunks[-1]["end_seconds"] if job.chunks else end_seconds
        )
        remaining = max(
            1.0,
            round(job.last_estimated_remaining or job.estimated_total_seconds, 1),
        )
        return {
            "status": "processing",
            "job_id": job.job_id,
            "source": source,
            "start_seconds": start_seconds,
            "end_seconds": processing_end,
            "coverage": job.coverage(),
            "total_chunks": job.total_chunks,
            "completed_chunks": job.completed_chunks,
            "next_since_chunk": len(job.available_sections),
            "has_more": True,
            "next_action": "get_job_status",
            "retry_after_seconds": remaining,
            "estimated_completion_seconds": round(job.estimated_total_seconds, 1),
            "initial_timeline": job.available_sections,
            "hint": f"Poll get_job_status(job_id='{job.job_id}', since_chunk={len(job.available_sections)})",
        }
    except (VideoAnalyzerFailure, SourceBackendFailure) as exc:
        payload = _error_payload(exc.error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except Exception as exc:
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="analyze_video",
            message=str(exc)[:2_048] or "failed to start analysis",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)


@mcp.tool(
    name="get_job_status",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_job_status(job_id: str, since_chunk: int = 0) -> dict[str, Any] | ToolResult:
    """Check the status of a background video analysis job and retrieve streaming timeline chunks.

    Supports incremental polling: pass since_chunk (e.g. 1) to retrieve only newly completed
    timeline sections, preventing token waste on repeated calls.
    """
    job = global_job_manager.get_job(job_id)
    if job is None:
        error = AnalysisError(
            code=ErrorCode.ARTIFACT_NOT_FOUND,
            stage="get_job_status",
            message=f"Job '{job_id}' was not found or has expired. Call analyze_video to start a new analysis.",
            retryable=False,
            diagnostics={
                "job_id": job_id,
                "next_action": "analyze_video",
                "retry_after_seconds": 0,
            },
        )
        payload = _error_payload(error)
        payload["next_action"] = "analyze_video"
        payload["retry_after_seconds"] = 0
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if job.status == "failed":
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="analyze_video",
            message=job.error or "analysis job failed",
            retryable=False,
        )
        payload = _error_payload(error)
        payload["job_id"] = job_id
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    return job.to_dict(since_chunk=since_chunk)


@mcp.tool(
    name="view_frame",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def view_frame(
    frame_id: Annotated[
        str | None,
        Field(
            description="Keyframe identifier from analyze_video timeline (mutually exclusive with source/timestamp)",
        ),
    ] = None,
    source: Annotated[
        str | None,
        Field(
            description="Video URL or path to extract a frame on demand (requires timestamp_seconds, mutually exclusive with frame_id)",
        ),
    ] = None,
    timestamp_seconds: Annotated[
        float | None,
        Field(
            ge=0.0,
            description="Timestamp in seconds to extract frame on demand (requires source, mutually exclusive with frame_id)",
        ),
    ] = None,
    max_width: Annotated[
        int,
        Field(
            gt=0,
            description="Maximum frame image width in pixels",
        ),
    ] = 1280,
) -> ToolResult:
    """View a video frame as a native MCP Image content block with OCR metadata.

    Can retrieve an extracted keyframe via frame_id, or extract a frame on demand
    given source and timestamp_seconds.
    """
    if frame_id is not None and (source is not None or timestamp_seconds is not None):
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="view_frame",
            message="Cannot provide both 'frame_id' and 'source'/'timestamp_seconds'. Provide either 'frame_id' to retrieve an existing keyframe, or ('source' and 'timestamp_seconds') to extract a frame at a specific timestamp.",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if (source is not None and timestamp_seconds is None) or (
        source is None and timestamp_seconds is not None
    ):
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="view_frame",
            message="Both 'source' and 'timestamp_seconds' must be provided when extracting a frame by timestamp.",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if frame_id is None and source is None and timestamp_seconds is None:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="view_frame",
            message="Must provide either 'frame_id' or ('source' and 'timestamp_seconds')",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if frame_id is not None:
        frame_data = global_job_manager.get_frame(frame_id)
        if frame_data is None:
            error = AnalysisError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                stage="view_frame",
                message=f"Frame '{frame_id}' not found or expired. Call analyze_video to start a new analysis.",
                retryable=False,
                diagnostics={
                    "frame_id": frame_id,
                    "next_action": "analyze_video",
                    "retry_after_seconds": 0,
                },
            )
            payload = _error_payload(error)
            payload["next_action"] = "analyze_video"
            payload["retry_after_seconds"] = 0
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
                    tasks={AnalysisTask.FRAMES},
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

                frame_files = sorted(
                    f
                    for f in output_dir.rglob("frame-*.jpg")
                    if "staging" not in f.parts
                )
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
    name="search_video",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def search_video(
    job_id: Annotated[
        str,
        Field(description="Analysis job ID from analyze_video to search transcript"),
    ],
    query: Annotated[
        str,
        Field(description="Text or regex pattern to search for in transcript"),
    ],
    is_regex: Annotated[
        bool,
        Field(description="Whether to treat query as a regular expression"),
    ] = False,
    case_sensitive: Annotated[
        bool,
        Field(description="Whether search should match case sensitivity"),
    ] = False,
    max_matches: Annotated[
        int,
        Field(gt=0, description="Maximum number of matches to return"),
    ] = 20,
) -> dict[str, Any] | ToolResult:
    """Search the transcript of an analyzed video using job_id.

    Requires a completed analysis job to prevent false negatives.
    Supports substring and regex matching.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="search_video",
            message="search query cannot be empty",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = (
            re.compile(cleaned_query, flags)
            if is_regex
            else re.compile(re.escape(cleaned_query), flags)
        )
    except re.error as exc:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="search_video",
            message=f"invalid regular expression: {exc}",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    job = global_job_manager.get_job(job_id)
    if job is None:
        error = AnalysisError(
            code=ErrorCode.ARTIFACT_NOT_FOUND,
            stage="search_video",
            message=f"Job '{job_id}' was not found or has expired. Call analyze_video to start a new analysis.",
            retryable=False,
            diagnostics={
                "job_id": job_id,
                "next_action": "analyze_video",
                "retry_after_seconds": 0,
            },
        )
        payload = _error_payload(error)
        payload["next_action"] = "analyze_video"
        payload["retry_after_seconds"] = 0
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if job.status == "failed":
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="search_video",
            message=job.error or "Analysis job failed.",
            retryable=False,
        )
        payload = _error_payload(error)
        payload["job_id"] = job_id
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if job.status != "completed":
        remaining = (
            max(1.0, round(job.last_estimated_remaining, 1))
            if job.last_estimated_remaining > 0
            else max(1.0, round(job.estimated_total_seconds, 1))
        )
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="search_video",
            message=(
                f"Job '{job_id}' is still processing ({round(job.progress_percentage)}% complete). "
                "search_video requires a completed job to prevent false-negative searches."
            ),
            retryable=True,
            diagnostics={
                "job_id": job_id,
                "status": job.status,
                "progress_percentage": round(job.progress_percentage, 1),
                "completed_chunks": job.completed_chunks,
                "total_chunks": job.total_chunks,
                "retry_after_seconds": remaining,
                "next_action": "get_job_status",
            },
        )
        payload = _error_payload(error)
        payload["retry_after_seconds"] = remaining
        payload["next_action"] = "get_job_status"
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    matches: list[dict[str, Any]] = []
    for seg in job.full_transcript:
        text = str(seg.get("text") or "").strip()
        if pattern.search(text):
            start_val = seg.get("start_seconds")
            if start_val is None:
                start_val = seg.get("start", 0.0)
            end_val = seg.get("end_seconds")
            if end_val is None:
                end_val = seg.get("end", start_val)
            matches.append(
                {
                    "start_seconds": round(float(start_val), 2),
                    "end_seconds": round(float(end_val), 2),
                    "formatted_time": format_timestamp(float(start_val)),
                    "snippet": text,
                }
            )
            if len(matches) >= max(1, max_matches):
                break

    matches.sort(key=lambda m: (m["start_seconds"], m["end_seconds"]))
    coverage_meta = job.coverage()
    res: dict[str, Any] = {
        "job_id": job_id,
        "query": query,
        "is_regex": is_regex,
        "case_sensitive": case_sensitive,
        "coverage": coverage_meta,
        "matches_count": len(matches),
        "matches": matches,
    }
    if len(matches) == 0:
        if not job.full_transcript:
            res["hint"] = (
                "Job has no speech transcript (visual-only analysis). "
                "Call view_frame(frame_id=...) to inspect frames."
            )
        else:
            if coverage_meta.get("transcript_end_seconds") is not None:
                res["hint"] = (
                    f"No transcript matches found for '{query}' in transcript range "
                    f"({coverage_meta['transcript_start_seconds']}s - {coverage_meta['transcript_end_seconds']}s, "
                    f"analyzed: {coverage_meta['analyzed_start_seconds']}s - {coverage_meta['analyzed_end_seconds']}s)."
                )
            else:
                res["hint"] = (
                    f"No transcript matches found for '{query}' in analyzed range "
                    f"({coverage_meta['analyzed_start_seconds']}s - {coverage_meta['analyzed_end_seconds']}s)."
                )
    return res


@mcp.tool(
    name="get_transcript",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_transcript(
    job_id: Annotated[
        str,
        Field(description="Analysis job ID from analyze_video to read transcript from"),
    ],
    start_seconds: Annotated[
        float,
        Field(ge=0.0, description="Start offset in seconds (default: 0.0)"),
    ] = 0.0,
    end_seconds: Annotated[
        float | None,
        Field(
            ge=0.0,
            description="End offset in seconds. If omitted, defaults to start_seconds + max_duration_seconds",
        ),
    ] = None,
    max_duration_seconds: Annotated[
        float,
        Field(
            gt=0.0,
            le=600.0,
            description="Maximum allowed transcript window duration in seconds (default: 300.0, max: 600.0)",
        ),
    ] = 300.0,
) -> dict[str, Any] | ToolResult:
    """Retrieve timestamped speech transcript segments and joined dialogue text for a video time range.

    Requires a completed analysis job. Bounded to max_duration_seconds (up to 600s/10min)
    to protect agent context windows.

    Filtering is overlap-based: segments that overlap [start_seconds, end_seconds] are
    returned (a segment may start slightly before start_seconds if it ends after start_seconds).
    Returned segments are guaranteed to be sorted monotonically by (start_seconds, end_seconds).
    """
    if start_seconds < 0.0:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="get_transcript",
            message="start_seconds must be >= 0.0",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if max_duration_seconds <= 0.0 or max_duration_seconds > 600.0:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="get_transcript",
            message="max_duration_seconds must be > 0.0 and <= 600.0",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    job = global_job_manager.get_job(job_id)
    if job is None:
        error = AnalysisError(
            code=ErrorCode.ARTIFACT_NOT_FOUND,
            stage="get_transcript",
            message=f"Job '{job_id}' was not found or has expired. Call analyze_video to start a new analysis.",
            retryable=False,
            diagnostics={
                "job_id": job_id,
                "next_action": "analyze_video",
                "retry_after_seconds": 0,
            },
        )
        payload = _error_payload(error)
        payload["next_action"] = "analyze_video"
        payload["retry_after_seconds"] = 0
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if job.status == "failed":
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="get_transcript",
            message=job.error or "Analysis job failed.",
            retryable=False,
        )
        payload = _error_payload(error)
        payload["job_id"] = job_id
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    if job.status != "completed":
        remaining = (
            max(1.0, round(job.last_estimated_remaining, 1))
            if job.last_estimated_remaining > 0
            else max(1.0, round(job.estimated_total_seconds, 1))
        )
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="get_transcript",
            message=(
                f"Job '{job_id}' is still processing ({round(job.progress_percentage)}% complete). "
                "get_transcript requires a completed job to ensure full transcript availability."
            ),
            retryable=True,
            diagnostics={
                "job_id": job_id,
                "status": job.status,
                "progress_percentage": round(job.progress_percentage, 1),
                "completed_chunks": job.completed_chunks,
                "total_chunks": job.total_chunks,
                "retry_after_seconds": remaining,
                "next_action": "get_job_status",
            },
        )
        payload = _error_payload(error)
        payload["retry_after_seconds"] = remaining
        payload["next_action"] = "get_job_status"
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    coverage_meta = job.coverage()
    job_analyzed_end = coverage_meta["analyzed_end_seconds"]

    if end_seconds is not None:
        if end_seconds <= start_seconds:
            error = AnalysisError(
                code=ErrorCode.INVALID_REQUEST,
                stage="get_transcript",
                message=f"end_seconds ({end_seconds}) must be greater than start_seconds ({start_seconds})",
                retryable=False,
            )
            payload = _error_payload(error)
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )

        if (end_seconds - start_seconds) > max_duration_seconds:
            error = AnalysisError(
                code=ErrorCode.INVALID_REQUEST,
                stage="get_transcript",
                message=(
                    f"Requested window ({end_seconds - start_seconds:.1f}s) exceeds "
                    f"max_duration_seconds ({max_duration_seconds:.1f}s). "
                    "Narrow start_seconds and end_seconds to protect context window."
                ),
                retryable=False,
            )
            payload = _error_payload(error)
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )
        resolved_end = end_seconds
    else:
        if job_analyzed_end > start_seconds:
            resolved_end = min(start_seconds + max_duration_seconds, job_analyzed_end)
        else:
            resolved_end = start_seconds + max_duration_seconds

    # Filter transcript segments overlapping [start_seconds, resolved_end]
    matching_segments: list[dict[str, Any]] = []
    for seg in job.full_transcript:
        s_start = seg.get("start_seconds")
        if s_start is None:
            s_start = seg.get("start", 0.0)
        s_end = seg.get("end_seconds")
        if s_end is None:
            s_end = seg.get("end", s_start)
        s_start_f = float(s_start)
        s_end_f = float(s_end)

        if s_end_f > start_seconds and s_start_f < resolved_end:
            text = str(seg.get("text") or "").strip()
            matching_segments.append(
                {
                    "start_seconds": round(s_start_f, 2),
                    "end_seconds": round(s_end_f, 2),
                    "formatted_time": format_timestamp(s_start_f),
                    "text": text,
                }
            )

    matching_segments.sort(key=lambda s: (s["start_seconds"], s["end_seconds"]))
    full_text = " ".join(s["text"] for s in matching_segments if s["text"]).strip()

    res: dict[str, Any] = {
        "job_id": job_id,
        "start_seconds": round(start_seconds, 2),
        "end_seconds": round(resolved_end, 2),
        "window_duration_seconds": round(resolved_end - start_seconds, 2),
        "coverage": coverage_meta,
        "segments_count": len(matching_segments),
        "segments": matching_segments,
        "text": full_text,
    }

    if len(matching_segments) == 0:
        if not job.full_transcript:
            res["hint"] = (
                "Job has no speech transcript (visual-only analysis). "
                "Call view_frame(frame_id=...) to inspect frames."
            )
        else:
            res["hint"] = (
                f"No speech segments found in range {format_timestamp(start_seconds)} - "
                f"{format_timestamp(resolved_end)}."
            )
    return res


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


@mcp.prompt(name="analyze_video_workflow")
def analyze_video_workflow(source: str) -> str:
    """Step-by-step guidance for analyzing a video with Vidscope."""
    return f"""To analyze and understand the video at '{source}':
1. Start with get_video_info(source='{source}') to inspect duration, chapters, and caption availability.
2. If chapter titles answer the question, refer to chapter timestamps directly.
3. Call analyze_video(source='{source}') to begin streaming timeline analysis.
4. If it transitions to background processing, poll get_job_status(job_id='...', since_chunk=...) using next_since_chunk.
5. Use get_transcript(job_id='...', start_seconds=..., end_seconds=...) to read exact speech/dialogue for any timestamp range.
6. Use view_frame(frame_id='...') to inspect high-resolution visual evidence for key timestamps.
"""


@mcp.prompt(name="search_video_workflow")
def search_video_workflow(source: str, query: str) -> str:
    """Step-by-step guidance for locating spoken keywords or topics in a video."""
    return f"""To search for '{query}' in '{source}':
1. Start analysis with analyze_video(source='{source}').
2. If background processing, poll get_job_status(job_id=..., since_chunk=next_since_chunk) incrementally until completed.
3. Once completed, search the transcript with search_video(job_id='...', query='{query}').
4. Read surrounding dialogue context using get_transcript(job_id='...', start_seconds=match['start_seconds'] - 15, end_seconds=match['end_seconds'] + 15).
5. Inspect matching frames using view_frame(source='{source}', timestamp_seconds=match['start_seconds']).
"""


@mcp.resource("vidscope://info", name="server_info")
def _server_info_resource() -> str:
    """Server capabilities, active version, and available resource URI templates."""
    capabilities = _get_host_capabilities()
    return json.dumps(
        {
            "name": "vidscope",
            "version": "3.4.7",
            "description": "Video understanding and visual intelligence MCP server",
            "capabilities": capabilities,
            "tools": [
                "get_video_info",
                "analyze_video",
                "get_job_status",
                "view_frame",
                "search_video",
                "get_transcript",
            ],
            "prompts": [
                "analyze_video_workflow",
                "search_video_workflow",
            ],
            "resource_templates": [
                "vidscope://runs/{run_id}/artifacts/{artifact_id}{?page,offset,limit}",
                "vidscope://runs/{run_id}/manifest",
                "vidscope://runs/{run_id}/plan",
            ],
        },
        indent=2,
    )


def _configure_tool_schemas() -> None:
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        async def _apply() -> None:
            vf = await mcp.get_tool("view_frame")
            if vf and hasattr(vf, "parameters") and isinstance(vf.parameters, dict):
                vf.parameters["oneOf"] = [
                    {"required": ["frame_id"]},
                    {"required": ["source", "timestamp_seconds"]},
                ]

        if loop and loop.is_running():
            loop.create_task(_apply())
        else:
            new_loop = asyncio.new_event_loop()
            try:
                new_loop.run_until_complete(_apply())
            finally:
                new_loop.close()
    except Exception:
        pass


_configure_tool_schemas()


def main() -> None:
    configure_logging()
    logging.getLogger("fastmcp").setLevel(logging.ERROR)
    logging.getLogger("fastmcp.server").setLevel(logging.ERROR)
    mcp.run(show_banner=False, log_level="ERROR")


__all__ = [
    "analyze_video",
    "analyze_video_workflow",
    "get_job_status",
    "get_transcript",
    "get_video_info",
    "main",
    "mcp",
    "read_artifact_resource",
    "search_video",
    "search_video_workflow",
    "view_frame",
]

if __name__ == "__main__":
    main()
