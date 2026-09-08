from __future__ import annotations

import json
import logging
import re
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
    AnalysisTask,
    AnalyzeVideoRequest,
    ErrorCode,
    TimeRange,
)
from .core import VideoAnalyzerFailure
from .core import analyze_video as core_analyze_video
from .jobs import format_timestamp, global_job_manager
from .logging import configure_logging
from .settings import get_settings

logger = logging.getLogger("vidscope.mcp")

mcp = FastMCP("vidscope")


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
                    tasks={
                        AnalysisTask.METADATA,
                        AnalysisTask.TRANSCRIPT,
                        AnalysisTask.FRAMES,
                    },
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
                    "start_seconds": round(start_sec, 2),
                    "end_seconds": round(end_sec, 2),
                    "formatted_range": (
                        f"{format_timestamp(start_sec)} - {format_timestamp(end_sec)}"
                    ),
                    "summary": full_transcript[:4_000],
                    "keyframes": keyframes,
                }
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
    source: str,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    chunk_duration_seconds: float = 180.0,
    sync_timeout_seconds: float = 5.0,
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

        # Wait synchronously up to sync_timeout_seconds for fast completion
        job.completed_event.wait(timeout=max(0.1, sync_timeout_seconds))

        if job.status == "completed":
            return {
                "status": "completed",
                "job_id": job.job_id,
                "source": source,
                "start_seconds": start_seconds,
                "end_seconds": resolved_end,
                "total_chunks": len(chunk_ranges),
                "timeline": job.available_sections,
                "hint": "Call view_frame(frame_id='<frame_id>') to inspect any keyframe image directly in conversation context.",
            }

        if job.status == "failed":
            error = AnalysisError(
                code=ErrorCode.INTERNAL_STAGE_FAILED,
                stage="analyze_video",
                message=job.error or "analysis failed",
                retryable=False,
            )
            payload = _error_payload(error)
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )

        # Longer task: return async handoff envelope with ETA
        return {
            "status": "processing",
            "job_id": job.job_id,
            "source": source,
            "start_seconds": start_seconds,
            "end_seconds": resolved_end,
            "total_chunks": len(chunk_ranges),
            "completed_chunks": job.completed_chunks,
            "estimated_completion_seconds": round(job.estimated_total_seconds, 1),
            "initial_timeline": job.available_sections,
            "message": (
                f"Analysis is processing in the background (estimated ~{int(job.estimated_total_seconds)}s). "
                f"Call get_job_status(job_id='{job.job_id}') to check progress and retrieve timeline sections."
            ),
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
            message=f"Job '{job_id}' was not found or has expired.",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    return job.to_dict(since_chunk=since_chunk)


@mcp.tool(
    name="view_frame",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def view_frame(
    frame_id: str | None = None,
    source: str | None = None,
    timestamp_seconds: float | None = None,
    max_width: int = 1280,
) -> ToolResult:
    """View a video frame as a native MCP Image content block with OCR metadata.

    Can retrieve an extracted keyframe via frame_id, or extract a frame on demand
    given source and timestamp_seconds.
    """
    if frame_id is not None:
        frame_data = global_job_manager.get_frame(frame_id)
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
    name="search_video",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def search_video(
    query: str,
    source: str | None = None,
    job_id: str | None = None,
    is_regex: bool = False,
    case_sensitive: bool = False,
    max_matches: int = 20,
) -> dict[str, Any] | ToolResult:
    """Grep across video subtitles or analyzed transcript with optional regex matching.

    Can search native captions upfront using source, or search the ASR transcript of an
    analyzed video using job_id.
    """
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

    segments_to_search: list[dict[str, Any]] = []

    if job_id is not None:
        job = global_job_manager.get_job(job_id)
        if job is None:
            error = AnalysisError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                stage="search_video",
                message=f"Job '{job_id}' was not found or has expired.",
                retryable=False,
            )
            payload = _error_payload(error)
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )

        if not job.full_transcript:
            return {
                "query": query,
                "job_id": job_id,
                "matches_count": 0,
                "matches": [],
                "message": (
                    f"No transcript segments available yet for job '{job_id}' "
                    f"(status: '{job.status}')."
                ),
            }
        segments_to_search = job.full_transcript

    elif source is not None:
        try:
            inspector = SourceInspector()
            inspection = inspector.inspect({"source": source})
            resolver = CaptionResolver()
            track = resolver.resolve(inspection, {"language": "en"})

            if track is None or not track.segments:
                return {
                    "source": source,
                    "query": query,
                    "matches_count": 0,
                    "matches": [],
                    "message": (
                        "No native captions available for this source. "
                        "Call analyze_video(source) to transcribe the speech first, "
                        "then search with job_id."
                    ),
                }
            segments_to_search = track.segments
        except (VideoAnalyzerFailure, SourceBackendFailure) as exc:
            payload = _error_payload(exc.error)
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )
        except Exception as exc:
            error = AnalysisError(
                code=ErrorCode.INTERNAL_STAGE_FAILED,
                stage="search_video",
                message=str(exc)[:2_048] or "search failed",
                retryable=False,
            )
            payload = _error_payload(error)
            return ToolResult(
                content=payload, structured_content=payload, is_error=True
            )

    else:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="search_video",
            message="Must provide either source (for native captions) or job_id (for analyzed transcript)",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)

    matches: list[dict[str, Any]] = []
    for seg in segments_to_search:
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

    return {
        "query": query,
        "is_regex": is_regex,
        "case_sensitive": case_sensitive,
        "target": "job" if job_id else "source",
        "target_id": job_id or source,
        "matches_count": len(matches),
        "matches": matches,
    }


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
    "main",
    "mcp",
    "read_artifact_resource",
    "search_video",
    "view_frame",
]
