"""Mock backend simulation for Vidscope FastMCP server."""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .jobs import format_timestamp, global_job_manager

logger = logging.getLogger(__name__)

# Valid 1x1 JPEG image bytes
TINY_JPEG_BYTES = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)


def is_mock_mcp_enabled() -> bool:
    """Check whether FastMCP mock mode is enabled via environment variable."""
    return os.environ.get("VIDSCOPE_MOCK_MCP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_mock_async_requested(
    source: str, start_seconds: float, end_seconds: float | None
) -> bool:
    """Determine if mock mode should simulate asynchronous in-flight background processing."""
    env_val = os.environ.get("VIDSCOPE_MOCK_ASYNC", "").strip().lower()
    if env_val in {"1", "true", "yes", "on", "async"}:
        return True
    if env_val in {"0", "false", "no", "off", "sync"}:
        return False
    if env_val in {"random", "rand"}:
        import random

        return random.random() < 0.5
    # Auto-detect: if source mentions 'async' or duration > 300s, simulate async
    if "async" in source.lower():
        return True
    dur = (end_seconds - start_seconds) if end_seconds else 180.0
    return dur > 300.0


def mock_get_video_info(source: str) -> dict[str, Any]:
    """Return synthetic preflight video metadata without downloading or probing."""
    return {
        "source": source,
        "title": "Mock Video Presentation: Deterministic Timeline",
        "duration_seconds": 180.0,
        "formatted_duration": "00:03:00",
        "uploader": "Vidscope Synthetic Test Suite",
        "available_languages": ["en"],
        "has_captions": True,
        "chapters": [
            {"title": "Introduction", "start_seconds": 0.0, "end_seconds": 60.0},
            {
                "title": "Core Presentation",
                "start_seconds": 60.0,
                "end_seconds": 120.0,
            },
            {
                "title": "Conclusion",
                "start_seconds": 120.0,
                "end_seconds": 180.0,
            },
        ],
        "runtime_capabilities": {
            "local_asr_available": True,
            "local_ocr_available": True,
            "bgutil_pot_available": False,
        },
    }


def mock_process_job_chunks(
    job_id: str,
    source: str,
    start_seconds: float,
    end_seconds: float | None,
    chunk_duration_seconds: float,
    sim_async: bool | None = None,
) -> None:
    """Simulate chunk processing without media downloads, FFmpeg, or OCR."""
    job = global_job_manager.get_job(job_id)
    if not job:
        return

    if sim_async is None:
        sim_async = is_mock_async_requested(source, start_seconds, end_seconds)

    # If async requested, sleep briefly to allow sync_timeout_seconds to expire
    if sim_async:
        job.estimated_total_seconds = 4.0
        job.last_estimated_remaining = 4.0
        time.sleep(0.5)

    cache_dir = Path(tempfile.gettempdir()) / "vidscope_mock_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Ensure at least one chunk exists
    if not job.chunks:
        resolved_end = (
            end_seconds if end_seconds is not None else (start_seconds + 60.0)
        )
        global_job_manager.set_job_chunks(
            job_id,
            [(start_seconds, resolved_end)],
            video_duration_seconds=resolved_end,
        )
        job = global_job_manager.get_job(job_id)
        if not job:
            return

    for index, chunk_meta in enumerate(job.chunks):
        chunk_start = chunk_meta["start_seconds"]
        chunk_end = chunk_meta["end_seconds"]

        # 1. Create mock frame
        frame_id = f"frame_mock_{job_id[:6]}_{int(chunk_start + 10.0)}"
        frame_file = cache_dir / f"{frame_id}.jpg"
        if not frame_file.exists():
            frame_file.write_bytes(TINY_JPEG_BYTES)

        ocr_text = f"Synthetic Slide Text for Chunk {index} ({int(chunk_start)}s)"
        global_job_manager.register_frame(
            frame_id,
            {
                "path": frame_file,
                "timestamp_seconds": chunk_start + 10.0,
                "ocr_text": ocr_text,
            },
            job_id=job_id,
        )

        keyframes = [
            {
                "frame_id": frame_id,
                "timestamp_seconds": round(chunk_start + 10.0, 2),
                "formatted_time": format_timestamp(chunk_start + 10.0),
                "ocr_text": ocr_text,
                "ocr_status": "detected",
            }
        ]

        # 2. Create mock transcript
        seg_text = (
            f"Speaker discusses topic section {index} starting at "
            f"{int(chunk_start)} seconds."
        )
        seg = {
            "start_seconds": round(chunk_start, 2),
            "end_seconds": round(chunk_end, 2),
            "formatted_time": format_timestamp(chunk_start),
            "text": seg_text,
        }
        transcript_segments = [seg]

        section = {
            "chunk_index": index,
            "mode": "speech_and_visual",
            "transcript_status": "completed",
            "start_seconds": round(chunk_start, 2),
            "end_seconds": round(chunk_end, 2),
            "formatted_range": (
                f"{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}"
            ),
            "summary": seg_text,
            "keyframes": keyframes,
        }

        global_job_manager.update_job_progress(
            job_id,
            section,
            index,
            transcript_segments=transcript_segments,
        )

    global_job_manager.complete_job(job_id)
