"""Thread-safe in-memory job state registry and frame cache for vidscope.

Enables asynchronous multi-chunk video processing and progressive result streaming,
avoiding transport timeouts on long-running operations.
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("vidscope.jobs")

DEFAULT_JOB_TTL_SECONDS: float = 7200.0  # 2 hours


def format_timestamp(seconds: float | None) -> str:
    """Format seconds into HH:MM:SS or MM:SS string."""
    if seconds is None or not math.isfinite(seconds):
        return "00:00"
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class JobState:
    """Thread-safe state for an ongoing or completed video analysis job."""

    job_id: str
    source: str
    status: str  # "processing", "completed", "failed"
    created_at: float
    updated_at: float
    progress_percentage: float
    completed_chunks: int
    total_chunks: int
    chunks: list[dict[str, float]]
    estimated_total_seconds: float = 0.0
    available_sections: list[dict[str, Any]] = field(default_factory=list)
    full_transcript: list[dict[str, Any]] = field(default_factory=list)
    frames: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None
    work_dir: Path | None = None
    completed_event: threading.Event = field(default_factory=threading.Event)
    job_key: str | None = None
    last_completed_at: float = 0.0
    last_estimated_remaining: float = 0.0
    chunk_durations: list[float] = field(default_factory=list)

    def to_dict(self, since_chunk: int = 0) -> dict[str, Any]:
        now = time.time()
        elapsed = now - self.created_at
        if self.status == "processing":
            if self.total_chunks > self.completed_chunks:
                remaining_chunks = self.total_chunks - self.completed_chunks
                if self.chunk_durations:
                    avg_chunk_dur = sum(self.chunk_durations) / len(
                        self.chunk_durations
                    )
                    in_flight_spent = max(
                        0.0, now - (self.last_completed_at or self.created_at)
                    )
                    current_chunk_remaining = max(0.5, avg_chunk_dur - in_flight_spent)
                    raw_remaining = (
                        current_chunk_remaining
                        + max(0, remaining_chunks - 1) * avg_chunk_dur
                    )
                else:
                    raw_remaining = max(1.0, self.estimated_total_seconds - elapsed)

                if self.last_estimated_remaining > 0.0:
                    remaining = min(
                        self.last_estimated_remaining * 1.15,
                        0.7 * self.last_estimated_remaining + 0.3 * raw_remaining,
                    )
                else:
                    remaining = raw_remaining
                self.last_estimated_remaining = remaining
            else:
                remaining = 1.0
        else:
            remaining = 0.0

        returned_sections = self.available_sections[since_chunk:]
        total_available = len(self.available_sections)
        has_more = (self.status == "processing") or (
            since_chunk + len(returned_sections) < self.total_chunks
        )

        if self.status == "processing":
            if len(returned_sections) == 0:
                message = (
                    f"Job is still processing ({round(self.progress_percentage)}% complete, "
                    f"{self.completed_chunks}/{self.total_chunks} chunks). "
                    f"No new timeline sections since chunk {since_chunk}. "
                    f"Continue polling with since_chunk={total_available}."
                )
            else:
                message = (
                    f"Job is processing ({round(self.progress_percentage)}% complete, "
                    f"{self.completed_chunks}/{self.total_chunks} chunks). "
                    f"Returned {len(returned_sections)} new timeline section(s). "
                    f"Next poll should use since_chunk={total_available}."
                )
        elif self.status == "completed":
            message = (
                f"Analysis completed successfully ({self.completed_chunks}/{self.total_chunks} chunks). "
                f"Returned {len(returned_sections)} timeline section(s)."
            )
        else:
            message = f"Job status: '{self.status}'."

        return {
            "job_id": self.job_id,
            "source": self.source,
            "status": self.status,
            "progress_percentage": round(self.progress_percentage, 1),
            "completed_chunks": self.completed_chunks,
            "total_chunks": self.total_chunks,
            "estimated_remaining_seconds": round(remaining, 1),
            "timeline": list(returned_sections),
            "timeline_chunks_returned": len(returned_sections),
            "total_timeline_chunks": total_available,
            "next_since_chunk": total_available,
            "has_more": has_more,
            "message": message,
            "error": self.error,
        }


class JobManager:
    """In-memory thread-safe registry for background analysis jobs and frame cache."""

    def __init__(self, ttl_seconds: float = DEFAULT_JOB_TTL_SECONDS) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        self._frames: dict[str, dict[str, Any]] = {}
        self._frame_expiry: dict[str, float] = {}
        self.ttl_seconds = ttl_seconds

    def create_job(
        self,
        source: str,
        chunk_ranges: list[tuple[float, float]],
        estimated_seconds_per_chunk: float = 12.0,
        job_key: str | None = None,
    ) -> JobState:
        with self._lock:
            self._cleanup_locked()
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            work_dir = Path(tempfile.gettempdir()) / "vidscope_jobs" / job_id
            work_dir.mkdir(parents=True, exist_ok=True)
            now = time.time()
            chunks = [
                {"start_seconds": round(s, 2), "end_seconds": round(e, 2)}
                for s, e in chunk_ranges
            ]
            estimated_total = max(1.0, len(chunk_ranges) * estimated_seconds_per_chunk)
            job = JobState(
                job_id=job_id,
                source=source,
                status="processing",
                created_at=now,
                updated_at=now,
                progress_percentage=0.0,
                completed_chunks=0,
                total_chunks=len(chunk_ranges),
                chunks=chunks,
                estimated_total_seconds=estimated_total,
                available_sections=[],
                full_transcript=[],
                frames={},
                error=None,
                work_dir=work_dir,
                completed_event=threading.Event(),
                job_key=job_key,
            )
            self._jobs[job_id] = job
            return job

    def find_job_by_key(self, job_key: str) -> JobState | None:
        with self._lock:
            self._cleanup_locked()
            for job in self._jobs.values():
                if job.job_key == job_key and job.status != "failed":
                    return job
            return None

    def set_job_chunks(
        self,
        job_id: str,
        chunk_ranges: list[tuple[float, float]],
        estimated_seconds_per_chunk: float = 12.0,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.total_chunks = len(chunk_ranges)
            job.chunks = [
                {"start_seconds": round(s, 2), "end_seconds": round(e, 2)}
                for s, e in chunk_ranges
            ]
            job.estimated_total_seconds = max(
                1.0, len(chunk_ranges) * estimated_seconds_per_chunk
            )
            job.updated_at = time.time()

    def get_job(self, job_id: str) -> JobState | None:
        with self._lock:
            self._cleanup_locked()
            return self._jobs.get(job_id)

    def update_job_progress(
        self,
        job_id: str,
        section: dict[str, Any],
        completed_chunks: int,
        transcript_segments: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.available_sections.append(section)
            if transcript_segments:
                job.full_transcript.extend(transcript_segments)
            job.completed_chunks = completed_chunks
            job.progress_percentage = (
                (completed_chunks / job.total_chunks) * 100.0
                if job.total_chunks > 0
                else 100.0
            )
            now = time.time()
            prev_time = (
                job.last_completed_at if job.last_completed_at > 0.0 else job.created_at
            )
            job.chunk_durations.append(max(0.5, now - prev_time))
            job.last_completed_at = now
            job.updated_at = now

    def complete_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "completed"
            job.progress_percentage = 100.0
            job.updated_at = time.time()
            job.completed_event.set()

    def fail_job(self, job_id: str, error_message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.error = error_message
            job.updated_at = time.time()
            job.completed_event.set()

    def register_frame(
        self,
        frame_id: str,
        frame_data: dict[str, Any],
        job_id: str | None = None,
    ) -> None:
        with self._lock:
            self._frames[frame_id] = frame_data
            self._frame_expiry[frame_id] = time.time() + self.ttl_seconds
            if job_id and job_id in self._jobs:
                self._jobs[job_id].frames[frame_id] = frame_data

    def get_frame(
        self,
        frame_id: str,
        job_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            self._cleanup_locked()
            if job_id and job_id in self._jobs:
                frame = self._jobs[job_id].frames.get(frame_id)
                if frame is not None:
                    return frame
            return self._frames.get(frame_id)

    def _cleanup_locked(self) -> None:
        now = time.time()
        expired_jobs = [
            jid
            for jid, job in self._jobs.items()
            if now - job.updated_at > self.ttl_seconds
        ]
        for jid in expired_jobs:
            job = self._jobs.pop(jid, None)
            if job and job.work_dir and job.work_dir.exists():
                try:
                    shutil.rmtree(job.work_dir, ignore_errors=True)
                except Exception as exc:
                    logger.warning("Failed to remove work dir for %s: %s", jid, exc)

        expired_frames = [
            fid for fid, expiry in self._frame_expiry.items() if now > expiry
        ]
        for fid in expired_frames:
            self._frames.pop(fid, None)
            self._frame_expiry.pop(fid, None)


global_job_manager = JobManager()

__all__ = [
    "DEFAULT_JOB_TTL_SECONDS",
    "JobManager",
    "JobState",
    "format_timestamp",
    "global_job_manager",
]
