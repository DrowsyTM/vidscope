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


@dataclass(slots=True)
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
    available_sections: list[dict[str, Any]] = field(default_factory=list)
    frames: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None
    work_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source": self.source,
            "status": self.status,
            "progress_percentage": round(self.progress_percentage, 1),
            "completed_chunks": self.completed_chunks,
            "total_chunks": self.total_chunks,
            "available_sections": list(self.available_sections),
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
                available_sections=[],
                frames={},
                error=None,
                work_dir=work_dir,
            )
            self._jobs[job_id] = job
            return job

    def get_job(self, job_id: str) -> JobState | None:
        with self._lock:
            self._cleanup_locked()
            return self._jobs.get(job_id)

    def update_job_progress(
        self,
        job_id: str,
        section: dict[str, Any],
        completed_chunks: int,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.available_sections.append(section)
            job.completed_chunks = completed_chunks
            job.progress_percentage = (
                (completed_chunks / job.total_chunks) * 100.0
                if job.total_chunks > 0
                else 100.0
            )
            job.updated_at = time.time()

    def complete_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "completed"
            job.progress_percentage = 100.0
            job.updated_at = time.time()

    def fail_job(self, job_id: str, error_message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.error = error_message
            job.updated_at = time.time()

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
