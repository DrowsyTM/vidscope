"""Execution telemetry, cross-platform memory tracking, and performance metrics."""

from __future__ import annotations

import sys

try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]


def get_peak_rss_mb() -> float:
    """Return the peak resident set size in megabytes across the process and its children.

    Handles platform differences:
    - Linux: ru_maxrss is returned in Kilobytes.
    - macOS (Darwin): ru_maxrss is returned in Bytes.
    - Windows: Returns 0.0 gracefully since the resource module is UNIX-only.
    """
    if resource is None:
        return 0.0

    try:
        ru_self = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        try:
            ru_children = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        except (ValueError, OSError):
            ru_children = 0.0
    except (AttributeError, OSError):
        return 0.0

    # macOS returns bytes; Linux returns KiB
    scale = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    self_mb = ru_self / scale
    children_mb = ru_children / scale
    return round(max(self_mb, children_mb), 2)


def compute_rtf(elapsed_seconds: float, window_duration_seconds: float) -> float:
    """Compute Real-Time Factor (processing time / window audio duration)."""
    duration = max(0.001, float(window_duration_seconds))
    return round(float(elapsed_seconds) / duration, 3)


def compute_ocr_fps(frame_count: int, elapsed_seconds: float) -> float:
    """Compute OCR throughput in frames processed per second."""
    seconds = max(0.001, float(elapsed_seconds))
    return round(float(frame_count) / seconds, 2)


def compute_download_throughput_mbps(
    download_bytes: int, elapsed_seconds: float
) -> float:
    """Compute network download throughput in Megabits per second (Mbps)."""
    seconds = max(0.001, float(elapsed_seconds))
    return round((float(download_bytes) * 8.0) / (seconds * 1_000_000.0), 2)


__all__ = [
    "compute_download_throughput_mbps",
    "compute_ocr_fps",
    "compute_rtf",
    "get_peak_rss_mb",
]
