"""Bounded local video analysis through a shared API, CLI, and FastMCP tool."""

from __future__ import annotations

from .contracts import (
    AnalysisError,
    AnalysisMetrics,
    AnalysisResult,
    AnalyzeVideoRequest,
    ArtifactRef,
    ErrorCode,
    TimeRange,
)
from .core import AnalysisContext, VideoAnalyzerFailure, analyze_video

__version__ = "0.1.0"

__all__ = [
    "AnalysisContext",
    "AnalysisError",
    "AnalysisMetrics",
    "AnalysisResult",
    "AnalyzeVideoRequest",
    "ArtifactRef",
    "ErrorCode",
    "TimeRange",
    "VideoAnalyzerFailure",
    "__version__",
    "analyze_video",
]
