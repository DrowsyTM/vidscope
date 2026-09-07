from __future__ import annotations

from pathlib import Path
from typing import Any

from video_analyzer.backends.source import (
    CaptionTrack,
    SourceInspection,
    _parse_caption_payload,
)
from video_analyzer.contracts import AnalyzeVideoRequest, TimeRange
from video_analyzer.planner import Capabilities, build_execution_plan
from video_analyzer.telemetry import compute_ocr_fps, compute_rtf, get_peak_rss_mb


def test_benchmark_dag_planning(benchmark: Any, tmp_path: Path) -> None:
    request = AnalyzeVideoRequest(
        source="https://example.com/video.mp4",
        time_range=TimeRange(start_seconds=0, end_seconds=60),
        tasks={"metadata", "transcript", "vad", "frames", "ocr"},
        output_directory=tmp_path,
        max_frames=6,
    )
    inspection = SourceInspection(
        source="https://example.com/video.mp4",
        is_url=True,
        duration_seconds=120.0,
        caption_tracks=[
            CaptionTrack(
                kind="manual", language="en", provider="yt-dlp", source_url=None
            )
        ],
        formats=[{"format_id": "18", "ext": "mp4"}],
    )
    capabilities = Capabilities(
        ffprobe=True,
        ffmpeg=True,
        asr=True,
        vad=True,
        tesseract=True,
        captions=True,
    )

    result = benchmark(build_execution_plan, request, inspection, capabilities)
    assert len(result.stages) > 0


def test_benchmark_request_parsing(benchmark: Any, tmp_path: Path) -> None:
    payload = {
        "source": "https://example.com/video.mp4",
        "time_range": {"start_seconds": 10.0, "end_seconds": 45.0},
        "tasks": ["metadata", "transcript", "frames"],
        "output_directory": str(tmp_path),
        "language": "en",
        "max_frames": 4,
    }

    def parse_and_dump() -> dict[str, Any]:
        req = AnalyzeVideoRequest.model_validate(payload)
        return req.model_dump(mode="json")

    result = benchmark(parse_and_dump)
    assert result["language"] == "en"


def test_benchmark_caption_parsing(benchmark: Any) -> None:
    vtt_payload = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello world, this is a benchmark test.

00:00:04.500 --> 00:00:08.000
Testing subtitle timestamp parsing speed.
"""

    def parse() -> list[dict[str, Any]]:
        return _parse_caption_payload(vtt_payload, extension="vtt")

    segments = benchmark(parse)
    assert len(segments) == 2


def test_benchmark_telemetry_metrics(benchmark: Any) -> None:
    def compute_all() -> tuple[float, float, float]:
        rss = get_peak_rss_mb()
        rtf = compute_rtf(elapsed_seconds=2.0, window_duration_seconds=10.0)
        fps = compute_ocr_fps(frame_count=10, elapsed_seconds=2.0)
        return rss, rtf, fps

    rss, rtf, fps = benchmark(compute_all)
    assert rss > 0
    assert rtf == 0.2
    assert fps == 5.0
