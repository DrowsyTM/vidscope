from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from video_analyzer.contracts import AnalyzeVideoRequest, TimeRange


def make_request(
    tmp_path: Path, source: str | None = None, **overrides: Any
) -> AnalyzeVideoRequest:
    values: dict[str, Any] = {
        "source": source or "https://example.com/video.mp4",
        "output_directory": tmp_path,
    }
    values.update(overrides)
    return AnalyzeVideoRequest(**values)


def assert_rejected(factory: Callable[[], object]) -> None:
    """Request validation is expected to surface as a Pydantic/value error."""
    with pytest.raises((ValidationError, ValueError)):
        factory()


def task_names(tasks: object) -> set[str]:
    values = tasks if isinstance(tasks, (set, frozenset, list, tuple)) else []
    return {str(getattr(task, "value", task)) for task in values}


def test_accepts_https_file_uri_and_absolute_local_file(tmp_path: Path) -> None:
    source_file = tmp_path / "fixture.mp4"
    source_file.write_bytes(b"fixture")

    https_request = make_request(tmp_path, "https://example.com/video.mp4")
    file_uri_request = make_request(tmp_path, source_file.as_uri())
    local_request = make_request(tmp_path, str(source_file))

    assert isinstance(https_request, AnalyzeVideoRequest)
    assert isinstance(file_uri_request, AnalyzeVideoRequest)
    assert isinstance(local_request, AnalyzeVideoRequest)
    assert all(
        getattr(request, "source", None)
        for request in (https_request, file_uri_request, local_request)
    )


@pytest.mark.parametrize(
    "source",
    [
        "https://user:password@example.com/video.mp4",
        "https://example.com/video.mp4#fragment",
        "relative/video.mp4",
        "ftp://example.com/video.mp4",
        "http://example.com/video.mp4",
        "data:text/plain,video",
    ],
)
def test_rejects_unsafe_or_non_absolute_sources(tmp_path: Path, source: str) -> None:
    assert_rejected(lambda: make_request(tmp_path, source))


def test_rejects_missing_absolute_local_source(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.mp4"
    assert_rejected(lambda: make_request(tmp_path, str(missing)))


def test_rejects_non_absolute_output_directory(tmp_path: Path) -> None:
    source_file = tmp_path / "fixture.mp4"
    source_file.write_bytes(b"fixture")

    assert_rejected(
        lambda: AnalyzeVideoRequest(
            source=str(source_file),
            output_directory=Path("relative-output"),
        )
    )


@pytest.mark.parametrize(
    "time_range",
    [
        {"start_seconds": -0.1, "end_seconds": 1.0},
        {"start_seconds": 1.0, "end_seconds": 1.0},
        {"start_seconds": 2.0, "end_seconds": 1.0},
        {"start_seconds": 0.0, "end_seconds": 181.0},
        {"start_seconds": float("nan"), "end_seconds": 1.0},
        {"start_seconds": 0.0, "end_seconds": float("inf")},
    ],
)
def test_rejects_invalid_or_overlong_time_windows(
    tmp_path: Path, time_range: dict[str, float]
) -> None:
    assert_rejected(lambda: make_request(tmp_path, time_range=time_range))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_frames", 0),
        ("max_frames", 13),
        ("max_frame_width", 0),
        ("max_frame_width", 1921),
        ("max_download_bytes", 0),
        ("max_download_bytes", 268_435_457),
        ("max_output_bytes", -1),
        ("max_output_bytes", 67_108_865),
        ("timeout_seconds", 0),
        ("timeout_seconds", 601),
    ],
)
def test_rejects_nonpositive_and_over_cap_limits(
    tmp_path: Path, field: str, value: int
) -> None:
    assert_rejected(lambda: make_request(tmp_path, **{field: value}))


def test_rejects_frame_timestamps_outside_window_or_nonfinite(tmp_path: Path) -> None:
    time_range = TimeRange(start_seconds=10.0, end_seconds=20.0)

    assert_rejected(
        lambda: make_request(
            tmp_path,
            time_range=time_range,
            frame_timestamps_seconds=(9.999,),
        )
    )
    assert_rejected(
        lambda: make_request(
            tmp_path,
            time_range=time_range,
            frame_timestamps_seconds=(float("nan"),),
        )
    )


def test_rejects_more_frame_timestamps_than_max_frames(tmp_path: Path) -> None:
    assert_rejected(
        lambda: make_request(
            tmp_path,
            time_range=TimeRange(start_seconds=0.0, end_seconds=10.0),
            max_frames=2,
            frame_timestamps_seconds=(1.0, 2.0, 3.0),
        )
    )


def test_validates_request_id_pattern_and_length(tmp_path: Path) -> None:
    valid = make_request(tmp_path, request_id="A_valid-request_123" + "x" * 45)
    assert valid.request_id is not None
    assert len(valid.request_id) == 64

    for request_id in ("", "contains space", "bad/slash", "bad.dot", "x" * 65):
        assert_rejected(
            lambda request_id=request_id: make_request(tmp_path, request_id=request_id)
        )


def test_rejects_input_symlink_that_escapes_allowed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_file = outside_root / "outside.mp4"
    outside_file.write_bytes(b"outside")
    escaped_link = allowed_root / "escaped.mp4"
    escaped_link.symlink_to(outside_file)
    monkeypatch.setenv("VIDEO_ANALYZER_ALLOWED_INPUT_ROOT", str(allowed_root))

    assert_rejected(lambda: make_request(tmp_path, str(escaped_link)))


def test_default_tasks_are_metadata_and_transcript(tmp_path: Path) -> None:
    request = make_request(tmp_path)

    assert task_names(request.tasks) == {"metadata", "transcript"}
