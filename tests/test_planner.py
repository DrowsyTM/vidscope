from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from vidscope.backends.source import CaptionTrack, SourceInspection
from vidscope.contracts import AnalyzeVideoRequest, TimeRange
from vidscope.planner import Capabilities, build_execution_plan


def as_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="python"))
    return dict(vars(value))


def make_request(
    tmp_path: Path,
    tasks: Iterable[str],
    **overrides: Any,
) -> AnalyzeVideoRequest:
    values: dict[str, Any] = {
        "source": "https://example.com/video.mp4",
        "output_directory": tmp_path,
        "tasks": set(tasks),
    }
    values.update(overrides)
    return AnalyzeVideoRequest(**values)


def make_caption(kind: str) -> CaptionTrack:
    return CaptionTrack(
        kind=kind,
        language="en",
        provider="fake",
        source_url=None,
        segments=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    )


def make_inspection(*tracks: CaptionTrack) -> SourceInspection:
    return SourceInspection(
        source="https://example.com/video.mp4",
        is_url=True,
        duration_seconds=120.0,
        caption_tracks=list(tracks),
        formats=[{"format_id": "fixture", "ext": "mp4"}],
        metadata={"title": "fixture"},
    )


def make_capabilities(**overrides: bool) -> Capabilities:
    values = {
        "ffprobe": True,
        "ffmpeg": True,
        "asr": True,
        "vad": True,
        "tesseract": True,
        "captions": True,
    }
    values.update(overrides)
    return Capabilities(**values)


def stages_by_name(plan: object) -> dict[str, object]:
    stages = plan.stages
    records = stages.values() if isinstance(stages, Mapping) else stages
    return {str(stage.name): stage for stage in records}


def dependencies(stage: object) -> set[str]:
    return set(stage.dependencies)


def route_value(plan: object) -> str:
    routes = as_mapping(plan.routes)
    for key in ("captions", "caption", "transcript"):
        if key not in routes:
            continue
        value = routes[key]
        if isinstance(value, Mapping):
            for nested_key in ("kind", "route", "provider", "name"):
                if nested_key in value:
                    value = value[nested_key]
                    break
        return str(getattr(value, "value", value)).lower()
    raise AssertionError("plan did not expose a caption/transcript route")


def error_code(exc: BaseException) -> str | None:
    candidate: object = exc
    for _ in range(3):
        if isinstance(candidate, Mapping):
            code = candidate.get("code")
            if code is not None:
                return str(getattr(code, "value", code))
        code = getattr(candidate, "code", None)
        if code is not None:
            return str(getattr(code, "value", code))
        next_candidate = None
        for attr in ("error", "analysis_error", "failure"):
            value = getattr(candidate, attr, None)
            if value is not None and value is not candidate:
                next_candidate = value
                break
        if next_candidate is None:
            break
        candidate = next_candidate
    return None


@pytest.mark.parametrize(
    ("tracks", "expected_route", "uses_asr"),
    [
        ([make_caption("manual"), make_caption("automatic")], "manual", False),
        ([make_caption("automatic")], "automatic", False),
        ([], "asr", True),
    ],
)
def test_caption_precedence_is_manual_then_automatic_then_asr(
    tmp_path: Path,
    tracks: list[CaptionTrack],
    expected_route: str,
    uses_asr: bool,
) -> None:
    request = make_request(tmp_path, {"transcript"})
    plan = build_execution_plan(
        request,
        make_inspection(*tracks),
        make_capabilities(),
    )
    stage_names = set(stages_by_name(plan))

    assert expected_route in route_value(plan)
    assert ("transcribe" in stage_names) is uses_asr
    assert ("captions" in stage_names) is not uses_asr


def test_no_captions_schedule_bounded_asr_and_vad_when_asr_is_enabled(
    tmp_path: Path,
) -> None:
    request = make_request(tmp_path, {"transcript"}, asr_enabled=True)
    plan = build_execution_plan(request, make_inspection(), make_capabilities())
    stage_names = set(stages_by_name(plan))

    assert {
        "acquire_media",
        "probe",
        "extract_audio",
        "vad",
        "transcribe",
    } <= stage_names
    assert "captions" not in stage_names
    assert "cloud" not in {name.lower() for name in stage_names}


def test_no_captions_fail_without_enabled_asr_before_media_work(tmp_path: Path) -> None:
    request = make_request(tmp_path, {"transcript"}, asr_enabled=False)

    with pytest.raises(Exception) as exc_info:
        build_execution_plan(request, make_inspection(), make_capabilities())

    assert error_code(exc_info.value) == "CAPTIONS_UNAVAILABLE"


def test_caption_transcript_does_not_add_vad_without_an_asr_fallback(
    tmp_path: Path,
) -> None:
    request = make_request(tmp_path, {"transcript"})
    plan = build_execution_plan(
        request,
        make_inspection(make_caption("manual")),
        make_capabilities(),
    )

    stage_names = set(stages_by_name(plan))
    assert "captions" in stage_names
    assert "vad" not in stage_names
    assert "extract_audio" not in stage_names


def test_youtube_url_always_routes_to_asr_even_with_metadata_captions(
    tmp_path: Path,
) -> None:
    request = make_request(tmp_path, {"transcript"}, asr_enabled=True)
    inspection = SourceInspection(
        source="https://www.youtube.com/watch?v=aircAruvnKk",
        is_url=True,
        duration_seconds=120.0,
        caption_tracks=[make_caption("manual")],
        formats=[{"format_id": "fixture", "ext": "mp4"}],
        metadata={"title": "fixture"},
    )
    plan = build_execution_plan(request, inspection, make_capabilities())
    stage_names = set(stages_by_name(plan))

    assert {
        "acquire_media",
        "probe",
        "extract_audio",
        "vad",
        "transcribe",
    } <= stage_names
    assert "captions" not in stage_names


def test_is_youtube_source_hostname_validation() -> None:
    from vidscope.planner import _is_youtube_source

    # Valid YouTube hosts
    assert _is_youtube_source("https://www.youtube.com/watch?v=aircAruvnKk") is True
    assert _is_youtube_source("https://youtube.com/watch?v=aircAruvnKk") is True
    assert _is_youtube_source("https://m.youtube.com/watch?v=aircAruvnKk") is True
    assert _is_youtube_source("https://youtu.be/aircAruvnKk") is True
    assert (
        _is_youtube_source("https://www.youtube-nocookie.com/embed/aircAruvnKk") is True
    )
    assert _is_youtube_source("youtube.com/watch?v=aircAruvnKk") is True
    assert _is_youtube_source("//youtube.com/watch?v=aircAruvnKk") is True

    # Non-HTTPS schemes
    assert _is_youtube_source("file://youtube.com/clip") is False
    assert _is_youtube_source("ftp://youtube.com/clip") is False
    assert _is_youtube_source("http://youtube.com/watch?v=aircAruvnKk") is False

    # Attack/false positive URLs containing youtube substrings
    assert _is_youtube_source("https://attacker.com/youtube.com") is False
    assert _is_youtube_source("https://attacker.com?v=youtube.com") is False
    assert _is_youtube_source("https://youtube.com.attacker.com") is False
    assert _is_youtube_source("https://notyoutube.com") is False
    assert _is_youtube_source("https://attacker.com/youtu.be") is False
    assert _is_youtube_source("https://attacker.com/youtube-nocookie.com") is False
    assert _is_youtube_source("/tmp/videos/youtube.com.mp4") is False
    assert _is_youtube_source("relative/youtube.com.mp4") is False


def test_non_youtube_url_with_youtube_in_path_keeps_captions(
    tmp_path: Path,
) -> None:
    request = make_request(tmp_path, {"transcript"}, asr_enabled=True)
    inspection = SourceInspection(
        source="https://example.com/videos/youtube.com/clip.mp4",
        is_url=True,
        duration_seconds=120.0,
        caption_tracks=[make_caption("manual")],
        formats=[{"format_id": "fixture", "ext": "mp4"}],
        metadata={"title": "fixture"},
    )
    plan = build_execution_plan(request, inspection, make_capabilities())
    stage_names = set(stages_by_name(plan))

    assert "captions" in stage_names
    assert "transcribe" not in stage_names


def test_ocr_adds_the_frame_dependency_and_no_unrequested_audio_branch(
    tmp_path: Path,
) -> None:
    request = make_request(tmp_path, {"ocr"})
    plan = build_execution_plan(request, make_inspection(), make_capabilities())
    stage_map = stages_by_name(plan)

    assert set(stage_map) == {
        "validate_source",
        "inspect_source",
        "persist_plan",
        "acquire_media",
        "probe",
        "extract_frames",
        "ocr",
    }
    assert dependencies(stage_map["extract_frames"]) == {"probe"}
    assert dependencies(stage_map["ocr"]) == {"extract_frames"}
    assert "extract_audio" not in stage_map
    assert "transcribe" not in stage_map


@pytest.mark.parametrize(
    ("tasks", "capability_overrides", "expected_code"),
    [
        ({"frames"}, {"ffmpeg": False}, "TOOL_UNAVAILABLE"),
        ({"ocr"}, {"tesseract": False}, "OCR_UNAVAILABLE"),
        ({"transcript"}, {"asr": False}, "ASR_MODEL_UNAVAILABLE"),
    ],
)
def test_unavailable_requested_capabilities_fail_before_media(
    tmp_path: Path,
    tasks: set[str],
    capability_overrides: dict[str, bool],
    expected_code: str,
) -> None:
    request = make_request(tmp_path, tasks)
    inspection = make_inspection()

    with pytest.raises(Exception) as exc_info:
        build_execution_plan(
            request,
            inspection,
            make_capabilities(**capability_overrides),
        )

    assert error_code(exc_info.value) == expected_code


def test_all_requested_stages_have_exact_bounded_dag_and_effective_limits(
    tmp_path: Path,
) -> None:
    request = make_request(
        tmp_path,
        {"metadata", "transcript", "vad", "frames", "ocr"},
        max_frames=4,
        max_frame_width=960,
        max_download_bytes=4_000_000,
        max_output_bytes=2_000_000,
        timeout_seconds=90,
    )
    plan = build_execution_plan(request, make_inspection(), make_capabilities())
    stage_map = stages_by_name(plan)

    assert set(stage_map) == {
        "validate_source",
        "inspect_source",
        "persist_plan",
        "metadata",
        "acquire_media",
        "probe",
        "extract_audio",
        "vad",
        "transcribe",
        "extract_frames",
        "ocr",
    }
    expected_dependencies = {
        "validate_source": set(),
        "inspect_source": {"validate_source"},
        "persist_plan": {"inspect_source"},
        "metadata": {"persist_plan"},
        "acquire_media": {"persist_plan"},
        "probe": {"acquire_media"},
        "extract_audio": {"probe"},
        "vad": {"extract_audio"},
        "transcribe": {"extract_audio"},
        "extract_frames": {"probe"},
        "ocr": {"extract_frames"},
    }
    assert {
        name: dependencies(stage) for name, stage in stage_map.items()
    } == expected_dependencies

    limits = as_mapping(plan.effective_limits)
    assert {
        key: limits[key]
        for key in (
            "max_frames",
            "max_frame_width",
            "max_download_bytes",
            "max_output_bytes",
            "timeout_seconds",
        )
    } == {
        "max_frames": 4,
        "max_frame_width": 960,
        "max_download_bytes": 4_000_000,
        "max_output_bytes": 2_000_000,
        "timeout_seconds": 90,
    }


def test_uniform_frame_timestamps_are_capped_by_max_frames(tmp_path: Path) -> None:
    request = make_request(
        tmp_path,
        {"frames"},
        time_range=TimeRange(start_seconds=20.0, end_seconds=80.0),
        max_frames=3,
    )
    plan = build_execution_plan(request, make_inspection(), make_capabilities())
    timestamps = tuple(plan.frame_timestamps_seconds)

    assert len(timestamps) == 3
    assert all(20.0 < timestamp < 80.0 for timestamp in timestamps)
    assert timestamps == tuple(sorted(timestamps))
    assert timestamps[1] - timestamps[0] == pytest.approx(timestamps[2] - timestamps[1])


def test_caption_selection_prefers_requested_language_and_manual_kind(
    tmp_path: Path,
) -> None:
    request = make_request(tmp_path, {"transcript"}, language="en")
    inspection = make_inspection(
        CaptionTrack(
            kind="manual",
            language="fr",
            provider="fake",
            source_url=None,
            segments=[{"start": 0.0, "end": 1.0, "text": "bonjour"}],
        ),
        make_caption("automatic"),
    )

    plan = build_execution_plan(request, inspection, make_capabilities())

    routes = as_mapping(plan.routes)
    transcript_route = as_mapping(routes["transcript"])
    assert transcript_route["kind"] == "automatic"
    assert "captions" in stages_by_name(plan)


def test_caption_only_transcript_excludes_all_media_stages(tmp_path: Path) -> None:
    request = make_request(tmp_path, {"transcript"})
    plan = build_execution_plan(
        request,
        make_inspection(make_caption("manual")),
        make_capabilities(),
    )

    assert not {"acquire_media", "probe", "extract_audio", "transcribe"} & set(
        stages_by_name(plan)
    )


def test_caller_supplied_frame_timestamps_are_preserved(tmp_path: Path) -> None:
    timestamps = (21.0, 42.0, 79.0)
    request = make_request(
        tmp_path,
        {"frames"},
        time_range=TimeRange(start_seconds=20.0, end_seconds=80.0),
        frame_timestamps_seconds=timestamps,
        max_frames=3,
    )

    plan = build_execution_plan(request, make_inspection(), make_capabilities())

    assert tuple(plan.frame_timestamps_seconds) == timestamps


def test_plan_stage_order_is_deterministic_for_task_set_order(tmp_path: Path) -> None:
    first = build_execution_plan(
        make_request(tmp_path, {"ocr", "metadata", "frames", "vad", "transcript"}),
        make_inspection(),
        make_capabilities(),
    )
    second = build_execution_plan(
        make_request(tmp_path, {"transcript", "vad", "frames", "metadata", "ocr"}),
        make_inspection(),
        make_capabilities(),
    )

    def order(plan: object) -> list[str]:
        stages = plan.stages
        values = stages.values() if isinstance(stages, Mapping) else stages
        return [str(stage.name) for stage in values]

    assert order(first) == order(second)


def test_manual_locale_match_beats_exact_automatic_match(tmp_path: Path) -> None:
    request = make_request(tmp_path, {"transcript"}, language="en")
    inspection = make_inspection(
        CaptionTrack(
            kind="manual",
            language="en-US",
            provider="fake",
            source_url=None,
            segments=[{"start": 0.0, "end": 1.0, "text": "manual"}],
        ),
        make_caption("automatic"),
    )

    plan = build_execution_plan(request, inspection, make_capabilities())

    route = as_mapping(as_mapping(plan.routes)["transcript"])
    assert route["kind"] == "manual"


@pytest.mark.parametrize(
    ("time_range", "timestamps"),
    [
        ({"start_seconds": -5.0, "end_seconds": 5.0}, None),
        ({"start_seconds": 0.0, "end_seconds": 181.0}, None),
        ({"start_seconds": 0.0, "end_seconds": 10.0}, (float("nan"),)),
        ({"start_seconds": 0.0, "end_seconds": 10.0}, (11.0,)),
        ({"start_seconds": 0.0, "end_seconds": 10.0}, (1.0, 2.0, 3.0)),
    ],
)
def test_planner_rejects_invalid_duck_typed_bounds_and_timestamps(
    tmp_path: Path,
    time_range: dict[str, float],
    timestamps: tuple[float, ...] | None,
) -> None:
    request = make_request(tmp_path, {"frames"})
    request = request.model_copy(
        update={
            "time_range": time_range,
            "frame_timestamps_seconds": timestamps,
            "max_frames": 2 if timestamps and len(timestamps) > 2 else 6,
        }
    )
    object.__setattr__(request, "time_range", time_range)
    object.__setattr__(request, "frame_timestamps_seconds", timestamps)

    with pytest.raises(Exception) as exc_info:
        build_execution_plan(request, make_inspection(), make_capabilities())

    assert error_code(exc_info.value) == "INVALID_REQUEST"
