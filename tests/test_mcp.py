from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest


def _construct(model: type[Any], **values: Any) -> Any:
    """Construct a contract model without coupling adapter tests to defaults."""
    fields = getattr(model, "model_fields", {})
    selected = {
        name: value for name, value in values.items() if not fields or name in fields
    }
    return model.model_construct(**selected)


def _request(source: Path, output: Path) -> Any:
    from vidscope.contracts import AnalyzeVideoRequest, TimeRange

    return AnalyzeVideoRequest(
        source=str(source),
        time_range=TimeRange(start_seconds=0, end_seconds=30),
        tasks={"metadata"},
        output_directory=output,
    )


def _success_result() -> Any:
    from vidscope.contracts import AnalysisResult, AnalysisSummary, ArtifactRef

    uri = "vidscope://runs/run-mcp/artifacts/transcript"
    artifact = _construct(
        ArtifactRef,
        artifact_id="transcript",
        uri=uri,
        media_type="application/jsonl",
        byte_size=32,
        sha256="b" * 64,
    )
    summary = _construct(
        AnalysisSummary,
        source="file:///tmp/clip.mp4",
        duration_seconds=30.0,
        task_count=1,
    )
    return _construct(
        AnalysisResult,
        ok=True,
        status="completed",
        summary=summary,
        stages=[],
        warnings=[],
        manifest_uri="vidscope://runs/run-mcp/manifest",
        artifacts=[artifact],
        artifact_refs=[artifact],
    )


def _error(
    code: str = "INTERNAL_STAGE_FAILED",
    stage: str = "transcribe",
    message: str = "deterministic test failure",
) -> Any:
    from vidscope.contracts import AnalysisError

    return _construct(
        AnalysisError,
        ok=False,
        status="failed",
        code=code,
        stage=stage,
        message=message,
        retryable=False,
        diagnostics={"detail": "fixture"},
        artifact_refs=[],
        manifest_uri="vidscope://runs/run-mcp/manifest",
    )


def _failure(error: Any) -> BaseException:
    from vidscope.core import VideoAnalyzerFailure

    return VideoAnalyzerFailure(error)


def _patch_core(monkeypatch: pytest.MonkeyPatch, replacement: Any) -> Any:
    """Patch both the core module and the common FastMCP import alias."""
    import vidscope.core as core_module
    import vidscope.mcp as mcp_module

    original = core_module.analyze_video
    monkeypatch.setattr(core_module, "analyze_video", replacement)
    for name, value in list(vars(mcp_module).items()):
        if name in {"core_analyze_video", "_core_analyze_video"} or value is original:
            monkeypatch.setattr(mcp_module, name, replacement)
    return mcp_module


def _callable_tool(tool: Any) -> Any:
    if callable(tool):
        return tool
    for name in ("fn", "function", "handler"):
        candidate = getattr(tool, name, None)
        if callable(candidate):
            return candidate
    raise AssertionError(f"registered analyze_video tool is not callable: {tool!r}")


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        value = await value
    task_result = getattr(value, "result", None)
    if callable(task_result):
        resolved = task_result()
        if inspect.isawaitable(resolved):
            value = await resolved
    return value


def _await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(_maybe_await(value))
    return value


def _content_payload(value: Any) -> Any:
    """Decode a ToolResult content block without requiring one wire shape."""
    if isinstance(value, Mapping):
        if value.get("type") == "text" and "text" in value:
            return _content_payload(value["text"])
        return dict(value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, (list, tuple)):
        decoded = [_content_payload(item) for item in value]
        return decoded[0] if len(decoded) == 1 else decoded
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _content_payload(model_dump(mode="json"))
    text = getattr(value, "text", None)
    if text is not None:
        return _content_payload(text)
    return value


def _assert_tool_result(
    returned: Any, expected: Mapping[str, Any], *, is_error: bool
) -> None:
    from fastmcp.tools.base import ToolResult

    assert isinstance(returned, ToolResult)
    assert returned.is_error is is_error
    assert returned.structured_content == expected
    assert _content_payload(returned.content) == expected


def _registered_tools(server: Any) -> list[Any] | None:
    manager = getattr(server, "_tool_manager", None)
    for owner in (manager, server):
        if owner is None:
            continue
        for method_name in ("list_tools", "get_tools"):
            method = getattr(owner, method_name, None)
            if not callable(method):
                continue
            try:
                value = _await(method())
            except (AttributeError, TypeError):
                continue
            if isinstance(value, Mapping):
                return list(value.values())
            return list(value)
        for attr_name in ("_tools", "tools"):
            value = getattr(owner, attr_name, None)
            if isinstance(value, Mapping):
                return list(value.values())
    return None


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _snake_annotation_name(name: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index:
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def _tool_annotation(tool: Any, name: str) -> Any:
    annotations = _field(tool, "annotations")
    if annotations is None:
        return None
    if isinstance(annotations, Mapping):
        return annotations.get(name, annotations.get(_snake_annotation_name(name)))
    return getattr(
        annotations, name, getattr(annotations, _snake_annotation_name(name), None)
    )


def test_mcp_registers_only_analyze_video_with_non_mutating_annotations() -> None:
    from vidscope.mcp import mcp

    list_tools = getattr(mcp, "list_tools", None)
    if callable(list_tools):
        tools = _await(list_tools())
        if isinstance(tools, Mapping):
            tools = list(tools.values())
    else:
        tools = _registered_tools(mcp)
    if tools is None:
        pytest.skip("FastMCP version does not expose tool introspection")
    tool_map = {_field(t, "name"): t for t in tools}
    assert "analyze_video" in tool_map
    assert "get_video_info" in tool_map
    assert "search_video" in tool_map
    assert "get_video_transcript" in tool_map
    assert "view_frame" in tool_map
    assert "get_video_timeline" in tool_map
    assert "start_video_analysis" in tool_map
    assert "get_job_status" in tool_map

    assert _tool_annotation(tool_map["analyze_video"], "readOnlyHint") is False
    assert _tool_annotation(tool_map["analyze_video"], "idempotentHint") is False
    assert _tool_annotation(tool_map["get_video_info"], "readOnlyHint") is True
    assert _tool_annotation(tool_map["get_video_info"], "idempotentHint") is True
    assert _tool_annotation(tool_map["search_video"], "readOnlyHint") is True
    assert _tool_annotation(tool_map["get_video_transcript"], "readOnlyHint") is True
    assert _tool_annotation(tool_map["view_frame"], "readOnlyHint") is True
    assert _tool_annotation(tool_map["get_video_timeline"], "readOnlyHint") is True
    assert _tool_annotation(tool_map["start_video_analysis"], "readOnlyHint") is False
    assert _tool_annotation(tool_map["get_job_status"], "readOnlyHint") is True


def test_mcp_success_returns_shared_result_unchanged_and_compact_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vidscope.mcp import analyze_video

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"deterministic fixture")
    output = tmp_path / "results"
    output.mkdir()
    request = _request(source, output)
    expected = _success_result()
    seen: list[Any] = []

    def fake_core(request_arg: Any, *, context: Any = None) -> Any:
        seen.append((request_arg, context))
        return expected

    _patch_core(monkeypatch, fake_core)
    returned = _callable_tool(analyze_video)(request)

    assert returned is expected
    assert len(seen) == 1
    assert seen[0][0] is request
    assert seen[0][1] is None
    assert request.tasks == {"metadata"}
    payload = returned.model_dump(mode="json")
    assert payload["ok"] is True
    assert payload["status"] == "completed"
    assert payload["manifest_uri"].startswith("vidscope://")
    assert all(item["uri"].startswith("vidscope://") for item in payload["artifacts"])


def test_mcp_terminal_failure_is_error_tool_result_with_shared_error_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vidscope.mcp import analyze_video

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"deterministic fixture")
    output = tmp_path / "results"
    output.mkdir()
    request = _request(source, output)
    error = _error()

    def fake_core(request_arg: Any, *, context: Any = None) -> Any:
        raise _failure(error)

    _patch_core(monkeypatch, fake_core)
    returned = _callable_tool(analyze_video)(request)

    expected = error.model_dump(mode="json")
    _assert_tool_result(returned, expected, is_error=True)
    assert returned.structured_content["ok"] is False
    assert returned.structured_content["code"] == "INTERNAL_STAGE_FAILED"


def test_mcp_empty_transcript_is_not_a_successful_tool_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vidscope.mcp import analyze_video

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"deterministic fixture")
    output = tmp_path / "results"
    output.mkdir()
    request = _request(source, output)
    error = _error(
        code="INTERNAL_STAGE_FAILED",
        stage="transcribe",
        message="empty transcript artifact rejected",
    )

    def fake_core(request_arg: Any, *, context: Any = None) -> Any:
        raise _failure(error)

    _patch_core(monkeypatch, fake_core)
    returned = _callable_tool(analyze_video)(request)

    expected = error.model_dump(mode="json")
    _assert_tool_result(returned, expected, is_error=True)
    assert returned.structured_content["ok"] is False
    assert returned.structured_content["status"] != "completed"
    assert returned.structured_content["stage"] == "transcribe"
    assert (
        returned.structured_content["message"] == "empty transcript artifact rejected"
    )
    assert returned.structured_content["code"] == "INTERNAL_STAGE_FAILED"
    assert returned.is_error is True
    expected = error.model_dump(mode="json")
    assert returned.structured_content == expected
    assert _content_payload(returned.content) == expected
    assert returned.structured_content["ok"] is False
    assert returned.structured_content["code"] == "INTERNAL_STAGE_FAILED"


def test_mcp_empty_transcript_returns_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import analyze_video

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"deterministic fixture")
    output = tmp_path / "results"
    output.mkdir()
    request = _request(source, output)
    error = _error(stage="transcribe", message="transcript artifact was empty")

    def fake_core(request_arg: Any, *, context: Any = None) -> Any:
        raise _failure(error)

    _patch_core(monkeypatch, fake_core)
    returned = _callable_tool(analyze_video)(request)

    assert isinstance(returned, ToolResult)
    assert returned.is_error is True
    assert returned.structured_content is not None
    assert returned.structured_content["ok"] is False
    assert returned.structured_content["stage"] == "transcribe"
    assert returned.structured_content["code"] == "INTERNAL_STAGE_FAILED"


def _resource_error_code(value: Any) -> str | None:
    if isinstance(value, BaseException):
        value = getattr(value, "error", value)
    if hasattr(value, "structured_content"):
        value = value.structured_content
    if isinstance(value, Mapping):
        return value.get("code")
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped.get("code")
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return _resource_error_code(decoded)
    return None


def _resource_lines(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [json.loads(line) for line in value.splitlines() if line.strip()]
    if isinstance(value, Mapping):
        for key in ("text", "data", "content"):
            if key in value:
                return _resource_lines(value[key])
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                rows.extend(_resource_lines(item))
            elif isinstance(item, Mapping):
                rows.append(dict(item))
            else:
                text = getattr(item, "text", None)
                if text is not None:
                    rows.extend(_resource_lines(text))
        return rows
    text = getattr(value, "text", None)
    if text is not None:
        return _resource_lines(text)
    raise AssertionError(f"resource did not return JSONL content: {value!r}")


def _write_resource_fixture(root: Path) -> str:
    run_id = "run-resource"
    artifact_id = "transcript"
    uri = f"vidscope://runs/{run_id}/artifacts/{artifact_id}"
    records = "".join(
        json.dumps({"index": index, "text": f"segment-{index}"}) + "\n"
        for index in range(450)
    )
    digest = hashlib.sha256(records.encode()).hexdigest()
    run_dir = root / run_id
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    # Keep the canonical path and compatibility copies so the fixture exercises
    # URI resolution rather than relying on a filesystem path being accepted.
    for path in (
        artifact_dir / "transcript.jsonl",
        artifact_dir / artifact_id,
        run_dir / "transcript.jsonl",
    ):
        path.write_text(records, encoding="utf-8")
    reference = {
        "artifact_id": artifact_id,
        "uri": uri,
        "path": "artifacts/transcript.jsonl",
        "relative_path": "artifacts/transcript.jsonl",
        "media_type": "application/jsonl",
        "byte_size": len(records.encode()),
        "sha256": digest,
    }
    manifest = {
        "run_id": run_id,
        "request_id": run_id,
        "artifacts": [reference],
        "artifact_refs": [reference],
        "stages": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return uri


def test_resource_pages_jsonl_and_rejects_invalid_uri_or_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vidscope.mcp as mcp_module

    monkeypatch.setenv("VIDSCOPE_ALLOWED_OUTPUT_ROOT", str(tmp_path))
    uri = _write_resource_fixture(tmp_path)
    reader = mcp_module.read_artifact_resource

    page_two = reader(uri, page=2, offset=0, limit=200)
    rows = _resource_lines(page_two)
    assert len(rows) == 200
    assert rows[0]["index"] == 200
    assert rows[-1]["index"] == 399

    offset_page = reader(uri, page=None, offset=400, limit=50)
    rows = _resource_lines(offset_page)
    assert len(rows) == 50
    assert rows[0]["index"] == 400

    missing = reader("file:///etc/passwd", page=1, offset=0, limit=10)
    assert _resource_error_code(missing) == "ARTIFACT_NOT_FOUND"

    invalid_range = reader(uri, page=1, offset=-1, limit=10)
    assert _resource_error_code(invalid_range) == "ARTIFACT_RANGE_INVALID"
    too_many = reader(uri, page=1, offset=0, limit=201)
    assert _resource_error_code(too_many) == "ARTIFACT_RANGE_INVALID"


def test_get_video_info_returns_metadata_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vidscope.backends.source import CaptionTrack, SourceInspection
    from vidscope.mcp import get_video_info

    inspection = SourceInspection(
        source="https://www.youtube.com/watch?v=aircAruvnKk",
        is_url=True,
        duration_seconds=120.0,
        caption_tracks=[
            CaptionTrack(kind="manual", language="en", provider="test", source_url=None)
        ],
        metadata={
            "title": "Sample Neural Networks",
            "chapters": [
                {"title": "Intro", "start_time": 0.0, "end_time": 60.0},
                {"title": "Deep Dive", "start_time": 60.0, "end_time": 120.0},
            ],
        },
    )
    monkeypatch.setattr(
        "vidscope.backends.source.SourceInspector.inspect",
        lambda self, req: inspection,
    )

    result = get_video_info("https://www.youtube.com/watch?v=aircAruvnKk")
    assert isinstance(result, dict)
    assert result["title"] == "Sample Neural Networks"
    assert result["duration_seconds"] == 120.0
    assert result["formatted_duration"] == "02:00"
    assert result["has_captions"] is True
    assert result["languages"] == ["en"]
    assert len(result["chapters"]) == 2
    assert result["chapters"][0]["title"] == "Intro"


def test_get_video_info_invalid_url_returns_tool_error() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import get_video_info

    res = get_video_info("ftp://invalid-url.com")
    assert isinstance(res, ToolResult)
    assert res.is_error is True


def test_search_video_finds_matching_snippets(monkeypatch: pytest.MonkeyPatch) -> None:
    from vidscope.backends.source import CaptionTrack, SourceInspection
    from vidscope.mcp import search_video

    track = CaptionTrack(
        kind="manual",
        language="en",
        provider="test",
        source_url=None,
        segments=[
            {"start": 10.0, "end": 15.0, "text": "Welcome to neural networks"},
            {"start": 65.0, "end": 70.0, "text": "This is backpropagation in action"},
            {"start": 90.0, "end": 95.0, "text": "Gradient descent optimization"},
        ],
    )
    monkeypatch.setattr(
        "vidscope.backends.source.SourceInspector.inspect",
        lambda self, req: SourceInspection(
            source="test", is_url=False, duration_seconds=100.0
        ),
    )
    monkeypatch.setattr(
        "vidscope.backends.source.CaptionResolver.resolve",
        lambda self, insp, req: track,
    )

    res = search_video("test.mp4", query="backpropagation")
    assert isinstance(res, dict)
    assert res["matches_count"] == 1
    assert res["matches"][0]["start_seconds"] == 65.0
    assert res["matches"][0]["formatted_time"] == "01:05"
    assert "backpropagation" in res["matches"][0]["snippet"]


def test_search_video_empty_query_returns_error() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import search_video

    res = search_video("test.mp4", query="   ")
    assert isinstance(res, ToolResult)
    assert res.is_error is True


def test_get_video_transcript_bounds_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vidscope.backends.source import CaptionTrack, SourceInspection
    from vidscope.mcp import get_video_transcript

    track = CaptionTrack(
        kind="manual",
        language="en",
        provider="test",
        source_url=None,
        segments=[
            {"start": 0.0, "end": 10.0, "text": "First part"},
            {"start": 20.0, "end": 30.0, "text": "Second part"},
            {"start": 100.0, "end": 110.0, "text": "Far part"},
        ],
    )
    monkeypatch.setattr(
        "vidscope.backends.source.SourceInspector.inspect",
        lambda self, req: SourceInspection(
            source="test", is_url=False, duration_seconds=200.0
        ),
    )
    monkeypatch.setattr(
        "vidscope.backends.source.CaptionResolver.resolve",
        lambda self, insp, req: track,
    )

    res = get_video_transcript("test.mp4", start_seconds=15.0, end_seconds=50.0)
    assert isinstance(res, dict)
    assert res["segment_count"] == 1
    assert res["segments"][0]["text"] == "Second part"
    assert res["full_text"] == "Second part"


def test_get_video_transcript_invalid_range() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import get_video_transcript

    res = get_video_transcript("test.mp4", start_seconds=50.0, end_seconds=20.0)
    assert isinstance(res, ToolResult)
    assert res.is_error is True


def test_view_frame_cached_frame(tmp_path: Path) -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.jobs import global_job_manager
    from vidscope.mcp import view_frame

    frame_file = tmp_path / "test_frame.jpg"
    frame_file.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00"
    )
    global_job_manager.register_frame(
        "frame_test_123",
        {"path": frame_file, "timestamp_seconds": 45.0, "ocr_text": "Sample Diagram"},
    )

    res = view_frame(frame_id="frame_test_123")
    assert isinstance(res, ToolResult)
    assert res.is_error is False
    assert len(res.content) == 2
    payload = json.loads(res.content[0].text)
    assert payload["timestamp_seconds"] == 45.0
    assert payload["ocr_text"] == "Sample Diagram"
    assert getattr(res.content[1], "type", None) == "image"


def test_view_frame_missing_frame() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import view_frame

    res = view_frame(frame_id="nonexistent_frame_999")
    assert isinstance(res, ToolResult)
    assert res.is_error is True


def test_get_video_timeline_rejects_exceeded_window() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import get_video_timeline

    res = get_video_timeline("test.mp4", start_seconds=0.0, end_seconds=200.0)
    assert isinstance(res, ToolResult)
    assert res.is_error is True


def test_start_video_analysis_and_get_job_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vidscope.backends.source import SourceInspection
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import get_job_status, start_video_analysis

    monkeypatch.setattr(
        "vidscope.backends.source.SourceInspector.inspect",
        lambda self, req: SourceInspection(
            source="test", is_url=False, duration_seconds=300.0
        ),
    )
    monkeypatch.setattr(
        "vidscope.mcp._process_job_chunks", lambda job_id, src, chunks: None
    )

    res = start_video_analysis(
        "test.mp4",
        start_seconds=0.0,
        end_seconds=300.0,
        chunk_duration_seconds=150.0,
    )
    assert isinstance(res, dict)
    job_id = res["job_id"]
    assert res["status"] == "processing"
    assert res["total_chunks"] == 2

    status = get_job_status(job_id)
    assert isinstance(status, dict)
    assert status["job_id"] == job_id
    assert status["status"] == "processing"

    global_job_manager.complete_job(job_id)
    status2 = get_job_status(job_id)
    assert isinstance(status2, dict)
    assert status2["status"] == "completed"
    assert status2["progress_percentage"] == 100.0
