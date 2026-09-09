from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest


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


def test_mcp_registers_expected_tools_with_annotations() -> None:
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
    assert len(tool_map) == 5
    assert "analyze_video" in tool_map
    assert "get_video_info" in tool_map
    assert "search_video" in tool_map
    assert "view_frame" in tool_map
    assert "get_job_status" in tool_map

    assert _tool_annotation(tool_map["get_video_info"], "readOnlyHint") is True
    assert _tool_annotation(tool_map["get_video_info"], "idempotentHint") is True
    assert _tool_annotation(tool_map["analyze_video"], "readOnlyHint") is False
    assert _tool_annotation(tool_map["analyze_video"], "idempotentHint") is False
    assert _tool_annotation(tool_map["get_job_status"], "readOnlyHint") is True
    assert _tool_annotation(tool_map["get_job_status"], "idempotentHint") is True
    assert _tool_annotation(tool_map["view_frame"], "readOnlyHint") is True
    assert _tool_annotation(tool_map["view_frame"], "idempotentHint") is True
    assert _tool_annotation(tool_map["search_video"], "readOnlyHint") is True
    assert _tool_annotation(tool_map["search_video"], "idempotentHint") is True


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
            {
                "start_seconds": 10.0,
                "end_seconds": 15.0,
                "text": "Welcome to neural networks",
            },
            {
                "start_seconds": 65.0,
                "end_seconds": 70.0,
                "text": "This is backpropagation in action",
            },
            {
                "start_seconds": 90.0,
                "end_seconds": 95.0,
                "text": "Gradient descent optimization",
            },
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

    res = search_video(query="backpropagation", source="test.mp4")
    assert isinstance(res, dict)
    assert res["matches_count"] == 1
    assert res["matches"][0]["start_seconds"] == 65.0
    assert res["matches"][0]["formatted_time"] == "01:05"
    assert "backpropagation" in res["matches"][0]["snippet"]


def test_search_video_regex_and_case_sensitivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vidscope.backends.source import CaptionTrack, SourceInspection
    from vidscope.mcp import search_video

    track = CaptionTrack(
        kind="manual",
        language="en",
        provider="test",
        source_url=None,
        segments=[
            {
                "start_seconds": 10.0,
                "end_seconds": 15.0,
                "text": "Error 404: Not Found",
            },
            {
                "start_seconds": 65.0,
                "end_seconds": 70.0,
                "text": "error 500: server failure",
            },
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

    # Regex test
    res = search_video(query=r"Error \d+", source="test.mp4", is_regex=True)
    assert isinstance(res, dict)
    assert res["matches_count"] == 2

    # Case-sensitive test
    res_case = search_video(query="Error", source="test.mp4", case_sensitive=True)
    assert isinstance(res_case, dict)
    assert res_case["matches_count"] == 1
    assert res_case["matches"][0]["start_seconds"] == 10.0


def test_search_video_empty_query_returns_error() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import search_video

    res = search_video(query="   ", source="test.mp4")
    assert isinstance(res, ToolResult)
    assert res.is_error is True


def test_search_video_invalid_regex_returns_error() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import search_video

    res = search_video(query="[unclosed-regex", source="test.mp4", is_regex=True)
    assert isinstance(res, ToolResult)
    assert res.is_error is True


def test_search_video_job_id_transcript() -> None:
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import search_video

    job = global_job_manager.create_job("test_src", [(0.0, 100.0)])
    global_job_manager.update_job_progress(
        job.job_id,
        section={"chunk_index": 1, "summary": "Intro"},
        completed_chunks=1,
        transcript_segments=[
            {
                "start_seconds": 12.5,
                "end_seconds": 18.0,
                "text": "Transformer architecture",
            },
            {"start_seconds": 25.0, "end_seconds": 30.0, "text": "Attention mechanism"},
        ],
    )
    res = search_video(query="Attention", job_id=job.job_id)
    assert isinstance(res, dict)
    assert res["matches_count"] == 1
    assert res["matches"][0]["start_seconds"] == 25.0
    assert "Attention mechanism" in res["matches"][0]["snippet"]


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
    from mcp.types import TextContent

    assert isinstance(res.content[0], TextContent)
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


def test_analyze_video_sync_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from vidscope.backends.source import SourceInspection
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import analyze_video

    monkeypatch.setattr(
        "vidscope.backends.source.SourceInspector.inspect",
        lambda self, req: SourceInspection(
            source="test", is_url=False, duration_seconds=60.0
        ),
    )

    def mock_process(job_id: str, src: str, *args: Any, **kwargs: Any) -> None:
        global_job_manager.set_job_chunks(job_id, [(0.0, 60.0)])
        global_job_manager.update_job_progress(
            job_id,
            section={
                "chunk_index": 1,
                "start_seconds": 0.0,
                "end_seconds": 60.0,
                "summary": "Full overview",
            },
            completed_chunks=1,
        )
        global_job_manager.complete_job(job_id)

    monkeypatch.setattr("vidscope.mcp._process_job_chunks", mock_process)

    res = analyze_video("test.mp4", sync_timeout_seconds=2.0)
    assert isinstance(res, dict)
    assert res["status"] == "completed"
    assert len(res["timeline"]) == 1
    assert res["timeline"][0]["summary"] == "Full overview"
    assert "hint" in res


def test_analyze_video_async_fallback_and_get_job_status_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from vidscope.backends.source import SourceInspection
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import analyze_video, get_job_status

    monkeypatch.setattr(
        "vidscope.backends.source.SourceInspector.inspect",
        lambda self, req: SourceInspection(
            source="test", is_url=False, duration_seconds=360.0
        ),
    )

    def slow_process(job_id: str, src: str, *args: Any, **kwargs: Any) -> None:
        global_job_manager.set_job_chunks(job_id, [(0.0, 180.0), (180.0, 360.0)])
        time.sleep(0.3)
        global_job_manager.update_job_progress(
            job_id,
            section={"chunk_index": 1, "summary": "Section 1"},
            completed_chunks=1,
        )
        time.sleep(0.3)
        global_job_manager.update_job_progress(
            job_id,
            section={"chunk_index": 2, "summary": "Section 2"},
            completed_chunks=2,
        )
        global_job_manager.complete_job(job_id)

    monkeypatch.setattr("vidscope.mcp._process_job_chunks", slow_process)

    # Sync timeout very small (0.05s) to trigger immediate async return
    res = analyze_video(
        "test.mp4",
        start_seconds=0.0,
        end_seconds=360.0,
        chunk_duration_seconds=180.0,
        sync_timeout_seconds=0.05,
    )
    assert isinstance(res, dict)
    assert res["status"] == "processing"
    job_id = res["job_id"]
    assert res["total_chunks"] == 2
    assert "estimated_completion_seconds" in res

    # Wait for completion
    job = global_job_manager.get_job(job_id)
    assert job is not None
    job.completed_event.wait(timeout=2.0)

    # Test incremental streaming with since_chunk
    status_all = get_job_status(job_id, since_chunk=0)
    assert isinstance(status_all, dict)
    assert status_all["status"] == "completed"
    assert status_all["timeline_chunks_returned"] == 2
    assert len(status_all["timeline"]) == 2

    # Polling newly completed chunks with cursor
    status_partial = get_job_status(job_id, since_chunk=1)
    assert isinstance(status_partial, dict)
    assert status_partial["timeline_chunks_returned"] == 1
    assert status_partial["timeline"][0]["summary"] == "Section 2"


def test_analyze_video_invalid_parameters() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import analyze_video

    res_neg = analyze_video("test.mp4", start_seconds=-10.0)
    assert isinstance(res_neg, ToolResult)
    assert res_neg.is_error is True

    res_zero_chunk = analyze_video("test.mp4", chunk_duration_seconds=0.0)
    assert isinstance(res_zero_chunk, ToolResult)
    assert res_zero_chunk.is_error is True

    res_large_chunk = analyze_video("test.mp4", chunk_duration_seconds=300.0)
    assert isinstance(res_large_chunk, ToolResult)
    assert res_large_chunk.is_error is True


def test_get_job_status_failed_returns_tool_result() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.jobs import global_job_manager
    from vidscope.mcp import get_job_status

    job = global_job_manager.create_job("test_fail.mp4", [(0.0, 60.0)])
    global_job_manager.fail_job(job.job_id, "Video decoding error")

    res = get_job_status(job.job_id)
    assert isinstance(res, ToolResult)
    assert res.is_error is True


def test_search_video_language_support(monkeypatch: pytest.MonkeyPatch) -> None:
    from vidscope.backends.source import CaptionTrack, SourceInspection
    from vidscope.mcp import search_video

    track_es = CaptionTrack(
        kind="manual",
        language="es",
        provider="test",
        source_url=None,
        segments=[
            {"start_seconds": 5.0, "end_seconds": 10.0, "text": "Hola mundo"},
        ],
    )
    monkeypatch.setattr(
        "vidscope.backends.source.SourceInspector.inspect",
        lambda self, req: SourceInspection(
            source="test",
            is_url=False,
            duration_seconds=100.0,
            caption_tracks=[track_es],
        ),
    )
    monkeypatch.setattr(
        "vidscope.backends.source.CaptionResolver.resolve",
        lambda self, insp, req: track_es if req.get("language") == "es" else None,
    )

    res_es = search_video(query="Hola", source="test.mp4", language="es")
    assert isinstance(res_es, dict)
    assert res_es["matches_count"] == 1
    assert res_es["matches"][0]["snippet"] == "Hola mundo"


def test_analyze_video_idempotency_returns_existing_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vidscope.backends.source import SourceInspection
    from vidscope.mcp import analyze_video

    monkeypatch.setattr(
        "vidscope.backends.source.SourceInspector.inspect",
        lambda self, req: SourceInspection(
            source="test", is_url=False, duration_seconds=60.0
        ),
    )
    monkeypatch.setattr(
        "vidscope.mcp._process_job_chunks", lambda *args, **kwargs: None
    )

    res1 = analyze_video(
        "idempotent_test.mp4",
        start_seconds=0.0,
        end_seconds=60.0,
        sync_timeout_seconds=0.05,
    )
    assert isinstance(res1, dict)
    job_id_1 = res1["job_id"]

    res2 = analyze_video(
        "idempotent_test.mp4",
        start_seconds=0.0,
        end_seconds=60.0,
        sync_timeout_seconds=0.05,
    )
    assert isinstance(res2, dict)
    job_id_2 = res2["job_id"]

    assert job_id_1 == job_id_2


def test_view_frame_mutual_exclusivity_and_validation() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.contracts import ErrorCode
    from vidscope.mcp import view_frame

    # Both frame_id and source/timestamp
    res1 = view_frame(frame_id="frame_123", source="test.mp4", timestamp_seconds=10.0)
    assert isinstance(res1, ToolResult)
    assert res1.is_error is True
    assert res1.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert "Cannot provide both" in res1.structured_content["message"]

    # Only source without timestamp
    res2 = view_frame(source="test.mp4")
    assert isinstance(res2, ToolResult)
    assert res2.is_error is True
    assert res2.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert "Both 'source' and 'timestamp_seconds'" in res2.structured_content["message"]

    # Only timestamp without source
    res3 = view_frame(timestamp_seconds=15.0)
    assert isinstance(res3, ToolResult)
    assert res3.is_error is True
    assert res3.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert "Both 'source' and 'timestamp_seconds'" in res3.structured_content["message"]

    # Empty invocation
    res4 = view_frame()
    assert isinstance(res4, ToolResult)
    assert res4.is_error is True
    assert res4.structured_content["code"] == ErrorCode.INVALID_REQUEST


def test_search_video_mutual_exclusivity_and_validation() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.contracts import ErrorCode
    from vidscope.mcp import search_video

    # Both source and job_id
    res1 = search_video(query="hello", source="test.mp4", job_id="job_123")
    assert isinstance(res1, ToolResult)
    assert res1.is_error is True
    assert res1.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert "Cannot provide both" in res1.structured_content["message"]

    # Neither source nor job_id
    res2 = search_video(query="hello")
    assert isinstance(res2, ToolResult)
    assert res2.is_error is True
    assert res2.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert "Must provide either" in res2.structured_content["message"]


def test_server_info_static_resource() -> None:
    from vidscope.mcp import _server_info_resource, mcp

    payload = json.loads(_server_info_resource())
    assert payload["name"] == "vidscope"
    assert "3.4.7" in payload["version"]
    assert "analyze_video" in payload["tools"]
    assert len(payload["resource_templates"]) >= 3

    # FastMCP list_resources returns server_info
    list_resources = getattr(mcp, "list_resources", None)
    if callable(list_resources):
        resources = _await(list_resources())
        if isinstance(resources, Mapping):
            resources = list(resources.values())
        resource_uris = [str(_field(r, "uri")) for r in resources]
        assert "vidscope://info" in resource_uris


def test_analyze_video_schema_constraints() -> None:
    from vidscope.mcp import mcp

    list_tools = getattr(mcp, "list_tools", None)
    if callable(list_tools):
        tools = _await(list_tools())
        if isinstance(tools, Mapping):
            tools = list(tools.values())
        tool_map = {_field(t, "name"): t for t in tools}
        analyze_tool = tool_map["analyze_video"]
        params = getattr(analyze_tool, "parameters", {})
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        chunk_prop = props.get("chunk_duration_seconds", {})
        assert chunk_prop.get("maximum") == 180.0
        assert chunk_prop.get("exclusiveMinimum") == 0.0


def test_search_video_language_normalization_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.backends.source import CaptionTrack, SourceInspection
    from vidscope.contracts import ErrorCode
    from vidscope.mcp import search_video

    mock_track = CaptionTrack(
        kind="manual",
        language="en",
        provider="test",
        source_url="http://test",
        segments=[{"text": "learning deep learning", "start": 1.0, "end": 2.0}],
    )
    mock_inspection = SourceInspection(
        source="https://example.com/video.mp4",
        is_url=True,
        duration_seconds=60.0,
        caption_tracks=[mock_track],
        streams=[],
        metadata={},
    )

    monkeypatch.setattr(
        "vidscope.mcp.SourceInspector.inspect",
        lambda self, req: mock_inspection,
    )
    monkeypatch.setattr(
        "vidscope.mcp.CaptionResolver.resolve",
        lambda self, insp, req: mock_track if req.get("language") == "en" else None,
    )

    # 1. Full language name "english" normalizes to "en" and matches
    res1 = search_video(
        query="learning",
        source="https://example.com/video.mp4",
        language="english",
    )
    assert not isinstance(res1, ToolResult)
    assert res1["language"] == "en"
    assert res1["matches_count"] == 1

    # 2. Invalid language string rejected as INVALID_REQUEST
    res2 = search_video(
        query="learning",
        source="https://example.com/video.mp4",
        language="invalid language with $$$",
    )
    assert isinstance(res2, ToolResult)
    assert res2.is_error is True
    assert res2.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert "Unsupported or invalid language" in res2.structured_content["message"]


def test_search_video_remote_caption_failure_informative_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.backends.source import CaptionTrack, SourceInspection
    from vidscope.mcp import search_video

    mock_meta_track = CaptionTrack(
        kind="manual",
        language="en",
        provider="yt-dlp",
        source_url="https://youtube.com/timedtext?v=123",
        segments=[],
    )
    mock_inspection = SourceInspection(
        source="https://www.youtube.com/watch?v=123",
        is_url=True,
        duration_seconds=60.0,
        caption_tracks=[mock_meta_track],
        streams=[],
        metadata={},
    )

    monkeypatch.setattr(
        "vidscope.mcp.SourceInspector.inspect",
        lambda self, req: mock_inspection,
    )
    monkeypatch.setattr(
        "vidscope.mcp.CaptionResolver.resolve",
        lambda self, insp, req: None,
    )

    res = search_video(
        query="test",
        source="https://www.youtube.com/watch?v=123",
        language="en",
    )
    assert not isinstance(res, ToolResult)
    assert res["matches_count"] == 0
    # Must explain that the track exists in metadata but could not be downloaded
    assert "is listed in video metadata" in res["message"]
    assert "remote provider" in res["message"]
    assert "analyze_video" in res["message"]


def test_job_status_continuation_contract_and_cursor() -> None:
    from vidscope.jobs import JobState

    job = JobState(
        job_id="job_test_cursor",
        source="video.mp4",
        status="processing",
        created_at=100.0,
        updated_at=105.0,
        progress_percentage=50.0,
        completed_chunks=1,
        total_chunks=2,
        chunks=[
            {"start_seconds": 0.0, "end_seconds": 30.0},
            {"start_seconds": 30.0, "end_seconds": 60.0},
        ],
        available_sections=[{"chunk_index": 1, "summary": "part 1"}],
    )

    # Initial poll at chunk 0
    poll0 = job.to_dict(since_chunk=0)
    assert poll0["next_since_chunk"] == 1
    assert poll0["has_more"] is True
    assert poll0["timeline_chunks_returned"] == 1
    assert "Job is processing" in poll0["message"]

    # Poll with since_chunk=1 (empty incremental response)
    poll1 = job.to_dict(since_chunk=1)
    assert poll1["next_since_chunk"] == 1
    assert poll1["has_more"] is True
    assert poll1["timeline_chunks_returned"] == 0
    assert "No new timeline sections since chunk 1" in poll1["message"]

    # When completed
    job.status = "completed"
    job.completed_chunks = 2
    job.available_sections.append({"chunk_index": 2, "summary": "part 2"})
    poll2 = job.to_dict(since_chunk=1)
    assert poll2["next_since_chunk"] == 2
    assert poll2["has_more"] is False
    assert poll2["timeline_chunks_returned"] == 1
    assert "Analysis completed successfully" in poll2["message"]


def test_tool_schema_oneof_exclusivity() -> None:
    from vidscope.mcp import mcp

    list_tools = getattr(mcp, "list_tools", None)
    if callable(list_tools):
        tools = _await(list_tools())
        if isinstance(tools, Mapping):
            tools = list(tools.values())
        tool_map = {_field(t, "name"): t for t in tools}

        vf = tool_map["view_frame"]
        params_vf = getattr(vf, "parameters", {})
        assert "oneOf" in params_vf
        vf_reqs = [set(o.get("required", [])) for o in params_vf["oneOf"]]
        assert {"frame_id"} in vf_reqs
        assert {"source", "timestamp_seconds"} in vf_reqs

        sv = tool_map["search_video"]
        params_sv = getattr(sv, "parameters", {})
        assert "oneOf" in params_sv
        sv_reqs = [set(o.get("required", [])) for o in params_sv["oneOf"]]
        assert {"query", "source"} in sv_reqs
        assert {"query", "job_id"} in sv_reqs


def test_mcp_prompt_registration_and_discovery() -> None:
    from vidscope.mcp import analyze_video_workflow, mcp, search_video_workflow

    # Direct invocations
    text_av = analyze_video_workflow("test.mp4")
    assert "get_video_info" in text_av
    assert "analyze_video" in text_av

    text_sv = search_video_workflow("test.mp4", "hello")
    assert "search_video" in text_sv
    assert "hello" in text_sv

    # FastMCP list_prompts discovery
    list_prompts = getattr(mcp, "list_prompts", None)
    if callable(list_prompts):
        prompts = _await(list_prompts())
        if isinstance(prompts, Mapping):
            prompts = list(prompts.values())
        names = [_field(p, "name") for p in prompts]
        assert "analyze_video_workflow" in names
        assert "search_video_workflow" in names
