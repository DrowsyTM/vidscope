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
    assert len(tool_map) == 6
    assert "analyze_video" in tool_map
    assert "get_video_info" in tool_map
    assert "search_video" in tool_map
    assert "view_frame" in tool_map
    assert "get_job_status" in tool_map
    assert "get_transcript" in tool_map

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
    assert _tool_annotation(tool_map["get_transcript"], "readOnlyHint") is True
    assert _tool_annotation(tool_map["get_transcript"], "idempotentHint") is True


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
    assert "capabilities" in result
    assert "local_asr_available" in result["capabilities"]
    assert "local_ocr_available" in result["capabilities"]
    assert "caption_tracks_listed" in result
    assert len(result["caption_tracks_listed"]) == 1
    assert result["caption_tracks_listed"][0]["language"] == "en"
    assert len(result["chapters"]) == 2
    assert result["chapters"][0]["title"] == "Intro"


def test_get_video_info_invalid_url_returns_tool_error() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import get_video_info

    res = get_video_info("ftp://invalid-url.com")
    assert isinstance(res, ToolResult)
    assert res.is_error is True


def test_search_video_finds_matching_snippets() -> None:
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import search_video

    job = global_job_manager.create_job(
        "test.mp4",
        [(0.0, 100.0)],
        video_duration_seconds=100.0,
    )
    global_job_manager.update_job_progress(
        job.job_id,
        section={"chunk_index": 1, "summary": "Intro"},
        completed_chunks=1,
        transcript_segments=[
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
    global_job_manager.complete_job(job.job_id)

    res = search_video(query="backpropagation", job_id=job.job_id)
    assert isinstance(res, dict)
    assert res["matches_count"] == 1
    assert res["matches"][0]["start_seconds"] == 65.0
    assert res["matches"][0]["formatted_time"] == "01:05"
    assert "backpropagation" in res["matches"][0]["snippet"]
    assert "coverage" in res
    assert res["coverage"]["is_full_video"] is True


def test_search_video_regex_and_case_sensitivity() -> None:
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import search_video

    job = global_job_manager.create_job(
        "test.mp4",
        [(0.0, 100.0)],
        video_duration_seconds=100.0,
    )
    global_job_manager.update_job_progress(
        job.job_id,
        section={"chunk_index": 1, "summary": "Errors"},
        completed_chunks=1,
        transcript_segments=[
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
    global_job_manager.complete_job(job.job_id)

    # Regex test
    res = search_video(query=r"Error \d+", job_id=job.job_id, is_regex=True)
    assert isinstance(res, dict)
    assert res["matches_count"] == 2

    # Case-sensitive test
    res_case = search_video(query="Error", job_id=job.job_id, case_sensitive=True)
    assert isinstance(res_case, dict)
    assert res_case["matches_count"] == 1
    assert res_case["matches"][0]["start_seconds"] == 10.0


def test_search_video_empty_query_returns_error() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import search_video

    res = search_video(query="   ", job_id="job_123")
    assert isinstance(res, ToolResult)
    assert res.is_error is True


def test_search_video_invalid_regex_returns_error() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.mcp import search_video

    res = search_video(query="[unclosed-regex", job_id="job_123", is_regex=True)
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
    global_job_manager.complete_job(job.job_id)
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
    assert res["next_since_chunk"] == 0
    assert res["has_more"] is True
    assert "estimated_remaining_seconds" in res

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


def test_search_video_rejects_inflight_processing_job() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.contracts import ErrorCode
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import search_video

    job = global_job_manager.create_job("test_inflight", [(0.0, 30.0), (30.0, 60.0)])
    res = search_video(query="something", job_id=job.job_id)
    assert isinstance(res, ToolResult)
    assert res.is_error is True
    assert res.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert res.structured_content["retryable"] is True
    assert res.structured_content["next_action"] == "get_job_status"
    assert res.structured_content["estimated_remaining_seconds"] > 0


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


def test_search_video_job_validation_and_not_found() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.contracts import ErrorCode
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import search_video

    # Missing / non-existent job_id
    res1 = search_video(query="hello", job_id="nonexistent_job_999")
    assert isinstance(res1, ToolResult)
    assert res1.is_error is True
    assert res1.structured_content["code"] == ErrorCode.ARTIFACT_NOT_FOUND
    assert res1.structured_content["next_action"] == "analyze_video"
    assert res1.structured_content["retryable"] is False

    # Failed job
    failed_job = global_job_manager.create_job("failed_src", [(0.0, 10.0)])
    global_job_manager.fail_job(failed_job.job_id, "decoder crash")
    res2 = search_video(query="hello", job_id=failed_job.job_id)
    assert isinstance(res2, ToolResult)
    assert res2.is_error is True
    assert res2.structured_content["code"] == ErrorCode.INTERNAL_STAGE_FAILED


def test_server_info_static_resource() -> None:
    from vidscope.mcp import _server_info_resource, mcp

    payload = json.loads(_server_info_resource())
    assert payload["name"] == "vidscope"
    assert "3.4.7" in payload["version"]
    assert "analyze_video" in payload["tools"]
    assert "get_transcript" in payload["tools"]
    assert "capabilities" in payload
    assert "local_asr_available" in payload["capabilities"]
    assert "local_ocr_available" in payload["capabilities"]
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


def test_search_video_zero_matches_hints() -> None:
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import search_video

    # 1. Zero matches with transcript present
    job1 = global_job_manager.create_job(
        "video1.mp4", [(0.0, 60.0)], video_duration_seconds=60.0
    )
    global_job_manager.update_job_progress(
        job1.job_id,
        section={"chunk_index": 1, "summary": "Intro"},
        completed_chunks=1,
        transcript_segments=[
            {"start_seconds": 0.0, "end_seconds": 10.0, "text": "hello world"}
        ],
    )
    global_job_manager.complete_job(job1.job_id)

    res1 = search_video(query="kubernetes", job_id=job1.job_id)
    assert isinstance(res1, dict)
    assert res1["matches_count"] == 0
    assert "No transcript matches found" in res1["hint"]

    # 2. Zero matches in visual-only job (no transcript)
    job2 = global_job_manager.create_job(
        "video2.mp4", [(0.0, 60.0)], video_duration_seconds=60.0
    )
    global_job_manager.update_job_progress(
        job2.job_id,
        section={"chunk_index": 1, "summary": "Visuals"},
        completed_chunks=1,
        transcript_segments=[],
    )
    global_job_manager.complete_job(job2.job_id)

    res2 = search_video(query="kubernetes", job_id=job2.job_id)
    assert isinstance(res2, dict)
    assert res2["matches_count"] == 0
    assert "visual-only" in res2["hint"]


def test_search_video_windowed_coverage_metadata() -> None:
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import search_video

    # Windowed analysis: analyzed seconds 30 to 60 of a 300-second video
    job = global_job_manager.create_job(
        "video_long.mp4",
        [(30.0, 60.0)],
        video_duration_seconds=300.0,
    )
    global_job_manager.update_job_progress(
        job.job_id,
        section={"chunk_index": 1, "summary": "Segment 30-60"},
        completed_chunks=1,
        transcript_segments=[
            {"start_seconds": 40.0, "end_seconds": 45.0, "text": "quantum computing"}
        ],
    )
    global_job_manager.complete_job(job.job_id)

    res = search_video(query="quantum", job_id=job.job_id)
    assert isinstance(res, dict)
    assert res["matches_count"] == 1
    cov = res["coverage"]
    assert cov["is_full_video"] is False
    assert cov["analyzed_start_seconds"] == 30.0
    assert cov["analyzed_end_seconds"] == 60.0
    assert cov["analyzed_duration_seconds"] == 30.0
    assert cov["video_duration_seconds"] == 300.0


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
    assert poll0["next_action"] == "get_job_status"
    assert poll0["estimated_remaining_seconds"] > 0
    assert "coverage" in poll0

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
    assert poll2["next_action"] is None
    assert "coverage" in poll2


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
        assert "oneOf" not in params_sv
        assert set(params_sv.get("required", [])) >= {"job_id", "query"}


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


@pytest.mark.anyio
async def test_mcp_call_tool_validation_error_normalized_by_middleware() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.contracts import ErrorCode
    from vidscope.mcp import mcp

    # start_seconds < 0 fails FastMCP / Pydantic schema validation
    res = await mcp.call_tool(
        "analyze_video", {"source": "test.mp4", "start_seconds": -5.0}
    )
    assert isinstance(res, ToolResult)
    assert res.is_error is True
    assert res.structured_content is not None
    assert res.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert res.structured_content["stage"] == "analyze_video"
    assert res.structured_content["retryable"] is False
    assert "start_seconds" in res.structured_content["message"]

    # chunk_duration_seconds > 180 fails FastMCP / Pydantic schema validation
    res2 = await mcp.call_tool(
        "analyze_video", {"source": "test.mp4", "chunk_duration_seconds": 250.0}
    )
    assert isinstance(res2, ToolResult)
    assert res2.is_error is True
    assert res2.structured_content is not None
    assert res2.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert "chunk_duration_seconds" in res2.structured_content["message"]


def test_eta_smoothing_stability() -> None:
    import time

    from vidscope.jobs import JobState

    now = time.time()
    job = JobState(
        job_id="job_eta_test",
        source="test.mp4",
        status="processing",
        created_at=now - 30.0,
        updated_at=now - 15.0,
        progress_percentage=20.0,
        completed_chunks=1,
        total_chunks=5,
        chunks=[{"start_seconds": 0.0, "end_seconds": 30.0}],
        chunk_durations=[15.0],
        last_completed_at=now - 15.0,
        last_estimated_remaining=60.0,
    )

    d1 = job.to_dict()
    rem1 = d1["estimated_remaining_seconds"]
    assert rem1 > 0
    # Remaining ETA should not spike wildly upwards from 60s
    assert rem1 <= 60.0 * 1.15


def test_get_transcript_success_and_windowing() -> None:
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import get_transcript

    job = global_job_manager.create_job(
        "test_transcript.mp4",
        [(0.0, 60.0)],
        video_duration_seconds=60.0,
    )
    global_job_manager.update_job_progress(
        job.job_id,
        section={"chunk_index": 1, "summary": "Full speech"},
        completed_chunks=1,
        transcript_segments=[
            {"start_seconds": 0.0, "end_seconds": 5.0, "text": "Hello world."},
            {
                "start_seconds": 10.0,
                "end_seconds": 15.0,
                "text": "This is segment one.",
            },
            {
                "start_seconds": 20.0,
                "end_seconds": 25.0,
                "text": "This is segment two.",
            },
            {"start_seconds": 40.0, "end_seconds": 50.0, "text": "Final thoughts."},
        ],
    )
    global_job_manager.complete_job(job.job_id)

    # Query narrow window 8.0 - 28.0 (should capture segments at 10-15 and 20-25)
    res = get_transcript(job_id=job.job_id, start_seconds=8.0, end_seconds=28.0)
    assert isinstance(res, dict)
    assert res["segments_count"] == 2
    assert res["start_seconds"] == 8.0
    assert res["end_seconds"] == 28.0
    assert res["window_duration_seconds"] == 20.0
    assert res["text"] == "This is segment one. This is segment two."
    assert len(res["segments"]) == 2
    assert res["segments"][0]["start_seconds"] == 10.0
    assert res["coverage"]["is_full_video"] is True


def test_get_transcript_default_end_seconds_clamps() -> None:
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import get_transcript

    job = global_job_manager.create_job(
        "short_video.mp4",
        [(0.0, 45.0)],
        video_duration_seconds=45.0,
    )
    global_job_manager.update_job_progress(
        job.job_id,
        section={"chunk_index": 1, "summary": "Speech"},
        completed_chunks=1,
        transcript_segments=[
            {"start_seconds": 5.0, "end_seconds": 10.0, "text": "Start speech."},
            {"start_seconds": 35.0, "end_seconds": 40.0, "text": "Ending speech."},
        ],
    )
    global_job_manager.complete_job(job.job_id)

    # Calling without end_seconds should clamp to analyzed_end_seconds (45.0)
    res = get_transcript(job_id=job.job_id, start_seconds=0.0)
    assert isinstance(res, dict)
    assert res["end_seconds"] == 45.0
    assert res["segments_count"] == 2
    assert "Start speech. Ending speech." in res["text"]


def test_get_transcript_rejects_inflight_processing_job() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.contracts import ErrorCode
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import get_transcript

    job = global_job_manager.create_job("inflight.mp4", [(0.0, 60.0)])
    res = get_transcript(job_id=job.job_id)
    assert isinstance(res, ToolResult)
    assert res.is_error is True
    assert res.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert res.structured_content["retryable"] is True
    assert res.structured_content["next_action"] == "get_job_status"
    assert res.structured_content["estimated_remaining_seconds"] > 0


def test_get_transcript_job_not_found_and_failed() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.contracts import ErrorCode
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import get_transcript

    # Not found
    res1 = get_transcript(job_id="job_missing_xyz")
    assert isinstance(res1, ToolResult)
    assert res1.is_error is True
    assert res1.structured_content["code"] == ErrorCode.ARTIFACT_NOT_FOUND
    assert res1.structured_content["next_action"] == "analyze_video"
    assert res1.structured_content["retryable"] is False

    # Failed
    job_failed = global_job_manager.create_job("fail.mp4", [(0.0, 30.0)])
    global_job_manager.fail_job(job_failed.job_id, "decoder error")
    res2 = get_transcript(job_id=job_failed.job_id)
    assert isinstance(res2, ToolResult)
    assert res2.is_error is True
    assert res2.structured_content["code"] == ErrorCode.INTERNAL_STAGE_FAILED


def test_get_transcript_validation_errors() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.contracts import ErrorCode
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import get_transcript

    job = global_job_manager.create_job("val.mp4", [(0.0, 60.0)])
    global_job_manager.complete_job(job.job_id)

    # start_seconds < 0
    res1 = get_transcript(job_id=job.job_id, start_seconds=-10.0)
    assert isinstance(res1, ToolResult)
    assert res1.structured_content["code"] == ErrorCode.INVALID_REQUEST

    # end_seconds <= start_seconds
    res2 = get_transcript(job_id=job.job_id, start_seconds=20.0, end_seconds=10.0)
    assert isinstance(res2, ToolResult)
    assert res2.structured_content["code"] == ErrorCode.INVALID_REQUEST

    # window > max_duration_seconds
    res3 = get_transcript(
        job_id=job.job_id,
        start_seconds=0.0,
        end_seconds=400.0,
        max_duration_seconds=300.0,
    )
    assert isinstance(res3, ToolResult)
    assert res3.structured_content["code"] == ErrorCode.INVALID_REQUEST


def test_get_transcript_zero_segments_hints() -> None:
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import get_transcript

    # 1. Speech exists, but outside requested range
    job1 = global_job_manager.create_job(
        "speech.mp4", [(0.0, 60.0)], video_duration_seconds=60.0
    )
    global_job_manager.update_job_progress(
        job1.job_id,
        section={"chunk_index": 1, "summary": "Speech"},
        completed_chunks=1,
        transcript_segments=[
            {"start_seconds": 10.0, "end_seconds": 20.0, "text": "Speech"}
        ],
    )
    global_job_manager.complete_job(job1.job_id)

    res1 = get_transcript(job_id=job1.job_id, start_seconds=30.0, end_seconds=50.0)
    assert isinstance(res1, dict)
    assert res1["segments_count"] == 0
    assert "No speech segments found in range" in res1["hint"]

    # 2. Visual-only (no transcript segments anywhere)
    job2 = global_job_manager.create_job(
        "visual.mp4", [(0.0, 60.0)], video_duration_seconds=60.0
    )
    global_job_manager.update_job_progress(
        job2.job_id,
        section={"chunk_index": 1, "summary": "Visual only"},
        completed_chunks=1,
        transcript_segments=[],
    )
    global_job_manager.complete_job(job2.job_id)

    res2 = get_transcript(job_id=job2.job_id, start_seconds=0.0, end_seconds=30.0)
    assert isinstance(res2, dict)
    assert res2["segments_count"] == 0
    assert "visual-only" in res2["hint"]


def test_chunk_section_transcript_status_labels() -> None:
    from vidscope.jobs import global_job_manager

    # 1. Speech completed
    job = global_job_manager.create_job(
        "test.mp4", [(0.0, 30.0)], video_duration_seconds=30.0
    )
    global_job_manager.update_job_progress(
        job.job_id,
        section={
            "chunk_index": 1,
            "mode": "speech_and_visual",
            "transcript_status": "completed",
            "summary": "Full speech transcript",
        },
        completed_chunks=1,
        transcript_segments=[{"start_seconds": 0.0, "end_seconds": 5.0, "text": "Hi"}],
    )
    global_job_manager.complete_job(job.job_id)
    state = global_job_manager.get_job(job.job_id)
    assert state is not None
    assert state.available_sections[0]["transcript_status"] == "completed"

    # 2. Visual-only fallback with error
    job_err = global_job_manager.create_job(
        "test2.mp4", [(0.0, 30.0)], video_duration_seconds=30.0
    )
    global_job_manager.update_job_progress(
        job_err.job_id,
        section={
            "chunk_index": 1,
            "mode": "visual_only",
            "transcript_status": "failed",
            "transcript_error": "CUDA out of memory",
            "summary": "Keyframe extraction only (4 frames). (Transcript extraction failed: CUDA out of memory)",
        },
        completed_chunks=1,
    )
    global_job_manager.complete_job(job_err.job_id)
    state_err = global_job_manager.get_job(job_err.job_id)
    assert state_err is not None
    assert state_err.available_sections[0]["transcript_status"] == "failed"
    assert state_err.available_sections[0]["transcript_error"] == "CUDA out of memory"
    assert "Keyframe extraction only" in state_err.available_sections[0]["summary"]


def test_stdio_validation_error_stderr_suppression() -> None:
    import json
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "vidscope.mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    reqs = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "search_video", "arguments": {}},
        },
    ]
    for r in reqs:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(r) + "\n")
        proc.stdin.flush()

    assert proc.stdout is not None
    _ = proc.stdout.readline()
    line2 = proc.stdout.readline()
    proc.terminate()
    assert proc.stderr is not None
    stderr = proc.stderr.read()

    assert stderr.strip() == ""
    parsed = json.loads(line2)
    assert parsed["result"]["isError"] is True


def test_get_transcript_sorts_segments_monotonically() -> None:
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import get_transcript

    job = global_job_manager.create_job("sort_test.mp4", [(0.0, 100.0)])
    # Add segments out of order
    out_of_order_segs = [
        {"start_seconds": 50.0, "end_seconds": 60.0, "text": "later dialogue"},
        {"start_seconds": 10.0, "end_seconds": 20.0, "text": "earlier dialogue"},
        {"start_seconds": 25.0, "end_seconds": 35.0, "text": "middle dialogue"},
    ]
    global_job_manager.update_job_progress(
        job.job_id,
        section={"chunk_index": 1, "transcript_status": "completed"},
        completed_chunks=1,
        transcript_segments=out_of_order_segs,
    )
    global_job_manager.complete_job(job.job_id)

    res = get_transcript(job_id=job.job_id, start_seconds=0.0, end_seconds=100.0)
    assert isinstance(res, dict)
    assert res["segments_count"] == 3
    starts = [s["start_seconds"] for s in res["segments"]]
    assert starts == [10.0, 25.0, 50.0]
    assert res["text"] == "earlier dialogue middle dialogue later dialogue"


def test_job_state_coverage_independent_transcript_metrics() -> None:
    from vidscope.jobs import global_job_manager

    job = global_job_manager.create_job("cov_test.mp4", [(0.0, 180.0), (180.0, 360.0)])
    # Initially pending with no transcript
    cov0 = job.coverage()
    assert cov0["transcript_start_seconds"] is None
    assert cov0["transcript_end_seconds"] is None
    assert cov0["transcript_segments_count"] == 0
    assert cov0["transcript_status"] == "pending"

    # Add chunk with speech
    global_job_manager.update_job_progress(
        job.job_id,
        section={"chunk_index": 1, "transcript_status": "completed"},
        completed_chunks=1,
        transcript_segments=[
            {"start_seconds": 5.0, "end_seconds": 25.0, "text": "speech here"}
        ],
    )
    global_job_manager.complete_job(job.job_id)
    cov1 = job.coverage()
    assert cov1["transcript_start_seconds"] == 5.0
    assert cov1["transcript_end_seconds"] == 25.0
    assert cov1["transcript_segments_count"] == 1
    assert cov1["transcript_status"] == "completed"


def test_search_video_zero_matches_hint_includes_transcript_range() -> None:
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import search_video

    job = global_job_manager.create_job("search_test.mp4", [(0.0, 180.0)])
    global_job_manager.update_job_progress(
        job.job_id,
        section={"chunk_index": 1, "transcript_status": "completed"},
        completed_chunks=1,
        transcript_segments=[
            {
                "start_seconds": 10.0,
                "end_seconds": 30.0,
                "text": "we discuss machine learning",
            }
        ],
    )
    global_job_manager.complete_job(job.job_id)

    res = search_video(job_id=job.job_id, query="quantum")
    assert isinstance(res, dict)
    assert res["matches_count"] == 0
    assert "transcript range (10.0s - 30.0s" in res["hint"]
    assert "analyzed: 0.0s - 180.0s" in res["hint"]


def test_get_job_status_missing_job_returns_actionable_next_action() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.contracts import ErrorCode
    from vidscope.mcp import get_job_status

    res = get_job_status(job_id="nonexistent_job_12345")
    assert isinstance(res, ToolResult)
    assert res.is_error is True
    assert res.structured_content["code"] == ErrorCode.ARTIFACT_NOT_FOUND
    assert res.structured_content["next_action"] == "analyze_video"
    assert res.structured_content["retryable"] is False


def test_view_frame_missing_frame_returns_actionable_next_action() -> None:
    from fastmcp.tools.base import ToolResult

    from vidscope.contracts import ErrorCode
    from vidscope.mcp import view_frame

    res = view_frame(frame_id="frame_missing_12345")
    assert isinstance(res, ToolResult)
    assert res.is_error is True
    assert res.structured_content["code"] == ErrorCode.ARTIFACT_NOT_FOUND
    assert res.structured_content["next_action"] == "analyze_video"
    assert res.structured_content["retryable"] is False


def test_reconcile_transcript_segments_word_level_timestamps() -> None:
    from vidscope.jobs import reconcile_transcript_segments

    existing = [
        {
            "start_seconds": 170.0,
            "end_seconds": 178.0,
            "text": "let's prepare",
            "words": [
                {"word": "let's", "start_seconds": 170.0, "end_seconds": 174.0},
                {"word": "prepare", "start_seconds": 174.0, "end_seconds": 178.0},
            ],
        },
        {
            "start_seconds": 178.0,
            "end_seconds": 181.5,
            "text": "for the next section",
            "words": [
                {"word": "for", "start_seconds": 178.0, "end_seconds": 178.8},
                {"word": "the", "start_seconds": 178.8, "end_seconds": 179.4},
                {"word": "next", "start_seconds": 179.4, "end_seconds": 180.2},
                {"word": "section", "start_seconds": 180.2, "end_seconds": 181.5},
            ],
        },
    ]

    incoming = [
        {
            "start_seconds": 179.0,
            "end_seconds": 183.0,
            "text": "the next section where we",
            "words": [
                {"word": "the", "start_seconds": 179.0, "end_seconds": 179.5},
                {"word": "next", "start_seconds": 179.5, "end_seconds": 180.3},
                {"word": "section", "start_seconds": 180.3, "end_seconds": 181.4},
                {"word": "where", "start_seconds": 181.6, "end_seconds": 182.2},
                {"word": "we", "start_seconds": 182.2, "end_seconds": 183.0},
            ],
        },
        {
            "start_seconds": 183.1,
            "end_seconds": 186.0,
            "text": "dive into details",
            "words": [
                {"word": "dive", "start_seconds": 183.1, "end_seconds": 184.0},
                {"word": "into", "start_seconds": 184.0, "end_seconds": 185.0},
                {"word": "details", "start_seconds": 185.0, "end_seconds": 186.0},
            ],
        },
    ]

    result = reconcile_transcript_segments(existing, incoming)
    # Total segments: 2 existing + 2 incoming (first incoming segment modified)
    assert len(result) == 4
    # Check the modified incoming segment
    modified = result[2]
    assert modified["text"] == "where we"
    assert modified["start_seconds"] == 181.6
    assert len(modified["words"]) == 2
    assert modified["words"][0]["word"] == "where"
    # Check the second incoming segment is untouched
    assert result[3]["text"] == "dive into details"


def test_reconcile_transcript_segments_text_only() -> None:
    from vidscope.jobs import reconcile_transcript_segments

    existing = [
        {"start_seconds": 175.0, "end_seconds": 181.2, "text": "we discuss attention"}
    ]
    incoming = [
        {
            "start_seconds": 180.0,
            "end_seconds": 185.0,
            "text": "attention and transformers",
        }
    ]

    result = reconcile_transcript_segments(existing, incoming)
    assert len(result) == 2
    assert result[0]["text"] == "we discuss attention"
    assert result[1]["text"] == "and transformers"
    assert result[1]["start_seconds"] == 181.2


def test_reconcile_transcript_segments_entire_segment_consumed() -> None:
    from vidscope.jobs import reconcile_transcript_segments

    existing = [
        {"start_seconds": 175.0, "end_seconds": 181.0, "text": "we discuss attention"}
    ]
    incoming = [
        {"start_seconds": 178.0, "end_seconds": 181.0, "text": "discuss attention"},
        {
            "start_seconds": 181.5,
            "end_seconds": 186.0,
            "text": "mechanisms in depth",
        },
    ]

    result = reconcile_transcript_segments(existing, incoming)
    assert len(result) == 2
    assert result[0]["text"] == "we discuss attention"
    assert result[1]["text"] == "mechanisms in depth"
    assert result[1]["start_seconds"] == 181.5


def test_reconcile_transcript_segments_no_overlap() -> None:
    from vidscope.jobs import reconcile_transcript_segments

    existing = [
        {"start_seconds": 0.0, "end_seconds": 10.0, "text": "first segment text"}
    ]
    incoming = [
        {"start_seconds": 12.0, "end_seconds": 20.0, "text": "second segment text"}
    ]

    result = reconcile_transcript_segments(existing, incoming)
    assert len(result) == 2
    assert result[0]["text"] == "first segment text"
    assert result[1]["text"] == "second segment text"
    assert result[1]["start_seconds"] == 12.0


def test_reconcile_transcript_segments_timestamp_overlap_no_word_match() -> None:
    from vidscope.jobs import reconcile_transcript_segments

    existing = [
        {"start_seconds": 0.0, "end_seconds": 10.5, "text": "first segment text"}
    ]
    incoming = [
        {"start_seconds": 10.0, "end_seconds": 20.0, "text": "totally different words"}
    ]

    result = reconcile_transcript_segments(existing, incoming)
    assert len(result) == 2
    assert result[0]["text"] == "first segment text"
    # No words dropped since tokens didn't match, but start_seconds clamped to last end
    assert result[1]["text"] == "totally different words"
    assert result[1]["start_seconds"] == 10.5


def test_job_multi_chunk_transcript_reconciliation() -> None:
    from vidscope.jobs import global_job_manager
    from vidscope.mcp import OVERLAP_BUFFER_SECONDS, get_transcript

    assert OVERLAP_BUFFER_SECONDS == 2.0

    job = global_job_manager.create_job(
        "overlap_test.mp4", [(0.0, 180.0), (180.0, 360.0)]
    )

    # Chunk 1 (with audio buffer up to 182.0s)
    chunk1_segments = [
        {
            "start_seconds": 176.0,
            "end_seconds": 181.8,
            "text": "we wrap up neural networks",
        }
    ]
    global_job_manager.update_job_progress(
        job.job_id,
        section={"chunk_index": 1, "summary": "chunk 1 summary"},
        completed_chunks=1,
        transcript_segments=chunk1_segments,
    )

    # Chunk 2 (starts at 180.0s, picks up duplicate words)
    chunk2_segments = [
        {
            "start_seconds": 180.0,
            "end_seconds": 184.0,
            "text": "neural networks and begin transformers",
        }
    ]
    global_job_manager.update_job_progress(
        job.job_id,
        section={"chunk_index": 2, "summary": "chunk 2 summary"},
        completed_chunks=2,
        transcript_segments=chunk2_segments,
    )
    global_job_manager.complete_job(job.job_id)

    res = get_transcript(job_id=job.job_id)
    assert isinstance(res, dict)
    assert res["segments_count"] == 2
    assert res["segments"][0]["text"] == "we wrap up neural networks"
    assert res["segments"][1]["text"] == "and begin transformers"
    assert res["text"] == "we wrap up neural networks and begin transformers"
