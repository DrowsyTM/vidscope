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
    from video_analyzer.contracts import AnalyzeVideoRequest, TimeRange

    return AnalyzeVideoRequest(
        source=str(source),
        time_range=TimeRange(start_seconds=0, end_seconds=30),
        tasks={"metadata"},
        output_directory=output,
    )


def _success_result() -> Any:
    from video_analyzer.contracts import AnalysisResult, AnalysisSummary, ArtifactRef

    uri = "video-analyzer://runs/run-mcp/artifacts/transcript"
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
        manifest_uri="video-analyzer://runs/run-mcp/manifest",
        artifacts=[artifact],
        artifact_refs=[artifact],
    )


def _error(
    code: str = "INTERNAL_STAGE_FAILED",
    stage: str = "transcribe",
    message: str = "deterministic test failure",
) -> Any:
    from video_analyzer.contracts import AnalysisError

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
        manifest_uri="video-analyzer://runs/run-mcp/manifest",
    )


def _failure(error: Any) -> BaseException:
    from video_analyzer.core import VideoAnalyzerFailure

    return VideoAnalyzerFailure(error)


def _patch_core(monkeypatch: pytest.MonkeyPatch, replacement: Any) -> Any:
    """Patch both the core module and the common FastMCP import alias."""
    import video_analyzer.core as core_module
    import video_analyzer.mcp as mcp_module

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
    from video_analyzer.mcp import mcp

    list_tools = getattr(mcp, "list_tools", None)
    if callable(list_tools):
        tools = _await(list_tools())
        if isinstance(tools, Mapping):
            tools = list(tools.values())
    else:
        tools = _registered_tools(mcp)
    if tools is None:
        pytest.skip("FastMCP version does not expose tool introspection")
    assert len(tools) == 1
    tool = tools[0]
    assert _field(tool, "name") == "analyze_video"
    assert _tool_annotation(tool, "readOnlyHint") is False
    assert _tool_annotation(tool, "idempotentHint") is False


def test_mcp_success_returns_shared_result_unchanged_and_compact_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_analyzer.mcp import analyze_video

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
    assert payload["manifest_uri"].startswith("video-analyzer://")
    assert all(
        item["uri"].startswith("video-analyzer://") for item in payload["artifacts"]
    )


def test_mcp_terminal_failure_is_error_tool_result_with_shared_error_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_analyzer.mcp import analyze_video

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
    from video_analyzer.mcp import analyze_video

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

    from video_analyzer.mcp import analyze_video

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
    uri = f"video-analyzer://runs/{run_id}/artifacts/{artifact_id}"
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
    import video_analyzer.mcp as mcp_module

    monkeypatch.setenv("VIDEO_ANALYZER_ALLOWED_OUTPUT_ROOT", str(tmp_path))
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
