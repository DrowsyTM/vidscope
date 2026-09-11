"""Unit tests for mock FastMCP mode simulation."""

from __future__ import annotations

import os
from unittest.mock import patch

from fastmcp.tools.base import ToolResult
from mcp.types import ImageContent

from vidscope.jobs import global_job_manager
from vidscope.mcp import (
    analyze_video,
    get_job_status,
    get_video_info,
    search_video,
    view_frame,
)
from vidscope.mock_mcp import (
    is_mock_async_requested,
    is_mock_mcp_enabled,
    mock_get_video_info,
    mock_process_job_chunks,
)


def test_is_mock_mcp_enabled() -> None:
    with patch.dict(os.environ, {"VIDSCOPE_MOCK_MCP": "1"}):
        assert is_mock_mcp_enabled() is True
    with patch.dict(os.environ, {"VIDSCOPE_MOCK_MCP": "true"}):
        assert is_mock_mcp_enabled() is True
    with patch.dict(os.environ, {"VIDSCOPE_MOCK_MCP": "0"}):
        assert is_mock_mcp_enabled() is False
    with patch.dict(os.environ, {}, clear=True):
        assert is_mock_mcp_enabled() is False


def test_is_mock_async_requested() -> None:
    # Explicit env values
    with patch.dict(os.environ, {"VIDSCOPE_MOCK_ASYNC": "1"}):
        assert is_mock_async_requested("normal.mp4", 0.0, 30.0) is True
    with patch.dict(os.environ, {"VIDSCOPE_MOCK_ASYNC": "async"}):
        assert is_mock_async_requested("normal.mp4", 0.0, 30.0) is True
    with patch.dict(os.environ, {"VIDSCOPE_MOCK_ASYNC": "0"}):
        assert is_mock_async_requested("async_test.mp4", 0.0, 30.0) is False
    with patch.dict(os.environ, {"VIDSCOPE_MOCK_ASYNC": "sync"}):
        assert is_mock_async_requested("async_test.mp4", 0.0, 30.0) is False

    # Random mode
    with patch.dict(os.environ, {"VIDSCOPE_MOCK_ASYNC": "random"}):
        with patch("random.random", return_value=0.2):
            assert is_mock_async_requested("normal.mp4", 0.0, 30.0) is True
        with patch("random.random", return_value=0.8):
            assert is_mock_async_requested("normal.mp4", 0.0, 30.0) is False

    # Auto detection via keyword or duration
    with patch.dict(os.environ, {}, clear=True):
        assert is_mock_async_requested("mock://async_video.mp4", 0.0, 30.0) is True
        assert is_mock_async_requested("mock://normal_video.mp4", 0.0, 30.0) is False
        assert is_mock_async_requested("mock://long_video.mp4", 0.0, 400.0) is True


def test_mock_get_video_info() -> None:
    info = mock_get_video_info("mock://presentation.mp4")
    assert info["duration_seconds"] == 180.0
    assert len(info["chapters"]) == 3
    assert info["runtime_capabilities"]["local_asr_available"] is True


def test_mock_process_job_chunks_direct() -> None:
    job = global_job_manager.create_job("mock://direct_test.mp4", [(0.0, 30.0)])
    mock_process_job_chunks(job.job_id, "mock://direct_test.mp4", 0.0, 30.0, 30.0)

    job_after = global_job_manager.get_job(job.job_id)
    assert job_after is not None
    assert job_after.status == "completed"
    assert len(job_after.available_sections) == 1
    assert len(job_after.full_transcript) == 1
    assert len(job_after.frames) == 1


def test_mcp_analyze_video_mock_sync() -> None:
    with patch.dict(os.environ, {"VIDSCOPE_MOCK_MCP": "1", "VIDSCOPE_MOCK_ASYNC": "0"}):
        # get_video_info returns mock metadata
        info_res = get_video_info("mock://sync_demo.mp4")
        assert isinstance(info_res, dict)
        assert info_res["duration_seconds"] == 180.0

        # analyze_video returns completed synchronously
        res = analyze_video(
            source="mock://sync_demo.mp4",
            start_seconds=0.0,
            end_seconds=30.0,
            sync_timeout_seconds=5.0,
        )
        assert isinstance(res, dict)
        assert res["status"] == "completed"
        assert res["has_more"] is False
        assert len(res["timeline"]) == 1
        assert "keyframes" in res["timeline"][0]

        frame_id = res["timeline"][0]["keyframes"][0]["frame_id"]
        # view_frame returns FastMCP Image
        v_res = view_frame(frame_id=frame_id)
        assert isinstance(v_res, ToolResult)
        assert any(isinstance(block, ImageContent) for block in v_res.content)

        # search_video searches synthetic transcript
        s_res = search_video(job_id=res["job_id"], query="Speaker")
        assert isinstance(s_res, dict)
        assert s_res["matches_count"] >= 1


def test_mcp_analyze_video_mock_async_and_polling() -> None:
    with patch.dict(os.environ, {"VIDSCOPE_MOCK_MCP": "1", "VIDSCOPE_MOCK_ASYNC": "1"}):
        # analyze_video with short timeout triggers async handoff
        res = analyze_video(
            source="mock://async_demo.mp4",
            start_seconds=0.0,
            end_seconds=60.0,
            sync_timeout_seconds=0.1,
        )
        assert isinstance(res, dict)
        assert res["status"] == "processing"
        assert res["next_action"] == "get_job_status"
        assert res["estimated_remaining_seconds"] > 0
        job_id = res["job_id"]

        # Wait for the simulated 0.5s mock sleep to complete
        import time

        time.sleep(0.7)

        # Poll status
        poll_res = get_job_status(job_id=job_id)
        assert isinstance(poll_res, dict)
        assert poll_res["status"] == "completed"
        assert len(poll_res["timeline"]) >= 1


def test_mcp_analyze_video_mock_random_mode() -> None:
    with patch.dict(
        os.environ, {"VIDSCOPE_MOCK_MCP": "1", "VIDSCOPE_MOCK_ASYNC": "random"}
    ):
        # When random returns 0.8 (>= 0.5), it should resolve synchronously
        with patch("random.random", return_value=0.8):
            res_sync = analyze_video(
                source="mock://random_test.mp4",
                start_seconds=0.0,
                end_seconds=30.0,
            )
            assert isinstance(res_sync, dict)
            assert res_sync["status"] == "completed"
            assert "timeline" in res_sync

        # When random returns 0.2 (< 0.5), it should resolve asynchronously (requiring get_job_status)
        with patch("random.random", return_value=0.2):
            res_async = analyze_video(
                source="mock://random_test_2.mp4",
                start_seconds=0.0,
                end_seconds=30.0,
            )
            assert isinstance(res_async, dict)
            assert res_async["status"] == "processing"
            assert res_async["next_action"] == "get_job_status"
            assert "estimated_remaining_seconds" in res_async
