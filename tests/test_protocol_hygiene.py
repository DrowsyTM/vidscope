"""Automated test suite for Phase 1: Protocol & Tool Calling Hygiene."""

from __future__ import annotations

import pytest
from fastmcp.tools.base import ToolResult

from vidscope.contracts import ErrorCode
from vidscope.eval_harness import (
    InvocationTrace,
    ProtocolScorecard,
    evaluate_invocation_traces,
)
from vidscope.jobs import global_job_manager
from vidscope.mcp import (
    get_job_status,
    get_transcript,
    mcp,
    search_video,
    view_frame,
)


@pytest.mark.anyio
async def test_schema_validation_tcer_normalized() -> None:
    """Test that invalid argument combinations are normalized into structured ToolResults."""
    # Test 1: Invalid chunk duration (out of bounds)
    res1 = await mcp.call_tool(
        "analyze_video",
        {"source": "test.mp4", "chunk_duration_seconds": 300.0},
    )
    assert isinstance(res1, ToolResult)
    assert res1.is_error is True
    assert res1.structured_content["code"] == ErrorCode.INVALID_REQUEST.value

    # Test 2: view_frame mutually exclusive arguments
    res2 = await mcp.call_tool(
        "view_frame",
        {
            "frame_id": "frame_0001",
            "source": "https://example.com/video.mp4",
            "timestamp_seconds": 12.0,
        },
    )
    assert isinstance(res2, ToolResult)
    assert res2.is_error is True
    assert res2.structured_content["code"] == ErrorCode.INVALID_REQUEST.value

    # Test 3: get_transcript duration exceeds 600s max
    res3 = await mcp.call_tool(
        "get_transcript",
        {"job_id": "dummy_job", "max_duration_seconds": 900.0},
    )
    assert isinstance(res3, ToolResult)
    assert res3.is_error is True
    assert res3.structured_content["code"] == ErrorCode.INVALID_REQUEST.value


def test_sequencing_tser_premature_querying() -> None:
    """Test that querying downstream artifacts on in-flight jobs returns actionable guidance."""
    job = global_job_manager.create_job("in_flight.mp4", [(0.0, 180.0)])
    job.last_estimated_remaining = 14.5

    # Attempting search before job completes
    res = search_video(query="key term", job_id=job.job_id)
    assert isinstance(res, ToolResult)
    assert res.is_error is True
    assert res.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert res.structured_content["retryable"] is True
    assert res.structured_content["next_action"] == "get_job_status"
    assert res.structured_content["estimated_remaining_seconds"] == 14.5

    # Attempting transcript before job completes
    res_t = get_transcript(job_id=job.job_id)
    assert isinstance(res_t, ToolResult)
    assert res_t.is_error is True
    assert res_t.structured_content["code"] == ErrorCode.INVALID_REQUEST
    assert res_t.structured_content["retryable"] is True
    assert res_t.structured_content["next_action"] == "get_job_status"
    assert res_t.structured_content["estimated_remaining_seconds"] == 14.5


def test_sequencing_tser_expired_or_missing_artifacts() -> None:
    """Test that querying non-existent artifacts guides the agent back to analyze_video."""
    res_status = get_job_status(job_id="job_nonexistent_999")
    assert isinstance(res_status, ToolResult)
    assert res_status.is_error is True
    assert res_status.structured_content["code"] == ErrorCode.ARTIFACT_NOT_FOUND
    assert res_status.structured_content["next_action"] == "analyze_video"
    assert res_status.structured_content["retryable"] is False

    res_frame = view_frame(frame_id="frame_nonexistent_999")
    assert isinstance(res_frame, ToolResult)
    assert res_frame.is_error is True
    assert res_frame.structured_content["code"] == ErrorCode.ARTIFACT_NOT_FOUND
    assert res_frame.structured_content["next_action"] == "analyze_video"
    assert res_frame.structured_content["retryable"] is False


def test_polling_cadence_evaluator_disciplined_agent() -> None:
    """Test scorecard computation for an agent respecting backoff estimates."""
    traces = [
        InvocationTrace(
            tool_name="analyze_video",
            arguments={"source": "video.mp4"},
            timestamp_seconds=0.0,
            response={"status": "processing", "estimated_remaining_seconds": 8.0},
            is_error=False,
            estimated_remaining_seconds=8.0,
        ),
        # Agent waits 8.0 seconds before polling
        InvocationTrace(
            tool_name="get_job_status",
            arguments={"job_id": "job_1"},
            timestamp_seconds=8.0,
            response={"status": "processing", "estimated_remaining_seconds": 3.0},
            is_error=False,
            estimated_remaining_seconds=3.0,
        ),
        # Agent waits 3.0 seconds before polling again
        InvocationTrace(
            tool_name="get_job_status",
            arguments={"job_id": "job_1"},
            timestamp_seconds=11.0,
            response={"status": "completed", "estimated_remaining_seconds": 0.0},
            is_error=False,
            estimated_remaining_seconds=0.0,
        ),
    ]

    card = evaluate_invocation_traces(traces)
    assert card.total_invocations == 3
    assert card.schema_validation_errors == 0
    assert card.sequencing_errors == 0
    assert card.cadence_violations == 0
    assert card.tcer == 0.0
    assert card.tser == 0.0
    assert card.cadence_violation_rate == 0.0
    assert "Tool Call Error Rate" in card.to_markdown_table()


def test_polling_cadence_evaluator_busy_wait_agent() -> None:
    """Test scorecard computation for an agent spamming status checks in a tight loop."""
    traces = [
        InvocationTrace(
            tool_name="analyze_video",
            arguments={"source": "video.mp4"},
            timestamp_seconds=0.0,
            response={"status": "processing", "estimated_remaining_seconds": 10.0},
            is_error=False,
            estimated_remaining_seconds=10.0,
        ),
        # Agent polls immediately after 0.1s instead of waiting ~10s
        InvocationTrace(
            tool_name="get_job_status",
            arguments={"job_id": "job_1"},
            timestamp_seconds=0.1,
            response={"status": "processing", "estimated_remaining_seconds": 9.9},
            is_error=False,
            estimated_remaining_seconds=9.9,
        ),
        # Agent polls again after 0.2s
        InvocationTrace(
            tool_name="get_job_status",
            arguments={"job_id": "job_1"},
            timestamp_seconds=0.3,
            response={"status": "processing", "estimated_remaining_seconds": 9.7},
            is_error=False,
            estimated_remaining_seconds=9.7,
        ),
    ]

    card = evaluate_invocation_traces(traces)
    assert card.total_invocations == 3
    assert card.cadence_violations == 2
    assert card.cadence_violation_rate == 1.0  # 2 violations out of 2 subsequent polls
    score_dict = card.to_dict()
    assert score_dict["cadence_violations"] == 2


def test_protocol_scorecard_tool_usage_metrics() -> None:
    traces = [
        InvocationTrace(
            tool_name="get_video_info",
            arguments={"source": "test.mp4"},
            timestamp_seconds=0.0,
            response={"duration_seconds": 60.0},
            is_error=False,
            latency_ms=120.0,
            response_words=50,
            response_tokens=65,
        ),
        InvocationTrace(
            tool_name="analyze_video",
            arguments={"source": "test.mp4"},
            timestamp_seconds=0.5,
            response={"status": "processing"},
            is_error=False,
            latency_ms=80.0,
            response_words=100,
            response_tokens=130,
        ),
        InvocationTrace(
            tool_name="get_job_status",
            arguments={"job_id": "job_1"},
            timestamp_seconds=1.5,
            response={"status": "completed"},
            is_error=False,
            latency_ms=25.0,
            response_words=30,
            response_tokens=40,
        ),
        InvocationTrace(
            tool_name="search_video",
            arguments={"job_id": "job_1", "query": "text"},
            timestamp_seconds=2.0,
            response={"error": "job not found"},
            is_error=True,
            error_code="ARTIFACT_NOT_FOUND",
            latency_ms=15.0,
            response_words=10,
            response_tokens=15,
        ),
        InvocationTrace(
            tool_name="search_video",
            arguments={"query": ""},
            timestamp_seconds=2.5,
            response={"error": "invalid parameter"},
            is_error=True,
            error_code="INVALID_REQUEST",
            latency_ms=10.0,
            response_words=8,
            response_tokens=12,
        ),
    ]

    card = evaluate_invocation_traces(traces)
    counts = card.tool_counts()
    assert counts["get_video_info"] == 1
    assert counts["analyze_video"] == 1
    assert counts["get_job_status"] == 1
    assert counts["search_video"] == 2

    error_counts = card.tool_error_counts()
    assert error_counts.get("get_video_info", 0) == 0
    assert error_counts["search_video"] == 2

    stats = card.tool_usage_stats(total_runs=2)
    assert len(stats) == 4
    tool_map = {s["tool_name"]: s for s in stats}
    # Check search_video separated form vs order errors
    search_stat = tool_map["search_video"]
    assert search_stat["total_calls"] == 2
    assert search_stat["avg_calls_per_run"] == 1.0
    assert search_stat["form_errors"] == 1
    assert search_stat["form_error_rate_percentage"] == 50.0
    assert search_stat["order_errors"] == 1
    assert search_stat["order_error_rate_percentage"] == 50.0
    assert search_stat["avg_latency_ms"] == 12.5

    table_md = card.tool_usage_markdown_table(total_runs=2)
    assert (
        "| Tool Name | Calls | Avg/Run | Latency (ms) | Words/Call | Est. Tokens | Form Err (%) | Order Err (%) |"
        in table_md
    )
    assert (
        "| `search_video` | 2 | 1.00 | 12.5 ms | 9 | ~14 | 50.0% | 50.0% |" in table_md
    )

    # Ensure empty traces produces safe fallback
    empty_card = ProtocolScorecard()
    assert empty_card.tool_usage_markdown_table() == "_No tool invocations recorded._"
