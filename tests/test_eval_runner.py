"""Tests for OMP benchmark evaluation runner and semantic scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vidscope.cli import app
from vidscope.eval_harness import (
    AbsentEvent,
    BenchmarkDataset,
    BenchmarkEvent,
    OrderingTask,
    RetrievalQuery,
    check_hallucination_rejection,
    compute_boundary_mae,
    compute_kendall_tau,
    compute_tiou,
    load_benchmark_dataset,
)
from vidscope.eval_runner import (
    OMPEvalRunner,
    extract_predicted_ordering,
    extract_predicted_time_range,
    parse_omp_stream,
)

runner = CliRunner()


def test_compute_tiou() -> None:
    # Exact match
    assert compute_tiou((10.0, 20.0), (10.0, 20.0)) == 1.0
    # Complete non-overlap
    assert compute_tiou((0.0, 5.0), (10.0, 20.0)) == 0.0
    # Partial overlap: [10, 20] & [15, 25] -> inter=5, union=15 -> 0.3333
    assert compute_tiou((10.0, 20.0), (15.0, 25.0)) == 0.3333
    # Subsumed: [10, 30] & [15, 25] -> inter=10, union=20 -> 0.5
    assert compute_tiou((10.0, 30.0), (15.0, 25.0)) == 0.5
    # Unsorted input
    assert compute_tiou((20.0, 10.0), (15.0, 25.0)) == 0.3333


def test_compute_boundary_mae() -> None:
    assert compute_boundary_mae((10.0, 20.0), (10.0, 20.0)) == 0.0
    # Pred: [12.0, 22.0], GT: [10.0, 20.0] -> (|12-10| + |22-20|)/2 = 2.0
    assert compute_boundary_mae((12.0, 22.0), (10.0, 20.0)) == 2.0


def test_compute_kendall_tau() -> None:
    # Perfect alignment
    assert compute_kendall_tau(["e1", "e2", "e3"], ["e1", "e2", "e3"]) == 1.0
    # Completely reversed
    assert compute_kendall_tau(["e3", "e2", "e1"], ["e1", "e2", "e3"]) == -1.0
    # One pair inverted: e1, e3, e2 vs e1, e2, e3 (pairs: (e1,e2) conc, (e1,e3) conc, (e2,e3) disc) -> (2-1)/3 = 0.3333
    assert compute_kendall_tau(["e1", "e3", "e2"], ["e1", "e2", "e3"]) == 0.3333
    # Single element
    assert compute_kendall_tau(["e1"], ["e1"]) == 1.0


def test_check_hallucination_rejection() -> None:
    assert (
        check_hallucination_rejection("The event did not occur in this video.") is True
    )
    assert check_hallucination_rejection("Status: EVENT_ABSENT.") is True
    assert (
        check_hallucination_rejection("No evidence was found in the visual timeline.")
        is True
    )
    assert (
        check_hallucination_rejection(
            "The event occurred at 01:23 when the speaker smiled."
        )
        is False
    )


def test_extract_predicted_time_range() -> None:
    assert extract_predicted_time_range("The event occurs at [14.5, 32.0].") == (
        14.5,
        32.0,
    )
    assert extract_predicted_time_range("Range: [10 - 25]") == (10.0, 25.0)
    assert extract_predicted_time_range("Occurred from 00:01:15 - 00:02:00") == (
        75.0,
        120.0,
    )
    assert extract_predicted_time_range("No timestamps here") is None


def test_extract_predicted_ordering() -> None:
    candidates = ["e1", "e2", "e3"]
    assert extract_predicted_ordering('["e2", "e1", "e3"]', candidates) == [
        "e2",
        "e1",
        "e3",
    ]
    assert extract_predicted_ordering(
        "First was e2, followed by e1, then e3", candidates
    ) == ["e2", "e1", "e3"]
    assert extract_predicted_ordering("Irrelevant text without IDs", candidates) == []


def test_parse_omp_stream() -> None:
    sample_lines = [
        json.dumps({"type": "session", "id": "test_sess"}),
        json.dumps(
            {
                "role": "toolResult",
                "timestamp": 1789090507674,
                "details": {
                    "xdev": {
                        "tool": "mcp__vidscope_get_job_status",
                        "args": {"job_id": "job_99"},
                        "inner": {
                            "serverName": "vidscope",
                            "mcpToolName": "get_job_status",
                            "isError": True,
                            "rawContent": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "ok": False,
                                            "code": "ARTIFACT_NOT_FOUND",
                                            "next_action": "analyze_video",
                                        }
                                    ),
                                }
                            ],
                        },
                    }
                },
                "isError": True,
            }
        ),
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Job job_99 was not found."}],
                },
            }
        ),
    ]

    traces, final_text, raw = parse_omp_stream(sample_lines)
    assert len(traces) == 1
    assert traces[0].tool_name == "get_job_status"
    assert traces[0].arguments == {"job_id": "job_99"}
    assert traces[0].is_error is True
    assert traces[0].error_code == "ARTIFACT_NOT_FOUND"
    assert traces[0].next_action == "analyze_video"
    assert final_text == "Job job_99 was not found."


def test_load_benchmark_dataset(tmp_path: Path) -> None:
    yaml_content = """
video_id: "test_vid"
metadata:
  title: "Test Video"
  source: "https://example.com/video.mp4"
  duration_seconds: 120.0
  domain: "screencast"

events:
  - id: "e1"
    description: "Intro slide"
    time_range: [0.0, 10.0]
  - id: "e2"
    description: "Main tutorial"
    time_range: [10.0, 90.0]

absent_events:
  - id: "neg1"
    description: "Cooking demonstration"

ordering_tasks:
  - task_id: "order_all"
    event_ids: ["e1", "e2"]

retrieval_queries:
  - query: "What was shown at start?"
    ground_truth_answer: "Intro slide"
"""
    dataset_file = tmp_path / "dataset.yaml"
    dataset_file.write_text(yaml_content, encoding="utf-8")

    ds = load_benchmark_dataset(str(dataset_file))
    assert ds.video_id == "test_vid"
    assert ds.title == "Test Video"
    assert ds.duration_seconds == 120.0
    assert len(ds.events) == 2
    assert len(ds.absent_events) == 1
    assert len(ds.ordering_tasks) == 1
    assert len(ds.retrieval_queries) == 1


@patch("subprocess.run")
def test_omp_runner_run_prompt(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        stdout=json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Event happens at [10.0, 20.0]"}
                    ],
                },
            }
        ),
        stderr="",
        returncode=0,
    )

    runner_inst = OMPEvalRunner(
        omp_bin=sys.executable,
        thinking="off",
    )
    res = runner_inst.run_prompt("Test prompt")
    assert res.final_text == "Event happens at [10.0, 20.0]"
    assert res.scorecard.total_invocations == 0


@patch("subprocess.run")
def test_omp_runner_run_dataset(mock_run: MagicMock) -> None:
    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> MagicMock:
        prompt_arg = cmd[-1]
        if "Identify the starting and ending timestamp" in prompt_arg:
            text = "Found at [5.0, 15.0]"
        elif "Order the following events" in prompt_arg:
            text = 'Order: ["e1", "e2"]'
        elif "explicitly state 'EVENT_ABSENT'" in prompt_arg:
            text = "This is EVENT_ABSENT."
        else:
            text = "Intro slide is shown."

        out_lines = [
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    },
                }
            )
        ]
        return MagicMock(stdout="\n".join(out_lines), stderr="", returncode=0)

    mock_run.side_effect = fake_subprocess_run

    dataset = BenchmarkDataset(
        video_id="v1",
        source="test.mp4",
        duration_seconds=60.0,
        events=[
            BenchmarkEvent(id="e1", description="Intro", time_range=(0.0, 10.0)),
            BenchmarkEvent(id="e2", description="Outro", time_range=(50.0, 60.0)),
        ],
        absent_events=[AbsentEvent(id="neg1", description="Alien attack")],
        ordering_tasks=[OrderingTask(task_id="t1", event_ids=["e1", "e2"])],
        retrieval_queries=[
            RetrievalQuery(query="What is at start?", ground_truth_answer="Intro slide")
        ],
    )

    runner_inst = OMPEvalRunner(
        omp_bin=sys.executable,
        thinking="off",
    )
    report = runner_inst.run_dataset(dataset)

    assert report.semantic_scorecard.localization_evals == 2
    assert report.semantic_scorecard.absent_rejections_correct == 1
    assert report.semantic_scorecard.retrieval_matches == 1
    assert "Benchmark Evaluation Report" in report.to_markdown()


def test_cli_eval_help_and_validation() -> None:
    # Missing prompt, dataset, and protocol-suite
    res = runner.invoke(app, ["eval"])
    assert res.exit_code == 1
    assert "Must provide either --prompt, --dataset, or --protocol-suite" in res.output


@patch("subprocess.run")
def test_omp_runner_run_repeated_prompt(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        stdout=json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Result text"}],
                },
            }
        ),
        stderr="",
        returncode=0,
    )
    runner_inst = OMPEvalRunner(
        omp_bin=sys.executable,
        thinking="off",
    )
    scorecard, results = runner_inst.run_repeated_prompt("Check prompt", runs=3)
    assert len(results) == 3
    assert scorecard.total_invocations == 0


@patch("subprocess.run")
def test_omp_runner_run_protocol_suite(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        stdout=json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Result"}],
                },
            }
        ),
        stderr="",
        returncode=0,
    )
    runner_inst = OMPEvalRunner(
        omp_bin=sys.executable,
        thinking="off",
    )
    report = runner_inst.run_protocol_suite(total_runs=10)
    assert report.total_runs == 10
    assert len(report.scenario_stats) > 0
    assert "Protocol Hygiene Benchmark Report (Phase 1)" in report.to_markdown()
    assert "Scenario Performance & Failure Rates" in report.to_markdown()
    assert (
        "| Scenario ID | Runs | Failed Runs | Failure Rate (%) |"
        in report.to_markdown()
    )


@patch("vidscope.eval_runner.OMPEvalRunner.run_protocol_suite")
def test_cli_eval_protocol_suite_mocked(mock_suite: MagicMock) -> None:
    from vidscope.eval_harness import ProtocolScorecard
    from vidscope.eval_runner import ProtocolSuiteReport

    mock_suite.return_value = ProtocolSuiteReport(
        total_runs=10,
        scorecard=ProtocolScorecard(total_invocations=10, successful_invocations=10),
        scenario_counts={"status_missing": 2},
        total_duration_seconds=5.2,
    )

    res = runner.invoke(
        app, ["eval", "--protocol-suite", "--runs", "10", "--concurrency", "4"]
    )
    assert res.exit_code == 0
    assert "Protocol Hygiene Benchmark Report" in res.output


@patch("subprocess.run")
def test_omp_runner_concurrency_execution(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        stdout=json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "OK"}],
                },
            }
        ),
        stderr="",
        returncode=0,
    )
    runner_inst = OMPEvalRunner(
        omp_bin=sys.executable,
        thinking="off",
    )
    # Repeated prompt with concurrency=3
    card, results = runner_inst.run_repeated_prompt("Prompt", runs=6, concurrency=3)
    assert len(results) == 6
    assert card.total_invocations == 0

    # Protocol suite with concurrency=2
    report = runner_inst.run_protocol_suite(total_runs=10, concurrency=2)
    assert report.total_runs == 10


@patch("subprocess.run")
def test_omp_runner_mock_async_env(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        stdout=json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done"}],
                },
            }
        ),
        stderr="",
        returncode=0,
    )
    runner_inst = OMPEvalRunner(
        omp_bin=sys.executable,
        thinking="off",
        mock_mcp=True,
        mock_async="random",
    )
    runner_inst.run_prompt("Test prompt")
    mock_run.assert_called_once()
    passed_env = mock_run.call_args[1]["env"]
    assert passed_env["VIDSCOPE_MOCK_MCP"] == "1"
    assert passed_env["VIDSCOPE_MOCK_ASYNC"] == "random"


def test_omp_runner_explicit_missing_binary() -> None:
    with pytest.raises(FileNotFoundError, match="OMP binary not found at"):
        OMPEvalRunner(omp_bin="/nonexistent/custom/omp_bin")


def test_omp_runner_missing_in_path_on_run() -> None:
    with (
        patch("shutil.which", return_value=None),
        patch("pathlib.Path.is_file", return_value=False),
    ):
        runner_inst = OMPEvalRunner()
        with pytest.raises(FileNotFoundError, match="OMP binary 'omp' not found"):
            runner_inst.run_prompt("Test")


def test_cli_eval_missing_binary_error() -> None:
    res = runner.invoke(
        app, ["eval", "--prompt", "test", "--omp-bin", "/nonexistent/custom/omp_bin"]
    )
    assert res.exit_code == 1
    assert "Error: OMP binary not found at '/nonexistent/custom/omp_bin'" in res.output
