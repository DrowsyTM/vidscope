"""OMP benchmark runner for agent evaluation across Vidscope MCP tools."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .eval_harness import (
    BenchmarkDataset,
    InvocationTrace,
    ProtocolScorecard,
    SemanticScorecard,
    check_hallucination_rejection,
    compute_boundary_mae,
    compute_kendall_tau,
    compute_tiou,
    evaluate_invocation_traces,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvalRunResult:
    """Outcome of running a single prompt through OMP."""

    prompt: str
    final_text: str
    traces: list[InvocationTrace]
    scorecard: ProtocolScorecard
    duration_seconds: float
    raw_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class DatasetEvaluationReport:
    """Comprehensive evaluation report across a benchmark dataset."""

    dataset: BenchmarkDataset
    protocol_scorecard: ProtocolScorecard
    semantic_scorecard: SemanticScorecard
    task_results: list[dict[str, Any]] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    def to_markdown(self) -> str:
        lines = [
            f"# Benchmark Evaluation Report: {self.dataset.title or self.dataset.video_id}",
            "",
            f"- **Video ID**: `{self.dataset.video_id}`",
            f"- **Source**: `{self.dataset.source}`",
            f"- **Domain**: `{self.dataset.domain}`",
            f"- **Duration**: `{self.dataset.duration_seconds}s`",
            f"- **Total Eval Time**: `{self.total_duration_seconds:.2f}s`",
            "",
            "## 1. MCP Protocol Hygiene Scorecard",
            "",
            self.protocol_scorecard.to_markdown_table(),
            "",
            "## 2. Timeline Semantic Comprehension Scorecard",
            "",
            self.semantic_scorecard.to_markdown_table(),
            "",
            "## 3. Individual Task Outcomes",
            "",
            "| Task Type | Task ID / Query | Result | Details |",
            "|:---|:---|:---|:---|",
        ]

        for t in self.task_results:
            task_type = t.get("type", "unknown")
            task_id = t.get("id", "-")
            status = "PASS" if t.get("passed", False) else "FAIL"
            details = t.get("details", "")
            lines.append(f"| `{task_type}` | {task_id} | **{status}** | {details} |")

        lines.append("")
        return "\n".join(lines)


@dataclass(slots=True)
class ProtocolScenario:
    """Standardized test scenario evaluating MCP protocol adherence."""

    id: str
    name: str
    prompt: str
    description: str = ""


PROTOCOL_SUITE_SCENARIOS: list[ProtocolScenario] = [
    ProtocolScenario(
        id="info_metadata",
        name="Preflight Video Inspection",
        prompt="What is the duration, uploader, and chapter breakdown of 'mock://presentation.mp4'?",
        description="Tests fast preflight metadata discovery using get_video_info.",
    ),
    ProtocolScenario(
        id="sync_view_frame",
        name="Synchronous Analysis & Keyframe Inspection",
        prompt="Analyze the first 30 seconds of 'mock://presentation.mp4' and inspect the slide keyframe image from that segment.",
        description="Tests multi-step synchronous analyze_video followed by view_frame.",
    ),
    ProtocolScenario(
        id="sync_search",
        name="Synchronous Analysis & Transcript Search",
        prompt="Analyze 'mock://presentation.mp4' and find what the speaker says about 'Speaker discusses topic'.",
        description="Tests multi-step synchronous analyze_video followed by search_video.",
    ),
    ProtocolScenario(
        id="sync_transcript",
        name="Synchronous Analysis & Transcript Export",
        prompt="Analyze 'mock://presentation.mp4' and retrieve the full spoken dialogue transcript for the first 30 seconds.",
        description="Tests multi-step synchronous analyze_video followed by get_transcript.",
    ),
    ProtocolScenario(
        id="async_search_workflow",
        name="Asynchronous Analysis, Polling & Search",
        prompt="Analyze the video 'mock://async_presentation.mp4' and find what the speaker says about section 0.",
        description="Tests async analyze_video handoff, agent-directed get_job_status polling, and search_video.",
    ),
    ProtocolScenario(
        id="async_frame_workflow",
        name="Asynchronous Analysis, Polling & Keyframe Inspection",
        prompt="Analyze the video 'mock://async_presentation.mp4' and view the presentation slide keyframe shown in the first section.",
        description="Tests async analyze_video handoff, agent-directed get_job_status polling, and view_frame.",
    ),
]


@dataclass(slots=True)
class ScenarioStats:
    """Performance and failure metrics for a specific protocol scenario."""

    scenario_id: str
    total_runs: int = 0
    failed_runs: int = 0
    failure_rate_percentage: float = 0.0
    form_errors: int = 0
    order_errors: int = 0
    cadence_violations: int = 0
    avg_duration_seconds: float = 0.0


@dataclass(slots=True)
class ProtocolSuiteReport:
    """Evaluation report across multiple protocol hygiene test runs."""

    total_runs: int
    scorecard: ProtocolScorecard
    scenario_counts: dict[str, int]
    total_duration_seconds: float = 0.0
    scenario_stats: dict[str, ScenarioStats] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            "# Protocol Hygiene Benchmark Report (Phase 1)",
            "",
            f"- **Total Invocations Evaluated**: `{self.scorecard.total_invocations}`",
            f"- **Total Agent Runs**: `{self.total_runs}`",
            f"- **Total Execution Duration**: `{self.total_duration_seconds:.2f}s`",
            "",
            "## 1. Protocol Hygiene Scorecard",
            "",
            self.scorecard.to_markdown_table(),
            "",
            "## 2. Tool Usage Statistics",
            "",
            self.scorecard.tool_usage_markdown_table(total_runs=self.total_runs),
            "",
            "## 3. Metric Breakdown",
            "",
            "- **Tool Call Error Rate (TCER)**: Percentage of tool invocations rejected due to schema/argument validation failures.",
            "- **Tool Sequencing Error Rate (TSER)**: Percentage of calls violating tool lifecycle prerequisites.",
            "- **Polling Cadence Violations**: Status polls executed prematurely without respecting estimated remaining time.",
            "",
        ]
        if self.scenario_stats:
            lines.extend(
                [
                    "## 4. Scenario Performance & Failure Rates",
                    "",
                    "| Scenario ID | Runs | Failed Runs | Failure Rate (%) | Form Errors | Order Errors | Cadence Violations | Avg Latency |",
                    "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
                ]
            )
            for sc_id, st in sorted(self.scenario_stats.items()):
                lines.append(
                    f"| `{sc_id}` | {st.total_runs} | {st.failed_runs} | {st.failure_rate_percentage:.1f}% | {st.form_errors} | {st.order_errors} | {st.cadence_violations} | {st.avg_duration_seconds:.2f}s |"
                )
        else:
            lines.extend(
                [
                    "## 4. Scenario Distribution",
                    "",
                    "| Scenario ID | Iterations |",
                    "|:---|:---|",
                ]
            )
            for sc_id, count in sorted(self.scenario_counts.items()):
                lines.append(f"| `{sc_id}` | {count} |")
        lines.append("")
        return "\n".join(lines)


def _extract_trace_from_tool_result(
    msg: dict[str, Any],
    call_start_timestamp_ms: int | None = None,
) -> InvocationTrace | None:
    """Extract InvocationTrace from a toolResult message if it pertains to Vidscope."""
    role = msg.get("role")
    if role != "toolResult":
        return None

    details = msg.get("details", {})
    xdev = details.get("xdev", {}) if isinstance(details, dict) else {}
    inner = xdev.get("inner", {}) if isinstance(xdev, dict) else {}
    server_name = inner.get("serverName") if isinstance(inner, dict) else None
    tool_raw = xdev.get("tool", "") if isinstance(xdev, dict) else ""

    is_vidscope = (
        server_name == "vidscope"
        or "vidscope" in tool_raw
        or tool_raw.startswith("mcp__vidscope_")
    )
    if not is_vidscope:
        return None

    mcp_tool_name = (
        inner.get("mcpToolName")
        if isinstance(inner, dict) and inner.get("mcpToolName")
        else tool_raw.replace("mcp__vidscope_", "")
    )
    args = xdev.get("args") or {}
    timestamp_ms = msg.get("timestamp") or 0
    timestamp_seconds = float(timestamp_ms) / 1000.0 if timestamp_ms else 0.0
    is_error = bool(
        msg.get("isError")
        or (inner.get("isError") if isinstance(inner, dict) else False)
    )

    response_payload: Any = None
    raw_content = (
        inner.get("rawContent")
        if isinstance(inner, dict) and inner.get("rawContent")
        else msg.get("content", [])
    )
    raw_text = ""
    if isinstance(raw_content, list) and raw_content:
        for block in raw_content:
            if isinstance(block, dict) and block.get("type") == "text":
                raw_text += block.get("text", "") + " "
        first_text = raw_content[0].get("text", "")
        if first_text.startswith("Error: "):
            first_text = first_text[len("Error: ") :]
        try:
            response_payload = json.loads(first_text)
        except (json.JSONDecodeError, TypeError):
            response_payload = first_text
    elif isinstance(raw_content, str):
        raw_text = raw_content
        try:
            response_payload = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            response_payload = raw_content

    response_words = len(raw_text.split()) if raw_text else 0
    response_tokens = max(1, len(raw_text.strip()) // 4) if raw_text.strip() else 0
    latency_ms = 0.0
    if call_start_timestamp_ms is not None and timestamp_ms:
        latency_ms = max(0.0, float(timestamp_ms - call_start_timestamp_ms))

    error_code: str | None = None
    next_action: str | None = None
    estimated_remaining_seconds: float | None = None

    if isinstance(response_payload, dict):
        error_code = response_payload.get("code")
        next_action = response_payload.get("next_action")
        if "estimated_remaining_seconds" in response_payload:
            estimated_remaining_seconds = float(
                response_payload["estimated_remaining_seconds"]
            )
        diagnostics = response_payload.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            if not next_action:
                next_action = diagnostics.get("next_action")
            if "estimated_remaining_seconds" in diagnostics:
                estimated_remaining_seconds = float(
                    diagnostics["estimated_remaining_seconds"]
                )

    return InvocationTrace(
        tool_name=str(mcp_tool_name),
        arguments=args if isinstance(args, dict) else {"raw": args},
        timestamp_seconds=timestamp_seconds,
        response=response_payload,
        is_error=is_error,
        error_code=error_code,
        next_action=next_action,
        estimated_remaining_seconds=estimated_remaining_seconds,
        latency_ms=latency_ms,
        response_words=response_words,
        response_tokens=response_tokens,
    )


def parse_omp_stream(
    lines: Iterable[str],
) -> tuple[list[InvocationTrace], str, list[dict[str, Any]]]:
    """Parse OMP JSON stream, extracting Vidscope MCP invocation traces and final response."""
    traces: list[InvocationTrace] = []
    seen_call_ids: set[str] = set()
    tool_call_timestamps: dict[str, int] = {}
    final_text_chunks: list[str] = []
    raw_events: list[dict[str, Any]] = []

    def _maybe_add_trace(msg_dict: dict[str, Any]) -> None:
        call_id = str(msg_dict.get("toolCallId") or msg_dict.get("timestamp") or "")
        if call_id in seen_call_ids:
            return
        start_ts = tool_call_timestamps.get(str(msg_dict.get("toolCallId") or ""))
        trace = _extract_trace_from_tool_result(
            msg_dict, call_start_timestamp_ms=start_ts
        )
        if trace is not None:
            seen_call_ids.add(call_id)
            traces.append(trace)

    for raw_line in lines:
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        raw_events.append(event)
        event_type = event.get("type")

        # 1. Capture assistant messages & toolCall start timestamps
        if event_type in {"turn_end", "message_end", "agent_end"}:
            msg = event.get("message")
            if isinstance(msg, dict):
                if msg.get("role") == "assistant":
                    content_items = msg.get("content", [])
                    for item in content_items:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text = item.get("text", "")
                                if text:
                                    final_text_chunks.append(text)
                            elif item.get("type") == "toolCall":
                                cid = item.get("id")
                                ts = msg.get("timestamp") or event.get("timestamp")
                                if cid and ts:
                                    tool_call_timestamps[str(cid)] = int(ts)
                elif msg.get("role") == "toolResult":
                    _maybe_add_trace(msg)

            if event_type == "agent_end":
                messages = event.get("messages", [])
                for m in messages:
                    if isinstance(m, dict):
                        if m.get("role") == "assistant":
                            for item in m.get("content", []):
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        text = item.get("text", "")
                                        if text:
                                            final_text_chunks.append(text)
                                    elif item.get("type") == "toolCall":
                                        cid = item.get("id")
                                        ts = m.get("timestamp") or event.get(
                                            "timestamp"
                                        )
                                        if cid and ts:
                                            tool_call_timestamps[str(cid)] = int(ts)
                        elif m.get("role") == "toolResult":
                            _maybe_add_trace(m)

        # 2. Check top-level toolResult
        if event.get("role") == "toolResult":
            _maybe_add_trace(event)

    final_text = final_text_chunks[-1] if final_text_chunks else ""
    return traces, final_text, raw_events


def extract_predicted_time_range(text: str) -> tuple[float, float] | None:
    """Extract [start_seconds, end_seconds] from LLM response text."""
    # Try bracketed format: [12.5, 34.0]
    match = re.search(
        r"\[\s*(\d+(?:\.\d+)?)\s*(?:,\s*|\s*-\s*|\s+to\s+)\s*(\d+(?:\.\d+)?)\s*\]",
        text,
    )
    if match:
        return float(match.group(1)), float(match.group(2))

    # Try formatted range: 00:01:23 - 00:01:45
    time_match = re.search(
        r"(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:-|to)\s*(\d{1,2}:\d{2}(?::\d{2})?)", text
    )
    if time_match:
        return _parse_timestamp(time_match.group(1)), _parse_timestamp(
            time_match.group(2)
        )

    return None


def _parse_timestamp(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def extract_predicted_ordering(text: str, candidate_ids: list[str]) -> list[str]:
    """Extract ordered event IDs from LLM response text."""
    # Try JSON array first
    json_match = re.search(r"\[\s*(?:\"[^\"]+\"(?:,\s*)?)+\s*\]", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list):
                extracted = [str(x) for x in parsed if str(x) in candidate_ids]
                if extracted:
                    return extracted
        except json.JSONDecodeError:
            pass

    # Match tokens matching candidate IDs in order of first appearance
    seen: list[str] = []
    pattern = r"\b(" + "|".join(re.escape(cid) for cid in candidate_ids) + r")\b"
    for match in re.finditer(pattern, text):
        cid = match.group(1)
        if cid not in seen:
            seen.append(cid)
    return seen


class OMPEvalRunner:
    """Evaluation runner that orchestrates headless agent queries via OMP."""

    def __init__(
        self,
        omp_bin: str | None = None,
        thinking: str = "off",
        approval_mode: str = "yolo",
        model: str | None = None,
        timeout_seconds: float = 180.0,
        mock_mcp: bool = True,
        mock_async: str | None = None,
    ) -> None:
        resolved_bin = omp_bin or shutil.which("omp") or "/home/dima/.bun/bin/omp"
        if not Path(resolved_bin).is_file():
            raise FileNotFoundError(
                f"OMP binary not found at '{resolved_bin}'. Please install OMP or provide valid path."
            )
        self.omp_bin = resolved_bin
        self.thinking = thinking
        self.approval_mode = approval_mode
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.mock_mcp = mock_mcp
        self.mock_async = mock_async

    def run_prompt(self, prompt: str) -> EvalRunResult:
        """Run a single prompt through OMP and extract invocation traces and response."""
        import os
        import time

        cmd = [
            self.omp_bin,
            "-p",
            "--mode=json",
            f"--thinking={self.thinking}",
            f"--approval-mode={self.approval_mode}",
        ]
        if self.model:
            cmd.append(f"--model={self.model}")
        cmd.append(prompt)

        env = {**os.environ}
        if self.mock_mcp:
            env["VIDSCOPE_MOCK_MCP"] = "1"
        if self.mock_async is not None:
            env["VIDSCOPE_MOCK_ASYNC"] = str(self.mock_async)

        start_time = time.perf_counter()
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        duration = time.perf_counter() - start_time

        traces, final_text, raw_events = parse_omp_stream(proc.stdout.splitlines())
        scorecard = evaluate_invocation_traces(traces)

        return EvalRunResult(
            prompt=prompt,
            final_text=final_text,
            traces=traces,
            scorecard=scorecard,
            duration_seconds=round(duration, 3),
            raw_events=raw_events,
        )

    def run_repeated_prompt(
        self,
        prompt: str,
        runs: int = 1,
        concurrency: int = 1,
        on_run_complete: Callable[[int, int, EvalRunResult], None] | None = None,
    ) -> tuple[ProtocolScorecard, list[EvalRunResult]]:
        """Run a prompt repeatedly, aggregating all traces into a cumulative scorecard."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_traces: list[InvocationTrace] = []
        results: list[EvalRunResult] = []

        if concurrency <= 1 or runs <= 1:
            for i in range(1, runs + 1):
                res = self.run_prompt(prompt)
                all_traces.extend(res.traces)
                results.append(res)
                if on_run_complete:
                    on_run_complete(i, runs, res)
        else:
            completed_count = 0
            with ThreadPoolExecutor(max_workers=min(concurrency, runs)) as executor:
                futures = [
                    executor.submit(self.run_prompt, prompt) for _ in range(runs)
                ]
                for fut in as_completed(futures):
                    res = fut.result()
                    all_traces.extend(res.traces)
                    results.append(res)
                    completed_count += 1
                    if on_run_complete:
                        on_run_complete(completed_count, runs, res)

        cumulative_scorecard = evaluate_invocation_traces(all_traces)
        return cumulative_scorecard, results

    def run_protocol_suite(
        self,
        total_runs: int = 100,
        concurrency: int = 1,
        scenarios: list[ProtocolScenario] | None = None,
        on_run_complete: (Callable[[str, int, int, EvalRunResult], None] | None) = None,
    ) -> ProtocolSuiteReport:
        """Run standard protocol scenarios distributed across total_runs with optional concurrency."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start_time = time.perf_counter()
        target_scenarios = scenarios or PROTOCOL_SUITE_SCENARIOS
        if not target_scenarios:
            target_scenarios = PROTOCOL_SUITE_SCENARIOS
        num_scenarios = len(target_scenarios)
        safe_total = max(1, total_runs)
        runs_per_scenario = safe_total // num_scenarios
        remainder = safe_total % num_scenarios

        all_traces: list[InvocationTrace] = []
        counts: dict[str, int] = {sc.id: 0 for sc in target_scenarios}
        scenario_results: dict[str, list[EvalRunResult]] = {
            sc.id: [] for sc in target_scenarios
        }
        completed = 0

        work_items: list[ProtocolScenario] = []
        for idx, sc in enumerate(target_scenarios):
            count = runs_per_scenario + (1 if idx < remainder else 0)
            for _ in range(count):
                work_items.append(sc)
        actual_total = len(work_items)

        if concurrency <= 1:
            for sc in work_items:
                res = self.run_prompt(sc.prompt)
                all_traces.extend(res.traces)
                counts[sc.id] += 1
                scenario_results[sc.id].append(res)
                completed += 1
                if on_run_complete:
                    on_run_complete(sc.id, completed, actual_total, res)
        else:
            with ThreadPoolExecutor(
                max_workers=min(concurrency, len(work_items))
            ) as executor:
                future_to_sc = {
                    executor.submit(self.run_prompt, sc.prompt): sc for sc in work_items
                }
                for fut in as_completed(future_to_sc):
                    sc = future_to_sc[fut]
                    res = fut.result()
                    all_traces.extend(res.traces)
                    counts[sc.id] += 1
                    scenario_results[sc.id].append(res)
                    completed += 1
                    if on_run_complete:
                        on_run_complete(sc.id, completed, actual_total, res)

        duration = time.perf_counter() - start_time
        scorecard = evaluate_invocation_traces(all_traces)

        stats: dict[str, ScenarioStats] = {}
        for sc_id, r_list in scenario_results.items():
            tot = len(r_list)
            if tot == 0:
                continue
            failed = sum(
                1
                for r in r_list
                if (
                    r.scorecard.schema_validation_errors > 0
                    or r.scorecard.sequencing_errors > 0
                    or r.scorecard.cadence_violations > 0
                )
            )
            form_errs = sum(r.scorecard.schema_validation_errors for r in r_list)
            order_errs = sum(r.scorecard.sequencing_errors for r in r_list)
            cadence_viols = sum(r.scorecard.cadence_violations for r in r_list)
            avg_dur = sum(r.duration_seconds for r in r_list) / tot
            stats[sc_id] = ScenarioStats(
                scenario_id=sc_id,
                total_runs=tot,
                failed_runs=failed,
                failure_rate_percentage=round((failed / tot * 100), 1),
                form_errors=form_errs,
                order_errors=order_errs,
                cadence_violations=cadence_viols,
                avg_duration_seconds=round(avg_dur, 2),
            )

        return ProtocolSuiteReport(
            total_runs=actual_total,
            scorecard=scorecard,
            scenario_counts=counts,
            total_duration_seconds=round(duration, 2),
            scenario_stats=stats,
        )

    def run_dataset(self, dataset: BenchmarkDataset) -> DatasetEvaluationReport:
        """Run all benchmark tasks in dataset and compile protocol & semantic scorecards."""
        import time

        overall_start = time.perf_counter()
        all_traces: list[InvocationTrace] = []
        task_outcomes: list[dict[str, Any]] = []

        tiou_scores: list[float] = []
        mae_scores: list[float] = []
        tau_scores: list[float] = []
        absent_correct = 0
        retrieval_matches = 0

        # 1. Temporal Localization Tasks
        for event in dataset.events:
            prompt = (
                f"Using the Vidscope MCP tools, analyze the video source '{dataset.source}'. "
                f"Identify the starting and ending timestamp in seconds for this event: '{event.description}'. "
                f"Provide your final answer with the exact numeric range in brackets: [start_seconds, end_seconds]."
            )
            result = self.run_prompt(prompt)
            all_traces.extend(result.traces)

            pred_range = extract_predicted_time_range(result.final_text)
            if pred_range is not None:
                tiou = compute_tiou(pred_range, event.time_range)
                mae = compute_boundary_mae(pred_range, event.time_range)
                tiou_scores.append(tiou)
                mae_scores.append(mae)
                passed = tiou >= 0.5
                task_outcomes.append(
                    {
                        "type": "temporal_localization",
                        "id": event.id,
                        "passed": passed,
                        "details": f"Pred: [{pred_range[0]}, {pred_range[1]}], GT: [{event.time_range[0]}, {event.time_range[1]}], tIoU: {tiou}, MAE: {mae}s",
                    }
                )
            else:
                tiou_scores.append(0.0)
                mae_scores.append(dataset.duration_seconds)
                task_outcomes.append(
                    {
                        "type": "temporal_localization",
                        "id": event.id,
                        "passed": False,
                        "details": "Failed to parse predicted time range [start, end]",
                    }
                )

        # 2. Event Ordering Tasks
        for ordering in dataset.ordering_tasks:
            events_in_task = [e for e in dataset.events if e.id in ordering.event_ids]
            # Provide descriptions with scrambled order
            scrambled = sorted(events_in_task, key=lambda x: x.description)
            desc_list = "\n".join(f"- ID `{e.id}`: {e.description}" for e in scrambled)
            prompt = (
                f"Using the Vidscope MCP tools, analyze the video source '{dataset.source}'. "
                f"Order the following events chronologically from first to last as they appear in the video:\n{desc_list}\n"
                f'Return your final answer strictly as a JSON array of event IDs, e.g. ["{scrambled[0].id}", ...].'
            )
            result = self.run_prompt(prompt)
            all_traces.extend(result.traces)

            # True chronological order
            gt_order = [
                e.id for e in sorted(events_in_task, key=lambda x: x.time_range[0])
            ]
            pred_order = extract_predicted_ordering(
                result.final_text, ordering.event_ids
            )
            tau = compute_kendall_tau(pred_order, gt_order)
            tau_scores.append(tau)
            passed = tau >= 0.9
            task_outcomes.append(
                {
                    "type": "event_ordering",
                    "id": ordering.task_id,
                    "passed": passed,
                    "details": f"Pred: {pred_order}, GT: {gt_order}, τ: {tau}",
                }
            )

        # 3. Absent Event Rejection Tasks
        for absent in dataset.absent_events:
            prompt = (
                f"Using the Vidscope MCP tools, inspect the video '{dataset.source}'. "
                f"At what time does this occur: '{absent.description}'? "
                f"If this event did NOT occur in the video, explicitly state 'EVENT_ABSENT'."
            )
            result = self.run_prompt(prompt)
            all_traces.extend(result.traces)

            rejected = check_hallucination_rejection(result.final_text)
            if rejected:
                absent_correct += 1
            task_outcomes.append(
                {
                    "type": "absent_rejection",
                    "id": absent.id,
                    "passed": rejected,
                    "details": f"Rejection Detected: {rejected}",
                }
            )

        # 4. Factual Retrieval Queries
        for ret in dataset.retrieval_queries:
            prompt = (
                f"Using the Vidscope MCP tools, inspect the video '{dataset.source}'. "
                f"Answer this question factually: '{ret.query}'."
            )
            result = self.run_prompt(prompt)
            all_traces.extend(result.traces)

            gt_norm = ret.ground_truth_answer.lower()
            resp_norm = result.final_text.lower()
            matched = gt_norm in resp_norm or any(
                token in resp_norm for token in gt_norm.split() if len(token) > 4
            )
            if matched:
                retrieval_matches += 1
            task_outcomes.append(
                {
                    "type": "factual_retrieval",
                    "id": ret.query[:30] + "...",
                    "passed": matched,
                    "details": f"Match: {matched}",
                }
            )

        total_duration = time.perf_counter() - overall_start

        # Aggregate Protocol & Semantic scorecards
        protocol_card = evaluate_invocation_traces(all_traces)
        semantic_card = SemanticScorecard(
            localization_evals=len(dataset.events),
            localization_tiou_avg=round(
                sum(tiou_scores) / len(tiou_scores) if tiou_scores else 0.0, 3
            ),
            localization_mae_avg=round(
                sum(mae_scores) / len(mae_scores) if mae_scores else 0.0, 2
            ),
            ordering_evals=len(dataset.ordering_tasks),
            ordering_tau_avg=round(
                sum(tau_scores) / len(tau_scores) if tau_scores else 0.0, 3
            ),
            absent_evals=len(dataset.absent_events),
            absent_rejections_correct=absent_correct,
            retrieval_evals=len(dataset.retrieval_queries),
            retrieval_matches=retrieval_matches,
        )

        return DatasetEvaluationReport(
            dataset=dataset,
            protocol_scorecard=protocol_card,
            semantic_scorecard=semantic_card,
            task_results=task_outcomes,
            total_duration_seconds=round(total_duration, 2),
        )
