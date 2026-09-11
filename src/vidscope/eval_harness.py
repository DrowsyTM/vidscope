"""Protocol hygiene evaluation and scoring for agent-MCP interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class InvocationTrace:
    """Record of a single MCP tool invocation by an agent."""

    tool_name: str
    arguments: dict[str, Any]
    timestamp_seconds: float
    response: Any
    is_error: bool = False
    error_code: str | None = None
    next_action: str | None = None
    estimated_remaining_seconds: float | None = None
    latency_ms: float = 0.0
    response_words: int = 0
    response_tokens: int = 0
    is_form_error: bool = False
    is_order_error: bool = False


@dataclass(slots=True)
class ProtocolScorecard:
    """Quantitative scorecard of agent protocol and tool-calling hygiene."""

    total_invocations: int = 0
    schema_validation_errors: int = 0
    sequencing_errors: int = 0
    cadence_violations: int = 0
    successful_invocations: int = 0
    traces: list[InvocationTrace] = field(default_factory=list)

    @property
    def tcer(self) -> float:
        """Tool Call Error Rate (validation failure percentage)."""
        if self.total_invocations == 0:
            return 0.0
        return round(self.schema_validation_errors / self.total_invocations, 4)

    @property
    def tser(self) -> float:
        """Tool Sequencing Error Rate (lifecycle violation percentage)."""
        if self.total_invocations == 0:
            return 0.0
        return round(self.sequencing_errors / self.total_invocations, 4)

    @property
    def cadence_violation_rate(self) -> float:
        """Rate of status polls executed faster than estimated remaining time."""
        status_polls = sum(1 for t in self.traces if t.tool_name == "get_job_status")
        if status_polls == 0:
            return 0.0
        return round(self.cadence_violations / status_polls, 4)

    def tool_counts(self) -> dict[str, int]:
        """Count total invocations per tool."""
        counts: dict[str, int] = {}
        for t in self.traces:
            counts[t.tool_name] = counts.get(t.tool_name, 0) + 1
        return counts

    def tool_error_counts(self) -> dict[str, int]:
        """Count error invocations per tool."""
        counts: dict[str, int] = {}
        for t in self.traces:
            if t.is_error:
                counts[t.tool_name] = counts.get(t.tool_name, 0) + 1
        return counts

    def tool_usage_stats(self, total_runs: int = 1) -> list[dict[str, Any]]:
        """Compute usage statistics per tool including average calls per run, latency, word/token counts, and form/order error rates."""
        safe_runs = max(1, total_runs)
        tools = sorted({t.tool_name for t in self.traces})
        stats: list[dict[str, Any]] = []
        for tool in tools:
            tool_traces = [t for t in self.traces if t.tool_name == tool]
            tot = len(tool_traces)
            form_errs = sum(1 for t in tool_traces if t.is_form_error)
            order_errs = sum(1 for t in tool_traces if t.is_order_error)
            avg_lat = (
                round(sum(t.latency_ms for t in tool_traces) / tot, 1)
                if tot > 0
                else 0.0
            )
            avg_words = (
                round(sum(t.response_words for t in tool_traces) / tot, 1)
                if tot > 0
                else 0.0
            )
            avg_tokens = (
                round(sum(t.response_tokens for t in tool_traces) / tot, 1)
                if tot > 0
                else 0.0
            )
            stats.append(
                {
                    "tool_name": tool,
                    "total_calls": tot,
                    "avg_calls_per_run": round(tot / safe_runs, 2),
                    "avg_latency_ms": avg_lat,
                    "avg_words": avg_words,
                    "avg_tokens": avg_tokens,
                    "form_errors": form_errs,
                    "form_error_rate_percentage": (
                        round((form_errs / tot * 100), 1) if tot > 0 else 0.0
                    ),
                    "order_errors": order_errs,
                    "order_error_rate_percentage": (
                        round((order_errs / tot * 100), 1) if tot > 0 else 0.0
                    ),
                }
            )
        return stats

    def tool_usage_markdown_table(self, total_runs: int = 1) -> str:
        """Render markdown table of tool usage, latency, payload size, and form/order error rates."""
        stats = self.tool_usage_stats(total_runs=total_runs)
        if not stats:
            return "_No tool invocations recorded._"
        lines = [
            "| Tool Name | Calls | Avg/Run | Latency (ms) | Words/Call | Est. Tokens | Form Err (%) | Order Err (%) |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for s in stats:
            lat_str = (
                f"{s['avg_latency_ms']:.1f} ms" if s["avg_latency_ms"] > 0 else "<1 ms"
            )
            lines.append(
                f"| `{s['tool_name']}` | {s['total_calls']} | {s['avg_calls_per_run']:.2f} | {lat_str} | {s['avg_words']:.0f} | ~{s['avg_tokens']:.0f} | {s['form_error_rate_percentage']:.1f}% | {s['order_error_rate_percentage']:.1f}% |"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_invocations": self.total_invocations,
            "successful_invocations": self.successful_invocations,
            "schema_validation_errors": self.schema_validation_errors,
            "sequencing_errors": self.sequencing_errors,
            "cadence_violations": self.cadence_violations,
            "tcer_percentage": round(self.tcer * 100, 2),
            "tser_percentage": round(self.tser * 100, 2),
            "cadence_violation_percentage": round(self.cadence_violation_rate * 100, 2),
            "tool_usage": self.tool_usage_stats(total_runs=1),
        }

    def to_markdown_table(self) -> str:
        status_polls = sum(1 for t in self.traces if t.tool_name == "get_job_status")
        lines = [
            "| Metric | Total | Violations | Rate (%) | Target |",
            "|:---|:---|:---|:---|:---|",
            f"| **Tool Call Error Rate (TCER)** | {self.total_invocations} | {self.schema_validation_errors} | {self.tcer * 100:.1f}% | 0.0% |",
            f"| **Tool Sequencing Error Rate (TSER)** | {self.total_invocations} | {self.sequencing_errors} | {self.tser * 100:.1f}% | 0.0% |",
            f"| **Polling Cadence Violations** | {status_polls} | {self.cadence_violations} | {self.cadence_violation_rate * 100:.1f}% | 0.0% |",
        ]
        return "\n".join(lines)


def evaluate_invocation_traces(traces: list[InvocationTrace]) -> ProtocolScorecard:
    """Evaluate an ordered list of tool call traces and compute protocol scorecard."""
    card = ProtocolScorecard(
        total_invocations=len(traces),
        traces=traces,
    )

    last_poll_time: float | None = None
    last_eta_seconds: float | None = None

    for trace in traces:
        if trace.is_error:
            # Check if this is a schema validation error (Form Error)
            if trace.error_code in {"INVALID_REQUEST", "VALIDATION_ERROR"}:
                # If error is due to in-flight processing or missing artifact, it's a sequencing error (Order Error)
                if trace.next_action in {"get_job_status", "analyze_video"}:
                    trace.is_order_error = True
                    card.sequencing_errors += 1
                else:
                    trace.is_form_error = True
                    card.schema_validation_errors += 1
            elif trace.error_code in {"ARTIFACT_NOT_FOUND"}:
                trace.is_order_error = True
                card.sequencing_errors += 1
            else:
                trace.is_form_error = True
                card.schema_validation_errors += 1
        else:
            card.successful_invocations += 1

        # Evaluate polling cadence
        if trace.tool_name in {"analyze_video", "get_job_status"}:
            if (
                trace.tool_name == "get_job_status"
                and last_poll_time is not None
                and last_eta_seconds is not None
            ):
                elapsed = trace.timestamp_seconds - last_poll_time
                # If agent polled prematurely (e.g. less than half the estimated remaining time)
                # and the ETA was > 1.0s, count as a cadence violation
                if last_eta_seconds > 1.0 and elapsed < (last_eta_seconds * 0.5):
                    card.cadence_violations += 1

            last_poll_time = trace.timestamp_seconds
            last_eta_seconds = trace.estimated_remaining_seconds

    return card


def compute_tiou(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    """Compute Temporal Intersection-over-Union between predicted and ground-truth intervals."""
    pred_start, pred_end = sorted((float(pred[0]), float(pred[1])))
    gt_start, gt_end = sorted((float(gt[0]), float(gt[1])))

    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 0.0:
        return 0.0
    return round(intersection / union, 4)


def compute_boundary_mae(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    """Compute Mean Absolute Error across start and end boundaries in seconds."""
    pred_start, pred_end = sorted((float(pred[0]), float(pred[1])))
    gt_start, gt_end = sorted((float(gt[0]), float(gt[1])))
    mae = (abs(pred_start - gt_start) + abs(pred_end - gt_end)) / 2.0
    return round(mae, 3)


def aggregate_protocol_scorecards(
    scorecards: list[ProtocolScorecard],
) -> ProtocolScorecard:
    """Aggregate individual run scorecards without cross-run cadence interference."""
    agg = ProtocolScorecard()
    for sc in scorecards:
        agg.total_invocations += sc.total_invocations
        agg.schema_validation_errors += sc.schema_validation_errors
        agg.sequencing_errors += sc.sequencing_errors
        agg.cadence_violations += sc.cadence_violations
        agg.successful_invocations += sc.successful_invocations
        agg.traces.extend(sc.traces)
    return agg


def compute_kendall_tau(pred_order: list[str], gt_order: list[str]) -> float:
    """Compute Kendall's tau rank correlation between predicted and true sequence."""
    if len(gt_order) <= 1:
        return 1.0 if pred_order == gt_order else -1.0

    total_gt_pairs = len(gt_order) * (len(gt_order) - 1) / 2.0
    pred_ranks = {item: idx for idx, item in enumerate(pred_order)}

    concordant = 0
    discordant = 0
    for i in range(len(gt_order)):
        for j in range(i + 1, len(gt_order)):
            item_a = gt_order[i]
            item_b = gt_order[j]
            if item_a in pred_ranks and item_b in pred_ranks:
                if pred_ranks[item_a] < pred_ranks[item_b]:
                    concordant += 1
                else:
                    discordant += 1
            else:
                discordant += 1

    tau = (concordant - discordant) / total_gt_pairs
    return round(tau, 4)


def check_hallucination_rejection(response: str) -> bool:
    """Check if model response rejects the occurrence of an absent event."""
    normalized = response.lower()
    rejection_phrases = [
        "event_absent",
        "did not occur",
        "does not occur",
        "did not happen",
        "does not happen",
        "not found",
        "not present",
        "not mentioned",
        "not observed",
        "no evidence",
        "never occurs",
        "never happened",
        "absent from the video",
        "was not shown",
        "not shown",
        "doesn't appear",
        "does not appear",
    ]
    return any(phrase in normalized for phrase in rejection_phrases)


@dataclass(slots=True)
class BenchmarkEvent:
    """Ground truth video event."""

    id: str
    description: str
    time_range: tuple[float, float]
    modality: str = "multimodal"
    keyframe_id: str | None = None
    visual_text: str | None = None
    supporting_quote: str | None = None


@dataclass(slots=True)
class AbsentEvent:
    """Plausible event that did NOT occur in the video (negative test)."""

    id: str
    description: str
    plausibility: str = "medium"


@dataclass(slots=True)
class OrderingTask:
    """Task requiring chronological sequencing of scrambled events."""

    task_id: str
    event_ids: list[str]


@dataclass(slots=True)
class RetrievalQuery:
    """Factual question about video content."""

    query: str
    ground_truth_answer: str
    verification_mode: str = "exact_or_llm_judge"


@dataclass(slots=True)
class BenchmarkDataset:
    """Full ground-truth evaluation specification for a video."""

    video_id: str
    source: str
    duration_seconds: float
    title: str = ""
    domain: str = "general"
    events: list[BenchmarkEvent] = field(default_factory=list)
    absent_events: list[AbsentEvent] = field(default_factory=list)
    ordering_tasks: list[OrderingTask] = field(default_factory=list)
    retrieval_queries: list[RetrievalQuery] = field(default_factory=list)


@dataclass(slots=True)
class SemanticScorecard:
    """Quantitative scorecard of agent timeline comprehension."""

    localization_evals: int = 0
    localization_tiou_avg: float = 0.0
    localization_mae_avg: float = 0.0
    ordering_evals: int = 0
    ordering_tau_avg: float = 0.0
    absent_evals: int = 0
    absent_rejections_correct: int = 0
    retrieval_evals: int = 0
    retrieval_matches: int = 0

    @property
    def absent_rejection_rate(self) -> float:
        if self.absent_evals == 0:
            return 0.0
        return round(self.absent_rejections_correct / self.absent_evals, 4)

    @property
    def retrieval_accuracy(self) -> float:
        if self.retrieval_evals == 0:
            return 0.0
        return round(self.retrieval_matches / self.retrieval_evals, 4)

    def to_markdown_table(self) -> str:
        lines = [
            "| Semantic Metric | Evaluated Tasks | Score | Target |",
            "|:---|:---|:---|:---|",
            f"| **Temporal Localization (tIoU)** | {self.localization_evals} | {self.localization_tiou_avg:.3f} | >= 0.500 |",
            f"| **Boundary Error (MAE)** | {self.localization_evals} | {self.localization_mae_avg:.1f}s | <= 5.0s |",
            f"| **Event Ordering (Kendall's τ)** | {self.ordering_evals} | {self.ordering_tau_avg:.3f} | >= 0.900 |",
            f"| **Absent Event Rejection Rate** | {self.absent_evals} | {self.absent_rejection_rate * 100:.1f}% | >= 95.0% |",
            f"| **Factual Retrieval Match Rate** | {self.retrieval_evals} | {self.retrieval_accuracy * 100:.1f}% | >= 90.0% |",
        ]
        return "\n".join(lines)


def load_benchmark_dataset(file_path: str) -> BenchmarkDataset:
    """Load benchmark dataset from a YAML or JSON file."""
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark dataset not found: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    events = [
        BenchmarkEvent(
            id=str(e.get("id")),
            description=str(e.get("description", "")),
            time_range=(float(e["time_range"][0]), float(e["time_range"][1])),
            modality=str(e.get("modality", "multimodal")),
            keyframe_id=e.get("keyframe_id"),
            visual_text=e.get("visual_text"),
            supporting_quote=e.get("supporting_quote"),
        )
        for e in raw.get("events", [])
    ]

    absent = [
        AbsentEvent(
            id=str(a.get("id")),
            description=str(a.get("description", "")),
            plausibility=str(a.get("plausibility", "medium")),
        )
        for a in raw.get("absent_events", [])
    ]

    ordering = [
        OrderingTask(
            task_id=str(o.get("task_id", f"task_{idx}")),
            event_ids=[str(x) for x in o.get("event_ids", [])],
        )
        for idx, o in enumerate(raw.get("ordering_tasks", []))
    ]

    retrievals = [
        RetrievalQuery(
            query=str(r.get("query", "")),
            ground_truth_answer=str(r.get("ground_truth_answer", "")),
            verification_mode=str(r.get("verification_mode", "exact_or_llm_judge")),
        )
        for r in raw.get("retrieval_queries", [])
    ]

    meta = raw.get("metadata", {})
    return BenchmarkDataset(
        video_id=str(raw.get("video_id", path.stem)),
        source=str(meta.get("source", raw.get("source", ""))),
        duration_seconds=float(
            meta.get("duration_seconds", raw.get("duration_seconds", 0.0))
        ),
        title=str(meta.get("title", "")),
        domain=str(meta.get("domain", "general")),
        events=events,
        absent_events=absent,
        ordering_tasks=ordering,
        retrieval_queries=retrievals,
    )
