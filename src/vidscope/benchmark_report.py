"""Generate BENCHMARKS.md and GitHub Actions Step Summary from pytest-benchmark results."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCHMARK_DESCRIPTIONS: dict[str, str] = {
    "test_benchmark_tesseract_ocr": "Tesseract 5.3 OCR Frame Recognition (1280x720 PNG)",
    "test_benchmark_dag_planning": "Execution Plan DAG Synthesis (build_execution_plan)",
    "test_benchmark_request_parsing": "Pydantic Request Validation & Normalization",
    "test_benchmark_caption_parsing": "WebVTT Subtitle Timestamp & Segment Parsing",
    "test_benchmark_telemetry_metrics": "Runtime Telemetry Metrics (RSS, RTF, OCR FPS)",
    "test_benchmark_wer_throughput": "Levenshtein Word Error Rate (WER) Computation",
    "test_benchmark_vad_segmentation": "Silero VAD Speech Segmentation (3.0s WAV)",
}


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable microseconds or milliseconds."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} µs"
    if seconds < 1.0:
        return f"{seconds * 1_000:.2f} ms"
    return f"{seconds:.3f} s"


def _format_ops(ops: float) -> str:
    """Format operations per second."""
    if ops >= 1_000_000:
        return f"{ops / 1_000_000:.2f} M/s"
    if ops >= 1_000:
        return f"{ops / 1_000:.2f} k/s"
    return f"{ops:.2f} /s"


def generate_benchmark_markdown(data: dict[str, Any]) -> str:
    """Render markdown benchmark report from pytest-benchmark JSON data."""
    now_iso = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    machine = data.get("machine_info", {})
    cpu_info = machine.get("cpu", {})
    benchmarks = data.get("benchmarks", [])

    cpu_brand = cpu_info.get("brand_raw") or machine.get("processor") or "Unknown CPU"
    cpu_cores = cpu_info.get("count", machine.get("cpu_count", "N/A"))
    arch = machine.get("machine") or machine.get("processor") or "x86_64"
    system_os = f"{machine.get('system', 'Linux')} {machine.get('release', '')}".strip()
    python_ver = machine.get("python_version", sys.version.split()[0])

    lines: list[str] = [
        "# Vidscope Performance Benchmarks & Pipeline Budget",
        "",
        f"> Last Updated: **{now_iso}**  ",
        f"> System: **{system_os}** (`{arch}`) | CPU: **{cpu_brand}** ({cpu_cores} cores) | Python: **{python_ver}**",
        "",
        "This document compiles measured microbenchmark timings (Section 1), architectural pipeline",
        "latency and memory budgets (Section 2), and nominal ASR accuracy reference standards (Section 3).",
        "",
        "---",
        "",
        "## 1. Microbenchmark Results (pytest-benchmark)",
        "",
        "Timings measured under native execution with deterministic fixtures:",
        "",
        "| Benchmark Target | Mean Latency | Median | Min / Max | Ops/sec | Rounds |",
        "|:---|:---|:---|:---|:---|:---|",
    ]

    for b in sorted(benchmarks, key=lambda item: str(item.get("name", ""))):
        name = str(b.get("name", ""))
        desc = BENCHMARK_DESCRIPTIONS.get(name, name)
        stats = b.get("stats", {})
        mean_str = _format_time(float(stats.get("mean", 0.0)))
        median_str = _format_time(float(stats.get("median", 0.0)))
        min_str = _format_time(float(stats.get("min", 0.0)))
        max_str = _format_time(float(stats.get("max", 0.0)))
        ops_str = _format_ops(float(stats.get("ops", 0.0)))
        rounds = int(stats.get("rounds", 0))

        lines.append(
            f"| **{desc}** (`{name}`) | `{mean_str}` | `{median_str}` | `{min_str}` / `{max_str}` | `{ops_str}` | `{rounds:,}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. End-to-End Pipeline Latency & Memory Budget",
            "",
            "> [!NOTE]",
            "> The stage figures below represent nominal engineering budgets and architectural targets.",
            "> Dynamic execution microbenchmarks are measured in Section 1.",
            "",
            "Vidscope structures video analysis into independent, bounded stages across 180-second chunks.",
            "Below is the nominal latency and resource overhead budget on standard commodity hardware:",
            "",
            "| Pipeline Stage | Nominal Latency (CPU) | Nominal Latency (CUDA) | Peak RSS Memory | Notes |",
            "|:---|:---|:---|:---|:---|",
            "| **Preflight Inspection** (`get_video_info`) | < 5 ms (local) / 150-300 ms (YouTube) | Same | ~45 MB | Non-downloading yt-dlp metadata extraction |",
            "| **Caption Acquisition** (Captions API) | 15 - 50 ms | Same | ~50 MB | Native YouTube timedtext or embedded tracks |",
            "| **DAG Synthesis & Planning** | < 0.6 ms | Same | ~50 MB | Pure-Python deterministic DAG dependency graph |",
            "| **Keyframe Extraction** (FFmpeg) | ~300 - 450 ms | ~100 - 200 ms | ~90 MB | 4 evenly-spaced frames per 180s chunk (1 frame / 36s) |",
            "| **Visual OCR** (Tesseract 5.3 LSTM) | ~160 - 250 ms / frame | N/A (CPU-bound) | ~120 MB | ~0.7 - 1.0s total OCR compute per 180s chunk |",
            "| **Local Speech ASR** (`tiny.en`) | ~4.5 - 7.5s (RTF: 0.15 - 0.25) | ~0.9 - 1.5s (RTF: 0.03 - 0.05) | ~250 MB | Fallback when captions/subtitles are unavailable |",
            "| **Local Speech ASR** (`small.en`) | ~18 - 25s (RTF: 0.60 - 0.85) | ~2.5 - 4.0s (RTF: 0.08 - 0.12) | ~850 MB | Higher accuracy fallback on challenging speech |",
            "| **VAD Segmentation** (Silero VAD) | ~25 - 45 ms / audio minute | ~8 - 15 ms | ~80 MB | Speech boundary pruning prior to ASR |",
            "",
            "---",
            "",
            "## 3. ASR Accuracy & Alignment Reference Targets",
            "",
            "> [!NOTE]",
            "> The metrics below establish nominal baseline thresholds and target reference standards for ASR evaluation",
            "> against human ground truth captions. Dynamic execution benchmark timings are captured in Section 1.",
            "",
            "Speech transcript accuracy is validated against reference timed captions using Levenshtein distance metrics:",
            "",
            "| Metric | Formula / Standard | Target Threshold | Nominal Value |",
            "|:---|:---|:---|:---|",
            "| **Word Error Rate (WER)** | `(Substitutions + Deletions + Insertions) / Total Words` | `< 0.10` (clean audio) | `0.00 - 0.04` |",
            "| **ASR Accuracy %** | `max(0.0, 1.0 - WER)` | `> 90.0%` | `96.0% - 100.0%` |",
            "| **Timestamp Drift (MAE)** | `mean(abs(ref_timestamp - hyp_timestamp))` | `< 0.200 s` | `0.035 s - 0.050 s` |",
            "| **Maximum Offset Drift** | `max(abs(ref_timestamp - hyp_timestamp))` | `< 0.500 s` | `< 0.050 s` |",
            "",
            "---",
            "",
            "## 4. How to Run & Regenerate",
            "",
            "Run the benchmark test suite and regenerate this report locally:",
            "",
            "```bash",
            "# Run pytest benchmarks and output JSON",
            "uv run pytest benchmarks/ --benchmark-only --benchmark-json=benchmark-results.json",
            "",
            "# Compile benchmark results into BENCHMARKS.md",
            "python scripts/generate_benchmark_report.py --input benchmark-results.json --output BENCHMARKS.md",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def run_benchmark_report(
    input_path: Path, output_path: Path, step_summary: bool = False
) -> int:
    """Read benchmark JSON and render markdown output."""
    if not input_path.is_file():
        sys.stderr.write(f"Error: Benchmark results file '{input_path}' not found.\n")
        sys.stderr.write(
            "Run: uv run pytest benchmarks/ --benchmark-only --benchmark-json=benchmark-results.json\n"
        )
        return 1

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    markdown = generate_benchmark_markdown(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown.strip() + "\n", encoding="utf-8")
    print(f"Successfully generated benchmark report in '{output_path}'.")

    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary and step_summary_path:
        with open(step_summary_path, "a", encoding="utf-8") as f:
            f.write("\n" + markdown.strip() + "\n")
        print(
            f"Appended benchmark summary to GITHUB_STEP_SUMMARY ({step_summary_path})."
        )

    return 0
