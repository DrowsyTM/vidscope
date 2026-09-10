#!/usr/bin/env python3
"""CLI wrapper to generate BENCHMARKS.md and GitHub Step Summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vidscope.benchmark_report import run_benchmark_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile pytest-benchmark JSON into formatted BENCHMARKS.md report."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("benchmark-results.json"),
        help="Input benchmark JSON file (default: benchmark-results.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("BENCHMARKS.md"),
        help="Output markdown file (default: BENCHMARKS.md)",
    )
    parser.add_argument(
        "--step-summary",
        action="store_true",
        help="Write markdown output to GITHUB_STEP_SUMMARY environment variable if present.",
    )
    args = parser.parse_args()
    return run_benchmark_report(
        input_path=args.input,
        output_path=args.output,
        step_summary=args.step_summary,
    )


if __name__ == "__main__":
    sys.exit(main())
