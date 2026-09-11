from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vidscope.benchmark_report import generate_benchmark_markdown


def test_generate_benchmark_markdown_structure() -> None:
    sample_data = {
        "machine_info": {
            "system": "Linux",
            "release": "6.8.0-generic",
            "processor": "x86_64",
            "python_version": "3.12.3",
            "cpu": {
                "brand_raw": "Intel Core i5-7400 CPU @ 3.00GHz",
                "count": 4,
            },
        },
        "benchmarks": [
            {
                "name": "test_benchmark_tesseract_ocr",
                "stats": {
                    "mean": 0.165,
                    "median": 0.163,
                    "min": 0.160,
                    "max": 0.170,
                    "ops": 6.06,
                    "rounds": 5,
                },
            },
            {
                "name": "test_benchmark_caption_parsing",
                "stats": {
                    "mean": 0.000017,
                    "median": 0.000015,
                    "min": 0.000014,
                    "max": 0.000040,
                    "ops": 58823.5,
                    "rounds": 2500,
                },
            },
        ],
    }

    markdown = generate_benchmark_markdown(sample_data)
    assert "# Vidscope Performance Benchmarks & Pipeline Budget" in markdown
    assert "Intel Core i5-7400 CPU @ 3.00GHz" in markdown
    assert "test_benchmark_tesseract_ocr" in markdown
    assert "165.00 ms" in markdown
    assert "17.00 µs" in markdown
    assert "## 2. End-to-End Pipeline Latency & Memory Budget" in markdown
    assert "The stage figures below represent nominal engineering budgets" in markdown
    assert "## 3. ASR Accuracy & Alignment Reference Targets" in markdown
    assert "The metrics below establish nominal baseline thresholds" in markdown


def test_run_benchmark_report_with_step_summary(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from vidscope.benchmark_report import run_benchmark_report

    input_json = tmp_path / "results.json"
    input_json.write_text(
        json.dumps(
            {
                "machine_info": {"cpu": {"brand_raw": "Test CPU"}},
                "benchmarks": [],
            }
        ),
        encoding="utf-8",
    )

    output_md = tmp_path / "BENCHMARKS.md"
    summary_file = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    # Test execution with step_summary=True
    code = run_benchmark_report(input_json, output_md, step_summary=True)
    assert code == 0
    assert output_md.is_file()
    assert "# Vidscope Performance Benchmarks" in output_md.read_text(encoding="utf-8")
    assert summary_file.is_file()
    assert "# Vidscope Performance Benchmarks" in summary_file.read_text(
        encoding="utf-8"
    )

    # Test execution with step_summary=False (default) does not write to GITHUB_STEP_SUMMARY
    output_md_2 = tmp_path / "BENCHMARKS_2.md"
    summary_file_2 = tmp_path / "step_summary_2.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file_2))
    code_2 = run_benchmark_report(input_json, output_md_2, step_summary=False)
    assert code_2 == 0
    assert output_md_2.is_file()
    assert not summary_file_2.exists()

    # Test execution when input file does not exist
    missing_file = tmp_path / "nonexistent.json"
    err_code = run_benchmark_report(missing_file, output_md)
    assert err_code == 1
