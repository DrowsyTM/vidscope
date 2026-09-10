from __future__ import annotations

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
    assert "## 3. ASR Accuracy & Alignment Evaluation" in markdown
