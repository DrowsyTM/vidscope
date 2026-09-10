# Vidscope Performance Benchmarks & Pipeline Budget

> Last Updated: **2026-09-10 00:36:43 UTC**
> System: **Linux 6.8.0-139-generic** (`x86_64`) | CPU: **Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz** (4 cores) | Python: **3.12.3**

This document publishes deterministic microbenchmark timings, OCR extraction throughput,
ASR accuracy benchmarks, and stage-by-stage pipeline latency budgets for Vidscope.

---

## 1. Microbenchmark Results (pytest-benchmark)

Timings measured under native execution with deterministic fixtures:

| Benchmark Target | Mean Latency | Median | Min / Max | Ops/sec | Rounds |
|:---|:---|:---|:---|:---|:---|
| **WebVTT Subtitle Timestamp & Segment Parsing** (`test_benchmark_caption_parsing`) | `17.24 µs` | `15.79 µs` | `14.91 µs` / `120.42 µs` | `57.99 k/s` | `2,943` |
| **Execution Plan DAG Synthesis (build_execution_plan)** (`test_benchmark_dag_planning`) | `560.13 µs` | `554.75 µs` | `522.32 µs` / `862.60 µs` | `1.79 k/s` | `1,182` |
| **Pydantic Request Validation & Normalization** (`test_benchmark_request_parsing`) | `86.60 µs` | `83.13 µs` | `81.17 µs` / `197.82 µs` | `11.55 k/s` | `4,822` |
| **Runtime Telemetry Metrics (RSS, RTF, OCR FPS)** (`test_benchmark_telemetry_metrics`) | `4.45 µs` | `4.18 µs` | `4.04 µs` / `73.63 µs` | `224.92 k/s` | `47,509` |
| **Tesseract 5.3 OCR Frame Recognition (1280x720 PNG)** (`test_benchmark_tesseract_ocr`) | `213.50 ms` | `165.08 ms` | `163.47 ms` / `406.00 ms` | `4.68 /s` | `6` |
| **Levenshtein Word Error Rate (WER) Computation** (`test_benchmark_wer_throughput`) | `214.24 µs` | `204.73 µs` | `202.90 µs` / `337.97 µs` | `4.67 k/s` | `3,007` |

---

## 2. End-to-End Pipeline Latency & Memory Budget

Vidscope structures video analysis into independent, bounded stages across 180-second chunks.
Below is the nominal latency and resource overhead budget on standard commodity hardware:

| Pipeline Stage | Nominal Latency (CPU) | Nominal Latency (CUDA) | Peak RSS Memory | Notes |
|:---|:---|:---|:---|:---|
| **Preflight Inspection** (`get_video_info`) | < 5 ms (local) / 150-300 ms (YouTube) | Same | ~45 MB | Non-downloading yt-dlp metadata extraction |
| **Caption Acquisition** (Captions API) | 15 - 50 ms | Same | ~50 MB | Native YouTube timedtext or embedded tracks |
| **DAG Synthesis & Planning** | < 0.6 ms | Same | ~50 MB | Pure-Python deterministic DAG dependency graph |
| **Keyframe Extraction** (FFmpeg) | ~300 - 450 ms | ~100 - 200 ms | ~90 MB | 4 evenly-spaced frames per 180s chunk (1 frame / 36s) |
| **Visual OCR** (Tesseract 5.3 LSTM) | ~160 - 250 ms / frame | N/A (CPU-bound) | ~120 MB | ~0.7 - 1.0s total OCR compute per 180s chunk |
| **Local Speech ASR** (`tiny.en`) | ~4.5 - 7.5s (RTF: 0.15 - 0.25) | ~0.9 - 1.5s (RTF: 0.03 - 0.05) | ~250 MB | Fallback when captions/subtitles are unavailable |
| **Local Speech ASR** (`small.en`) | ~18 - 25s (RTF: 0.60 - 0.85) | ~2.5 - 4.0s (RTF: 0.08 - 0.12) | ~850 MB | Higher accuracy fallback on challenging speech |
| **VAD Segmentation** (Silero VAD) | ~25 - 45 ms / audio minute | ~8 - 15 ms | ~80 MB | Speech boundary pruning prior to ASR |

---

## 3. ASR Accuracy & Alignment Evaluation

Speech transcript accuracy is validated against reference timed captions using Levenshtein distance metrics:

| Metric | Formula / Standard | Target Threshold | Nominal Value |
|:---|:---|:---|:---|
| **Word Error Rate (WER)** | `(Substitutions + Deletions + Insertions) / Total Words` | `< 0.10` (clean audio) | `0.00 - 0.04` |
| **ASR Accuracy %** | `max(0.0, 1.0 - WER)` | `> 90.0%` | `96.0% - 100.0%` |
| **Timestamp Drift (MAE)** | `mean(abs(ref_timestamp - hyp_timestamp))` | `< 0.200 s` | `0.035 s - 0.050 s` |
| **Maximum Offset Drift** | `max(abs(ref_timestamp - hyp_timestamp))` | `< 0.500 s` | `< 0.050 s` |

---

## 4. How to Run & Regenerate

Run the benchmark test suite and regenerate this report locally:

```bash
# Run pytest benchmarks and output JSON
uv run pytest benchmarks/ --benchmark-only --benchmark-json=benchmark-results.json

# Compile benchmark results into BENCHMARKS.md
python scripts/generate_benchmark_report.py --input benchmark-results.json --output BENCHMARKS.md
```
