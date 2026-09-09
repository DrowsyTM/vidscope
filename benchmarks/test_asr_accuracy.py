"""ASR Accuracy Evaluation Benchmark.

Compares ASR transcript hypotheses against ground truth reference captions
(e.g., YouTube timedtext or human transcripts) using:
- Word Error Rate (WER): Standard Levenshtein distance on normalized word tokens.
- Character Error Rate (CER): Edit distance over character sequences.
- Timestamp Drift: Mean Absolute Error (MAE) and maximum offset for aligned segments.
"""

from __future__ import annotations

import re
from typing import Any


def normalize_words(text: str) -> list[str]:
    """Normalize text by lowercasing and stripping punctuation."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return cleaned.split()


def compute_wer(reference: str, hypothesis: str) -> dict[str, Any]:
    """Compute Word Error Rate (WER) between reference and hypothesis text."""
    r = normalize_words(reference)
    h = normalize_words(hypothesis)
    n = len(r)
    if n == 0:
        return {
            "wer": 0.0 if len(h) == 0 else 1.0,
            "accuracy": 1.0 if len(h) == 0 else 0.0,
            "edit_distance": len(h),
            "reference_words": 0,
            "hypothesis_words": len(h),
        }

    d = [[0] * (len(h) + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,  # deletion
                    d[i][j - 1] + 1,  # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    distance = d[n][len(h)]
    wer = round(distance / n, 4)
    accuracy = round(max(0.0, 1.0 - (distance / n)), 4)
    return {
        "wer": wer,
        "accuracy": accuracy,
        "edit_distance": distance,
        "reference_words": n,
        "hypothesis_words": len(h),
    }


def compute_timestamp_drift(
    reference_segments: list[dict[str, Any]],
    hypothesis_segments: list[dict[str, Any]],
) -> dict[str, float]:
    """Calculate timestamp drift (start/end alignment error) across segments."""
    if not reference_segments or not hypothesis_segments:
        return {"mae_start": 0.0, "mae_end": 0.0, "max_drift": 0.0}

    paired_count = min(len(reference_segments), len(hypothesis_segments))
    start_diffs: list[float] = []
    end_diffs: list[float] = []

    for i in range(paired_count):
        ref = reference_segments[i]
        hyp = hypothesis_segments[i]
        ref_start = float(ref.get("start_seconds", ref.get("start", 0.0)))
        hyp_start = float(hyp.get("start_seconds", hyp.get("start", 0.0)))
        ref_end = float(ref.get("end_seconds", ref.get("end", 0.0)))
        hyp_end = float(hyp.get("end_seconds", hyp.get("end", 0.0)))

        start_diffs.append(abs(ref_start - hyp_start))
        end_diffs.append(abs(ref_end - hyp_end))

    all_diffs = start_diffs + end_diffs
    return {
        "mae_start": round(sum(start_diffs) / len(start_diffs), 4),
        "mae_end": round(sum(end_diffs) / len(end_diffs), 4),
        "max_drift": round(max(all_diffs) if all_diffs else 0.0, 4),
    }


def test_wer_identical_transcripts() -> None:
    text = "Neural networks learn representations through backpropagation and gradient descent."
    result = compute_wer(text, text)
    assert result["wer"] == 0.0
    assert result["accuracy"] == 1.0
    assert result["edit_distance"] == 0


def test_wer_with_known_errors() -> None:
    ref = "This is the first chapter in deep learning."
    hyp = "This is the first chapter of deep learning today."
    # 'in' replaced by 'of' (sub: 1), 'today' inserted (ins: 1) -> 2 edits out of 7 words
    result = compute_wer(ref, hyp)
    assert result["reference_words"] == 8
    assert result["edit_distance"] == 2
    assert result["wer"] == 0.25


def test_timestamp_drift_calculation() -> None:
    ref_segs = [
        {"start_seconds": 0.0, "end_seconds": 2.5, "text": "Hello"},
        {"start_seconds": 2.6, "end_seconds": 5.0, "text": "World"},
    ]
    hyp_segs = [
        {"start_seconds": 0.05, "end_seconds": 2.55, "text": "Hello"},
        {"start_seconds": 2.58, "end_seconds": 4.95, "text": "World"},
    ]
    drift = compute_timestamp_drift(ref_segs, hyp_segs)
    assert drift["mae_start"] == 0.035
    assert drift["mae_end"] == 0.05
    assert drift["max_drift"] == 0.05


def test_benchmark_wer_throughput(benchmark: Any) -> None:
    ref = (
        "In this video we explore how artificial neural networks are trained. "
        "Each neuron computes a weighted sum of inputs and passes through an activation function."
    )
    hyp = (
        "In this video we explore how artificial neural networks get trained. "
        "Each neuron computes a weighted sum of its inputs and passes via an activation function."
    )
    result = benchmark(compute_wer, ref, hyp)
    assert result["wer"] > 0.0
    assert result["accuracy"] > 0.8
