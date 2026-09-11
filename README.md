# vidscope

[![CI](https://github.com/DrowsyTM/vidscope/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/DrowsyTM/vidscope/actions/workflows/ci.yml)
[![Security Analysis](https://github.com/DrowsyTM/vidscope/actions/workflows/security.yml/badge.svg?branch=master)](https://github.com/DrowsyTM/vidscope/actions/workflows/security.yml)
[![PyPI version](https://img.shields.io/pypi/v/vidscope.svg)](https://pypi.org/project/vidscope/)
[![Python versions](https://img.shields.io/pypi/pyversions/vidscope.svg)](https://pypi.org/project/vidscope/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/DrowsyTM/vidscope/badge)](https://scorecard.dev/viewer/?uri=github.com/DrowsyTM/vidscope)

> Local-first, sandboxed video analysis runtime exposing a unified Python API, CLI, and FastMCP server for LLM agents.

---

## Overview

`vidscope` provides deterministic, bounded multimodal video inspection without relying on third-party cloud APIs. Designed specifically for AI agent integration (such as Claude Desktop, Cursor, and custom agentic frameworks), `vidscope` analyzes video streams, extracts keyframes, performs optical character recognition (OCR), extracts native subtitles, detects speech activity (VAD), and runs local automatic speech recognition (ASR) fallback.

### Key Highlights

- **Standard FastMCP Tool Server**: First-class MCP server over standard I/O (`vidscope mcp`), allowing coding assistants and LLMs to inspect videos directly through function calling.
- **Pure Standard Output**: Strict separation of communication channels. FastMCP JSON-RPC messages and CLI JSON envelopes are emitted cleanly to standard output, while diagnostic and operational logs route to standard error.
- **Subprocess Sandboxing**: Hardened execution using `-nostdin` and `-protocol_whitelist "file,pipe,crypto,data"` across FFmpeg and FFprobe backends, plus POSIX option terminators (`--`) for Tesseract.
- **Layered SSRF Protection**: Inbound URL validation rejects private IP ranges (RFC 1918), loopback, link-local (AWS/GCP/Azure instance metadata endpoints), and malformed schemes before any network resolution occurs.
- **Deterministic DAG Planning**: Execution requests are compiled into an ordered dependency graph before processing begins, ensuring bounded execution windows (<=180 seconds), capped frame counts, and strict memory limits.
- **Modular Optional Extras**: Lean base footprint with modular ML extras (`[asr]`, `[vad]`, `[all]`) to prevent unneeded heavy model downloads.

---

## Architecture

```mermaid
flowchart TD
    subgraph Clients["Clients & Integration"]
        MCP["MCP Client (Claude / Cursor)"]
        CLI["Typer CLI"]
        SDK["Python SDK"]
    end

    subgraph Core["Core Engine"]
        Planner["Deterministic DAG Planner"]
        Capabilities["Capability Discovery"]
        Telemetry["Telemetry & RSS Monitor"]
    end

    subgraph SandboxedBackends["Sandboxed Backends"]
        Media["FFmpeg / FFprobe (Sandboxed Subprocess)"]
        OCR["Tesseract OCR (POSIX Terminated)"]
        VAD["Silero VAD (ONNX / CPU)"]
        ASR["Faster-Whisper (Int8 / CPU)"]
        Captions["YouTube & WebVTT Captions"]
    end

    subgraph Storage["Artifact & Output Layer"]
        Artifacts["ArtifactStore (SHA-256 Hashed)"]
        Manifest["RunManifest & Metrics"]
    end

    MCP --> Planner
    CLI --> Planner
    SDK --> Planner

    Capabilities --> Planner
    Planner --> Media
    Planner --> Captions
    Media --> OCR
    Media --> VAD
    Media --> ASR

    Media --> Artifacts
    OCR --> Artifacts
    ASR --> Artifacts
    Artifacts --> Manifest
    Telemetry --> Manifest
```

---

## System Prerequisites

`vidscope` relies on standard system media utilities:

### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-eng
```

### macOS (Homebrew)
```bash
brew install ffmpeg tesseract
```

---

## Installation

### Core Package (Lightweight CLI & FastMCP Server)
```bash
pip install vidscope
# or using uv:
uv add vidscope
```

### With Speech & Transcription Extras
```bash
# Local speech transcription (Faster-Whisper + Silero VAD)
pip install 'vidscope[asr]'

# Voice activity detection only
pip install 'vidscope[vad]'

# All features and extras
pip install 'vidscope[all]'
```

### System Diagnostics (`vidscope doctor`)
Verify that your local system has required media binaries, speech models, and token providers:
```bash
vidscope doctor
```
Outputs an actionable diagnostic checklist:
```text
Vidscope System Diagnostics
===========================
[✓] FFmpeg: ffmpeg version 6.1.1
[✓] FFprobe: ffprobe version 6.1.1
[✓] Tesseract OCR: tesseract 5.3.4
[✓] Speech-to-Text (ASR): faster-whisper + silero-vad ready [CPU (int8 quantized)]
[✓] JavaScript Runtime: Node.js v22.22.2 (/usr/bin/node)
[✓] PO Token Generator: Ready: v2.0.0 (~/bgutil-ytdlp-pot-provider/server/build/generate_once.js)
[✓] Cookies File: Not configured (anonymous mode)

System is fully configured and ready for video analysis.
```

### Optional: Proof-of-Origin (PO) Token Setup (`vidscope setup-pot`)
If you run Vidscope on cloud or datacenter IPs (AWS, GCP, Hetzner) where YouTube blocks video stream formats, compile the on-demand PO token generation script with a single command:
```bash
vidscope setup-pot
```
This automatically clones and compiles the standalone script into `~/bgutil-ytdlp-pot-provider/server/build/generate_once.js`. `yt-dlp` will automatically invoke and cache tokens on demand without running any background Docker containers.

---

## FastMCP Configuration (Claude Desktop & Cursor)

### Claude Desktop
Add `vidscope` to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vidscope": {
      "command": "uvx",
      "args": ["--from", "vidscope[all]", "vidscope", "mcp"]
    }
  }
}
```

### Cursor
Add `vidscope` to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "vidscope": {
      "command": "uvx",
      "args": ["--from", "vidscope[all]", "vidscope", "mcp"]
    }
  }
}
```

---

## FastMCP Tool Suite for AI Agents

`vidscope` exposes an agent-optimized FastMCP interface designed for multimodal LLMs (Claude 3.5 Sonnet, Gemini, GPT-4o). Rather than dumping raw files to disk and requiring local filesystem access, tools return bounded, structured metadata and **native MCP `Image` content blocks (`image/jpeg`)** directly inline.

### Available MCP Tools

| Tool | Purpose | Mode | Typical Use Case |
|---|---|---|---|
| `get_video_info` | Fast preflight inspection | Sync (~500ms) | Inspect video title, duration, languages, chapters, and host runtime capabilities (`local_asr_available`, `local_ocr_available`). |
| `analyze_video` | Single video analysis front door | Sync (<5s) / Async | Analyze video and extract condensed visual & speech timeline chunks with explicit `coverage` metadata. Short clips return complete timeline immediately; longer videos seamlessly hand off to background with `job_id`, ETA, and `next_action`. |
| `get_job_status` | Incremental streaming status | Instant | Check background job status and retrieve newly completed timeline chunks using `since_chunk` cursor. Emits structured `next_action`, `retry_after_seconds`, and `coverage` metadata. |
| `view_frame` | Multimodal frame delivery | Sync (~100ms–2s) | View any video frame directly in conversation context as a native MCP `Image` (`image/jpeg`) block with OCR metadata, via `frame_id` or `(source, timestamp)`. |
| `search_video` | Post-analysis transcript grep | Sync (~100ms) | Grep across analyzed speech transcripts (`job_id`) with regex, case-sensitivity, and windowed `coverage` metadata. Strictly gated to completed jobs. |
| `get_transcript` | Targeted dialogue retrieval | Sync (~50ms) | Retrieve timestamped dialogue segments and joined text for a specific time window (`start_seconds`, `end_seconds`, `max_duration_seconds`). Strictly gated to completed jobs. |

> For complete JSON schemas, parameter constraints, error envelopes, and realistic input/output fixtures, see the [MCP Tool Reference](docs/MCP_REFERENCE.md).
> For microbenchmark measurements, ASR accuracy metrics, and stage latency budgets, see [Performance Benchmarks](BENCHMARKS.md).

### Recommended Agent Workflow

```mermaid
flowchart TD
    A["get_video_info(source)"] --> B["analyze_video(source)"]
    B --> C{"Finished in <=5s?"}
    C -- Yes --> D["Timeline returned directly"]
    C -- No --> E["get_job_status(job_id, since_chunk=...)"]
    E --> D
    D --> F{"Inspect keyframe visually?"}
    F -- Yes --> G["view_frame(frame_id)"]
    F -- No --> H["Answer User Query"]
    G --> H
    D --> I{"Search spoken topics/keywords?"}
    I -- Yes --> J["search_video(job_id, query)"]
    J --> H
    D --> K{"Read verbatim dialogue?"}
    K -- Yes --> L["get_transcript(job_id, start_seconds, end_seconds)"]
    L --> H
```

---

## CLI Usage

### Analyze Video Window
```bash
vidscope analyze-video \
  --source "https://example.com/clip.mp4" \
  --out "./output_dir" \
  --start-seconds 0 \
  --end-seconds 60 \
  --task metadata \
  --task transcript \
  --task frames \
  --task ocr \
  --max-frames 6
```

### Run MCP Server
```bash
vidscope mcp
```

### Agent Evaluation Harness (`vidscope eval`)
Run automated agent evaluation benchmarks (Phase 1 protocol hygiene and timeline semantic comprehension) via headless MCP drivers:

```bash
# Run Phase 1 protocol hygiene suite (6 core scenarios, 100 runs, 4 parallel workers)
vidscope eval --protocol-suite --runs 100 --concurrency 4 --thinking off --output PROTOCOL_REPORT.md

# Stress-test a single prompt repeatedly with concurrent execution
vidscope eval -p "Check get_job_status for job_id 'test_123'" --runs 20 --concurrency 4

# Test stochastic async handoff flows (randomly simulate sync vs async job status workflows)
vidscope eval -p "Analyze video mock://demo.mp4 and summarize it" --runs 10 --mock-async random

# Run timeline semantic comprehension benchmark against a ground-truth dataset
vidscope eval -d benchmarks/eval_harness/datasets/demo.yaml --output EVAL_REPORT.md
```

#### Evaluation Options (`vidscope eval`)
- `--protocol-suite`: Standard Phase 1 protocol hygiene evaluation suite.
- `--runs` / `-n`: Number of iterations to run (default: 1 for prompt, 100 for suite).
- `--concurrency` / `-c`: Number of parallel worker threads (default: 4).
- `--thinking`: OMP thinking level (`off`, `minimal`, `low`, `medium`, `high`, `max`; default: `off`).
- `--mock / --no-mock`: Fast zero-cost FastMCP mock simulation mode (default: `--mock`).
- `--mock-async`: Mock async handoff mode: `auto` (default, detects via URL/duration), `sync` (forces immediate return), `async` (forces `get_job_status` background handoff), or `random` (50% stochastic sync vs async).
- `--output` / `-o`: Markdown report export path.

### CLI Options
- `--verbose` / `-v`: Enable verbose debug logging to standard error.
- `--quiet` / `-q`: Suppress non-error logging to standard error.
- `--source`: Local file path or HTTPS video URL.
- `--out`: Staging and artifact output directory.
- `--start-seconds`: Window start time in seconds (default: 0.0).
- `--end-seconds`: Window end time in seconds (default: 180.0, max 180s duration).
- `--task`: Repeatable task flag (`metadata`, `transcript`, `vad`, `frames`, `ocr`).


---

## Python API Usage

```python
from pathlib import Path
from vidscope import (
    AnalysisContext,
    AnalyzeVideoRequest,
    TimeRange,
    analyze_video,
)

request = AnalyzeVideoRequest(
    source="https://example.com/lecture.mp4",
    time_range=TimeRange(start_seconds=10.0, end_seconds=70.0),
    tasks={"metadata", "transcript", "frames", "ocr"},
    output_directory=Path("./runs/run_01"),
    max_frames=6,
    max_frame_width=1280,
    language="en",
)

result = analyze_video(request)

if result.ok:
    print(f"Status: {result.status}")
    print(f"Manifest URI: {result.manifest_uri}")
    for artifact in result.artifacts:
        print(
            f"Artifact: {artifact.artifact_id} -> {artifact.uri} ({artifact.byte_size} bytes)"
        )
    if result.metrics:
        print(f"Peak RSS: {result.metrics.peak_rss_mb} MB")
        print(f"Total Duration: {result.metrics.total_elapsed_ms:.1f} ms")
```

---

## Docker Container

A production container image with non-root security boundaries and pre-installed FFmpeg/Tesseract is available:

```bash
# Pull and run container in MCP mode
docker run -i --rm ghcr.io/drowsytm/vidscope:latest
```

Building locally:
```bash
docker build -t vidscope:latest .
```

---

## Development & Testing

```bash
git clone https://github.com/DrowsyTM/vidscope.git vidscope
cd vidscope

# Sync all development dependencies
uv sync --all-extras

# Linting & code formatting
uv run ruff check .
uv run ruff format --check .

# Static type checking
uv run mypy src tests

# Unit test suite with coverage
uv run pytest --cov=vidscope --cov-report=term-missing

# Run benchmarks
uv run pytest benchmarks/ --benchmark-only
```

---

## Governance & Community

- [Contributing Guidelines](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Agent & Developer Manual](AGENTS.md)
- [Changelog](CHANGELOG.md)

---

## License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for details.
