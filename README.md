# video-analyzer

[![CI](https://github.com/dima/video-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/dima/video-analyzer/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/video-analyzer.svg)](https://pypi.org/project/video-analyzer/)
[![Python versions](https://img.shields.io/pypi/pyversions/video-analyzer.svg)](https://pypi.org/project/video-analyzer/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/dima/video-analyzer/badge)](https://scorecard.dev/viewer/?pup=github.com/dima/video-analyzer)

> Local-first, sandboxed video analysis runtime exposing a unified Python API, CLI, and FastMCP server for LLM agents.

---

## Overview

`video-analyzer` provides deterministic, bounded multimodal video inspection without relying on third-party cloud APIs. Designed specifically for AI agent integration (such as Claude Desktop, Cursor, and custom agentic frameworks), `video-analyzer` analyzes video streams, extracts keyframes, performs optical character recognition (OCR), extracts native subtitles, detects speech activity (VAD), and runs local automatic speech recognition (ASR) fallback.

### Key Highlights

- **Standard FastMCP Tool Server**: First-class MCP server over standard I/O (`video-analyzer mcp`), allowing coding assistants and LLMs to inspect videos directly through function calling.
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

`video-analyzer` relies on standard system media utilities:

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
pip install video-analyzer
# or using uv:
uv add video-analyzer
```

### With Speech & Transcription Extras
```bash
# Local speech transcription (Faster-Whisper + Silero VAD)
pip install 'video-analyzer[asr]'

# Voice activity detection only
pip install 'video-analyzer[vad]'

# All features and extras
pip install 'video-analyzer[all]'
```

---

## FastMCP Configuration (Claude Desktop & Cursor)

### Claude Desktop
Add `video-analyzer` to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "video-analyzer": {
      "command": "uvx",
      "args": ["--from", "video-analyzer[all]", "video-analyzer", "mcp"]
    }
  }
}
```

### Cursor
Add `video-analyzer` to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "video-analyzer": {
      "command": "uvx",
      "args": ["--from", "video-analyzer[all]", "video-analyzer", "mcp"]
    }
  }
}
```

---

## CLI Usage

### Analyze Video Window
```bash
video-analyzer analyze-video \
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
video-analyzer mcp
```

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
from video_analyzer import (
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
docker run -i --rm ghcr.io/dima/video-analyzer:latest
```

Building locally:
```bash
docker build -t video-analyzer:latest .
```

---

## Development & Testing

```bash
# Clone repository
git clone https://github.com/dima/video-analyzer.git
cd video-analyzer

# Sync all development dependencies
uv sync --all-extras

# Linting & code formatting
uv run ruff check .
uv run ruff format --check .

# Static type checking
uv run mypy src tests

# Unit test suite with coverage
uv run pytest --cov=video_analyzer --cov-report=term-missing

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
