# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Agentic FastMCP tool suite for zero-filesystem, multimodal agent interaction:
  - `get_video_info`: Fast preflight metadata, subtitles, and native chapters discovery.
  - `search_video`: Sub-second keyword grep across video captions returning exact timestamps and formatted time codes.
  - `get_video_transcript`: Strictly window-bounded speech transcript extraction without full-track download overhead.
  - `view_frame`: Native MCP `Image` content block delivery (`image/jpeg`) with inline OCR and timestamp metadata.
  - `get_video_timeline`: Synchronous bounded timeline fusing dialogue, keyframes, and OCR text.
  - `start_video_analysis` and `get_job_status`: 2-step asynchronous multi-chunk job execution with progressive section streaming.
- Thread-safe in-memory `JobManager` with automatic 2-hour TTL cleanup for background video processing and keyframe image caching.

### Changed
- Refactored FastMCP server to deliver direct multimodal data blocks instead of requiring local filesystem artifact traversal.

### Fixed
- Fixed `ffprobe` execution failures by removing the unsupported `-nostdin` argument in `backends/media.py` and `backends/source.py`.
- Corrected frame extraction seek offset math for nonzero start offsets in `backends/media.py`.
- Fixed WebVTT timestamp decimal format to use standard `.` instead of `,`.
- Bounded caption track export to requested window range in `core.py`.
- Added missing FastMCP resource endpoints for `/manifest` and `/plan` URIs.
- Fixed Dockerfile installation by removing `--no-index` from local package install, enabling PEP 517 build backend resolution.
- Enforced `AnalysisTask` enum type safety and `TextContent` union narrowing for mypy compliance.

### Removed
- Removed legacy `analyze_video` MCP tool in favor of the modular agentic tool suite.

## [0.1.0] - 2026-09-07

### Added
- Standard FastMCP server implementation over standard I/O (`vidscope mcp`).
- CLI tool powered by Typer (`vidscope analyze-video`) with JSON envelope emissions and logging flags (`--verbose`, `--quiet`).
- Bounded DAG execution planner enforcing local capabilities, bounded time ranges (<=180s), frame count limits, and download/output size caps.
- Layered SSRF protection blocking loopback, link-local, and RFC 1918 private subnets for network URL sources and captions.
- Hardened subprocess sandboxing with `-nostdin` and `-protocol_whitelist "file,pipe,crypto,data"` across FFmpeg and FFprobe backends.
- POSIX argument termination (`--`) and timeout boundaries across Tesseract OCR invocations.
- Optional ML dependency extras (`[asr]`, `[vad]`, `[all]`, `[dev]`) for faster-whisper and silero-vad runtimes.
- Telemetry helpers for memory RSS monitoring (KiB to MB scaling on Linux, Darwin byte scaling) and throughput calculations.
- Comprehensive test suite and benchmark harness.
