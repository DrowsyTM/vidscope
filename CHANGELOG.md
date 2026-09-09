# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Unified Agentic FastMCP tool suite (5 tools) for zero-filesystem, multimodal agent interaction:
  - `get_video_info`: Fast preflight metadata, subtitles, and native chapters discovery.
  - `analyze_video`: Unified video analysis entry point with sync-then-async fallback (<5s sync window for fast timeline return; background handoff with ETA and `job_id` for longer runs).
  - `get_job_status`: Incremental streaming job status with cursor pagination (`since_chunk`) to prevent context window token bloat.
  - `view_frame`: Native MCP `Image` content block delivery (`image/jpeg`) with inline OCR and timestamp metadata.
  - `search_video`: Video grep engine supporting regex and case-sensitive matching across native captions (`source`) or analyzed speech transcripts (`job_id`).
- Thread-safe in-memory `JobManager` with completion signaling (`completed_event`), incremental cursor pagination, transcript accumulation, and automatic 2-hour TTL cleanup.

### Changed
- Consolidated video analysis into a single entry point (`analyze_video`), removing duplicate paths (`get_video_timeline`, `get_video_transcript`, `start_video_analysis`).
- Refactored FastMCP server to deliver direct multimodal data blocks instead of requiring local filesystem artifact traversal.

### Fixed
- Fixed `view_frame` and chunk keyframe artifact glob resolution when extracting frames into `<output_dir>/<request_id>/frames/`.
- Prevented network duration inspection from blocking the `analyze_video` synchronous timeout window by moving probe inspection into background workers.
- Added graceful degradation to visual-only keyframe analysis in `analyze_video` when remote caption fetching fails or is rate-limited and local ASR is unavailable.
- Ensured terminal job failures in `get_job_status` and `analyze_video` emit standard `is_error=True` ToolResult envelopes.
- Added explicit mutual exclusivity validation and schema constraints (`Field`) in `view_frame` and `search_video`.
- Enforced schema-level `oneOf` mutual exclusivity on `view_frame` and `search_video` tool parameters.
- Added explicit cursor pagination (`next_since_chunk`), continuation flag (`has_more`), and actionable polling progress messages to `get_job_status`.
- Replaced static job completion estimates with dynamic remaining time estimation based on rolling chunk execution latency.
- Added natural language normalization (`english` -> `en`) and ISO format validation in `search_video`.
- Added `ocr_status` tracking on keyframes and informative chunk summaries detailing when OCR is disabled due to missing host binary.
- Registered standard MCP workflow prompts (`analyze_video_workflow`, `search_video_workflow`) for agent prompt discovery.
- Registered static `vidscope://info` resource for discovery via standard `resources/list` protocol calls.
- Suppressed verbose FastMCP stderr startup banner on server launch.
- Added `language` parameter and available language hints to `search_video`, distinguishing between unlisted tracks and remote provider retrieval failures (e.g. HTTP 429).
- Fixed `ffprobe` execution failures by removing the unsupported `-nostdin` argument in `backends/media.py` and `backends/source.py`.
- Corrected frame extraction seek offset math for nonzero start offsets in `backends/media.py`.
- Fixed WebVTT timestamp decimal format to use standard `.` instead of `,`.
- Bounded caption track export to requested window range in `core.py`.
- Added missing FastMCP resource endpoints for `/manifest` and `/plan` URIs.
- Fixed Dockerfile installation by removing `--no-index` from local package install, enabling PEP 517 build backend resolution.
- Enforced `AnalysisTask` enum type safety and `TextContent` union narrowing for mypy compliance.

### Removed
- Removed legacy file-dumping `analyze_video` tool and redundant intermediate MCP tools (`get_video_timeline`, `get_video_transcript`, `start_video_analysis`).

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
