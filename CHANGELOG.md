# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Unified Agentic FastMCP tool suite (6 tools) for zero-filesystem, multimodal agent interaction:
  - `get_video_info`: Fast preflight metadata, subtitles, and native chapters discovery.
  - `analyze_video`: Unified video analysis entry point with sync-then-async fallback (<5s sync window for fast timeline return; background handoff with ETA and `job_id` for longer runs).
  - `get_job_status`: Incremental streaming job status with cursor pagination (`since_chunk`) to prevent context window token bloat.
  - `view_frame`: Native MCP `Image` content block delivery (`image/jpeg`) with inline OCR and timestamp metadata.
  - `search_video`: Post-analysis transcript grep engine supporting substring and regex matching across analyzed speech transcripts (`job_id`).
  - `get_transcript`: Targeted dialogue and verbatim transcript retrieval with bounded time windows (`start_seconds`, `end_seconds`, `max_duration_seconds`) and dual representation (joined text and timestamped segments).
- Thread-safe in-memory `JobManager` with completion signaling (`completed_event`), incremental cursor pagination, transcript accumulation, and automatic 2-hour TTL cleanup.
- Integrated `curl-cffi` and `bgutil-ytdlp-pot-provider` dependencies for browser TLS fingerprint impersonation and Proof-of-Origin (PO) token generation.
- Replaced persistent Docker PO container with native on-demand script provider (`bgutil:script-node` via `generate_once.js`), eliminating persistent background containers.
- Added `VIDSCOPE_COOKIES_FILE` environment configuration and `Settings.cookies_file` support for optional Netscape cookie-jar authentication across yt-dlp, curl-cffi, and youtube-transcript-api.
- Added `VIDSCOPE_WHISPER_MODEL`, `VIDSCOPE_WHISPER_DEVICE`, and `VIDSCOPE_WHISPER_COMPUTE_TYPE` settings for flexible ASR configuration.
- Added automatic device detection (`cuda` when CUDA GPU is available, fallback to `cpu` with `int8` quantization), VRAM-aware default model selection (`base.en`/`base` on CUDA with $\ge 1\text{ GB}$ VRAM; `tiny.en`/`tiny` on CPU or low-VRAM), and multilingual model fallback (`base`/`tiny` when `language != "en"`).
- Added `vidscope doctor` CLI diagnostic command checking FFmpeg, FFprobe, Tesseract, Whisper/CUDA device, Node.js runtime, PO token provider status, and cookie files with `--json` support.
- Added `vidscope setup-pot` CLI helper command to automatically clone and build the standalone on-demand PO token generator (`generate_once.js`).
- Added ASR accuracy evaluation benchmark (`benchmarks/test_asr_accuracy.py`) calculating Word Error Rate (WER) and timestamp drift against reference transcripts.
- Added DASH video and audio stream pairing in `_format_choice` to download and mux separate video-only and audio-only tracks within size constraints, enabling local ASR transcription for modern YouTube sources.
- Added `soundfile` waveform loading in `SileroVadBackend` with channel-averaging and resampling fallback, removing runtime dependency on `torchcodec` under torchaudio >= 2.6.
- Added explicit `transcript_status` (`"completed"`, `"no_speech_detected"`, `"failed"`) and `transcript_error` tracking on timeline chunk sections.

### Changed
- Refactored `search_video` to be strictly post-analysis (`job_id` required; removed `source` and `language`), eliminating upfront agent bypass antipatterns and token-wasting caption fetch attempts.
- Enforced strict completion gating on `search_video`, rejecting in-flight jobs (`status != "completed"`) with typed `retryable=True` errors, `retry_after_seconds`, and `next_action="get_job_status"` to eliminate false negatives and agent hallucinations.
- Added structured timeline coverage metadata (`coverage`: `is_full_video`, `analyzed_start_seconds`, `analyzed_end_seconds`, `analyzed_duration_seconds`, `video_duration_seconds`) across `analyze_video`, `get_job_status`, and `search_video` to explicitly distinguish partial window analyses from whole-video coverage.
- Stripped conversational filler and narrative prose across MCP tools (`analyze_video`, `get_job_status`, `search_video`), replacing verbose summaries with dense, agent-first structured keys (`next_action`, `retry_after_seconds`, `coverage`) and concise hints.
- Disabled remote YouTube-native caption fetching in favor of direct local Whisper ASR for all YouTube sources, preventing rate-limiting (429) errors, avoiding video-only stream selection during media acquisition, and eliminating premature fallback to visual-only summaries.
- Enhanced `CaptionResolver` to utilize `curl_cffi` browser sessions with Chrome impersonation and cookie jar integration.
- Upgraded `FasterWhisperBackend` with dynamic device, compute type, and language-aware model resolution.
- Consolidated video analysis into a single entry point (`analyze_video`), removing duplicate paths (`get_video_timeline`, `get_video_transcript`, `start_video_analysis`).
- Refactored FastMCP server to deliver direct multimodal data blocks instead of requiring local filesystem artifact traversal.
- Refined visual-only chunk summaries to explicitly identify keyframe extraction and surface underlying causes when speech transcript or OCR is unavailable.
- Configured FastMCP loggers to `ERROR` during stdio server launch, ensuring zero stderr leakage during client argument validation failures.
- Added standard `if __name__ == "__main__": main()` entrypoint guard to `vidscope.mcp`.

### Fixed
- Fixed `view_frame` and chunk keyframe artifact glob resolution when extracting frames into `<output_dir>/<request_id>/frames/`.
- Prevented network duration inspection from blocking the `analyze_video` synchronous timeout window by moving probe inspection into background workers.
- Added graceful degradation to visual-only keyframe analysis in `analyze_video` when remote caption fetching fails or is rate-limited and local ASR is unavailable.
- Ensured terminal job failures in `get_job_status` and `analyze_video` emit standard `is_error=True` ToolResult envelopes.
- Added explicit mutual exclusivity validation and schema constraints (`Field`) in `view_frame` and `search_video`.
- Enforced schema-level `oneOf` mutual exclusivity on `view_frame` and `search_video` tool parameters.
- Added explicit cursor pagination (`next_since_chunk`), continuation flag (`has_more`), and actionable polling progress messages to `get_job_status`, and unified the cursor contract by including `next_since_chunk` and `has_more` in `analyze_video` initial processing and completion responses.
- Normalized FastMCP and Pydantic argument validation errors into Vidscope's typed `ToolResult(is_error=True, structured_content=...)` error envelope via `ErrorNormalizationMiddleware`.
- Stabilized dynamic ETA calculation in `JobState` using finished chunk duration averages, in-flight latency subtraction, and smoothing to eliminate erratic upward spikes.
- Exposed host runtime capabilities (`local_asr_available`, `local_ocr_available`) and structured `caption_tracks_listed` in `get_video_info` and `vidscope://info` to clarify preflight environment state before long analyses.
- Added explicit `"mode": "speech_and_visual" | "visual_only"` section tags and enriched visual-only summaries with on-screen OCR text or explicit host binary unavailability notes.
- Replaced static job completion estimates with dynamic remaining time estimation based on rolling chunk execution latency.
- Added natural language normalization (`english` -> `en`) and ISO format validation in `search_video`.
- Added `ocr_status` tracking on keyframes and informative chunk summaries detailing when OCR is disabled due to missing host binary.
- Registered standard MCP workflow prompts (`analyze_video_workflow`, `search_video_workflow`) for agent prompt discovery.
- Registered static `vidscope://info` resource for discovery via standard `resources/list` protocol calls.
- Suppressed FastMCP stderr startup banner and transport info logs by configuring log level to `WARNING` during stdio server launch.
- Added `language` parameter and available language hints to `search_video`, distinguishing between unlisted tracks and remote provider retrieval failures (e.g. HTTP 429).
- Fixed `ffprobe` execution failures by removing the unsupported `-nostdin` argument in `backends/media.py` and `backends/source.py`.
- Corrected frame extraction seek offset math for nonzero start offsets in `backends/media.py`.
- Fixed WebVTT timestamp decimal format to use standard `.` instead of `,`.
- Bounded caption track export to requested window range in `core.py`.
- Added missing FastMCP resource endpoints for `/manifest` and `/plan` URIs.
- Fixed Dockerfile installation by removing `--no-index` from local package install, enabling PEP 517 build backend resolution.
- Enforced `AnalysisTask` enum type safety and `TextContent` union narrowing for mypy compliance.
- Fixed audio track selection in `_format_choice` with multi-tier language-aware ranking, prioritizing requested language tracks, `original`, and `default` audio over foreign language dubs with slightly higher bitrates.
- Fixed chunk timestamp offset in `core.py` transcribe stage so transcribed segments and word timestamps are absolute (`offset = float(request.time_range.start_seconds)`), resolving overlapping duplicate segments and enabling full multi-chunk transcript aggregation.
- Fixed Silero VAD timestamp calculation in `SileroVadBackend.detect` for chunk offsets (`start_seconds > 0`).
- Added independent transcript coverage metrics to `JobState.coverage()`: `transcript_start_seconds`, `transcript_end_seconds`, `transcript_segments_count`, and `transcript_status`.
- Ensured strictly monotonic segment ordering by `(start_seconds, end_seconds)` in `JobState.full_transcript` and `get_transcript`.
- Added machine-readable recovery actions (`next_action: "analyze_video"`, `retry_after_seconds: 0`) to static missing/expired job errors across `get_job_status`, `search_video`, `get_transcript`, and `view_frame`.
- Enhanced `search_video` zero-match hint to expose actual transcript coverage range when searching completed jobs.
- Added audio overlap buffer (2.0s) and conservative exact-token boundary deduplication (`reconcile_transcript_segments`) across adjacent analysis chunks, eliminating boundary phrase truncation and duplicate transcript segments across seams with zero information loss.
- Replaced loose URL substring matching for YouTube detection with centralized strict hostname parsing and domain suffix validation (`is_youtube_source` in `vidscope.contracts`), resolving CodeQL `py/incomplete-url-substring-sanitization` alerts and preventing false-positive routing on non-YouTube sources.

### Removed
- Removed legacy file-dumping `analyze_video` tool and redundant intermediate MCP tools (`get_video_timeline`, `get_video_transcript`, `start_video_analysis`).
- Removed obsolete private URL parser `_transcript_video_id` in `backends/source.py`.

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
