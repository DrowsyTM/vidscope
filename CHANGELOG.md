# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
