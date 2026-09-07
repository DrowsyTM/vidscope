# Agent & Developer Operating Manual: vidscope

This document establishes operational constraints, architectural invariants, and verification workflows for autonomous AI coding agents and human contributors interacting with `vidscope`.

---

## 1. Core Architectural Invariants

### Pure Standard Output (MCP & CLI)
- **Stdout is reserved exclusively for structured protocol traffic**:
  - In MCP mode (`vidscope mcp`), stdout carries framed JSON-RPC messages.
  - In CLI mode (`vidscope analyze-video`), stdout carries exactly one single-line JSON envelope upon completion.
- **Stderr is the only permitted channel for logging**:
  - `vidscope.logging.configure_logging()` strictly routes all application and library logs to `sys.stderr`.
  - Subprocess callers and third-party libraries (e.g. `yt-dlp`, FFmpeg) must have standard output redirected or muffled so stdout remains uncontaminated.

### Subprocess Sandboxing & Hardening
- **Subprocess flags**:
  - Every invocation of `ffmpeg` and `ffprobe` MUST include:
    - `-nostdin` to prevent subprocesses from hanging on standard input.
    - `-protocol_whitelist "file,pipe,crypto,data"` to prevent arbitrary protocol schemes.
  - Every invocation of `tesseract` MUST include the POSIX option terminator `--` immediately before the input frame path:
    ```python
    command = [tesseract_bin, "--", str(frame_path), "stdout", ...]
    ```
- **Execution timeouts**:
  - Subprocesses must be strictly bounded with timeouts (e.g. 300s for media, 60s for OCR).

### Layered SSRF Prevention
- Network URL inputs and caption downloads must be validated by `_validate_network_url`:
  - Enforce HTTPS scheme.
  - Reject user credentials, query fragments, and localhost references.
  - Disallow loopback, link-local, multicast, and RFC 1918 private IPv4/IPv6 address spaces.

### Deterministic DAG Planning
- The planner in `vidscope.planner` builds an ordered DAG of stages before executing media workloads.
- The planner inspects declared `Capabilities` and enforces hard limits:
  - Time range must be strictly bounded (`0 <= start < end`, `end - start <= 180s`).
  - Frame count is capped at `12`.
  - Size and output bytes are strictly checked.
- Missing capabilities produce actionable errors guiding users to install optional extras (e.g. `pip install 'vidscope[asr]'`).

---

## 2. Dependency Structure & Optional Extras

Heavy ML runtimes are separated into optional package extras to preserve lean container images and CLI environments:

- `vidscope`: Core CLI, FastMCP server, metadata extraction, native caption parsing.
- `vidscope[asr]`: Speech transcription with `faster-whisper` and voice activity detection with `silero-vad`.
- `vidscope[vad]`: Voice activity detection only (`silero-vad`).
- `vidscope[all]`: All optional extras combined.
- `vidscope[dev]`: Testing, coverage (`pytest-cov`), and benchmarking (`pytest-benchmark`, `pillow`).

---

## 3. Development & Verification Workflow

Always verify changes using the standardized verification commands:

```bash
# Linting and formatting
uv run ruff check .
uv run ruff format --check .

# Type checking
uv run mypy src tests

# Unit test suite with coverage
uv run pytest --cov=vidscope --cov-report=term-missing

# Performance benchmarks
uv run pytest benchmarks/ --benchmark-skip
```
