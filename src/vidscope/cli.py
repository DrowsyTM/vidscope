from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from .contracts import AnalysisError, AnalyzeVideoRequest, ErrorCode, TimeRange
from .core import VideoAnalyzerFailure, analyze_video

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _root(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose debug logging to stderr."),
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress non-error logging output.")
    ] = False,
) -> None:
    """Video analysis command group."""
    import logging

    from .logging import configure_logging

    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    configure_logging(level=level)
    return


def _emit(value: object) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _invalid_request(error: BaseException) -> AnalysisError:
    return AnalysisError(
        code=ErrorCode.INVALID_REQUEST,
        stage="validate_source",
        message=str(error)[:2_048] or "request validation failed",
        retryable=False,
    )


@app.command("analyze-video")
def analyze_video_command(
    source: Annotated[
        str, typer.Option("--source", help="Local path or HTTPS video URL.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")],
    start_seconds: Annotated[float, typer.Option("--start-seconds")] = 0.0,
    end_seconds: Annotated[float, typer.Option("--end-seconds")] = 180.0,
    task: Annotated[
        list[str] | None,
        typer.Option("--task", help="Task; repeat for multiple tasks."),
    ] = None,
    frame_timestamp: Annotated[
        list[float] | None,
        typer.Option("--frame-timestamp", help="Explicit frame timestamp; repeatable."),
    ] = None,
    language: Annotated[str, typer.Option("--language")] = "en",
    caption_preference: Annotated[
        str, typer.Option("--caption-preference")
    ] = "manual_then_automatic_then_asr",
    asr_enabled: Annotated[bool, typer.Option("--asr-enabled/--no-asr")] = True,
    max_frames: Annotated[int, typer.Option("--max-frames")] = 6,
    max_frame_width: Annotated[int, typer.Option("--max-frame-width")] = 1_280,
    max_download_bytes: Annotated[
        int, typer.Option("--max-download-bytes")
    ] = 268_435_456,
    max_output_bytes: Annotated[int, typer.Option("--max-output-bytes")] = 67_108_864,
    timeout_seconds: Annotated[int, typer.Option("--timeout-seconds")] = 600,
    request_id: Annotated[str | None, typer.Option("--request-id")] = None,
) -> None:
    try:
        values: dict[str, Any] = {
            "source": source,
            "time_range": TimeRange(
                start_seconds=start_seconds, end_seconds=end_seconds
            ),
            "tasks": set(task or ()) if task else {"metadata", "transcript"},
            "output_directory": out.expanduser().resolve(strict=False),
            "language": language,
            "caption_preference": caption_preference,
            "asr_enabled": asr_enabled,
            "frame_timestamps_seconds": tuple(frame_timestamp or ())
            if frame_timestamp
            else None,
            "max_frames": max_frames,
            "max_frame_width": max_frame_width,
            "max_download_bytes": max_download_bytes,
            "max_output_bytes": max_output_bytes,
            "timeout_seconds": timeout_seconds,
            "request_id": request_id,
        }
        request = AnalyzeVideoRequest(**values)
    except (ValidationError, ValueError, TypeError) as exc:
        _emit(_invalid_request(exc))
        raise typer.Exit(code=2) from exc
    try:
        result = analyze_video(request)
    except VideoAnalyzerFailure as exc:
        _emit(exc.error)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        _emit(
            AnalysisError(
                code=ErrorCode.INTERNAL_STAGE_FAILED,
                stage="orchestration",
                message=str(exc)[:2_048] or "analysis failed",
                retryable=False,
            )
        )
        raise typer.Exit(code=1) from exc
    _emit(result)


@app.command("mcp")
def mcp_command() -> None:
    """Run FastMCP server over standard I/O."""
    from .mcp import main

    main()


@app.command("doctor")
def doctor_command(
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit diagnostic report as JSON.")
    ] = False,
) -> None:
    """Check system requirements, media binaries, and YouTube token configuration."""
    from .doctor import run_doctor

    report = run_doctor()
    if as_json:
        _emit(report)
    else:
        typer.echo("Vidscope System Diagnostics")
        typer.echo("===========================")
        for check in report.checks:
            if check.status == "ok":
                symbol = typer.style("[✓]", fg=typer.colors.GREEN, bold=True)
            elif check.status == "warning":
                symbol = typer.style("[!]", fg=typer.colors.YELLOW, bold=True)
            elif check.status == "missing":
                symbol = typer.style("[✗]", fg=typer.colors.RED, bold=True)
            else:  # optional_missing
                symbol = typer.style("[-]", fg=typer.colors.BLUE)

            typer.echo(f"{symbol} {check.name}: {check.message}")
            if check.recommendation:
                hint = (
                    typer.style("    Tip: ", fg=typer.colors.CYAN)
                    + check.recommendation
                )
                typer.echo(hint)

        typer.echo("")
        if report.ok:
            status_text = typer.style(report.summary, fg=typer.colors.GREEN, bold=True)
        else:
            status_text = typer.style(report.summary, fg=typer.colors.RED, bold=True)
        typer.echo(status_text)

    if not report.ok:
        raise typer.Exit(code=1)


@app.command("setup-pot")
def setup_pot_command(
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            help="Target directory for bgutil provider (defaults to ~/bgutil-ytdlp-pot-provider).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force rebuild even if already present."),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit setup result as JSON.")
    ] = False,
) -> None:
    """Clone and compile the bgutil on-demand PO token generation script."""
    from .doctor import setup_pot_provider

    if not as_json:
        typer.echo("Setting up Proof-of-Origin (PO) token provider...")

    result = setup_pot_provider(target_dir=path, force=force)

    if as_json:
        _emit(result)
    else:
        if result.get("ok"):
            typer.echo(
                typer.style(
                    f"[✓] {result.get('message')}", fg=typer.colors.GREEN, bold=True
                )
            )
            typer.echo(f"    Script path: {result.get('path')}")
            typer.echo("    yt-dlp will automatically invoke this script on-demand.")
        else:
            typer.echo(
                typer.style(
                    f"[✗] Setup failed: {result.get('error')}",
                    fg=typer.colors.RED,
                    bold=True,
                )
            )

    if not result.get("ok"):
        raise typer.Exit(code=1)


@app.command("docs")
def docs_command(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Target markdown file path."),
    ] = Path("docs/MCP_REFERENCE.md"),
    check: Annotated[
        bool,
        typer.Option("--check", help="Check if existing documentation is up to date."),
    ] = False,
) -> None:
    """Generate or verify MCP tool documentation."""
    from .docs import run_docs_generation

    code = run_docs_generation(output=output, check=check)
    if code != 0:
        raise typer.Exit(code=code)


@app.command("eval")
def eval_command(
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            "-p",
            help="Single evaluation prompt to test agent tool execution.",
        ),
    ] = None,
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Path to YAML benchmark dataset file.",
        ),
    ] = None,
    protocol_suite: Annotated[
        bool,
        typer.Option(
            "--protocol-suite",
            help="Run standardized Phase 1 protocol hygiene test suite.",
        ),
    ] = False,
    runs: Annotated[
        int,
        typer.Option(
            "--runs",
            "-n",
            help="Number of iterations to execute for prompt or protocol suite.",
        ),
    ] = 1,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            "-c",
            help="Number of concurrent worker threads for batched execution.",
        ),
    ] = 4,
    thinking: Annotated[
        str,
        typer.Option(
            "--thinking",
            help="Thinking level for OMP: off, minimal, low, medium, high, max.",
        ),
    ] = "off",
    approval_mode: Annotated[
        str,
        typer.Option("--approval-mode", help="OMP approval mode."),
    ] = "yolo",
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout", help="Timeout per evaluation task in seconds."),
    ] = 180.0,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="OMP model override."),
    ] = None,
    omp_bin: Annotated[
        str | None,
        typer.Option("--omp-bin", help="Path to OMP binary."),
    ] = None,
    mock: Annotated[
        bool,
        typer.Option(
            "--mock/--no-mock",
            help="Use mock FastMCP mode for rapid zero-cost protocol evaluations.",
        ),
    ] = True,
    mock_async: Annotated[
        str | None,
        typer.Option(
            "--mock-async",
            help="Mock async behavior: 'auto', 'sync', 'async', or 'random'.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Target markdown file to save report."),
    ] = None,
) -> None:
    """Run agent timeline comprehension and protocol hygiene evaluations via OMP."""
    from .eval_harness import load_benchmark_dataset
    from .eval_runner import OMPEvalRunner

    if not prompt and not dataset and not protocol_suite:
        typer.echo(
            "Error: Must provide either --prompt, --dataset, or --protocol-suite.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        runner = OMPEvalRunner(
            omp_bin=omp_bin,
            thinking=thinking,
            approval_mode=approval_mode,
            model=model,
            timeout_seconds=timeout_seconds,
            mock_mcp=mock,
            mock_async=mock_async,
        )

        if protocol_suite:
            target_runs = runs if runs > 1 else 100
            typer.echo(
                f"Running Phase 1 Protocol Hygiene Suite ({target_runs} runs, concurrency={concurrency}, thinking={thinking})..."
            )

            def _on_suite_run(sc_id: str, done: int, total: int, res: Any) -> None:
                status = "OK" if not res.scorecard.schema_validation_errors else "FAIL"
                typer.echo(
                    f"  [{done}/{total}] scenario={sc_id} calls={len(res.traces)} ({status}) in {res.duration_seconds:.2f}s"
                )

            suite_report = runner.run_protocol_suite(
                total_runs=target_runs,
                concurrency=concurrency,
                on_run_complete=_on_suite_run,
            )
            report_md = suite_report.to_markdown()
            typer.echo("\n" + report_md)

            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(report_md, encoding="utf-8")
                typer.echo(f"\nReport written to: {output}")
            return

        if prompt:
            if runs > 1:
                typer.echo(
                    f"Running prompt repeatedly ({runs} runs, concurrency={concurrency}, thinking={thinking})..."
                )

                def _on_prompt_run(cur: int, tot: int, res: Any) -> None:
                    typer.echo(
                        f"  [{cur}/{tot}] calls={len(res.traces)} errors={res.scorecard.schema_validation_errors} in {res.duration_seconds:.2f}s"
                    )

                scorecard, _ = runner.run_repeated_prompt(
                    prompt,
                    runs=runs,
                    concurrency=concurrency,
                    on_run_complete=_on_prompt_run,
                )
                report_md = (
                    f"# Repeated Prompt Protocol Scorecard ({runs} runs)\n\n"
                    + "## 1. Protocol Hygiene Scorecard\n\n"
                    + scorecard.to_markdown_table()
                    + "\n\n## 2. Tool Usage Statistics\n\n"
                    + scorecard.tool_usage_markdown_table(total_runs=runs)
                )
                typer.echo("\n" + report_md)

                if output:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(report_md, encoding="utf-8")
                    typer.echo(f"\nReport written to: {output}")
            else:
                typer.echo(f"Running prompt via OMP (thinking={thinking})...")
                res = runner.run_prompt(prompt)
                typer.echo("\n### Agent Response:")
                typer.echo(res.final_text)
                typer.echo("\n### Protocol Scorecard:")
                typer.echo(res.scorecard.to_markdown_table())
                typer.echo("\n### Tool Usage Statistics:")
                typer.echo(res.scorecard.tool_usage_markdown_table(total_runs=1))
            return

        if dataset:
            ds = load_benchmark_dataset(str(dataset))
            typer.echo(
                f"Running benchmark dataset '{ds.video_id}' via OMP (thinking={thinking})..."
            )
            report = runner.run_dataset(ds)
            report_md = report.to_markdown()
            typer.echo("\n" + report_md)

            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(report_md, encoding="utf-8")
                typer.echo(f"\nReport written to: {output}")
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()


__all__ = ["app"]
