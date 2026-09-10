from __future__ import annotations

import asyncio
from pathlib import Path

from typer.testing import CliRunner

from vidscope.cli import app
from vidscope.docs import generate_mcp_markdown, run_docs_generation


def test_generate_mcp_markdown_contains_all_tools() -> None:
    content = asyncio.run(generate_mcp_markdown())
    assert "# Vidscope MCP Tool Reference" in content
    assert "### `get_video_info`" in content
    assert "### `analyze_video`" in content
    assert "### `get_job_status`" in content
    assert "### `get_transcript`" in content
    assert "### `search_video`" in content
    assert "### `view_frame`" in content
    assert "## MCP Resources" in content
    assert "## MCP Prompts" in content
    assert "Realistic Request Example" in content
    assert "Realistic Response Example" in content


def test_docs_reference_is_current() -> None:
    doc_path = Path(__file__).resolve().parent.parent / "docs" / "MCP_REFERENCE.md"
    assert doc_path.is_file(), "docs/MCP_REFERENCE.md should exist"
    code = run_docs_generation(output=doc_path, check=True)
    assert code == 0


def test_docs_check_detects_drift(tmp_path: Path) -> None:
    out = tmp_path / "STALE.md"
    out.write_text("Old content that has drifted", encoding="utf-8")
    code = run_docs_generation(output=out, check=True)
    assert code == 1


def test_cli_docs_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["docs", "--check"])
    assert result.exit_code == 0
