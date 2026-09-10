"""FastMCP documentation generator with schema introspection and realistic fixtures."""

from __future__ import annotations

import asyncio
import difflib
import json
import sys
from pathlib import Path
from typing import Any

from .mcp import mcp

# Curated realistic fixtures paired with each MCP tool
TOOL_EXAMPLES: dict[str, dict[str, Any]] = {
    "get_video_info": {
        "description": "Fast preflight metadata discovery without downloading media.",
        "request": {
            "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        },
        "response": {
            "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "duration_seconds": 213.0,
            "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
            "available_subtitles": [
                {
                    "language": "en",
                    "kind": "manual",
                    "provider": "youtube",
                    "source_url": None,
                }
            ],
            "chapters": [
                {"title": "Intro", "start_seconds": 0.0, "end_seconds": 18.5},
                {"title": "Chorus", "start_seconds": 18.5, "end_seconds": 43.0},
                {"title": "Verse 2", "start_seconds": 43.0, "end_seconds": 85.0},
            ],
            "formats": [
                {"format_id": "18", "ext": "mp4", "resolution": "640x360"},
                {"format_id": "22", "ext": "mp4", "resolution": "1280x720"},
                {"format_id": "140", "ext": "m4a", "resolution": "audio-only"},
            ],
        },
    },
    "analyze_video": {
        "description": "Bounded local video analysis across timeline chunks with visual keyframes and speech transcript.",
        "request": {
            "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "start_seconds": 0.0,
            "end_seconds": 180.0,
            "chunk_duration_seconds": 180.0,
            "sync_timeout_seconds": 5.0,
        },
        "response": {
            "job_id": "job_e7b29a14-8f43-4c9b-98f2-1d573be04f21",
            "status": "processing",
            "progress": 0.45,
            "current_stage": "extract_frames",
            "estimated_remaining_seconds": 6.2,
            "coverage": {
                "start_seconds": 0.0,
                "end_seconds": 180.0,
                "duration_seconds": 180.0,
            },
            "timeline": [
                {
                    "chunk_index": 0,
                    "time_range": {"start_seconds": 0.0, "end_seconds": 180.0},
                    "transcript_status": "completed",
                    "keyframes": [
                        {
                            "frame_id": "frame_0000_003600",
                            "timestamp_seconds": 36.0,
                            "ocr_text": "Rick Astley - Whenever You Need Somebody",
                            "ocr_confidence": 0.94,
                        },
                        {
                            "frame_id": "frame_0000_007200",
                            "timestamp_seconds": 72.0,
                            "ocr_text": "RCA RECORDS 1987",
                            "ocr_confidence": 0.89,
                        },
                    ],
                }
            ],
        },
    },
    "get_job_status": {
        "description": "Incremental status check and chunk retrieval for background video analysis jobs.",
        "request": {
            "job_id": "job_e7b29a14-8f43-4c9b-98f2-1d573be04f21",
            "since_chunk": 0,
        },
        "response": {
            "job_id": "job_e7b29a14-8f43-4c9b-98f2-1d573be04f21",
            "status": "completed",
            "progress": 1.0,
            "total_chunks": 1,
            "completed_chunks": 1,
            "timeline": [
                {
                    "chunk_index": 0,
                    "time_range": {"start_seconds": 0.0, "end_seconds": 180.0},
                    "transcript_status": "completed",
                    "keyframes": [
                        {
                            "frame_id": "frame_0000_003600",
                            "timestamp_seconds": 36.0,
                            "ocr_text": "Rick Astley - Whenever You Need Somebody",
                            "ocr_confidence": 0.94,
                        },
                        {
                            "frame_id": "frame_0000_007200",
                            "timestamp_seconds": 72.0,
                            "ocr_text": "RCA RECORDS 1987",
                            "ocr_confidence": 0.89,
                        },
                        {
                            "frame_id": "frame_0000_010800",
                            "timestamp_seconds": 108.0,
                            "ocr_text": "",
                            "ocr_confidence": 0.0,
                        },
                        {
                            "frame_id": "frame_0000_014400",
                            "timestamp_seconds": 144.0,
                            "ocr_text": "",
                            "ocr_confidence": 0.0,
                        },
                    ],
                }
            ],
        },
    },
    "get_transcript": {
        "description": "Retrieve timestamped dialogue text and word segments within a bounded time window.",
        "request": {
            "job_id": "job_e7b29a14-8f43-4c9b-98f2-1d573be04f21",
            "start_seconds": 15.0,
            "end_seconds": 45.0,
            "max_duration_seconds": 300.0,
        },
        "response": {
            "job_id": "job_e7b29a14-8f43-4c9b-98f2-1d573be04f21",
            "start_seconds": 15.0,
            "end_seconds": 45.0,
            "segment_count": 2,
            "dialogue_text": "We're no strangers to love. You know the rules and so do I.",
            "segments": [
                {
                    "start_seconds": 18.5,
                    "end_seconds": 22.1,
                    "text": "We're no strangers to love",
                    "confidence": 0.98,
                },
                {
                    "start_seconds": 22.8,
                    "end_seconds": 26.4,
                    "text": "You know the rules and so do I",
                    "confidence": 0.97,
                },
            ],
        },
    },
    "search_video": {
        "description": "Locate keywords or regex patterns in transcript speech and on-screen OCR text.",
        "request": {
            "job_id": "job_e7b29a14-8f43-4c9b-98f2-1d573be04f21",
            "query": "strangers to love",
            "is_regex": False,
            "case_sensitive": False,
            "max_matches": 10,
        },
        "response": {
            "job_id": "job_e7b29a14-8f43-4c9b-98f2-1d573be04f21",
            "query": "strangers to love",
            "total_matches": 1,
            "matches": [
                {
                    "kind": "transcript",
                    "timestamp_seconds": 18.5,
                    "text": "We're no strangers to love",
                    "context": "...We're no strangers to love. You know the rules...",
                    "confidence": 0.98,
                }
            ],
        },
    },
    "view_frame": {
        "description": "View a video frame as a native MCP Image content block with JPEG encoding.",
        "request": {
            "frame_id": "frame_0000_003600",
            "max_width": 1280,
        },
        "response": {
            "frame_id": "frame_0000_003600",
            "timestamp_seconds": 36.0,
            "width": 1280,
            "height": 720,
            "mime_type": "image/jpeg",
            "image_data": "<base64_encoded_jpeg_bytes>",
        },
    },
}

ERROR_RESPONSE_EXAMPLE = {
    "is_error": True,
    "content": {
        "code": "INVALID_REQUEST",
        "stage": "validate_source",
        "message": "source must be an existing local file or an HTTPS video URL",
        "retryable": False,
        "diagnostics": {
            "source": "ftp://invalid-url.com",
            "supported_schemes": ["https", "http", "file"],
        },
    },
}


def _format_type(schema: dict[str, Any]) -> str:
    """Format JSON Schema type into human-readable representation."""
    if "anyOf" in schema:
        types = [_format_type(s) for s in schema["anyOf"]]
        return " \\| ".join(t for t in types if t)
    if "type" in schema:
        t = schema["type"]
        if t == "null":
            return "null"
        return str(t)
    return "any"


def _format_constraints(schema: dict[str, Any]) -> str:
    """Extract numeric/range constraints from schema."""
    parts: list[str] = []
    if "minimum" in schema:
        parts.append(f">= {schema['minimum']}")
    if "exclusiveMinimum" in schema:
        parts.append(f"> {schema['exclusiveMinimum']}")
    if "maximum" in schema:
        parts.append(f"<= {schema['maximum']}")
    if "exclusiveMaximum" in schema:
        parts.append(f"< {schema['exclusiveMaximum']}")
    return ", ".join(parts) if parts else "-"


async def generate_mcp_markdown(server: Any = None) -> str:
    """Generate markdown documentation from the FastMCP server instance."""
    active_mcp = server if server is not None else mcp
    tools = await active_mcp.list_tools()
    tool_map = {t.name: t for t in tools}

    ordered_tool_names = [
        "get_video_info",
        "analyze_video",
        "get_job_status",
        "get_transcript",
        "search_video",
        "view_frame",
    ]

    resources = (
        await active_mcp.list_resources()
        if hasattr(active_mcp, "list_resources")
        else []
    )
    resource_templates = (
        await active_mcp.list_resource_templates()
        if hasattr(active_mcp, "list_resource_templates")
        else []
    )
    prompts = (
        await active_mcp.list_prompts() if hasattr(active_mcp, "list_prompts") else []
    )

    lines: list[str] = [
        "# Vidscope MCP Tool Reference",
        "",
        "> **Notice**: This document is auto-generated by `scripts/generate_mcp_docs.py` (or `vidscope docs`).",
        "> Any edits should be made directly to tool schemas in [`src/vidscope/mcp.py`](../src/vidscope/mcp.py).",
        "",
        "Vidscope exposes a Model Context Protocol (MCP) server over standard I/O (`stdio`).",
        "Clients and agent harnesses can connect via `vidscope-mcp` or `vidscope mcp`.",
        "",
        "## Interactive Inspection",
        "",
        "To inspect tools, schemas, and live executions interactively using the official MCP Inspector GUI:",
        "",
        "```bash",
        "npx @modelcontextprotocol/inspector uv run vidscope-mcp",
        "```",
        "",
        "---",
        "",
        "## Standard Error Envelope",
        "",
        "All tools normalize unexpected exceptions or schema validation failures into a unified error envelope",
        "with `is_error: true` and a structured `AnalysisError` dictionary:",
        "",
        "```json",
        json.dumps(ERROR_RESPONSE_EXAMPLE, indent=2),
        "```",
        "",
        "---",
        "",
        "## Tools",
        "",
    ]

    for name in ordered_tool_names:
        t = tool_map.get(name)
        if not t:
            continue

        lines.append(f"### `{name}`")
        lines.append("")
        doc = (t.description or "").strip()
        lines.append(doc)
        lines.append("")

        annotations = getattr(t, "annotations", None)
        hints: list[str] = []
        if getattr(annotations, "readOnlyHint", False):
            hints.append("`readOnly: true`")
        if getattr(annotations, "idempotentHint", False):
            hints.append("`idempotent: true`")
        if hints:
            lines.append(f"**Hints**: {', '.join(hints)}")
            lines.append("")

        params_schema = getattr(t, "parameters", {}) or {}
        properties = params_schema.get("properties", {})
        required_fields = set(params_schema.get("required", []))

        lines.append("#### Input Parameters")
        lines.append("")
        lines.append(
            "| Parameter | Type | Required / Default | Constraints | Description |"
        )
        lines.append("|:---|:---|:---|:---|:---|")

        for prop_name, prop_meta in properties.items():
            type_str = f"`{_format_type(prop_meta)}`"
            if prop_name in required_fields:
                req_str = "**Required**"
            else:
                default_val = prop_meta.get("default")
                req_str = f"Optional (default: `{default_val!r}`)"
            constraints = _format_constraints(prop_meta)
            description = (
                prop_meta.get("description", "-").replace("\n", " ").replace("|", "\\|")
            )
            lines.append(
                f"| `{prop_name}` | {type_str} | {req_str} | {constraints} | {description} |"
            )

        lines.append("")

        one_of = params_schema.get("oneOf")
        if one_of:
            lines.append("**Mutual Exclusivity Constraints (`oneOf`)**:")
            for clause in one_of:
                reqs = ", ".join(f"`{r}`" for r in clause.get("required", []))
                lines.append(f"- Requires: {reqs}")
            lines.append("")

        example_data = TOOL_EXAMPLES.get(name)
        if example_data:
            lines.append("#### Realistic Request Example")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(example_data["request"], indent=2))
            lines.append("```")
            lines.append("")

            lines.append("#### Realistic Response Example")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(example_data["response"], indent=2))
            lines.append("```")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## MCP Resources")
    lines.append("")
    lines.append(
        "Vidscope provides static and dynamic resources for inspecting runs, DAG plans, and server telemetry."
    )
    lines.append("")
    lines.append("| Resource URI / Template | Name | Description |")
    lines.append("|:---|:---|:---|")

    for r in resources:
        lines.append(
            f"| `{getattr(r, 'uri', '')}` | `{r.name}` | Server identity, capabilities, and tool status. |"
        )
    for rt in resource_templates:
        lines.append(
            f"| `{rt.uri_template}` | `{rt.name}` | Local run artifacts and DAG execution graphs. |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## MCP Prompts")
    lines.append("")
    lines.append(
        "Vidscope registers guided prompt workflows to help agentic orchestrators complete common video tasks."
    )
    lines.append("")
    lines.append("| Prompt Name | Description |")
    lines.append("|:---|:---|")
    for p in prompts:
        desc = (p.description or "-").replace("\n", " ")
        lines.append(f"| `{p.name}` | {desc} |")

    lines.append("")
    return "\n".join(lines)


def run_docs_generation(output: Path, check: bool = False) -> int:
    """Generate or check documentation file against current FastMCP schemas."""
    content = asyncio.run(generate_mcp_markdown())

    if check:
        if not output.is_file():
            sys.stderr.write(
                f"Error: Target file '{output}' does not exist. Run without --check to generate it.\n"
            )
            return 1
        existing = output.read_text(encoding="utf-8")
        if existing.strip() != content.strip():
            sys.stderr.write(
                f"Error: '{output}' is out of date with current MCP schemas.\n"
            )
            diff = difflib.unified_diff(
                existing.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=str(output),
                tofile="generated",
            )
            sys.stderr.writelines(diff)
            return 1
        print(f"Documentation check passed: '{output}' is up to date.")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Successfully generated MCP documentation in '{output}'.")
    return 0
