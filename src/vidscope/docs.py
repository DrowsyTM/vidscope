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
            "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
            "duration_seconds": 213.0,
            "formatted_duration": "00:03:33",
            "has_captions": True,
            "languages": ["en"],
            "caption_tracks_listed": [
                {
                    "language": "en",
                    "kind": "manual",
                    "provider": "youtube",
                }
            ],
            "capabilities": {
                "local_asr_available": True,
                "local_ocr_available": True,
            },
            "chapters": [
                {
                    "title": "Intro",
                    "start_seconds": 0.0,
                    "end_seconds": 18.5,
                    "formatted_start": "00:00:00",
                    "formatted_end": "00:00:18",
                },
                {
                    "title": "Chorus",
                    "start_seconds": 18.5,
                    "end_seconds": 43.0,
                    "formatted_start": "00:00:18",
                    "formatted_end": "00:00:43",
                },
                {
                    "title": "Verse 2",
                    "start_seconds": 43.0,
                    "end_seconds": 85.0,
                    "formatted_start": "00:00:43",
                    "formatted_end": "00:01:25",
                },
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
            "status": "completed",
            "job_id": "job_e7b29a14-8f43-4c9b-98f2-1d573be04f21",
            "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "start_seconds": 0.0,
            "end_seconds": 180.0,
            "coverage": {
                "is_full_video": False,
                "analyzed_start_seconds": 0.0,
                "analyzed_end_seconds": 180.0,
                "analyzed_duration_seconds": 180.0,
                "video_duration_seconds": 213.0,
                "transcript_start_seconds": 18.5,
                "transcript_end_seconds": 175.2,
                "transcript_segments_count": 42,
                "transcript_status": "completed",
            },
            "total_chunks": 1,
            "completed_chunks": 1,
            "next_since_chunk": 1,
            "has_more": False,
            "timeline": [
                {
                    "chunk_index": 0,
                    "mode": "speech_and_visual",
                    "transcript_status": "completed",
                    "start_seconds": 0.0,
                    "end_seconds": 180.0,
                    "formatted_range": "00:00:00 - 00:03:00",
                    "summary": "We're no strangers to love. You know the rules and so do I...",
                    "keyframes": [
                        {
                            "frame_id": "frame_0000_003600",
                            "timestamp_seconds": 36.0,
                            "formatted_time": "00:00:36",
                            "ocr_text": "Rick Astley - Whenever You Need Somebody",
                            "ocr_confidence": 0.94,
                        },
                        {
                            "frame_id": "frame_0000_007200",
                            "timestamp_seconds": 72.0,
                            "formatted_time": "00:01:12",
                            "ocr_text": "RCA RECORDS 1987",
                            "ocr_confidence": 0.89,
                        },
                    ],
                }
            ],
            "hint": "Use search_video(job_id=...) to search transcript or view_frame(frame_id=...) to inspect frames.",
        },
        "async_response": {
            "status": "processing",
            "job_id": "job_e7b29a14-8f43-4c9b-98f2-1d573be04f21",
            "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "start_seconds": 0.0,
            "end_seconds": 180.0,
            "coverage": {
                "is_full_video": False,
                "analyzed_start_seconds": 0.0,
                "analyzed_end_seconds": 0.0,
                "analyzed_duration_seconds": 0.0,
                "video_duration_seconds": 213.0,
                "transcript_start_seconds": None,
                "transcript_end_seconds": None,
                "transcript_segments_count": 0,
                "transcript_status": "pending",
            },
            "total_chunks": 1,
            "completed_chunks": 0,
            "next_since_chunk": 0,
            "has_more": True,
            "next_action": "get_job_status",
            "estimated_remaining_seconds": 6.2,
            "initial_timeline": [],
            "hint": "Poll get_job_status(job_id='job_e7b29a14-8f43-4c9b-98f2-1d573be04f21')",
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
            "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "status": "completed",
            "progress_percentage": 100.0,
            "completed_chunks": 1,
            "total_chunks": 1,
            "estimated_remaining_seconds": 0.0,
            "coverage": {
                "is_full_video": False,
                "analyzed_start_seconds": 0.0,
                "analyzed_end_seconds": 180.0,
                "analyzed_duration_seconds": 180.0,
                "video_duration_seconds": 213.0,
                "transcript_start_seconds": 18.5,
                "transcript_end_seconds": 175.2,
                "transcript_segments_count": 42,
                "transcript_status": "completed",
            },
            "timeline": [
                {
                    "chunk_index": 0,
                    "mode": "speech_and_visual",
                    "transcript_status": "completed",
                    "start_seconds": 0.0,
                    "end_seconds": 180.0,
                    "formatted_range": "00:00:00 - 00:03:00",
                    "summary": "We're no strangers to love. You know the rules and so do I...",
                    "keyframes": [
                        {
                            "frame_id": "frame_0000_003600",
                            "timestamp_seconds": 36.0,
                            "formatted_time": "00:00:36",
                            "ocr_text": "Rick Astley - Whenever You Need Somebody",
                            "ocr_status": "completed",
                            "surrounding_dialogue": "We're no strangers to love",
                        },
                        {
                            "frame_id": "frame_0000_007200",
                            "timestamp_seconds": 72.0,
                            "formatted_time": "00:01:12",
                            "ocr_text": "RCA RECORDS 1987",
                            "ocr_status": "completed",
                            "surrounding_dialogue": "Inside we both know what's been going on",
                        },
                    ],
                }
            ],
            "timeline_chunks_returned": 1,
            "total_timeline_chunks": 1,
            "next_since_chunk": 1,
            "has_more": False,
            "next_action": None,
            "retry_after_seconds": 0.0,
            "message": "Analysis completed successfully (1/1 chunks). Returned 1 timeline section(s).",
            "error": None,
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
            "window_duration_seconds": 30.0,
            "coverage": {
                "is_full_video": False,
                "analyzed_start_seconds": 0.0,
                "analyzed_end_seconds": 180.0,
                "analyzed_duration_seconds": 180.0,
                "video_duration_seconds": 213.0,
                "transcript_start_seconds": 18.5,
                "transcript_end_seconds": 175.2,
                "transcript_segments_count": 42,
                "transcript_status": "completed",
            },
            "segments_count": 2,
            "segments": [
                {
                    "start_seconds": 18.5,
                    "end_seconds": 22.1,
                    "formatted_time": "00:00:18",
                    "text": "We're no strangers to love",
                },
                {
                    "start_seconds": 22.8,
                    "end_seconds": 26.4,
                    "formatted_time": "00:00:22",
                    "text": "You know the rules and so do I",
                },
            ],
            "text": "We're no strangers to love You know the rules and so do I",
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
            "is_regex": False,
            "case_sensitive": False,
            "coverage": {
                "is_full_video": False,
                "analyzed_start_seconds": 0.0,
                "analyzed_end_seconds": 180.0,
                "analyzed_duration_seconds": 180.0,
                "video_duration_seconds": 213.0,
                "transcript_start_seconds": 18.5,
                "transcript_end_seconds": 175.2,
                "transcript_segments_count": 42,
                "transcript_status": "completed",
            },
            "matches_count": 1,
            "matches": [
                {
                    "start_seconds": 18.5,
                    "end_seconds": 22.1,
                    "formatted_time": "00:00:18",
                    "snippet": "We're no strangers to love",
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
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "frame_id": "frame_0000_003600",
                            "timestamp_seconds": 36.0,
                            "formatted_time": "00:00:36",
                            "ocr_text": "Rick Astley - Whenever You Need Somebody",
                        },
                        indent=2,
                    ),
                },
                {
                    "type": "image",
                    "mimeType": "image/jpeg",
                    "data": "<base64_encoded_jpeg_bytes>",
                },
            ]
        },
    },
}

ERROR_RESPONSE_EXAMPLE = {
    "is_error": True,
    "structured_content": {
        "ok": False,
        "status": "failed",
        "code": "URL_SCHEME_NOT_ALLOWED",
        "stage": "validate_source",
        "message": "source URL scheme is not allowed",
        "retryable": False,
        "diagnostics": {
            "details": [],
        },
        "artifact_refs": [],
        "artifacts": [],
        "manifest_uri": None,
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
    """Extract numeric/range constraints from schema, including anyOf variants."""
    sub_schemas = [schema]
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        sub_schemas.extend(s for s in schema["anyOf"] if isinstance(s, dict))
    if "allOf" in schema and isinstance(schema["allOf"], list):
        sub_schemas.extend(s for s in schema["allOf"] if isinstance(s, dict))

    parts: list[str] = []
    for s in sub_schemas:
        if "minimum" in s:
            parts.append(f">= {s['minimum']}")
        if "exclusiveMinimum" in s:
            parts.append(f"> {s['exclusiveMinimum']}")
        if "maximum" in s:
            parts.append(f"<= {s['maximum']}")
        if "exclusiveMaximum" in s:
            parts.append(f"< {s['exclusiveMaximum']}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_parts: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)

    return ", ".join(unique_parts) if unique_parts else "-"


async def generate_mcp_markdown(server: Any = None) -> str:
    """Generate markdown documentation from the FastMCP server instance."""
    active_mcp = server if server is not None else mcp
    tools = await active_mcp.list_tools()
    tool_map = {t.name: t for t in tools}

    preferred_order = [
        "get_video_info",
        "analyze_video",
        "get_job_status",
        "get_transcript",
        "search_video",
        "view_frame",
    ]
    ordered_tool_names = [name for name in preferred_order if name in tool_map]
    for name in sorted(tool_map.keys()):
        if name not in ordered_tool_names:
            ordered_tool_names.append(name)

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

            if "async_response" in example_data:
                lines.append(
                    "#### Realistic Response Example (Synchronous Completion <= 5.0s)"
                )
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(example_data["response"], indent=2))
                lines.append("```")
                lines.append("")

                lines.append(
                    "#### Realistic Response Example (Async Background Handoff > 5.0s)"
                )
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(example_data["async_response"], indent=2))
                lines.append("```")
                lines.append("")
            else:
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
        desc = (
            (
                getattr(r, "description", None)
                or "Server capabilities, active version, and available resource URI templates."
            )
            .strip()
            .replace("\n", " ")
        )
        lines.append(f"| `{getattr(r, 'uri', '')}` | `{r.name}` | {desc} |")
    for rt in resource_templates:
        fallback = {
            "read_artifact": "Retrieve raw artifact content or paginated text by run ID and artifact ID.",
            "read_manifest": "Retrieve execution manifest and output file metadata for an analysis run.",
            "read_plan": "Retrieve deterministic DAG execution plan JSON for an analysis run.",
        }.get(rt.name, "Local run artifacts and DAG execution graphs.")
        desc = (getattr(rt, "description", None) or fallback).strip().replace("\n", " ")
        lines.append(f"| `{rt.uri_template}` | `{rt.name}` | {desc} |")

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
