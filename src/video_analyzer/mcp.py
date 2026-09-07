from __future__ import annotations

from typing import Any

from fastmcp import FastMCP
from fastmcp.tools.base import ToolResult
from pydantic import ValidationError

from .artifacts import ArtifactStoreFailure, read_artifact_resource
from .contracts import AnalysisError, AnalysisResult, AnalyzeVideoRequest, ErrorCode
from .core import AnalysisContext, VideoAnalyzerFailure
from .core import analyze_video as core_analyze_video
from .logging import configure_logging

mcp = FastMCP("video-analyzer")


def _error_payload(error: AnalysisError) -> dict[str, Any]:
    return error.model_dump(mode="json")


@mcp.tool(
    name="analyze_video",
    annotations={"readOnlyHint": False, "idempotentHint": False},
)
def analyze_video(
    request: AnalyzeVideoRequest,
    ctx: Any = None,
) -> AnalysisResult | ToolResult:
    """Analyze one bounded local or HTTPS video through the shared core API."""

    context: AnalysisContext | None = None
    if ctx is not None:

        def progress(stage: str, progress: float, total: float, message: str) -> None:
            try:
                if hasattr(ctx, "info"):
                    ctx.info(f"[{stage}] {message}")
            except Exception:
                pass

        context = AnalysisContext(progress_callback=progress)

    try:
        return core_analyze_video(request, context=context)
    except VideoAnalyzerFailure as exc:
        payload = _error_payload(exc.error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except (ValidationError, ValueError, TypeError) as exc:
        error = AnalysisError(
            code=ErrorCode.INVALID_REQUEST,
            stage="validate_source",
            message=str(exc)[:2_048] or "request validation failed",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)
    except Exception as exc:  # noqa: BLE001 - adapter boundary must return a typed envelope
        error = AnalysisError(
            code=ErrorCode.INTERNAL_STAGE_FAILED,
            stage="orchestration",
            message=str(exc)[:2_048] or "analysis failed",
            retryable=False,
        )
        payload = _error_payload(error)
        return ToolResult(content=payload, structured_content=payload, is_error=True)


@mcp.resource(
    "video-analyzer://runs/{run_id}/artifacts/{artifact_id}{?page,offset,limit}",
    name="read_artifact",
    mime_type="application/jsonl",
)
def _read_artifact_resource(
    run_id: str,
    artifact_id: str,
    page: int | None = None,
    offset: int = 0,
    limit: int = 200,
) -> str | bytes | ArtifactStoreFailure:
    uri = f"video-analyzer://runs/{run_id}/artifacts/{artifact_id}"
    return read_artifact_resource(uri, page=page, offset=offset, limit=limit)


def main() -> None:
    configure_logging()
    mcp.run()


__all__ = ["analyze_video", "main", "mcp", "read_artifact_resource"]
