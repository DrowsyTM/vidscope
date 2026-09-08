from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools.base import ToolResult
from pydantic import ValidationError

from .artifacts import ArtifactStore, ArtifactStoreFailure, read_artifact_resource
from .contracts import AnalysisError, AnalysisResult, AnalyzeVideoRequest, ErrorCode
from .core import AnalysisContext, VideoAnalyzerFailure
from .core import analyze_video as core_analyze_video
from .logging import configure_logging
from .settings import get_settings

mcp = FastMCP("vidscope")


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
    "vidscope://runs/{run_id}/artifacts/{artifact_id}{?page,offset,limit}",
    name="read_artifact",
)
def _read_artifact_resource(
    run_id: str,
    artifact_id: str,
    page: int | None = None,
    offset: int = 0,
    limit: int = 200,
) -> str | bytes:
    uri = f"vidscope://runs/{run_id}/artifacts/{artifact_id}"
    result = read_artifact_resource(uri, page=page, offset=offset, limit=limit)
    if isinstance(result, ArtifactStoreFailure):
        return json.dumps(result.error.model_dump(mode="json"))
    return result


@mcp.resource("vidscope://runs/{run_id}/manifest", name="read_manifest")
def _read_manifest_resource(run_id: str) -> str:
    settings = get_settings()
    root = settings.allowed_output_root or Path.cwd()
    try:
        _, manifest = ArtifactStore._load_resource_manifest(root, run_id)
        return json.dumps(manifest, indent=2)
    except ArtifactStoreFailure as exc:
        return json.dumps(exc.error.model_dump(mode="json"))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("vidscope://runs/{run_id}/plan", name="read_plan")
def _read_plan_resource(run_id: str) -> str:
    settings = get_settings()
    root = settings.allowed_output_root or Path.cwd()
    try:
        run_dir, _ = ArtifactStore._load_resource_manifest(root, run_id)
        plan_path = run_dir / "plan.json"
        if not plan_path.is_file():
            return json.dumps({"error": "plan not found"})
        return plan_path.read_text(encoding="utf-8")
    except ArtifactStoreFailure as exc:
        return json.dumps(exc.error.model_dump(mode="json"))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def main() -> None:
    configure_logging()
    mcp.run()


__all__ = ["analyze_video", "main", "mcp", "read_artifact_resource"]
