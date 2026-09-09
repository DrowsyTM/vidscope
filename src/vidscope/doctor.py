from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .settings import Settings, get_settings


class DiagnosticItem(BaseModel):
    name: str
    status: str = Field(description="'ok', 'warning', 'missing', or 'optional_missing'")
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    recommendation: str | None = None


class DoctorReport(BaseModel):
    ok: bool
    summary: str
    checks: list[DiagnosticItem]
    system_info: dict[str, Any] = Field(default_factory=dict)


def _run_cmd(
    cmd: list[str],
    *,
    runner: Callable[..., Any] | None = None,
    timeout: float = 10.0,
) -> tuple[int, str]:
    if runner is not None:
        res = runner(cmd)
        code = getattr(res, "returncode", 0)
        out = str(getattr(res, "stdout", "") or getattr(res, "stderr", ""))
        return code, out.strip()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
        return proc.returncode, out
    except Exception as exc:
        return 1, str(exc)


def run_doctor(
    settings: Settings | None = None,
    *,
    runner: Callable[..., Any] | None = None,
) -> DoctorReport:
    """Run comprehensive environment diagnostics for local media and remote extraction."""
    cfg = settings or get_settings()
    checks: list[DiagnosticItem] = []

    # 1. FFmpeg
    ffmpeg_path = str(cfg.ffmpeg_bin) if cfg.ffmpeg_bin else shutil.which("ffmpeg")
    if ffmpeg_path:
        code, out = _run_cmd([ffmpeg_path, "-version"], runner=runner)
        first_line = out.splitlines()[0] if out else "ffmpeg (unknown version)"
        checks.append(
            DiagnosticItem(
                name="FFmpeg",
                status="ok" if code == 0 else "warning",
                message=first_line,
                details={"path": ffmpeg_path, "exit_code": code},
            )
        )
    else:
        checks.append(
            DiagnosticItem(
                name="FFmpeg",
                status="missing",
                message="Binary not found on PATH or in VIDSCOPE_FFMPEG_BIN",
                recommendation=(
                    "Install FFmpeg via your system package manager "
                    "(e.g. 'sudo apt install ffmpeg' or 'brew install ffmpeg'). "
                    "Required for frame extraction and audio slicing."
                ),
            )
        )

    # 2. FFprobe
    ffprobe_path = str(cfg.ffprobe_bin) if cfg.ffprobe_bin else shutil.which("ffprobe")
    if ffprobe_path:
        code, out = _run_cmd([ffprobe_path, "-version"], runner=runner)
        first_line = out.splitlines()[0] if out else "ffprobe (unknown version)"
        checks.append(
            DiagnosticItem(
                name="FFprobe",
                status="ok" if code == 0 else "warning",
                message=first_line,
                details={"path": ffprobe_path, "exit_code": code},
            )
        )
    else:
        checks.append(
            DiagnosticItem(
                name="FFprobe",
                status="missing",
                message="Binary not found on PATH or in VIDSCOPE_FFPROBE_BIN",
                recommendation=(
                    "Install FFprobe (typically included with the ffmpeg package). "
                    "Required for media container probing."
                ),
            )
        )

    # 3. Tesseract OCR
    tess_path = (
        str(cfg.tesseract_bin) if cfg.tesseract_bin else shutil.which("tesseract")
    )
    if tess_path:
        code, out = _run_cmd([tess_path, "--version"], runner=runner)
        first_line = out.splitlines()[0] if out else "tesseract (unknown version)"
        checks.append(
            DiagnosticItem(
                name="Tesseract OCR",
                status="ok" if code == 0 else "warning",
                message=first_line,
                details={
                    "path": tess_path,
                    "tessdata_prefix": str(cfg.tessdata_prefix)
                    if cfg.tessdata_prefix
                    else None,
                },
            )
        )
    else:
        checks.append(
            DiagnosticItem(
                name="Tesseract OCR",
                status="optional_missing",
                message="Binary not found on PATH or in VIDSCOPE_TESSERACT_BIN",
                recommendation=(
                    "Install Tesseract OCR ('sudo apt install tesseract-ocr' or 'brew install tesseract') "
                    "for on-screen text extraction. Frame extraction will proceed without OCR if omitted."
                ),
            )
        )

    # 4. Speech-to-Text / ASR
    has_faster_whisper = False
    has_silero_vad = False
    cuda_count = 0
    try:
        import faster_whisper  # noqa: F401

        has_faster_whisper = True
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import silero_vad  # noqa: F401

        has_silero_vad = True
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import ctranslate2  # type: ignore[import-untyped]

        if hasattr(ctranslate2, "get_cuda_device_count"):
            cuda_count = int(ctranslate2.get_cuda_device_count())
    except Exception:
        cuda_count = 0

    if has_faster_whisper and has_silero_vad:
        device_label = (
            f"CUDA ({cuda_count} GPU device{'s' if cuda_count != 1 else ''})"
            if cuda_count > 0
            else "CPU (int8 quantized)"
        )
        checks.append(
            DiagnosticItem(
                name="Speech-to-Text (ASR)",
                status="ok",
                message=f"faster-whisper + silero-vad ready [{device_label}]",
                details={
                    "cuda_devices": cuda_count,
                    "device": "cuda" if cuda_count > 0 else "cpu",
                    "whisper_model": cfg.whisper_model
                    or ("base.en" if cuda_count > 0 else "tiny.en"),
                },
            )
        )
    elif has_faster_whisper and not has_silero_vad:
        checks.append(
            DiagnosticItem(
                name="Speech-to-Text (ASR)",
                status="warning",
                message="faster-whisper installed but silero-vad is missing",
                recommendation="Install silero-vad for voice activity detection: 'pip install silero-vad'.",
            )
        )
    else:
        checks.append(
            DiagnosticItem(
                name="Speech-to-Text (ASR)",
                status="optional_missing",
                message="faster-whisper not installed",
                recommendation=(
                    "Install local ASR speech transcription: 'pip install \"vidscope[asr]\"'. "
                    "Allows offline audio transcription when remote YouTube captions are unavailable."
                ),
            )
        )

    # 5. JavaScript Runtime
    node_path = shutil.which("node")
    deno_path = shutil.which("deno")
    if node_path:
        code, out = _run_cmd([node_path, "--version"], runner=runner)
        checks.append(
            DiagnosticItem(
                name="JavaScript Runtime",
                status="ok" if code == 0 else "warning",
                message=f"Node.js {out} ({node_path})",
                details={"runtime": "node", "path": node_path, "version": out},
            )
        )
    elif deno_path:
        code, out = _run_cmd([deno_path, "--version"], runner=runner)
        first_line = out.splitlines()[0] if out else "deno"
        checks.append(
            DiagnosticItem(
                name="JavaScript Runtime",
                status="ok" if code == 0 else "warning",
                message=f"{first_line} ({deno_path})",
                details={"runtime": "deno", "path": deno_path, "version": first_line},
            )
        )
    else:
        checks.append(
            DiagnosticItem(
                name="JavaScript Runtime",
                status="optional_missing",
                message="Neither Node.js (>=20) nor Deno found on PATH",
                recommendation=(
                    "Install Node.js ('sudo apt install nodejs' or via nvm) for yt-dlp "
                    "JavaScript challenge solving and PO token generation."
                ),
            )
        )

    # 6. PO Token Generator Script
    default_pot_dir = Path.home() / "bgutil-ytdlp-pot-provider" / "server"
    script_path = default_pot_dir / "build" / "generate_once.js"
    if script_path.is_file() and node_path:
        code, out = _run_cmd([node_path, str(script_path), "--version"], runner=runner)
        if code == 0:
            checks.append(
                DiagnosticItem(
                    name="PO Token Generator",
                    status="ok",
                    message=f"Ready: v{out} ({script_path})",
                    details={"path": str(script_path), "version": out},
                )
            )
        else:
            checks.append(
                DiagnosticItem(
                    name="PO Token Generator",
                    status="warning",
                    message=f"Script exists but returned non-zero ({code}): {out}",
                    recommendation="Rebuild the PO token provider by running 'vidscope setup-pot --force'.",
                )
            )
    else:
        checks.append(
            DiagnosticItem(
                name="PO Token Generator",
                status="optional_missing",
                message="Not configured (optional; only needed if datacenter IP is blocked by YouTube)",
                recommendation=(
                    "Run 'vidscope setup-pot' to automatically clone and build the on-demand "
                    "PO token generator."
                ),
            )
        )

    # 7. Netscape Cookies File
    if cfg.cookies_file:
        if cfg.cookies_file.is_file() and cfg.cookies_file.stat().st_size > 0:
            checks.append(
                DiagnosticItem(
                    name="Cookies File",
                    status="ok",
                    message=f"Configured: {cfg.cookies_file} ({cfg.cookies_file.stat().st_size} bytes)",
                    details={"path": str(cfg.cookies_file)},
                )
            )
        else:
            checks.append(
                DiagnosticItem(
                    name="Cookies File",
                    status="warning",
                    message=f"VIDSCOPE_COOKIES_FILE={cfg.cookies_file} does not exist or is empty",
                    recommendation="Ensure the specified cookie file is an exported Netscape HTTP cookie file.",
                )
            )
    else:
        checks.append(
            DiagnosticItem(
                name="Cookies File",
                status="ok",
                message="Not configured (anonymous mode). Set VIDSCOPE_COOKIES_FILE if encountering YouTube 429 rate limits.",
            )
        )

    # Overall health: critical dependencies are ffmpeg & ffprobe
    has_critical_missing = any(c.status == "missing" for c in checks)
    ok = not has_critical_missing
    summary = (
        "System is fully configured and ready for video analysis."
        if ok
        else "System is missing critical dependencies required for video processing."
    )

    return DoctorReport(
        ok=ok,
        summary=summary,
        checks=checks,
        system_info={
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "os_name": os.name,
        },
    )


def setup_pot_provider(
    *,
    target_dir: Path | None = None,
    force: bool = False,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Clone and build the bgutil-ytdlp-pot-provider on-demand script."""
    dest = (target_dir or (Path.home() / "bgutil-ytdlp-pot-provider")).expanduser()
    server_dir = dest / "server"
    script_path = server_dir / "build" / "generate_once.js"

    node_bin = shutil.which("node")
    if not node_bin:
        return {
            "ok": False,
            "error": "Node.js executable ('node') was not found on PATH. Please install Node.js >= 20.",
        }

    npm_bin = shutil.which("npm")
    if not npm_bin:
        return {
            "ok": False,
            "error": "npm executable was not found on PATH. Please install npm (included with Node.js).",
        }

    git_bin = shutil.which("git")
    if not git_bin and not dest.exists():
        return {
            "ok": False,
            "error": "git executable was not found on PATH. Please install git to clone the provider repository.",
        }

    # If already built and not force, verify working
    if script_path.is_file() and not force:
        code, out = _run_cmd([node_bin, str(script_path), "--version"], runner=runner)
        if code == 0:
            return {
                "ok": True,
                "already_built": True,
                "path": str(script_path),
                "version": out,
                "message": f"PO token provider is already built and working (v{out}). Use --force to rebuild.",
            }

    # Clone repository if needed
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = [
            str(git_bin),
            "clone",
            "--single-branch",
            "--branch",
            "2.0.0",
            "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git",
            str(dest),
        ]
        code, out = _run_cmd(clone_cmd, runner=runner, timeout=120.0)
        if code != 0:
            return {
                "ok": False,
                "error": f"Failed to clone bgutil-ytdlp-pot-provider: {out}",
            }

    if not server_dir.exists():
        return {
            "ok": False,
            "error": f"Expected server directory does not exist at {server_dir}",
        }

    # npm ci
    npm_ci_cmd = [str(npm_bin), "ci", "--prefix", str(server_dir)]
    code, out = _run_cmd(npm_ci_cmd, runner=runner, timeout=180.0)
    if code != 0:
        return {
            "ok": False,
            "error": f"Failed to install npm dependencies ('npm ci'): {out}",
        }

    # npx tsc
    npx_bin = shutil.which("npx") or str(Path(npm_bin).parent / "npx")
    tsc_cmd = [npx_bin, "--prefix", str(server_dir), "tsc"]
    code, out = _run_cmd(tsc_cmd, runner=runner, timeout=120.0)
    if code != 0:
        return {
            "ok": False,
            "error": f"Failed to compile TypeScript ('npx tsc'): {out}",
        }

    # Validate generated script
    code, out = _run_cmd([node_bin, str(script_path), "--version"], runner=runner)
    if code != 0:
        return {
            "ok": False,
            "error": f"Built script failed version verification: {out}",
        }

    return {
        "ok": True,
        "already_built": False,
        "path": str(script_path),
        "version": out,
        "message": f"Successfully built on-demand PO token generator (v{out}).",
    }


__all__ = [
    "DiagnosticItem",
    "DoctorReport",
    "run_doctor",
    "setup_pot_provider",
]
