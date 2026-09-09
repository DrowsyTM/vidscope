from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..contracts import AnalysisError, ErrorCode
from ..settings import get_settings


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _error(code: str, message: str) -> AnalysisError:
    return AnalysisError(
        code=ErrorCode(code), stage="transcribe", message=message, retryable=False
    )


class AsrBackendFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.error = _error(code, message)
        self.analysis_error = self.error
        self.code = self.error.code
        super().__init__(message)


@dataclass(slots=True)
class TranscriptArtifact:
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        del mode
        return {"segments": self.segments, "metadata": self.metadata}


def _preload_cuda_libraries() -> None:
    """Preload installed nvidia wheel libraries for ctranslate2 / CUDA support."""
    import ctypes
    import sys

    for p in sys.path:
        nv_dir = Path(p) / "nvidia"
        if nv_dir.is_dir():
            for sub in ("cublas", "cudnn", "cuda_nvrtc"):
                lib_dir = nv_dir / sub / "lib"
                if lib_dir.is_dir():
                    cur = os.environ.get("LD_LIBRARY_PATH", "")
                    if str(lib_dir) not in cur:
                        os.environ["LD_LIBRARY_PATH"] = (
                            f"{lib_dir}:{cur}" if cur else str(lib_dir)
                        )
                    # cublasLt must be loaded before cublas
                    cublas_lt = lib_dir / "libcublasLt.so.12"
                    if cublas_lt.is_file():
                        with contextlib.suppress(Exception):
                            ctypes.CDLL(str(cublas_lt), mode=ctypes.RTLD_GLOBAL)
                    for so in sorted(lib_dir.glob("*.so*")):
                        with contextlib.suppress(Exception):
                            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)


def _detect_device(configured: str | None) -> str:
    if configured and configured.lower() not in ("auto", ""):
        return configured.lower()
    _preload_cuda_libraries()
    try:
        import ctranslate2  # type: ignore[import-untyped]

        if (
            hasattr(ctranslate2, "get_cuda_device_count")
            and ctranslate2.get_cuda_device_count() > 0
        ):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_cuda_vram_bytes() -> int:
    """Return total CUDA VRAM in bytes, or 0 if no CUDA device."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2.0,
        ).strip()
        first_line = out.splitlines()[0]
        return int(first_line) * 1024 * 1024
    except Exception:
        pass
    return 0


def _resolve_model_name(
    configured: str | None,
    language: str,
    device: str = "cpu",
) -> str:
    if configured and configured.strip():
        return configured.strip()
    norm_lang = language.strip().lower()
    is_english = norm_lang in ("en", "english")

    # If running on CUDA with at least 1 GB of VRAM, use base model for higher accuracy
    if device == "cuda" and _get_cuda_vram_bytes() >= 1024 * 1024 * 1024:
        return "base.en" if is_english else "base"

    return "tiny.en" if is_english else "tiny"


def _resolve_compute_type(configured: str | None, device: str) -> str:
    if configured and configured.strip():
        return configured.strip()
    if device == "cuda":
        _preload_cuda_libraries()
        try:
            import ctranslate2

            supported = ctranslate2.get_supported_compute_types("cuda")
            if "float16" in supported:
                return "float16"
            if "int8_float32" in supported:
                return "int8_float32"
            if "int8" in supported:
                return "int8"
            return "float32"
        except Exception:
            return "default"
    return "int8"


class FasterWhisperBackend:
    def __init__(
        self,
        settings: Any = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.model_factory = model_factory

    def _factory(self) -> Callable[..., Any]:
        if self.model_factory is not None:
            return self.model_factory
        _preload_cuda_libraries()
        try:
            from faster_whisper import WhisperModel
        except (ImportError, ModuleNotFoundError) as exc:
            raise AsrBackendFailure(
                "ASR_MODEL_UNAVAILABLE", "faster-whisper is unavailable"
            ) from exc
        return cast(Callable[..., Any], WhisperModel)

    def transcribe(
        self,
        audio: Any,
        request: Any = None,
        *,
        language: str | None = None,
    ) -> TranscriptArtifact:
        audio_path = Path(
            audio if isinstance(audio, (str, Path)) else _field(audio, "path", audio)
        )
        language_value = language or _field(request, "language", "en") or "en"
        settings = self.settings or get_settings()
        model_cache = _field(settings, "model_cache", None)
        if model_cache:
            cache = Path(model_cache).expanduser()
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache))
            os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))

        configured_model = _field(settings, "whisper_model", None)
        configured_device = _field(settings, "whisper_device", None)
        configured_compute_type = _field(settings, "whisper_compute_type", None)

        device = _detect_device(configured_device)
        compute_type = _resolve_compute_type(configured_compute_type, device)
        model_name = _resolve_model_name(
            configured_model, language_value, device=device
        )
        cpu_threads = 4 if device == "cpu" else 0
        num_workers = 1

        try:
            model = self._factory()(
                model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )
        except AsrBackendFailure:
            raise
        except Exception as exc:
            raise AsrBackendFailure(
                "ASR_MODEL_UNAVAILABLE",
                f"faster-whisper model '{model_name}' could not be loaded: {exc}",
            ) from exc
        try:
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=language_value,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
            )
            rows: list[dict[str, Any]] = []
            for index, segment in enumerate(segments_iter):
                text = str(_field(segment, "text", "") or "").strip()
                if not text:
                    continue
                words: list[dict[str, Any]] = []
                for word in _field(segment, "words", ()) or ():
                    words.append(
                        {
                            "start_seconds": float(_field(word, "start", 0.0)),
                            "end_seconds": float(_field(word, "end", 0.0)),
                            "word": str(_field(word, "word", "")),
                            "probability": float(_field(word, "probability", 0.0)),
                        }
                    )
                rows.append(
                    {
                        "id": int(_field(segment, "id", index)),
                        "start_seconds": float(_field(segment, "start", 0.0)),
                        "end_seconds": float(_field(segment, "end", 0.0)),
                        "text": text,
                        "words": words,
                    }
                )
        except Exception as exc:
            raise AsrBackendFailure(
                "INTERNAL_STAGE_FAILED", "faster-whisper transcription failed"
            ) from exc
        if not rows:
            raise AsrBackendFailure(
                "INTERNAL_STAGE_FAILED", "transcript output was empty"
            )
        metadata = {
            "provider": "faster-whisper",
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": cpu_threads,
            "num_workers": num_workers,
            "language": _field(info, "language", language_value),
            "language_probability": _field(info, "language_probability", None),
            "segment_count": len(rows),
            "word_count": sum(len(row["words"]) for row in rows),
            "model_cache": str(model_cache) if model_cache else None,
        }
        return TranscriptArtifact(rows, metadata)


__all__ = ["AsrBackendFailure", "FasterWhisperBackend", "TranscriptArtifact"]
