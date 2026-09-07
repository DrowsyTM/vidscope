from __future__ import annotations

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
        model_cache = _field(self.settings, "model_cache", None) or _field(
            get_settings(), "model_cache", None
        )
        if model_cache:
            cache = Path(model_cache).expanduser()
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache))
            os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
        try:
            model = self._factory()(
                "tiny.en",
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=1,
            )
        except AsrBackendFailure:
            raise
        except Exception as exc:
            raise AsrBackendFailure(
                "ASR_MODEL_UNAVAILABLE", "faster-whisper model could not be loaded"
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
            "model": "tiny.en",
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": 4,
            "num_workers": 1,
            "language": _field(info, "language", language_value),
            "language_probability": _field(info, "language_probability", None),
            "segment_count": len(rows),
            "word_count": sum(len(row["words"]) for row in rows),
            "model_cache": str(model_cache) if model_cache else None,
        }
        return TranscriptArtifact(rows, metadata)


__all__ = ["AsrBackendFailure", "FasterWhisperBackend", "TranscriptArtifact"]
