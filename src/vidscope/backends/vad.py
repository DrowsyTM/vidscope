from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import AnalysisError, ErrorCode
from ..settings import get_settings


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class VadBackendFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.error = AnalysisError(
            code=ErrorCode(code), stage="vad", message=message, retryable=False
        )
        self.analysis_error = self.error
        self.code = self.error.code
        super().__init__(message)


@dataclass(slots=True)
class VadArtifact:
    intervals: list[dict[str, float]]
    metadata: dict[str, Any]

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        del mode
        return {"intervals": self.intervals, "metadata": self.metadata}


class SileroVadBackend:
    def __init__(
        self, settings: Any = None, loader: Any = None, module: Any = None
    ) -> None:
        self.settings = settings
        self.loader = loader
        self.module = module

    def _module(self) -> Any:
        if self.module is not None:
            return self.module
        try:
            import silero_vad
        except (ImportError, ModuleNotFoundError) as exc:
            raise VadBackendFailure(
                "TOOL_UNAVAILABLE", "silero-vad is unavailable"
            ) from exc
        return silero_vad

    def detect(
        self,
        audio: Any,
        request: Any = None,
        *,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> VadArtifact:
        audio_path = Path(
            audio if isinstance(audio, (str, Path)) else _field(audio, "path", audio)
        )
        start = float(
            start_seconds
            if start_seconds is not None
            else _field(_field(request, "time_range", None), "start_seconds", 0.0)
        )
        end = float(
            end_seconds
            if end_seconds is not None
            else _field(_field(request, "time_range", None), "end_seconds", 180.0)
        )
        module = self._module()
        try:
            load = self.loader or module.load_silero_vad
            model = load(onnx=False)
            get_timestamps = module.get_speech_timestamps
            waveform: Any = None
            try:
                import soundfile as sf  # type: ignore[import-untyped]
                import torch

                data, _sr = sf.read(str(audio_path), dtype="float32")
                wav = torch.from_numpy(data)
                if wav.ndim > 1:
                    wav = wav.mean(dim=-1)
                waveform = wav
            except Exception:
                read_audio = getattr(module, "read_audio", None)
                if callable(read_audio):
                    waveform = read_audio(str(audio_path), sampling_rate=16_000)
                else:
                    raise
            raw = get_timestamps(
                waveform, model, sampling_rate=16_000, return_seconds=True
            )
        except VadBackendFailure:
            raise
        except Exception as exc:
            raise VadBackendFailure(
                "INTERNAL_STAGE_FAILED", "silero-vad detection failed"
            ) from exc
        intervals: list[dict[str, float]] = []
        for item in raw or ():
            try:
                rel_start = float(
                    _field(item, "start", _field(item, "start_seconds", 0.0))
                )
                rel_end = float(_field(item, "end", _field(item, "end_seconds", 0.0)))
                item_start = max(start, round(start + rel_start, 3))
                item_end = min(end, round(start + rel_end, 3))
            except (TypeError, ValueError):
                continue
            if item_end > item_start:
                intervals.append({"start_seconds": item_start, "end_seconds": item_end})
        intervals.sort(key=lambda value: (value["start_seconds"], value["end_seconds"]))
        return VadArtifact(
            intervals,
            {
                "provider": "silero-vad",
                "model": "silero-vad",
                "onnx": False,
                "device": "cpu",
                "sampling_rate": 16_000,
                "speech_seconds": sum(
                    item["end_seconds"] - item["start_seconds"] for item in intervals
                ),
                "model_cache": str(
                    _field(
                        self.settings,
                        "model_cache",
                        _field(get_settings(), "model_cache", None),
                    )
                    or ""
                ),
            },
        )


__all__ = ["SileroVadBackend", "VadArtifact", "VadBackendFailure"]
