from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import AnalysisError, ErrorCode
from ..settings import get_settings


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class OcrBackendFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.error = AnalysisError(code=ErrorCode(code), stage="ocr", message=message, retryable=False)
        self.analysis_error = self.error
        self.code = self.error.code
        super().__init__(message)


@dataclass(slots=True)
class OcrArtifact:
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        del mode
        return {"rows": self.rows, "metadata": self.metadata}


class TesseractBackend:
    def __init__(
        self,
        settings: Any = None,
        runner: Callable[..., Any] | None = None,
        binary: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.binary = str(binary) if binary else None

    def _binary(self) -> str:
        configured = self.binary or _field(self.settings, "tesseract_bin", None) or _field(get_settings(), "tesseract_bin", None)
        value = str(configured) if configured else shutil.which("tesseract")
        if not value:
            raise OcrBackendFailure("OCR_UNAVAILABLE", "tesseract binary is unavailable")
        return value

    def recognize(self, frame: Any, language: str = "eng", request: Any = None, **_: Any) -> OcrArtifact:
        frame_path = Path(frame if isinstance(frame, (str, Path)) else _field(frame, "path", frame))
        if not frame_path.is_file() or frame_path.stat().st_size <= 0:
            raise OcrBackendFailure("OCR_UNAVAILABLE", "frame is unavailable")
        command = [self._binary(), str(frame_path), "stdout", "--psm", "6", "--oem", "1", "-l", language, "tsv"]
        environment = os.environ.copy()
        tessdata = _field(self.settings, "tessdata_prefix", None) or _field(get_settings(), "tessdata_prefix", None)
        if tessdata:
            environment["TESSDATA_PREFIX"] = str(tessdata)
        try:
            if self.runner is None:
                result = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
            else:
                try:
                    result = self.runner(command, capture_output=True, text=True, env=environment, check=False)
                except TypeError:
                    result = self.runner(command)
        except FileNotFoundError as exc:
            raise OcrBackendFailure("OCR_UNAVAILABLE", "tesseract binary is unavailable") from exc
        except OSError as exc:
            raise OcrBackendFailure("OCR_UNAVAILABLE", "tesseract could not be started") from exc
        returncode = int(getattr(result, "returncode", result.get("returncode", 0) if isinstance(result, Mapping) else 0))
        stdout = getattr(result, "stdout", result.get("stdout", "") if isinstance(result, Mapping) else "")
        stderr = getattr(result, "stderr", result.get("stderr", "") if isinstance(result, Mapping) else "")
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if returncode:
            raise OcrBackendFailure("OCR_UNAVAILABLE", f"tesseract failed: {str(stderr)[:512]}")
        try:
            reader = csv.DictReader(io.StringIO(str(stdout)), delimiter="\t")
            rows: list[dict[str, Any]] = []
            for row in reader:
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                rows.append(
                    {
                        "text": text,
                        "confidence": float(row["conf"]),
                        "left": int(row["left"]),
                        "top": int(row["top"]),
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                    }
                )
        except (KeyError, TypeError, ValueError, csv.Error) as exc:
            raise OcrBackendFailure("INTERNAL_STAGE_FAILED", "tesseract returned malformed TSV") from exc
        if not rows:
            raise OcrBackendFailure("OCR_UNAVAILABLE", "tesseract returned no text")
        return OcrArtifact(
            rows,
            {
                "provider": "tesseract",
                "language": language,
                "psm": 6,
                "oem": 1,
                "word_count": len(rows),
            },
        )


__all__ = ["OcrArtifact", "OcrBackendFailure", "TesseractBackend"]
