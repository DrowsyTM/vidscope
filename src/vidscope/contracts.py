"""Shared request, execution-plan, result, and error contracts.

The contracts in this module are deliberately independent of media backends.  A
request can therefore be validated by the CLI, MCP adapter, or the core API
without importing ffmpeg, OCR, ASR, or any other optional implementation.
"""

from __future__ import annotations

# Pydantic validators intentionally raise ValueError for user-facing validation.
# ruff: noqa: TRY004
import json
import math
import ntpath
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Final, Literal
from urllib.parse import urlsplit
from urllib.request import url2pathname

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Public hard limits.  Keeping them in one place prevents adapters from
# accidentally accepting a larger budget than the shared request contract.
MAX_TIME_RANGE_SECONDS = 180.0
MAX_FRAMES = 12
MAX_FRAME_WIDTH = 1_920
MAX_DOWNLOAD_BYTES = 268_435_456
MAX_OUTPUT_BYTES = 67_108_864
MAX_TIMEOUT_SECONDS = 600
MAX_REQUEST_ID_LENGTH = 64
MAX_DIAGNOSTIC_KEYS = 32
MAX_DIAGNOSTIC_STRING_LENGTH = 2_048
MAX_DIAGNOSTIC_BYTES = 16_384
MAX_ARTIFACT_REFS = 128

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_ARTIFACT_URI_RE = re.compile(
    r"^vidscope://runs/[A-Za-z0-9._-]{1,128}/artifacts/[A-Za-z0-9._-]{1,128}$"
)
_MANIFEST_URI_RE = re.compile(r"^vidscope://runs/[A-Za-z0-9._-]{1,128}/manifest$")
_LANGUAGE_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,15}(?:\+[a-zA-Z0-9][a-zA-Z0-9_-]{1,15})*$"
)


class _StableStringEnum(str, Enum):
    """A string enum whose textual form is its wire value.

    Python's default ``Enum.__str__`` includes the class name.  Error codes are
    emitted in logs, JSON envelopes, and MCP payloads, so their string form is
    intentionally stable and identical to their value.
    """

    def __str__(self) -> str:
        return str(self.value)


class AnalysisTask(_StableStringEnum):
    """Public analysis tasks accepted by :class:`AnalyzeVideoRequest`."""

    METADATA = "metadata"
    TRANSCRIPT = "transcript"
    VAD = "vad"
    FRAMES = "frames"
    OCR = "ocr"


class ErrorCode(_StableStringEnum):
    """Stable terminal failure codes exposed by every adapter."""

    INVALID_REQUEST = "INVALID_REQUEST"
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    SOURCE_NOT_ALLOWED = "SOURCE_NOT_ALLOWED"
    URL_SCHEME_NOT_ALLOWED = "URL_SCHEME_NOT_ALLOWED"
    NETWORK_DISABLED = "NETWORK_DISABLED"
    DOWNLOAD_LIMIT_EXCEEDED = "DOWNLOAD_LIMIT_EXCEEDED"
    OUTPUT_LIMIT_EXCEEDED = "OUTPUT_LIMIT_EXCEEDED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    OUTPUT_NOT_ALLOWED = "OUTPUT_NOT_ALLOWED"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    CAPTIONS_UNAVAILABLE = "CAPTIONS_UNAVAILABLE"
    ASR_MODEL_UNAVAILABLE = "ASR_MODEL_UNAVAILABLE"
    MEDIA_DECODE_FAILED = "MEDIA_DECODE_FAILED"
    FRAME_LIMIT_EXCEEDED = "FRAME_LIMIT_EXCEEDED"
    OCR_UNAVAILABLE = "OCR_UNAVAILABLE"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
    ARTIFACT_RANGE_INVALID = "ARTIFACT_RANGE_INVALID"
    INTERNAL_STAGE_FAILED = "INTERNAL_STAGE_FAILED"


# Descriptive aliases retained as import-friendly names.  They all refer to
# the one enum, so there cannot be divergent wire values.
AnalysisErrorCode = ErrorCode
FailureCode = ErrorCode


class StageStatus(_StableStringEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class AnalysisStatus(_StableStringEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class CaptionKind(_StableStringEnum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class SourceKind(_StableStringEnum):
    LOCAL = "local"
    URL = "url"


class ResultStatus(_StableStringEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"


class ErrorStatus(_StableStringEnum):
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


AnalysisResultStatus = ResultStatus
AnalysisErrorStatus = ErrorStatus


class _ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        populate_by_name=True,
    )


# Pydantic's JSON type aliases are intentionally not used for the output
# dictionaries: accepting a bounded, cleaned mapping gives callers useful
# diagnostics while ensuring model_dump(mode="json") remains serializable.
def _clean_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        raise ValueError("nested diagnostic/settings values are too deep")
    if value is None or isinstance(value, (str, int, bool)):
        if isinstance(value, str) and len(value) > MAX_DIAGNOSTIC_STRING_LENGTH:
            raise ValueError("diagnostic string is too long")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("diagnostic values must be finite")
        return value
    if isinstance(value, Enum):
        return _clean_json_value(value.value, depth=depth + 1)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        if len(value) > MAX_DIAGNOSTIC_KEYS:
            raise ValueError("diagnostic mapping has too many keys")
        cleaned: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise ValueError("diagnostic keys must be short strings")
            cleaned[key] = _clean_json_value(child, depth=depth + 1)
        return cleaned
    if isinstance(value, (list, tuple, set, frozenset)):
        if len(value) > MAX_DIAGNOSTIC_KEYS:
            raise ValueError("diagnostic sequence has too many values")
        return [_clean_json_value(child, depth=depth + 1) for child in value]
    raise ValueError(f"value of type {type(value).__name__} is not JSON serializable")


def _clean_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    cleaned = _clean_json_value(value)
    assert isinstance(cleaned, dict)
    encoded = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_DIAGNOSTIC_BYTES:
        raise ValueError(f"{label} exceeds the bounded JSON size")
    return cleaned


def _bounded_strings(value: Any, *, label: str, maximum: int = 64) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"{label} must be a sequence")
    if len(value) > maximum:
        raise ValueError(f"{label} has too many entries")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or len(item) > 256:
            raise ValueError(f"{label} entries must be short strings")
        result.append(item)
    return result


def _root_from_environment(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return root.resolve(strict=False)


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} is outside the configured allowed root") from exc


def _is_windows_drive_path(source: str) -> bool:
    drive, tail = ntpath.splitdrive(source)
    return bool(drive) and tail.startswith(("/", "\\"))


def _is_windows_drive_authority(authority: str) -> bool:
    return (
        len(authority) == 2
        and authority[0].isascii()
        and authority[0].isalpha()
        and authority[1] == ":"
    )


def _local_source_path(source: str) -> Path | None:
    """Return a validated local path, or ``None`` for an HTTPS source."""

    if _is_windows_drive_path(source):
        path = Path(source)
        if not path.is_absolute():
            raise ValueError("local source path must be absolute")
        return path

    try:
        parsed = urlsplit(source)
    except ValueError as exc:
        raise ValueError("source is not a valid URI or absolute path") from exc
    scheme = parsed.scheme.lower()

    if scheme == "https":
        if any(character.isspace() for character in source):
            raise ValueError("HTTPS source must not contain whitespace")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("source URL has an invalid port") from exc
        if (
            not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "HTTPS source must not contain credentials and must have a host"
            )
        if parsed.fragment:
            raise ValueError("source URL fragments are not allowed")
        return None

    if scheme == "file":
        if any(character.isspace() for character in source):
            raise ValueError("file URI must percent-encode whitespace")
        if not source.lower().startswith("file://"):
            raise ValueError("only file:// URIs are accepted for local sources")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("file URI credentials are not allowed")
        if parsed.fragment:
            raise ValueError("file URI fragments are not allowed")
        # A host-bearing file URI can only target this machine.  Credentials
        # were rejected above; accepting localhost keeps standard file URIs
        # useful without permitting UNC/network paths.
        drive_authority = _is_windows_drive_authority(parsed.netloc)
        if (
            parsed.netloc
            and parsed.netloc.lower() != "localhost"
            and not drive_authority
        ):
            raise ValueError("file URI host is not allowed")
        uri_path = parsed.path
        if drive_authority:
            uri_path = f"{parsed.netloc}{uri_path}"
        path = Path(url2pathname(uri_path))
        if not path.is_absolute():
            raise ValueError("file URI path must be absolute")
        return path

    if scheme:
        raise ValueError("URL scheme is not allowed")

    path = Path(source)
    if not path.is_absolute():
        raise ValueError("local source path must be absolute")
    return path


def _validate_existing_local_source(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("local source does not exist") from exc
    if (
        not resolved.is_file()
        or not os.access(resolved, os.R_OK)
        or not (resolved.stat().st_mode & 0o444)
    ):
        raise ValueError("local source must be a readable file")
    allowed_root = _root_from_environment("VIDSCOPE_ALLOWED_INPUT_ROOT")
    if allowed_root is not None:
        _require_within(resolved, allowed_root, label="source")
    return resolved


YOUTUBE_DOMAINS: Final[frozenset[str]] = frozenset(
    {"youtube.com", "youtu.be", "youtube-nocookie.com"}
)


def is_youtube_source(source: str) -> bool:
    """Determine whether source is a YouTube URL by validating its parsed hostname."""
    try:
        url = (
            source
            if (
                "://" in source
                or source.startswith("//")
                or not source.startswith(("/", "."))
            )
            else ""
        )
        if url and "://" not in url and not url.startswith("//"):
            url = "//" + url
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"", "https"}:
            return False
        host = (parsed.hostname or "").lower().rstrip(".")
        return any(
            host == domain or host.endswith(f".{domain}") for domain in YOUTUBE_DOMAINS
        )
    except Exception:
        return False


_is_youtube_source = is_youtube_source


class TimeRange(_ContractModel):
    """Finite, strictly increasing media window no longer than three minutes."""

    start_seconds: float = 0.0
    end_seconds: float = MAX_TIME_RANGE_SECONDS

    @field_validator("start_seconds", "end_seconds", mode="before")
    @classmethod
    def _coerce_finite_float(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("time bounds must be numbers")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("time bounds must be numbers") from exc
        if not math.isfinite(result):
            raise ValueError("time bounds must be finite")
        return result

    @model_validator(mode="after")
    def _validate_window(self) -> TimeRange:
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.end_seconds - self.start_seconds > MAX_TIME_RANGE_SECONDS:
            raise ValueError("time range cannot exceed 180 seconds")
        return self


class AnalyzeVideoRequest(_ContractModel):
    """Canonical bounded request shared by CLI, MCP, and the core API."""

    source: str
    time_range: TimeRange = Field(default_factory=TimeRange)
    tasks: set[AnalysisTask] = Field(
        default_factory=lambda: {AnalysisTask.METADATA, AnalysisTask.TRANSCRIPT}
    )
    output_directory: Path
    language: str = "en"
    caption_preference: Literal["manual_then_automatic_then_asr"] = (
        "manual_then_automatic_then_asr"
    )
    asr_enabled: bool = True
    frame_timestamps_seconds: tuple[float, ...] | None = None
    max_frames: int = Field(default=6, gt=0, le=MAX_FRAMES)
    max_frame_width: int = Field(default=1_280, gt=0, le=MAX_FRAME_WIDTH)
    max_download_bytes: int = Field(
        default=MAX_DOWNLOAD_BYTES, gt=0, le=MAX_DOWNLOAD_BYTES
    )
    max_output_bytes: int = Field(default=MAX_OUTPUT_BYTES, gt=0, le=MAX_OUTPUT_BYTES)
    timeout_seconds: int = Field(
        default=MAX_TIMEOUT_SECONDS, gt=0, le=MAX_TIMEOUT_SECONDS
    )
    request_id: str | None = Field(default=None, max_length=MAX_REQUEST_ID_LENGTH)

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_source(cls, value: Any) -> str:
        if isinstance(value, Path):
            value = str(value)
        if not isinstance(value, str) or not value:
            raise ValueError("source must be a non-empty string")
        if "\x00" in value:
            raise ValueError("source must not contain NUL bytes")
        return value

    @field_validator("output_directory", mode="after")
    @classmethod
    def _validate_output_directory(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("output_directory must be absolute")
        resolved = value.resolve(strict=False)
        if value.exists() and not value.is_dir():
            raise ValueError("output_directory must be a directory")
        allowed_root = _root_from_environment("VIDSCOPE_ALLOWED_OUTPUT_ROOT")
        if allowed_root is not None:
            _require_within(resolved, allowed_root, label="output_directory")
        return value

    @field_validator("language", mode="before")
    @classmethod
    def _validate_language(cls, value: Any) -> str:
        if not isinstance(value, str) or not value or len(value) > 64:
            raise ValueError("language must be a short non-empty string")
        if any(character.isspace() for character in value):
            raise ValueError("language must not contain whitespace")
        if _LANGUAGE_RE.fullmatch(value) is None:
            raise ValueError(
                "language must match BCP-47 or alphanumeric language tokens (e.g. 'en', 'eng', 'en-US', 'eng+fra') and cannot start with a hyphen"
            )
        return value

    @field_validator("request_id")
    @classmethod
    def _validate_request_id(cls, value: str | None) -> str | None:
        if value is not None and _REQUEST_ID_RE.fullmatch(value) is None:
            raise ValueError("request_id must match [A-Za-z0-9_-]{1,64}")
        return value

    @field_validator(
        "max_frames",
        "max_frame_width",
        "max_download_bytes",
        "max_output_bytes",
        "timeout_seconds",
        mode="before",
    )
    @classmethod
    def _reject_boolean_limits(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("limits must be integers")
        return value

    @field_validator("frame_timestamps_seconds", mode="before")
    @classmethod
    def _coerce_timestamps(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            raise ValueError("frame_timestamps_seconds must be a sequence of numbers")
        try:
            return tuple(value)
        except TypeError as exc:
            raise ValueError("frame_timestamps_seconds must be a sequence") from exc

    @model_validator(mode="after")
    def _validate_request(self) -> AnalyzeVideoRequest:
        if not self.tasks:
            raise ValueError("at least one analysis task is required")

        local_path = _local_source_path(self.source)
        if local_path is not None:
            _validate_existing_local_source(local_path)

        timestamps = self.frame_timestamps_seconds
        if timestamps is not None:
            if len(timestamps) > self.max_frames:
                raise ValueError("frame timestamp count cannot exceed max_frames")
            for timestamp in timestamps:
                if isinstance(timestamp, bool):
                    raise ValueError("frame timestamps must be finite numbers")
                try:
                    numeric_timestamp = float(timestamp)
                except (TypeError, ValueError) as exc:
                    raise ValueError("frame timestamps must be finite numbers") from exc
                if not math.isfinite(numeric_timestamp):
                    raise ValueError("frame timestamps must be finite")
                if not (
                    self.time_range.start_seconds
                    <= numeric_timestamp
                    <= self.time_range.end_seconds
                ):
                    raise ValueError("frame timestamp is outside time_range")
        return self

    @property
    def source_path(self) -> Path | None:
        """Resolved local source path, while ``source`` remains caller supplied."""

        local = _local_source_path(self.source)
        return _validate_existing_local_source(local) if local is not None else None

    @property
    def local_source_path(self) -> Path | None:
        return self.source_path

    @property
    def is_url(self) -> bool:
        return _local_source_path(self.source) is None


class ArtifactRef(_ContractModel):
    """Bounded metadata for a persisted run artifact (never its body)."""

    artifact_id: str
    uri: str
    media_type: str
    byte_size: int = Field(gt=0)
    sha256: str
    name: str | None = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("artifact_id")
    @classmethod
    def _validate_artifact_id(cls, value: str) -> str:
        if _ARTIFACT_ID_RE.fullmatch(value) is None:
            raise ValueError("artifact_id contains unsafe characters")
        return value

    @field_validator("uri")
    @classmethod
    def _validate_artifact_uri(cls, value: str) -> str:
        if _ARTIFACT_URI_RE.fullmatch(value) is None:
            raise ValueError("artifact uri is not a persisted vidscope artifact URI")
        return value

    @field_validator("media_type")
    @classmethod
    def _validate_media_type(cls, value: str) -> str:
        if (
            not value
            or len(value) > 128
            or any(character.isspace() for character in value)
        ):
            raise ValueError("media_type must be a short non-empty token")
        return value

    @field_validator("sha256")
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> dict[str, Any]:
        return _clean_mapping(value, label="artifact metadata")


class AnalysisError(_ContractModel):
    """Typed terminal/stage failure envelope shared by all adapters."""

    ok: Literal[False] = False
    status: Literal["failed", "cancelled", "timed_out"] = "failed"
    code: ErrorCode
    stage: str = "unknown"
    message: str = "analysis failed"
    retryable: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    manifest_uri: str | None = None

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, value: str) -> str:
        if (
            not value
            or len(value) > 128
            or any(character.isspace() for character in value)
        ):
            raise ValueError("stage must be a short identifier")
        return value

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        if not value or len(value) > MAX_DIAGNOSTIC_STRING_LENGTH:
            raise ValueError("error message is too long or empty")
        return value

    @field_validator("diagnostics", mode="before")
    @classmethod
    def _validate_diagnostics(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping) or value is None:
            return _clean_mapping(value, label="diagnostics")
        if isinstance(value, (list, tuple, set, frozenset)):
            return _clean_mapping({"messages": list(value)}, label="diagnostics")
        return _clean_mapping({"detail": value}, label="diagnostics")

    @field_validator("artifact_refs", "artifacts", mode="before")
    @classmethod
    def _validate_artifacts(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)) or len(value) > MAX_ARTIFACT_REFS:
            raise ValueError("artifact references are bounded")
        return list(value)

    @field_validator("manifest_uri")
    @classmethod
    def _validate_manifest_uri(cls, value: str | None) -> str | None:
        if value is not None and _MANIFEST_URI_RE.fullmatch(value) is None:
            raise ValueError("manifest_uri is not a persisted manifest URI")
        return value

    @model_validator(mode="after")
    def _mirror_artifact_fields(self) -> AnalysisError:
        if self.artifact_refs and not self.artifacts:
            self.artifacts = list(self.artifact_refs)
        elif self.artifacts and not self.artifact_refs:
            self.artifact_refs = list(self.artifacts)
        return self


class StageRecord(_ContractModel):
    """Persisted status and bounded settings for one execution stage."""

    name: str = ""
    status: StageStatus = StageStatus.PLANNED
    dependencies: list[str] = Field(default_factory=list)
    start_timestamp: datetime | str | None = None
    end_timestamp: datetime | str | None = None
    elapsed_ms: float | None = Field(default=None, ge=0)
    settings: dict[str, Any] = Field(default_factory=dict)
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: AnalysisError | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if len(value) > 128 or any(character.isspace() for character in value):
            raise ValueError("stage name must be a short identifier")
        return value

    @field_validator(
        "dependencies", "input_artifact_ids", "output_artifact_ids", mode="before"
    )
    @classmethod
    def _validate_ids(cls, value: Any) -> list[str]:
        return _bounded_strings(value, label="stage identifiers")

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_warnings(cls, value: Any) -> list[str]:
        return _bounded_strings(value, label="warnings")

    @field_validator("settings", mode="before")
    @classmethod
    def _validate_settings(cls, value: Any) -> dict[str, Any]:
        return _clean_mapping(value, label="stage settings")


class AnalysisPlan(_ContractModel):
    """Deterministic, persisted execution plan produced before media work."""

    source: str = ""
    source_identity: dict[str, Any] = Field(default_factory=dict)
    request: AnalyzeVideoRequest | dict[str, Any] = Field(default_factory=dict)
    normalized_request: dict[str, Any] = Field(default_factory=dict)
    effective_limits: dict[str, int] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    routes: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    stages: list[StageRecord] | dict[str, StageRecord] = Field(default_factory=list)
    frame_timestamps_seconds: tuple[float, ...] = ()
    cloud_policy: Literal["deny"] = "deny"
    timestamp: datetime | str = Field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime | str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "source_identity",
        "normalized_request",
        "capabilities",
        "routes",
        "diagnostics",
        mode="before",
    )
    @classmethod
    def _validate_plan_mapping(cls, value: Any) -> dict[str, Any]:
        return _clean_mapping(value, label="plan mapping")

    @field_validator("request", mode="before")
    @classmethod
    def _validate_plan_request(cls, value: Any) -> AnalyzeVideoRequest | dict[str, Any]:
        if isinstance(value, AnalyzeVideoRequest):
            return value
        return _clean_mapping(value, label="plan request")

    @field_validator("effective_limits", mode="before")
    @classmethod
    def _validate_limits_mapping(cls, value: Any) -> dict[str, int]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("effective_limits must be a mapping")
        result: dict[str, int] = {}
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not isinstance(item, int)
                or isinstance(item, bool)
            ):
                raise ValueError("effective limits must contain integer values")
            result[key] = item
        return result

    @field_validator("frame_timestamps_seconds", mode="before")
    @classmethod
    def _validate_plan_timestamps(cls, value: Any) -> tuple[float, ...]:
        if value is None:
            return ()
        result = tuple(value)
        if len(result) > MAX_FRAMES:
            raise ValueError("planned frame timestamps are bounded")
        converted: list[float] = []
        for item in result:
            number = float(item)
            if not math.isfinite(number):
                raise ValueError("planned frame timestamps must be finite")
            converted.append(number)
        return tuple(converted)

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_plan_warnings(cls, value: Any) -> list[str]:
        return _bounded_strings(value, label="plan warnings")


class AnalysisSummary(_ContractModel):
    """Small summary returned alongside persisted artifact references."""

    source: str = ""
    duration_seconds: float | None = None
    task_count: int = Field(default=0, ge=0)
    requested_task_count: int | None = Field(default=None, ge=0)
    completed_task_count: int | None = Field(default=None, ge=0)
    failed_task_count: int | None = Field(default=None, ge=0)
    requested_tasks: list[AnalysisTask] = Field(default_factory=list)
    completed_tasks: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)
    caption_kind: CaptionKind | str | None = None
    caption_provider: str | None = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _validate_duration(cls, value: Any) -> Any:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("duration_seconds must be finite and non-negative")
        return number

    @field_validator("completed_tasks", "failed_tasks", mode="before")
    @classmethod
    def _validate_task_names(cls, value: Any) -> list[str]:
        return _bounded_strings(value, label="summary tasks")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_summary_metadata(cls, value: Any) -> dict[str, Any]:
        return _clean_mapping(value, label="summary metadata")


class AnalysisMetrics(_ContractModel):
    """Execution telemetry and throughput metrics."""

    total_elapsed_ms: float = Field(default=0.0, ge=0)
    peak_rss_mb: float = Field(default=0.0, ge=0)
    rtf: float | None = Field(default=None, ge=0)
    ocr_fps: float | None = Field(default=None, ge=0)
    download_throughput_mbps: float | None = Field(default=None, ge=0)
    stage_elapsed_ms: dict[str, float] = Field(default_factory=dict)


class AnalysisResult(_ContractModel):
    """Successful analysis envelope, with completed or useful partial status."""

    ok: Literal[True] = True
    status: Literal["completed", "partial"] = "completed"
    summary: AnalysisSummary | None = None
    stages: list[StageRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: AnalysisMetrics | None = None
    manifest_uri: str | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_result_warnings(cls, value: Any) -> list[str]:
        return _bounded_strings(value, label="result warnings")

    @field_validator("artifacts", "artifact_refs", mode="before")
    @classmethod
    def _validate_result_artifacts(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)) or len(value) > MAX_ARTIFACT_REFS:
            raise ValueError("artifact references are bounded")
        return list(value)

    @field_validator("manifest_uri")
    @classmethod
    def _validate_result_manifest_uri(cls, value: str | None) -> str | None:
        if value is not None and _MANIFEST_URI_RE.fullmatch(value) is None:
            raise ValueError("manifest_uri is not a persisted manifest URI")
        return value

    @model_validator(mode="after")
    def _mirror_artifact_fields(self) -> AnalysisResult:
        if self.artifacts and not self.artifact_refs:
            self.artifact_refs = list(self.artifacts)
        elif self.artifact_refs and not self.artifacts:
            self.artifacts = list(self.artifact_refs)
        return self


type AnalysisResponse = Annotated[
    AnalysisResult | AnalysisError, Field(discriminator="ok")
]

# Resolve recursive/forward annotations for Pydantic v2 and for adapters that
# call model_json_schema() before constructing a model.
AnalysisError.model_rebuild()
StageRecord.model_rebuild()
AnalysisPlan.model_rebuild()
AnalysisMetrics.model_rebuild()
AnalysisResult.model_rebuild()


__all__ = [
    "MAX_DOWNLOAD_BYTES",
    "MAX_FRAMES",
    "MAX_FRAME_WIDTH",
    "MAX_OUTPUT_BYTES",
    "MAX_REQUEST_ID_LENGTH",
    "MAX_TIMEOUT_SECONDS",
    "MAX_TIME_RANGE_SECONDS",
    "AnalysisError",
    "AnalysisErrorCode",
    "AnalysisErrorStatus",
    "AnalysisMetrics",
    "AnalysisPlan",
    "AnalysisResponse",
    "AnalysisResult",
    "AnalysisResultStatus",
    "AnalysisStatus",
    "AnalysisTask",
    "AnalyzeVideoRequest",
    "ArtifactRef",
    "CaptionKind",
    "ErrorCode",
    "ErrorStatus",
    "FailureCode",
    "ResultStatus",
    "SourceKind",
    "StageRecord",
    "StageStatus",
    "TimeRange",
    "YOUTUBE_DOMAINS",
    "is_youtube_source",
]
