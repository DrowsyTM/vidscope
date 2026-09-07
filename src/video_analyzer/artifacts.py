from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from .contracts import AnalysisError, ArtifactRef, ErrorCode, StageRecord
from .settings import get_settings

_MAX_RESOURCE_BYTES = 1_048_576
_MAX_JSONL_RECORDS = 200
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_relative(value: str | Path) -> Path:
    candidate = Path(str(value).replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError("artifact path must be relative and contained")
    if any(part in {"", "."} for part in candidate.parts):
        raise ValueError("artifact path contains an invalid component")
    return candidate


def _error(
    code: str,
    message: str,
    *,
    stage: str = "persist_artifact",
    diagnostics: list[str] | None = None,
) -> AnalysisError:
    return AnalysisError(
        code=ErrorCode(code),
        stage=stage,
        message=message[:2_048],
        retryable=False,
        diagnostics={"details": diagnostics or []},
    )


class ArtifactStoreFailure(RuntimeError):
    def __init__(self, error: AnalysisError) -> None:
        self.error = error
        self.analysis_error = error
        self.failure = error
        self.code = error.code
        super().__init__(error.message)


@dataclass(slots=True)
class RunManifest:
    run_id: str
    request_id: str
    created_at: str = field(default_factory=_now)
    plan_uri: str | None = None
    manifest_uri: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] | None = None

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        del mode
        data: dict[str, Any] = {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "created_at": self.created_at,
            "plan_uri": self.plan_uri,
            "manifest_uri": self.manifest_uri,
            "stages": _json_value(self.stages),
            "artifacts": _json_value(self.artifacts),
            "warnings": list(self.warnings),
        }
        if self.metrics is not None:
            data["metrics"] = _json_value(self.metrics)
        return data


class ArtifactStore:
    """Run-local, atomic persistence for plans, manifests, and artifacts."""

    def __init__(
        self,
        output_directory: Path,
        request_id: str | None = None,
        max_output_bytes: int = 67_108_864,
    ) -> None:
        root = Path(output_directory).expanduser()
        if not root.is_absolute():
            raise ArtifactStoreFailure(
                _error("OUTPUT_NOT_ALLOWED", "output directory must be absolute")
            )
        self.output_directory = root.resolve(strict=False)
        self.request_id = request_id or uuid.uuid4().hex
        if not _RUN_ID_RE.fullmatch(self.request_id):
            raise ArtifactStoreFailure(
                _error("OUTPUT_NOT_ALLOWED", "request id is not a safe run id")
            )
        self.run_id = self.request_id
        self.run_directory = self.output_directory / self.run_id
        self.run_dir = self.run_directory
        self.staging_directory = self.run_directory / "staging"
        self.root = self.run_directory
        self.max_output_bytes = max(1, int(max_output_bytes))
        self.manifest_uri = f"video-analyzer://runs/{self.run_id}/manifest"
        self._manifest: dict[str, Any] = {}
        self._artifact_paths: dict[str, Path] = {}
        self._artifact_bytes = 0
        self._created = False

    @property
    def manifest_path(self) -> Path:
        return self.run_directory / "manifest.json"

    @property
    def plan_path(self) -> Path:
        return self.run_directory / "plan.json"

    @property
    def artifacts(self) -> dict[str, dict[str, Any]]:
        return dict(self._manifest.get("artifacts", {}))

    @property
    def manifest(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(json.dumps(self._manifest)))

    def create(self) -> ArtifactStore:
        if self.run_directory.exists():
            try:
                occupied = any(self.run_directory.iterdir())
            except OSError as exc:
                raise ArtifactStoreFailure(
                    _error(
                        "OUTPUT_NOT_ALLOWED",
                        "run directory cannot be inspected",
                        diagnostics=[str(exc)],
                    )
                ) from exc
            if occupied:
                raise ArtifactStoreFailure(
                    _error(
                        "OUTPUT_NOT_ALLOWED",
                        "run directory already exists and is nonempty",
                    )
                )
        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            self.run_directory.mkdir(parents=False, exist_ok=True)
            self.staging_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactStoreFailure(
                _error(
                    "OUTPUT_NOT_ALLOWED",
                    "run directory cannot be created",
                    diagnostics=[str(exc)],
                )
            ) from exc
        self._created = True
        self._manifest = RunManifest(
            run_id=self.run_id,
            request_id=self.request_id,
            manifest_uri=self.manifest_uri,
        ).model_dump()
        self.write_manifest()
        return self

    def _require_created(self) -> None:
        if not self._created:
            self.create()

    def _atomic_write(self, target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _write_json_atomic(self, target: Path, value: Any) -> None:
        payload = json.dumps(
            _json_value(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        ).encode("utf-8")
        self._atomic_write(target, payload)

    def write_plan(self, plan: Any) -> Path:
        self._require_created()
        self._write_json_atomic(self.plan_path, plan)
        self._manifest["plan_uri"] = f"video-analyzer://runs/{self.run_id}/plan"
        self.write_manifest()
        return self.plan_path

    def initialize_manifest(
        self,
        stages: list[Any] | Mapping[str, Any] | None = None,
        *,
        warnings: list[str] | None = None,
        **extra: Any,
    ) -> Path:
        self._require_created()
        if stages is not None:
            values = stages.values() if isinstance(stages, Mapping) else stages
            self._manifest["stages"] = [_json_value(value) for value in values]
        if warnings is not None:
            self._manifest["warnings"] = list(warnings)
        self._manifest.update({key: _json_value(value) for key, value in extra.items()})
        return self.write_manifest()

    def update_stage(
        self,
        stage: str | StageRecord,
        *,
        status: str | None = None,
        error: AnalysisError | None = None,
        **updates: Any,
    ) -> Path:
        self._require_created()
        stages = self._manifest.setdefault("stages", [])
        if isinstance(stage, StageRecord):
            value = _json_value(stage)
            name = value.get("name", "")
        else:
            name = stage
            value = None
        target: dict[str, Any] | None = None
        for item in stages:
            if isinstance(item, dict) and item.get("name") == name:
                target = item
                break
        if target is None:
            target = value or {
                "name": name,
                "dependencies": [],
                "settings": {},
                "warnings": [],
            }
            stages.append(target)
        if value is not None:
            target.clear()
            target.update(value)
        if status is not None:
            target["status"] = getattr(status, "value", status)
        if error is not None:
            target["error"] = _json_value(error)
        target.update({key: _json_value(item) for key, item in updates.items()})
        return self.write_manifest()

    def set_metrics(self, metrics: Any) -> Path:
        self._manifest["metrics"] = _json_value(metrics)
        return self.write_manifest()

    def write_manifest(self, manifest: Any | None = None) -> Path:
        self._require_created_for_manifest()
        if manifest is not None:
            self._manifest = _json_value(manifest)
        self._manifest.setdefault("run_id", self.run_id)
        self._manifest.setdefault("request_id", self.request_id)
        self._manifest.setdefault("manifest_uri", self.manifest_uri)
        self._write_json_atomic(self.manifest_path, self._manifest)
        return self.manifest_path

    def _require_created_for_manifest(self) -> None:
        if not self._created:
            self.create()

    def _artifact_target(self, name: str | None, source: Path) -> tuple[Path, Path]:
        chosen = name or source.name
        relative = _safe_relative(chosen)
        target = (self.run_directory / relative).resolve(strict=False)
        if not target.is_relative_to(self.run_directory.resolve()):
            raise ArtifactStoreFailure(
                _error("OUTPUT_NOT_ALLOWED", "artifact path escapes run directory")
            )
        return relative, target

    def publish_file(
        self,
        path: Path,
        *,
        name: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        self._require_created()
        source = Path(path)
        if not source.is_file():
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "artifact source file was not found")
            )
        try:
            size = source.stat().st_size
        except OSError as exc:
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "artifact source cannot be read")
            ) from exc
        if size <= 0:
            raise ArtifactStoreFailure(
                _error(
                    "INTERNAL_STAGE_FAILED", "zero-byte artifacts are not publishable"
                )
            )
        if (
            size > self.max_output_bytes
            or self._artifact_bytes + size > self.max_output_bytes
        ):
            raise ArtifactStoreFailure(
                _error("OUTPUT_LIMIT_EXCEEDED", "artifact output budget exceeded")
            )
        relative, target = self._artifact_target(name, source)
        if target.exists() and target.resolve() != source.resolve():
            stem, suffix = target.stem, target.suffix
            relative = relative.with_name(f"{stem}-{uuid.uuid4().hex[:8]}{suffix}")
            target = self.run_directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            if source.resolve() != target.resolve():
                shutil.copyfile(source, temporary)
                os.replace(temporary, target)
            else:
                target = source
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            actual_size = target.stat().st_size
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise ArtifactStoreFailure(
                _error(
                    "INTERNAL_STAGE_FAILED",
                    "artifact publication failed",
                    diagnostics=[str(exc)],
                )
            ) from exc
        if actual_size <= 0:
            target.unlink(missing_ok=True)
            raise ArtifactStoreFailure(
                _error(
                    "INTERNAL_STAGE_FAILED", "zero-byte artifacts are not publishable"
                )
            )
        if (
            actual_size > self.max_output_bytes
            or self._artifact_bytes + actual_size > self.max_output_bytes
        ):
            target.unlink(missing_ok=True)
            raise ArtifactStoreFailure(
                _error("OUTPUT_LIMIT_EXCEEDED", "artifact output budget exceeded")
            )
        artifact_id = (
            f"artifact-{len(self._artifact_paths) + 1:04d}-{uuid.uuid4().hex[:8]}"
        )
        artifact_metadata = dict(metadata or {})
        artifact_metadata["relative_path"] = relative.as_posix()
        ref = ArtifactRef(
            artifact_id=artifact_id,
            uri=f"video-analyzer://runs/{self.run_id}/artifacts/{artifact_id}",
            media_type=media_type
            or mimetypes.guess_type(target.name)[0]
            or "application/octet-stream",
            byte_size=actual_size,
            sha256=digest,
            name=relative.as_posix(),
            metadata=artifact_metadata,
        )
        self._artifact_paths[artifact_id] = target
        self._artifact_bytes += actual_size
        self._manifest.setdefault("artifacts", {})[artifact_id] = ref.model_dump(
            mode="json"
        )
        self.write_manifest()
        return ref

    def artifact_from_path(self, path: Path, **kwargs: Any) -> ArtifactRef:
        return self.publish_file(path, **kwargs)

    def add_artifact(self, path: Path, **kwargs: Any) -> ArtifactRef:
        return self.publish_file(path, **kwargs)

    register_artifact = add_artifact
    create_artifact = add_artifact
    persist_artifact = add_artifact
    add_file = add_artifact

    def write_bytes(
        self,
        payload: bytes,
        *,
        name: str,
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        self._require_created()
        if (
            len(payload) > self.max_output_bytes
            or self._artifact_bytes + len(payload) > self.max_output_bytes
        ):
            raise ArtifactStoreFailure(
                _error("OUTPUT_LIMIT_EXCEEDED", "artifact output budget exceeded")
            )
        fd, temporary = tempfile.mkstemp(prefix="payload-", dir=self.staging_directory)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
            return self.publish_file(
                temporary_path, name=name, media_type=media_type, metadata=metadata
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    def write_text(
        self,
        payload: str,
        *,
        name: str,
        media_type: str = "text/plain",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        return self.write_bytes(
            payload.encode("utf-8"), name=name, media_type=media_type, metadata=metadata
        )

    def write_json(
        self,
        value: Any,
        *,
        name: str,
        media_type: str = "application/json",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        return self.write_text(
            json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            name=name,
            media_type=media_type,
            metadata=metadata,
        )

    def write_jsonl(
        self,
        rows: Any,
        *,
        name: str,
        media_type: str = "application/x-ndjson",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        payload = "".join(
            json.dumps(_json_value(row), ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for row in rows
        )
        return self.write_text(
            payload, name=name, media_type=media_type, metadata=metadata
        )

    def resolve_artifact_path(self, ref: ArtifactRef | Mapping[str, Any] | str) -> Path:
        artifact_id = (
            ref
            if isinstance(ref, str)
            else (
                ref.get("artifact_id") if isinstance(ref, Mapping) else ref.artifact_id
            )
        )
        artifacts = self._manifest.get("artifacts", {})
        entry = artifacts.get(artifact_id) if isinstance(artifacts, Mapping) else None
        if not isinstance(entry, Mapping):
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "artifact id is not in the manifest")
            )
        metadata = entry.get("metadata")
        relative = (
            metadata.get("relative_path") if isinstance(metadata, Mapping) else None
        )
        if not isinstance(relative, str):
            raise ArtifactStoreFailure(
                _error(
                    "ARTIFACT_NOT_FOUND", "artifact path is missing from the manifest"
                )
            )
        try:
            relative_path = _safe_relative(relative)
        except ValueError as exc:
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "artifact path is invalid")
            ) from exc
        path = (self.run_directory / relative_path).resolve(strict=False)
        if not path.is_relative_to(self.run_directory.resolve()) or not path.is_file():
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "artifact file is not available")
            )
        self._artifact_paths[str(artifact_id)] = path
        return path

    @staticmethod
    def _load_resource_manifest(
        output_root: Path, run_id: str
    ) -> tuple[Path, dict[str, Any]]:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "run id is invalid")
            )
        root = output_root.expanduser().resolve(strict=False)
        run_directory = (root / run_id).resolve(strict=False)
        if not run_directory.is_relative_to(root) or not run_directory.is_dir():
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "run is not available")
            )
        manifest_path = run_directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "run manifest is unavailable")
            ) from exc
        if not isinstance(manifest, dict):
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "run manifest is invalid")
            )
        return run_directory, manifest

    @classmethod
    def read_uri(
        cls,
        output_root: Path,
        uri: str,
        *,
        page: int | None = None,
        offset: int = 0,
        limit: int = 200,
    ) -> str | bytes:
        parsed = urlsplit(uri)
        if parsed.scheme != "video-analyzer" or parsed.netloc != "runs":
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "artifact URI is invalid")
            )
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 3 or parts[1] != "artifacts":
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "artifact URI is invalid")
            )
        run_id, artifact_id = parts[0], parts[2]
        run_directory, manifest = cls._load_resource_manifest(Path(output_root), run_id)
        try:
            offset = int(offset)
            limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ArtifactStoreFailure(
                _error("ARTIFACT_RANGE_INVALID", "artifact range must be numeric")
            ) from exc
        if offset < 0:
            raise ArtifactStoreFailure(
                _error("ARTIFACT_RANGE_INVALID", "offset must be non-negative")
            )
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError) as exc:
                raise ArtifactStoreFailure(
                    _error("ARTIFACT_RANGE_INVALID", "page must be numeric")
                ) from exc
            if page < 1:
                raise ArtifactStoreFailure(
                    _error("ARTIFACT_RANGE_INVALID", "page must be positive")
                )
            offset = (page - 1) * limit
        if offset < 0 or limit <= 0 or limit > _MAX_JSONL_RECORDS:
            raise ArtifactStoreFailure(
                _error("ARTIFACT_RANGE_INVALID", "artifact range is outside bounds")
            )
        artifacts = manifest.get("artifacts")
        if isinstance(artifacts, Mapping):
            entry = artifacts.get(artifact_id)
        elif isinstance(artifacts, list):
            entry = next(
                (
                    candidate
                    for candidate in artifacts
                    if isinstance(candidate, Mapping)
                    and candidate.get("artifact_id") == artifact_id
                ),
                None,
            )
        else:
            entry = None
        if not isinstance(entry, Mapping):
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "artifact id is not in the manifest")
            )
        metadata = entry.get("metadata")
        relative = (
            metadata.get("relative_path") if isinstance(metadata, Mapping) else None
        )
        if not isinstance(relative, str):
            relative = entry.get("relative_path", entry.get("path"))
        try:
            relative_path = _safe_relative(relative)
        except (TypeError, ValueError) as exc:
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "artifact path is invalid")
            ) from exc
        target = (run_directory / relative_path).resolve(strict=False)
        if not target.is_relative_to(run_directory.resolve()) or not target.is_file():
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "artifact file is unavailable")
            )
        if (
            target.suffix.lower() in {".jsonl", ".ndjson"}
            or entry.get("media_type") == "application/x-ndjson"
        ):
            try:
                records = target.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                raise ArtifactStoreFailure(
                    _error("ARTIFACT_NOT_FOUND", "artifact cannot be read")
                ) from exc
            return "\n".join(records[offset : offset + limit]) + (
                "\n" if records[offset : offset + limit] else ""
            )
        if (
            offset > _MAX_RESOURCE_BYTES
            or limit > _MAX_RESOURCE_BYTES
            or offset + limit > _MAX_RESOURCE_BYTES
        ):
            raise ArtifactStoreFailure(
                _error("ARTIFACT_RANGE_INVALID", "artifact byte range is too large")
            )
        try:
            with target.open("rb") as stream:
                stream.seek(offset)
                return stream.read(limit)
        except OSError as exc:
            raise ArtifactStoreFailure(
                _error("ARTIFACT_NOT_FOUND", "artifact cannot be read")
            ) from exc

    def read_resource(
        self, uri: str, *, page: int | None = None, offset: int = 0, limit: int = 200
    ) -> str | bytes:
        return self.read_uri(
            self.output_directory, uri, page=page, offset=offset, limit=limit
        )


def read_artifact_resource(
    uri: str,
    *,
    page: int | None = None,
    offset: int = 0,
    limit: int = 200,
    output_root: Path | None = None,
) -> str | bytes | ArtifactStoreFailure:
    root = output_root
    if root is None:
        settings = get_settings()
        root = settings.allowed_output_root or Path.cwd()
    try:
        return ArtifactStore.read_uri(root, uri, page=page, offset=offset, limit=limit)
    except ArtifactStoreFailure as exc:
        return exc


__all__ = [
    "ArtifactStore",
    "ArtifactStoreFailure",
    "RunManifest",
    "read_artifact_resource",
]
