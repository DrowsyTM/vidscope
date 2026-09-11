from __future__ import annotations

import base64
import builtins
import hashlib
import http.client
import importlib.util
import json
import os
import re
import socket
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from vidscope.backends.source import CaptionTrack, SourceInspection
from vidscope.contracts import AnalyzeVideoRequest, TimeRange
from vidscope.core import AnalysisContext, VideoAnalyzerFailure, analyze_video

ARTIFACT_URI = re.compile(r"^vidscope://runs/[^/]+/artifacts/[^/]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(autouse=True)
def _block_network_and_cloud_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every core test fail on accidental network/cloud provider use."""

    def fail_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unexpected network access in deterministic core test")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
    monkeypatch.setattr(urllib.request, "urlopen", fail_network)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", fail_network)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", fail_network)

    forbidden_prefixes = (
        "anthropic",
        "boto3",
        "botocore",
        "cohere",
        "google.cloud",
        "google.generativeai",
        "litellm",
        "openai",
        "vertexai",
    )
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        ):
            raise AssertionError(f"unexpected cloud provider import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@dataclass
class EventLog:
    """State observed by fakes while a single run is executing."""

    root: Path
    events: list[str] = field(default_factory=list)
    manifest_snapshots: list[bytes] = field(default_factory=list)
    plan_seen_at_media: bool = False
    plan_mtime_at_media: int | None = None
    url_env_value: str | None = None
    cloud_calls: list[str] = field(default_factory=list)
    published_media_ref: Any | None = None

    def run_dir(self, request: AnalyzeVideoRequest) -> Path:
        request_id = request.request_id or ""
        return request.output_directory / request_id

    def capture_manifest(self, request: AnalyzeVideoRequest) -> None:
        manifest = self.run_dir(request) / "manifest.json"
        if manifest.exists():
            raw = manifest.read_bytes()
            # An atomic update must never expose a partially written JSON file.
            json.loads(raw)
            self.manifest_snapshots.append(raw)


@dataclass
class FakeSourceInspector:
    log: EventLog
    tracks: list[CaptionTrack] = field(default_factory=list)
    on_inspect: Callable[[], None] | None = None
    is_url: bool = False

    def inspect(
        self, request: AnalyzeVideoRequest, *args: Any, **kwargs: Any
    ) -> SourceInspection:
        self.log.events.append("inspect_source")
        # Planning belongs after inspection, not before it.
        assert not (self.log.run_dir(request) / "plan.json").exists()
        if self.on_inspect is not None:
            self.on_inspect()
        return SourceInspection(
            source=request.source,
            is_url=self.is_url,
            duration_seconds=4.0,
            caption_tracks=self.tracks,
            formats=[
                {"format_id": "fixture", "ext": "mp4", "width": 320, "height": 180}
            ],
            metadata={"title": "deterministic fixture"},
        )


@dataclass
class FakeCaptionResolver:
    log: EventLog
    track: CaptionTrack | None

    def resolve(
        self,
        inspection: SourceInspection,
        request: AnalyzeVideoRequest,
        *args: Any,
        **kwargs: Any,
    ) -> CaptionTrack | None:
        self.log.events.append("resolve_captions")
        return self.track


@dataclass
class FakeMediaAcquirer:
    log: EventLog
    media_path: Path

    def acquire_window(
        self,
        inspection: SourceInspection,
        request: AnalyzeVideoRequest,
        store: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.log.events.append("acquire_media")
        run_dir = self.log.run_dir(request)
        plan = run_dir / "plan.json"
        self.log.plan_seen_at_media = plan.exists()
        self.log.plan_mtime_at_media = (
            plan.stat().st_mtime_ns if plan.exists() else None
        )
        self.log.url_env_value = os.environ.get("YTDLP_IGNORE_CONFIG")
        if inspection.is_url:
            # Assert the source-layer boundary, not merely a process-wide
            # environment value observed after the run.
            assert self.log.url_env_value == "1"
        self.log.capture_manifest(request)
        publish_file = getattr(store, "publish_file", None)
        if callable(publish_file):
            self.log.published_media_ref = publish_file(
                self.media_path,
                name="media.mp4",
                media_type="video/mp4",
                metadata={"source": "fake-acquirer"},
            )
            return self.log.published_media_ref
        # Keep this fake usable with small/simple store doubles that have no
        # artifact publication API.
        return self.media_path


@dataclass
class FakeMediaBackend:
    log: EventLog
    work_dir: Path
    fail_probe: bool = False
    fail_frames: bool = False
    empty_frames: bool = False
    active: int = 0
    max_active: int = 0
    _active_lock: threading.Lock = field(default_factory=threading.Lock)

    def _enter(self) -> None:
        with self._active_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _leave(self) -> None:
        with self._active_lock:
            self.active -= 1

    def probe(self, media: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        request = _request_argument(args, kwargs)
        self.log.events.append("probe")
        if request is not None:
            self.log.capture_manifest(request)
        if self.fail_probe:
            raise RuntimeError("fixture probe failure")
        return {
            "duration_seconds": 4.0,
            "streams": [{"codec_type": "video", "width": 320, "height": 180}],
        }

    def extract_audio(self, media: Any, *args: Any, **kwargs: Any) -> Path:
        request = _request_argument(args, kwargs)
        self.log.events.append("extract_audio")
        if request is not None:
            self.log.capture_manifest(request)
        self._enter()
        try:
            time.sleep(0.005)
            audio = self.work_dir / "audio.wav"
            audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt fixture-audio")
            return audio
        finally:
            self._leave()

    def extract_frames(self, media: Any, *args: Any, **kwargs: Any) -> list[Path]:
        request = _request_argument(args, kwargs)
        self.log.events.append("extract_frames")
        if request is not None:
            self.log.capture_manifest(request)
        self._enter()
        try:
            time.sleep(0.005)
            frame = self.work_dir / "frame-0001.jpg"
            frame.write_bytes(b"JPEG-FIXTURE-NOT-BASE64")
            if self.fail_frames:
                raise RuntimeError("fixture frame failure")
            if self.empty_frames:
                frame.write_bytes(b"")
            return [frame]
        finally:
            self._leave()


@dataclass
class FakeAsrBackend:
    log: EventLog
    empty: bool = False
    fail: bool = False
    segments: list[dict[str, Any]] | None = None

    def transcribe(
        self, audio: Any, request: AnalyzeVideoRequest, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        self.log.events.append("transcribe")
        self.log.capture_manifest(request)
        if self.fail:
            raise RuntimeError("fixture ASR failure")
        if self.segments is not None:
            return {
                "segments": [dict(s) for s in self.segments],
                "metadata": {"language": request.language, "provider": "fake-local"},
            }
        segments: list[dict[str, Any]] = []
        if not self.empty:
            segments.append(
                {
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "text": "asr body must remain in the artifact",
                    "words": [],
                }
            )
        return {
            "segments": segments,
            "metadata": {"language": request.language, "provider": "fake-local"},
        }


@dataclass
class FakeVadBackend:
    log: EventLog

    def detect(
        self,
        audio: Any,
        request: AnalyzeVideoRequest | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.log.events.append("vad")
        if request is not None:
            self.log.capture_manifest(request)
        return {
            "intervals": [{"start_seconds": 0.0, "end_seconds": 1.0}],
            "metadata": {"provider": "fake-local"},
        }


@dataclass
class FakeOcrBackend:
    log: EventLog
    fail: bool = False

    def recognize(
        self, frame: Any, language: str = "eng", *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        self.log.events.append("ocr")
        request = _request_argument(args, kwargs)
        if request is not None:
            self.log.capture_manifest(request)
        if self.fail:
            raise RuntimeError("fixture OCR failure")
        return {
            "rows": [
                {
                    "text": "fixture OCR",
                    "confidence": 98.0,
                    "left": 1,
                    "top": 2,
                    "width": 20,
                    "height": 8,
                }
            ],
            "metadata": {"provider": "fake-local"},
        }


def _request_argument(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> AnalyzeVideoRequest | None:
    for value in (*args, *kwargs.values()):
        if isinstance(value, AnalyzeVideoRequest):
            return value
    return None


def _caption_track(
    kind: str,
    *,
    text: str = "caption body must remain in the artifact",
    segments: list[dict[str, Any]] | None = None,
) -> CaptionTrack:
    return CaptionTrack(
        kind=kind,
        language="en",
        provider="fake-caption-provider",
        source_url=None,
        segments=segments
        if segments is not None
        else [{"start_seconds": 0.0, "end_seconds": 1.0, "text": text, "words": []}],
    )


def _request(
    tmp_path: Path,
    source: str | Path,
    *,
    tasks: set[str],
    request_id: str = "core-test-run",
    **overrides: Any,
) -> AnalyzeVideoRequest:
    output_directory = tmp_path / "output"
    output_directory.mkdir(exist_ok=True)
    values: dict[str, Any] = {
        "source": str(source),
        "time_range": TimeRange(start_seconds=0.0, end_seconds=4.0),
        "tasks": tasks,
        "output_directory": output_directory,
        "language": "en",
        "request_id": request_id,
    }
    values.update(overrides)
    return AnalyzeVideoRequest(**values)


def _context(
    log: EventLog,
    source_inspector: FakeSourceInspector,
    caption_resolver: FakeCaptionResolver,
    media_acquirer: FakeMediaAcquirer,
    media_backend: FakeMediaBackend,
    asr_backend: FakeAsrBackend | None = None,
    vad_backend: FakeVadBackend | None = None,
    ocr_backend: FakeOcrBackend | None = None,
    *,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> AnalysisContext:
    values: dict[str, Any] = {
        "source_inspector": source_inspector,
        "caption_resolver": caption_resolver,
        "media_acquirer": media_acquirer,
        "media_backend": media_backend,
        "asr_backend": asr_backend or FakeAsrBackend(log),
        "vad_backend": vad_backend or FakeVadBackend(log),
        "ocr_backend": ocr_backend or FakeOcrBackend(log),
    }
    if cancel_event is not None:
        values["cancel_event"] = cancel_event
    if deadline is not None:
        values["deadline"] = deadline
    return AnalysisContext(**values)


def _run_dir(request: AnalyzeVideoRequest) -> Path:
    assert request.request_id is not None
    return request.output_directory / request.request_id


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _stage_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    stages = manifest["stages"]
    if isinstance(stages, Mapping):
        return [value for value in stages.values() if isinstance(value, dict)]
    return list(stages)


def _stage(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    for record in _stage_records(manifest):
        if record.get("name") == name:
            return record
    raise AssertionError(f"stage {name!r} was not persisted")


def _dump(result: Any) -> dict[str, Any]:
    dumped = result.model_dump(mode="json")
    assert isinstance(dumped, dict)
    return dumped


def _artifact_refs(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        required = {"artifact_id", "uri", "media_type", "byte_size", "sha256"}
        if required <= set(value):
            found.append(dict(value))
        for child in value.values():
            found.extend(_artifact_refs(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_artifact_refs(child))
    return found


def _assert_bounded_result(
    result: Any, *, absent_markers: tuple[str, ...]
) -> dict[str, Any]:
    dumped = _dump(result)
    serialized = json.dumps(dumped, sort_keys=True)
    assert len(serialized.encode("utf-8")) < 16_384
    for marker in absent_markers:
        assert marker not in serialized
    refs = _artifact_refs(dumped)
    assert refs, (
        "successful analysis must return artifact references, not inline payloads"
    )
    for ref in refs:
        assert ARTIFACT_URI.fullmatch(ref["uri"])
        assert SHA256.fullmatch(ref["sha256"])
        assert isinstance(ref["byte_size"], int) and ref["byte_size"] > 0
        assert isinstance(ref["media_type"], str) and ref["media_type"]
    return dumped


def _manifest_artifact_refs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, Mapping), "manifest must persist an artifacts mapping"
    refs = _artifact_refs(artifacts)
    assert refs, "manifest must contain persisted artifact references"
    return refs


def _assert_artifacts_resolve(
    request: AnalyzeVideoRequest,
    result: Any,
    manifest: Mapping[str, Any],
) -> None:
    run_dir = _run_dir(request).resolve()
    manifest_refs = _manifest_artifact_refs(manifest)
    by_id = {ref["artifact_id"]: ref for ref in manifest_refs}
    assert len(by_id) == len(manifest_refs), "manifest artifact IDs must be unique"
    for ref in manifest_refs:
        metadata = ref.get("metadata")
        assert isinstance(metadata, Mapping)
        relative_value = metadata.get("relative_path")
        assert isinstance(relative_value, str) and relative_value
        relative_path = Path(relative_value)
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        artifact_path = (run_dir / relative_path).resolve()
        assert artifact_path.is_relative_to(run_dir)
        assert artifact_path.is_file()
        payload = artifact_path.read_bytes()
        assert len(payload) == ref["byte_size"]
        assert hashlib.sha256(payload).hexdigest() == ref["sha256"]
        assert ARTIFACT_URI.fullmatch(ref["uri"])

    result_refs = _artifact_refs(_dump(result))
    assert result_refs, "result must return references to persisted artifacts"
    for ref in result_refs:
        persisted = by_id.get(ref["artifact_id"])
        assert persisted is not None
        assert ref["uri"] == persisted["uri"]
        assert ref["sha256"] == persisted["sha256"]
        assert ref["byte_size"] == persisted["byte_size"]


def test_plan_is_persisted_before_media_and_manifest_updates_are_atomic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"small deterministic media placeholder")
    log = EventLog(tmp_path)
    inspector = FakeSourceInspector(log)
    resolver = FakeCaptionResolver(log, None)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    request = _request(tmp_path, source, tasks={"metadata", "frames"})

    result = analyze_video(
        request,
        context=_context(log, inspector, resolver, media, media_backend),
    )

    dumped = _dump(result)
    assert dumped["ok"] is True
    assert dumped["status"] == "completed"
    assert log.events.index("inspect_source") < log.events.index("acquire_media")
    assert log.plan_seen_at_media is True
    assert log.plan_mtime_at_media is not None

    run_dir = _run_dir(request)
    plan = run_dir / "plan.json"
    manifest = run_dir / "manifest.json"
    assert plan.exists()
    assert manifest.exists()
    assert plan.stat().st_mtime_ns <= log.plan_mtime_at_media

    records = _stage_records(_read_json(manifest))
    for name in (
        "validate_source",
        "inspect_source",
        "persist_plan",
        "acquire_media",
        "probe",
        "extract_frames",
    ):
        assert _stage(_read_json(manifest), name)["status"] == "completed"
    assert len(log.manifest_snapshots) >= 2
    assert len(set(log.manifest_snapshots)) >= 2
    snapshot_statuses: list[set[str]] = []
    for snapshot in log.manifest_snapshots:
        parsed = json.loads(snapshot)
        assert isinstance(parsed, dict)
        snapshot_statuses.append(
            {record["status"] for record in _stage_records(parsed)}
        )
    assert any("running" in statuses for statuses in snapshot_statuses)
    assert not list(run_dir.glob("manifest.json.*"))
    terminal_statuses = {"completed", "skipped", "failed", "cancelled", "timed_out"}
    assert records and all(record["status"] in terminal_statuses for record in records)
    assert all(
        record.get("start_timestamp") and record.get("end_timestamp")
        for record in records
    )
    _assert_artifacts_resolve(request, result, _read_json(manifest))


@pytest.mark.parametrize("kind", ["manual", "automatic"])
def test_caption_transcript_is_artifact_only_with_complete_hash_metadata(
    tmp_path: Path, kind: str
) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"caption-only source")
    log = EventLog(tmp_path)
    track = _caption_track(kind)
    inspector = FakeSourceInspector(log, tracks=[track])
    resolver = FakeCaptionResolver(log, track)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    request = _request(tmp_path, source, tasks={"metadata", "transcript"})

    result = analyze_video(
        request, context=_context(log, inspector, resolver, media, media_backend)
    )

    dumped = _assert_bounded_result(
        result,
        absent_markers=(
            "caption body must remain in the artifact",
            base64.b64encode(b"caption body").decode(),
        ),
    )
    assert dumped["ok"] is True
    assert dumped["status"] == "completed"
    assert str(dumped["manifest_uri"]).startswith("vidscope://runs/")

    run_dir = _run_dir(request)
    transcript = run_dir / "transcript.jsonl"
    assert transcript.exists()
    lines = [
        json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()
    ]
    assert lines and lines[0]["text"] == "caption body must remain in the artifact"
    expected_hash = hashlib.sha256(transcript.read_bytes()).hexdigest()
    refs = _artifact_refs(dumped)
    transcript_refs = [ref for ref in refs if ref["sha256"] == expected_hash]
    assert transcript_refs
    assert transcript_refs[0]["byte_size"] == transcript.stat().st_size
    manifest = _read_json(run_dir / "manifest.json")
    assert _stage(manifest, "captions")["status"] == "completed"
    plan = _read_json(run_dir / "plan.json")
    assert plan["routes"]["transcript"]["kind"] == kind
    assert _stage(plan, "captions")["settings"]["kind"] == kind
    _assert_artifacts_resolve(request, result, manifest)


def test_missing_captions_use_bounded_local_asr_and_vad_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"asr source")
    log = EventLog(tmp_path)
    inspector = FakeSourceInspector(log, tracks=[])
    resolver = FakeCaptionResolver(log, None)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    request = _request(
        tmp_path, source, tasks={"metadata", "transcript"}, asr_enabled=True
    )

    result = analyze_video(
        request, context=_context(log, inspector, resolver, media, media_backend)
    )

    dumped = _assert_bounded_result(
        result, absent_markers=("asr body must remain in the artifact",)
    )
    assert dumped["status"] == "completed"
    for name in ("acquire_media", "probe", "extract_audio", "transcribe", "vad"):
        assert name in log.events
    assert (
        log.events.index("acquire_media")
        < log.events.index("probe")
        < log.events.index("extract_audio")
    )
    assert log.events.index("extract_audio") < log.events.index("transcribe")
    manifest = _read_json(_run_dir(request) / "manifest.json")
    for name in ("acquire_media", "probe", "extract_audio", "transcribe", "vad"):
        assert _stage(manifest, name)["status"] == "completed"
    assert "cloud" not in " ".join(log.cloud_calls).lower()


def test_no_caption_transcript_and_frames_keep_media_concurrency_bounded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "combined-source.mp4"
    source.write_bytes(b"combined transcript and frames source")
    log = EventLog(tmp_path)
    inspector = FakeSourceInspector(log, tracks=[])
    resolver = FakeCaptionResolver(log, None)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    request = _request(
        tmp_path,
        source,
        tasks={"metadata", "transcript", "frames"},
        asr_enabled=True,
        request_id="combined-run",
    )

    result = analyze_video(
        request, context=_context(log, inspector, resolver, media, media_backend)
    )

    dumped = _assert_bounded_result(
        result, absent_markers=("asr body must remain in the artifact",)
    )
    assert dumped["status"] == "completed"
    assert media_backend.max_active <= 2
    assert {
        "acquire_media",
        "probe",
        "extract_audio",
        "transcribe",
        "vad",
        "extract_frames",
    } <= set(log.events)
    manifest = _read_json(_run_dir(request) / "manifest.json")
    for name in ("transcribe", "vad", "extract_frames"):
        assert _stage(manifest, name)["status"] == "completed"
    _assert_artifacts_resolve(request, result, manifest)


def test_ocr_request_extracts_frames_then_persists_nonempty_ocr_artifact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ocr-source.mp4"
    source.write_bytes(b"ocr source")
    log = EventLog(tmp_path)
    inspector = FakeSourceInspector(log)
    resolver = FakeCaptionResolver(log, None)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    request = _request(
        tmp_path, source, tasks={"metadata", "ocr"}, request_id="ocr-run"
    )

    result = analyze_video(
        request, context=_context(log, inspector, resolver, media, media_backend)
    )

    dumped = _assert_bounded_result(result, absent_markers=("fixture OCR",))
    assert dumped["status"] == "completed"
    assert log.events.index("extract_frames") < log.events.index("ocr")
    run_dir = _run_dir(request)
    ocr_path = run_dir / "ocr.jsonl"
    assert ocr_path.is_file()
    ocr_lines = [
        json.loads(line) for line in ocr_path.read_text(encoding="utf-8").splitlines()
    ]
    assert ocr_lines and any(
        "fixture OCR" in json.dumps(line, sort_keys=True) for line in ocr_lines
    )
    manifest = _read_json(run_dir / "manifest.json")
    assert _stage(manifest, "extract_frames")["status"] == "completed"
    assert _stage(manifest, "ocr")["status"] == "completed"
    ocr_refs = [
        ref
        for ref in _artifact_refs(dumped)
        if ref.get("metadata", {}).get("relative_path") == "ocr.jsonl"
    ]
    assert ocr_refs, "successful OCR must return an ocr.jsonl artifact reference"
    _assert_artifacts_resolve(request, result, manifest)


def test_optional_visual_failure_returns_useful_partial_result(tmp_path: Path) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"partial source")
    log = EventLog(tmp_path)
    track = _caption_track("manual")
    inspector = FakeSourceInspector(log, tracks=[track])
    resolver = FakeCaptionResolver(log, track)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work", fail_frames=True)
    media_backend.work_dir.mkdir()
    request = _request(tmp_path, source, tasks={"metadata", "transcript", "frames"})

    result = analyze_video(
        request, context=_context(log, inspector, resolver, media, media_backend)
    )

    dumped = _dump(result)
    assert dumped["ok"] is True
    assert dumped["status"] == "partial"
    assert "caption body must remain in the artifact" not in json.dumps(dumped)
    assert str(dumped["manifest_uri"]).startswith("vidscope://runs/")
    manifest = _read_json(_run_dir(request) / "manifest.json")
    assert _stage(manifest, "captions")["status"] == "completed"
    assert _stage(manifest, "extract_frames")["status"] == "failed"
    assert _stage(manifest, "extract_frames").get("error")
    assert "ocr" not in log.events


def test_terminal_stage_error_raises_with_manifest_reference(tmp_path: Path) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"terminal source")
    log = EventLog(tmp_path)
    inspector = FakeSourceInspector(log, tracks=[])
    resolver = FakeCaptionResolver(log, None)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    request = _request(tmp_path, source, tasks={"metadata", "transcript"})
    asr = FakeAsrBackend(log, fail=True)

    with pytest.raises(VideoAnalyzerFailure) as raised:
        analyze_video(
            request,
            context=_context(
                log, inspector, resolver, media, media_backend, asr_backend=asr
            ),
        )

    failure = raised.value
    assert failure.error.code == "INTERNAL_STAGE_FAILED"
    assert failure.error.stage == "transcribe"
    assert failure.manifest_uri
    assert str(failure.manifest_uri).startswith("vidscope://runs/")
    manifest = _read_json(_run_dir(request) / "manifest.json")
    failed = _stage(manifest, "transcribe")
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "INTERNAL_STAGE_FAILED"


def test_zero_byte_transcript_is_a_typed_failure_not_success(tmp_path: Path) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"empty transcript source")
    log = EventLog(tmp_path)
    inspector = FakeSourceInspector(log, tracks=[])
    resolver = FakeCaptionResolver(log, None)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    request = _request(tmp_path, source, tasks={"metadata", "transcript"})

    with pytest.raises(VideoAnalyzerFailure) as raised:
        analyze_video(
            request,
            context=_context(
                log,
                inspector,
                resolver,
                media,
                media_backend,
                asr_backend=FakeAsrBackend(log, empty=True),
            ),
        )

    assert raised.value.error.code == "INTERNAL_STAGE_FAILED"
    assert raised.value.error.stage == "transcribe"
    assert raised.value.manifest_uri
    refs = _artifact_refs(_read_json(_run_dir(request) / "manifest.json"))
    assert all(ref["byte_size"] > 0 for ref in refs)
    assert (
        _stage(_read_json(_run_dir(request) / "manifest.json"), "transcribe")["status"]
        == "failed"
    )


def test_zero_byte_frame_is_a_typed_failure_not_success(tmp_path: Path) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"empty frame source")
    log = EventLog(tmp_path)
    inspector = FakeSourceInspector(log)
    resolver = FakeCaptionResolver(log, None)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work", empty_frames=True)
    media_backend.work_dir.mkdir()
    request = _request(tmp_path, source, tasks={"metadata", "frames"})

    with pytest.raises(VideoAnalyzerFailure) as raised:
        analyze_video(
            request, context=_context(log, inspector, resolver, media, media_backend)
        )

    assert raised.value.error.code == "INTERNAL_STAGE_FAILED"
    assert raised.value.error.stage == "extract_frames"
    assert raised.value.manifest_uri
    manifest = _read_json(_run_dir(request) / "manifest.json")
    assert _stage(manifest, "extract_frames")["status"] == "failed"
    assert not any(ref["byte_size"] == 0 for ref in _artifact_refs(manifest))


def test_cancellation_after_inspection_maps_to_cancelled_with_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"cancel source")
    log = EventLog(tmp_path)
    cancel_event = threading.Event()
    inspector = FakeSourceInspector(log, on_inspect=cancel_event.set)
    resolver = FakeCaptionResolver(log, None)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    request = _request(tmp_path, source, tasks={"metadata", "frames"})

    with pytest.raises(VideoAnalyzerFailure) as raised:
        analyze_video(
            request,
            context=_context(
                log,
                inspector,
                resolver,
                media,
                media_backend,
                cancel_event=cancel_event,
            ),
        )

    assert raised.value.error.code == "CANCELLED"
    assert raised.value.manifest_uri
    manifest = _read_json(_run_dir(request) / "manifest.json")
    assert any(record["status"] == "cancelled" for record in _stage_records(manifest))
    assert "acquire_media" not in log.events


def test_expired_deadline_maps_to_timeout_with_manifest(tmp_path: Path) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"deadline source")
    log = EventLog(tmp_path)
    inspector = FakeSourceInspector(log)
    resolver = FakeCaptionResolver(log, None)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    request = _request(tmp_path, source, tasks={"metadata"})

    with pytest.raises(VideoAnalyzerFailure) as raised:
        analyze_video(
            request,
            context=_context(
                log,
                inspector,
                resolver,
                media,
                media_backend,
                deadline=time.monotonic() - 1.0,
            ),
        )

    assert raised.value.error.code == "TIMEOUT"
    assert raised.value.manifest_uri
    manifest = _read_json(_run_dir(request) / "manifest.json")
    assert any(record["status"] == "timed_out" for record in _stage_records(manifest))


def test_output_limit_is_typed_and_preserves_partial_manifest(tmp_path: Path) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"bounded output source")
    log = EventLog(tmp_path)
    huge_text = "x" * 256
    segments = [
        {
            "start_seconds": index / 20,
            "end_seconds": (index + 1) / 20,
            "text": huge_text,
            "words": [],
        }
        for index in range(20)
    ]
    track = _caption_track("manual", text=huge_text, segments=segments)
    inspector = FakeSourceInspector(log, tracks=[track])
    resolver = FakeCaptionResolver(log, track)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    request = _request(
        tmp_path,
        source,
        tasks={"metadata", "transcript"},
        max_output_bytes=1_024,
    )

    with pytest.raises(VideoAnalyzerFailure) as raised:
        analyze_video(
            request, context=_context(log, inspector, resolver, media, media_backend)
        )

    assert raised.value.error.code == "OUTPUT_LIMIT_EXCEEDED"
    assert raised.value.manifest_uri
    manifest = _read_json(_run_dir(request) / "manifest.json")
    assert any(record["status"] == "failed" for record in _stage_records(manifest))
    assert _run_dir(request).joinpath("manifest.json").exists()
    published_refs = _artifact_refs(manifest.get("artifacts", {}))
    assert all(ref["byte_size"] <= request.max_output_bytes for ref in published_refs)
    transcript_files = list(_run_dir(request).rglob("transcript.jsonl"))
    assert all(
        path.stat().st_size <= request.max_output_bytes for path in transcript_files
    )


def test_url_acquisition_sets_ytdlp_ignore_config_and_never_calls_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"URL acquisition fixture")
    log = EventLog(tmp_path)
    inspector = FakeSourceInspector(log, is_url=True)
    resolver = FakeCaptionResolver(log, None)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    request = _request(
        tmp_path,
        "https://example.test/video?id=fixture",
        tasks={"metadata", "frames"},
        request_id="url-run",
    )
    monkeypatch.delenv("YTDLP_IGNORE_CONFIG", raising=False)

    result = analyze_video(
        request, context=_context(log, inspector, resolver, media, media_backend)
    )

    assert _dump(result)["ok"] is True
    assert log.url_env_value == "1"
    assert log.cloud_calls == []
    assert (
        _stage(_read_json(_run_dir(request) / "manifest.json"), "acquire_media")[
            "status"
        ]
        == "completed"
    )


def test_transcribe_offsets_segments_and_words_by_start_seconds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "chunk-source.mp4"
    source.write_bytes(b"chunk source")
    log = EventLog(tmp_path)
    inspector = FakeSourceInspector(log, tracks=[])
    resolver = FakeCaptionResolver(log, None)
    media = FakeMediaAcquirer(log, source)
    media_backend = FakeMediaBackend(log, tmp_path / "media-work")
    media_backend.work_dir.mkdir()
    asr_backend = FakeAsrBackend(
        log,
        segments=[
            {
                "start_seconds": 2.5,
                "end_seconds": 8.0,
                "text": "spoken dialogue in chunk",
                "words": [
                    {"start_seconds": 2.5, "end_seconds": 4.0, "word": "spoken"},
                    {"start_seconds": 4.1, "end_seconds": 8.0, "word": "dialogue"},
                ],
            }
        ],
    )
    request = _request(
        tmp_path,
        source,
        tasks={"metadata", "transcript"},
        time_range=TimeRange(start_seconds=180.0, end_seconds=360.0),
        asr_enabled=True,
    )

    result = analyze_video(
        request,
        context=_context(
            log,
            inspector,
            resolver,
            media,
            media_backend,
            asr_backend=asr_backend,
        ),
    )

    assert result.status == "completed"
    transcript_files = list(_run_dir(request).rglob("transcript.jsonl"))
    assert len(transcript_files) == 1
    lines = [
        json.loads(line)
        for line in transcript_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    row = lines[0]
    assert row["start_seconds"] == 182.5
    assert row["end_seconds"] == 188.0
    assert row["words"][0]["start_seconds"] == 182.5
    assert row["words"][0]["end_seconds"] == 184.0
    assert row["words"][1]["start_seconds"] == 184.1
    assert row["words"][1]["end_seconds"] == 188.0


def test_present_but_broken_module_reports_unimportable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vidscope.core import _module_importable

    pkg = tmp_path / "broken_probe_mod"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "raise ImportError('simulated broken native dependency')\n",
        encoding="utf-8",
    )
    # Native shared-library load failures surface as OSError, not ImportError.
    os_pkg = tmp_path / "broken_native_mod"
    os_pkg.mkdir()
    (os_pkg / "__init__.py").write_text(
        "raise OSError('simulated missing shared library')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    # Present on disk, so spec discovery succeeds ...
    assert importlib.util.find_spec("broken_probe_mod") is not None
    assert importlib.util.find_spec("broken_native_mod") is not None
    _module_importable.cache_clear()
    try:
        # ... but the import itself fails, so the capability is unavailable.
        assert _module_importable("broken_probe_mod") is False
        assert _module_importable("broken_native_mod") is False
        assert _module_importable("json") is True
    finally:
        _module_importable.cache_clear()
