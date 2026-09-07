from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from vidscope.artifacts import ArtifactStore, ArtifactStoreFailure


def _code(exc: BaseException) -> str:
    value: Any = getattr(exc, "error", exc)
    return str(getattr(value, "code", getattr(exc, "code", "")))


def test_store_publishes_hashed_artifact_and_atomic_manifest(tmp_path: Path) -> None:
    output = tmp_path / "output"
    source = tmp_path / "source.txt"
    source.write_text("one\ntwo\n", encoding="utf-8")
    store = ArtifactStore(output, "run-1", max_output_bytes=1024).create()

    ref = store.publish_file(
        source, name="records.jsonl", media_type="application/x-ndjson"
    )

    payload = (store.run_directory / "records.jsonl").read_bytes()
    assert ref.byte_size == len(payload)
    assert ref.sha256 == hashlib.sha256(payload).hexdigest()
    assert ref.uri == "vidscope://runs/run-1/artifacts/" + ref.artifact_id
    assert "one" not in json.dumps(ref.model_dump(mode="json"))
    manifest = json.loads((store.run_directory / "manifest.json").read_text())
    assert (
        manifest["artifacts"][ref.artifact_id]["metadata"]["relative_path"]
        == "records.jsonl"
    )
    assert not list(store.run_directory.glob("manifest.json.*"))


def test_store_rejects_conflicts_empty_files_and_output_budget(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    conflict = output / "run-1"
    conflict.mkdir()
    (conflict / "existing").write_text("occupied")
    with pytest.raises(ArtifactStoreFailure) as conflict_error:
        ArtifactStore(output, "run-1").create()
    assert _code(conflict_error.value) == "OUTPUT_NOT_ALLOWED"

    store = ArtifactStore(output, "run-2", max_output_bytes=3).create()
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    with pytest.raises(ArtifactStoreFailure) as empty_error:
        store.publish_file(empty)
    assert _code(empty_error.value) == "INTERNAL_STAGE_FAILED"
    large = tmp_path / "large"
    large.write_bytes(b"1234")
    with pytest.raises(ArtifactStoreFailure) as budget_error:
        store.publish_file(large)
    assert _code(budget_error.value) == "OUTPUT_LIMIT_EXCEEDED"


def test_store_pages_jsonl_and_rejects_invalid_ranges_or_ids(tmp_path: Path) -> None:
    output = tmp_path / "output"
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"i": i}) + "\n" for i in range(5)), encoding="utf-8"
    )
    store = ArtifactStore(output, "run-3", max_output_bytes=4096).create()
    ref = store.publish_file(
        source, name="records.jsonl", media_type="application/x-ndjson"
    )

    assert json.loads(
        store.read_resource(ref.uri, offset=2, limit=2).splitlines()[0]
    ) == {"i": 2}
    assert len(store.read_resource(ref.uri, page=1, limit=2).splitlines()) == 2
    with pytest.raises(ArtifactStoreFailure) as invalid_id:
        store.read_resource(ref.uri.replace(ref.artifact_id, "missing"))
    assert _code(invalid_id.value) == "ARTIFACT_NOT_FOUND"
    with pytest.raises(ArtifactStoreFailure) as invalid_range:
        store.read_resource(ref.uri, limit=201)
    assert _code(invalid_range.value) == "ARTIFACT_RANGE_INVALID"
