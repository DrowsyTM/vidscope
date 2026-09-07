from __future__ import annotations

from pathlib import Path

from vidscope.artifacts import ArtifactStore


def test_vidscope_public_package_is_importable() -> None:
    import vidscope

    assert vidscope.__version__ == "0.1.0"


def test_artifact_store_uses_vidscope_uris(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, request_id="run-identity")
    store.create()
    source = tmp_path / "identity.txt"
    source.write_bytes(b"identity")
    reference = store.publish_file(source, name="payload")

    assert store.manifest_uri == "vidscope://runs/run-identity/manifest"
    assert reference.uri.startswith("vidscope://runs/run-identity/artifacts/")
    assert ArtifactStore.read_uri(tmp_path, reference.uri) == b"identity"
