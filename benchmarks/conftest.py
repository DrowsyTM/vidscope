from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "tests" / "fixtures"


@pytest.fixture
def speech_sample_wav(fixtures_dir: Path) -> Path:
    return fixtures_dir / "speech_sample.wav"
