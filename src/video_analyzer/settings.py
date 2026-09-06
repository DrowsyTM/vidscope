"""Explicit, local-only configuration for the video analyzer.

This module deliberately does not inspect the host for executables, create cache
or output directories, or import optional media/model dependencies.  Adapters
can use the configured binary paths when present and otherwise perform their
own PATH lookup.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ALLOWED_INPUT_ROOT_ENV: Final = "VIDEO_ANALYZER_ALLOWED_INPUT_ROOT"
TESSERACT_BIN_ENV: Final = "VIDEO_ANALYZER_TESSERACT_BIN"
TESSDATA_PREFIX_ENV: Final = "VIDEO_ANALYZER_TESSDATA_PREFIX"
FFMPEG_BIN_ENV: Final = "VIDEO_ANALYZER_FFMPEG_BIN"
FFPROBE_BIN_ENV: Final = "VIDEO_ANALYZER_FFPROBE_BIN"
MODEL_CACHE_ENV: Final = "VIDEO_ANALYZER_MODEL_CACHE"
ALLOWED_OUTPUT_ROOT_ENV: Final = "VIDEO_ANALYZER_ALLOWED_OUTPUT_ROOT"
LEGACY_TESSDATA_PREFIX_ENV: Final = "TESSDATA_PREFIX"


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved local configuration.

    ``None`` for a binary means that the corresponding adapter may use its
    normal PATH lookup.  ``cloud_allowed`` is intentionally not configurable:
    this package is local-only.
    """

    allowed_input_root: Path | None = None
    tesseract_bin: Path | None = None
    tessdata_prefix: Path | None = None
    ffmpeg_bin: Path | None = None
    ffprobe_bin: Path | None = None
    model_cache: Path | None = None
    allowed_output_root: Path | None = None
    cloud_allowed: bool = False

    def __post_init__(self) -> None:
        """Keep the local-only policy invariant for direct construction."""

        object.__setattr__(self, "cloud_allowed", False)


def _normalize_path(
    raw_value: str | None,
    *,
    variable_name: str,
    require_absolute: bool = False,
) -> Path | None:
    """Normalize one configured path without requiring it to exist.

    Empty environment values are treated as unset.  User-home expansion is
    useful for explicit local configuration.  Existing absolute paths are
    resolved with ``strict=False`` so symlinked policy roots compare
    consistently, while relative binary names (for example ``ffmpeg``) stay
    relative for the adapter's PATH lookup.
    """

    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise TypeError(f"{variable_name} must be a string")

    value = raw_value.strip()
    if not value:
        return None

    try:
        path = Path(value).expanduser()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{variable_name} is not a valid path") from exc

    if require_absolute and not path.is_absolute():
        raise ValueError(f"{variable_name} must be an absolute path")

    # Keep bare executable names usable by subprocess/PATH lookup.  For
    # absolute paths, resolve lexical ``..`` and symlink components without
    # requiring the path (or its parent) to exist.
    if path.is_absolute():
        try:
            path = path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"{variable_name} is not a resolvable path") from exc
    return path


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Load deterministic local settings from ``environ`` or ``os.environ``.

    Only the explicit ``VIDEO_ANALYZER_*`` variables are consumed.  The bare
    ``TESSDATA_PREFIX`` variable is accepted solely as a compatibility fallback
    when ``VIDEO_ANALYZER_TESSDATA_PREFIX`` is absent; an explicitly empty
    prefixed value therefore still takes precedence and disables the fallback.
    """

    values: Mapping[str, str] = os.environ if environ is None else environ

    if TESSDATA_PREFIX_ENV in values:
        tessdata_prefix_value = values.get(TESSDATA_PREFIX_ENV)
    else:
        tessdata_prefix_value = values.get(LEGACY_TESSDATA_PREFIX_ENV)

    return Settings(
        allowed_input_root=_normalize_path(
            values.get(ALLOWED_INPUT_ROOT_ENV),
            variable_name=ALLOWED_INPUT_ROOT_ENV,
            require_absolute=True,
        ),
        tesseract_bin=_normalize_path(
            values.get(TESSERACT_BIN_ENV), variable_name=TESSERACT_BIN_ENV
        ),
        tessdata_prefix=_normalize_path(
            tessdata_prefix_value, variable_name=TESSDATA_PREFIX_ENV
        ),
        ffmpeg_bin=_normalize_path(
            values.get(FFMPEG_BIN_ENV), variable_name=FFMPEG_BIN_ENV
        ),
        ffprobe_bin=_normalize_path(
            values.get(FFPROBE_BIN_ENV), variable_name=FFPROBE_BIN_ENV
        ),
        model_cache=_normalize_path(
            values.get(MODEL_CACHE_ENV), variable_name=MODEL_CACHE_ENV
        ),
        allowed_output_root=_normalize_path(
            values.get(ALLOWED_OUTPUT_ROOT_ENV),
            variable_name=ALLOWED_OUTPUT_ROOT_ENV,
            require_absolute=True,
        ),
    )


def get_settings() -> Settings:
    """Return settings for the current process environment.

    Settings are loaded on demand rather than at import time, so importing this
    module has no environment, filesystem, or executable-discovery side effect.
    """

    return load_settings()


__all__ = [
    "ALLOWED_INPUT_ROOT_ENV",
    "ALLOWED_OUTPUT_ROOT_ENV",
    "FFMPEG_BIN_ENV",
    "FFPROBE_BIN_ENV",
    "LEGACY_TESSDATA_PREFIX_ENV",
    "MODEL_CACHE_ENV",
    "TESSDATA_PREFIX_ENV",
    "TESSERACT_BIN_ENV",
    "Settings",
    "get_settings",
    "load_settings",
]
