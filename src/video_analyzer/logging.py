"""Structured logging configuration for video_analyzer.

Adheres to PEP 282 by attaching a NullHandler by default so importing
video_analyzer as a library never hijacks application logging.
Strictly isolates diagnostics and progress to sys.stderr to preserve
sys.stdout as a pure JSON / JSON-RPC communication transport for CLI
piping and FastMCP stdio.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

LOGGER_NAME: Final = "video_analyzer"

logger = logging.getLogger(LOGGER_NAME)
logger.addHandler(logging.NullHandler())


def configure_logging(
    level: int | str | None = None,
    *,
    force_handler: bool = False,
) -> logging.Logger:
    """Configure the package logger to output formatted records strictly to sys.stderr.

    Args:
        level: Optional log level name or integer. Defaults to the
            VIDEO_ANALYZER_LOG_LEVEL environment variable or INFO.
        force_handler: If True, adds a stderr StreamHandler even if handlers exist.
    """
    resolved_level: int
    if level is not None:
        if isinstance(level, str):
            resolved_level = getattr(logging, level.upper(), logging.INFO)
        else:
            resolved_level = level
    else:
        env_level = os.environ.get("VIDEO_ANALYZER_LOG_LEVEL", "INFO").strip().upper()
        resolved_level = getattr(logging, env_level, logging.INFO)

    logger.setLevel(resolved_level)

    # Check if a non-NullHandler is already attached
    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.NullHandler)
        for handler in logger.handlers
    )

    if not has_stream_handler or force_handler:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


__all__ = ["LOGGER_NAME", "configure_logging", "logger"]
