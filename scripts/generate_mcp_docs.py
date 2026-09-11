#!/usr/bin/env python3
"""Auto-generates docs/MCP_REFERENCE.md from FastMCP tool definitions and contracts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vidscope.docs import run_docs_generation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify Vidscope MCP tool documentation."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("docs/MCP_REFERENCE.md"),
        help="Target output markdown path (default: docs/MCP_REFERENCE.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify if target documentation is up to date without overwriting.",
    )
    args = parser.parse_args()
    return run_docs_generation(output=args.output, check=args.check)


if __name__ == "__main__":
    sys.exit(main())
