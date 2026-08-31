"""Logging configuration. Strictly stderr-only — stdout is reserved for MCP JSON-RPC."""

from __future__ import annotations

import logging
import sys

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def setup_logging(level: str = "INFO") -> None:
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    normalized = level.upper()
    if normalized in VALID_LOG_LEVELS:
        effective = normalized
        warn_message: str | None = None
    else:
        effective = "INFO"
        warn_message = (
            f"CDESK_LOG_LEVEL={level!r} is not one of {VALID_LOG_LEVELS}; defaulting to INFO."
        )

    logging.basicConfig(
        level=getattr(logging, effective),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if warn_message:
        logging.getLogger(__name__).warning(warn_message)
