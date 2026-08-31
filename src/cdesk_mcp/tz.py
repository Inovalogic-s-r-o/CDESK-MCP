"""Tenant-local timezone handling for naive datetimes.

WHY THIS EXISTS (bug found live 2026-06-05): CDESK's v3 API interprets a
NAIVE datetime ("2026-06-05T08:00:00") as UTC — a task created with naive
8:00–12:00 was stored as 08:00+00:00 and the tenant UI (Europe/Bratislava,
CEST) displayed it as 10:00–14:00, two hours off what the user asked for.
An EXPLICIT offset is honored correctly (08:00+02:00 → stored 06:00+00:00 →
displayed 08:00). Users and LLMs naturally speak in wall-clock tenant time,
so the MCP attaches the tenant offset to naive datetimes before sending.

The tenant timezone comes from the CDESK_TIMEZONE env var (IANA name, e.g.
"Europe/Bratislava" — the default, matching cmpp.seal.sk). The `tzdata`
package provides the IANA database on Windows.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, tzinfo
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_TENANT_TZ = "Europe/Bratislava"


@lru_cache(maxsize=1)
def tenant_timezone() -> tzinfo:
    """The tenant's wall-clock timezone (CDESK_TIMEZONE env, IANA name).

    Falls back to DEFAULT_TENANT_TZ on an unknown name, and to UTC only if
    even the default can't load (missing tzdata) — logged, never raised, so
    a misconfigured timezone degrades to the old naive-as-UTC behavior
    instead of breaking every datetime-carrying tool. Cached for the process
    lifetime; tests clear via tenant_timezone.cache_clear()."""
    name = os.getenv("CDESK_TIMEZONE", "").strip() or DEFAULT_TENANT_TZ
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        logger.warning(
            "CDESK_TIMEZONE=%r is not a known IANA timezone; "
            "falling back to %r", name, DEFAULT_TENANT_TZ,
        )
    try:
        return ZoneInfo(DEFAULT_TENANT_TZ)
    except ZoneInfoNotFoundError:  # no tzdata at all
        logger.warning(
            "IANA timezone database unavailable (install `tzdata`); naive "
            "datetimes will be treated as UTC, matching how CDESK reads them."
        )
        # NOT ZoneInfo("UTC") — zoneinfo has no built-in zones, so on a host
        # without tzdata that lookup raises too and the safety net would
        # throw. datetime.timezone.utc needs no database.
        return timezone.utc


def localize_naive_datetime(field_name: str, value: str | None) -> str | None:
    """Validate an ISO-8601 date/datetime; attach the tenant offset if naive.

    - None/empty → None (field omitted).
    - invalid → ValueError naming the field (fail fast, no opaque 400).
    - datetime WITH an offset → unchanged (the caller said what they meant).
    - NAIVE datetime (has a time part, no offset) → tenant-local: the exact
      same wall-clock time with the tenant offset for that date (DST-aware),
      e.g. '2026-06-05T08:00:00' → '2026-06-05T08:00:00+02:00'.
    - bare DATE (no time part) → unchanged: dates are calendar-level and
      CDESK's date-only parsing is fine; forcing midnight+offset would turn
      "the 5th" into "23:00 on the 4th" UTC-side for fullday semantics.
    """
    if not value:
        return None
    raw = value.strip()
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as e:
        raise ValueError(
            f"{field_name}={value!r} is not a valid ISO-8601 date or datetime "
            f"(e.g. '2026-05-28' or '2026-05-28T10:00:00+02:00'): {e}"
        ) from e
    # Bare date: no time part at all (no 'T'/space/':' — structural, matching
    # filters._w3c_datetime so '20260604' basic form counts as a date too).
    if "T" not in raw and " " not in raw and ":" not in raw:
        return raw
    if dt.tzinfo is not None:
        return raw
    return dt.replace(tzinfo=tenant_timezone()).isoformat()
