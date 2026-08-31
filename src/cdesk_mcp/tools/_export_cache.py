"""Per-tenant, TTL-bounded store for a bulk CMDB CI-export snapshot.

The CNB export endpoint (``GET /cnb/cmdb/export``) returns the WHOLE CI dataset
with no server-side paging or field selection, so ``export_cmdb_ci`` fetches it
once, parks the row list here under a generated ``export_id``, and
``get_cmdb_export_page`` serves slices from it. That keeps the bulk data out of
the LLM context window and avoids re-streaming the full dump on every page.

Isolation: snapshots are bucketed per CALLER (``resolve_cache_key`` — the same
mechanism the enum caches use in ``oauth/proxy.py:RequestScopedEnumCache``).
The key is base_url + login, NOT the base URL alone: one CDESK host holds many
environments (``admin_id``), so a host-only key put every environment in one
bucket. That was never a disclosure here — ``get`` also requires the exact
16-hex ``export_id``, which only the creator receives — but environment B's
export OVERWROTE environment A's snapshot, so A's next page fetch found
nothing. An ``export_id`` minted by one caller is simply not found in
another's bucket. Only the most-recent snapshot per caller is kept
(a fresh export evicts the old one), an LRU cap bounds the number of tenants,
and entries older than the TTL are treated as absent — all to keep what is a
large in-memory object from accumulating.
"""

from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

_DEFAULT_MAX_TENANTS = 32
_DEFAULT_TTL_SECONDS = 300.0  # 5 minutes


class ExportSnapshotStore:
    def __init__(
        self,
        *,
        resolve_cache_key: Callable[[], str],
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        max_tenants: int = _DEFAULT_MAX_TENANTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._resolve_cache_key = resolve_cache_key
        self._ttl = ttl_seconds
        self._max = max_tenants
        self._clock = clock
        # cache key -> (export_id, created_at, rows). One snapshot per caller.
        self._snaps: OrderedDict[str, tuple[str, float, list[Any]]] = OrderedDict()

    def put(self, rows: list[Any]) -> str:
        """Store ``rows`` for the current caller and return a fresh export_id.

        Replaces any previous snapshot for the same caller (we only ever need
        the latest) and evicts the least-recently-used caller past the cap."""
        key = self._resolve_cache_key()
        export_id = secrets.token_hex(8)
        self._snaps[key] = (export_id, self._clock(), rows)
        self._snaps.move_to_end(key)
        while len(self._snaps) > self._max:
            self._snaps.popitem(last=False)  # evict least-recently-used caller
        return export_id

    def get(self, export_id: str) -> list[Any] | None:
        """Return the rows for ``export_id`` if it's the current caller's live
        snapshot, else None (unknown/stale id, expired TTL, or a different caller).
        An expired entry is dropped on access so its memory is reclaimed."""
        key = self._resolve_cache_key()
        entry = self._snaps.get(key)
        if entry is None:
            return None
        stored_id, created_at, rows = entry
        if stored_id != export_id:
            return None
        if self._clock() - created_at > self._ttl:
            del self._snaps[key]
            return None
        self._snaps.move_to_end(key)
        return rows
