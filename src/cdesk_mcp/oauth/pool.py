"""Bounded per-replica LRU cache of live ``CdeskClient`` objects.

Live clients are NOT persisted — the credential material travels inside the
self-encoded access token, and each replica reconstructs a client on demand by
decrypting it. This pool caches the reconstructed clients keyed by access token,
bounds memory/connections via LRU eviction, and closes clients on eviction /
shutdown. It is a pure performance cache with no correctness role.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cdesk_mcp.cdesk_client import CdeskClient

_DEFAULT_MAXSIZE = 1024


async def _safe_close(client: CdeskClient) -> None:
    try:
        await client.close()
    except Exception:  # pragma: no cover - best effort
        pass


class ClientPool:
    """LRU map ``access_token -> CdeskClient``. Single-threaded asyncio use."""

    def __init__(self, maxsize: int = _DEFAULT_MAXSIZE) -> None:
        self._max = maxsize
        self._data: OrderedDict[str, CdeskClient] = OrderedDict()

    def get(self, key: str) -> CdeskClient | None:
        client = self._data.get(key)
        if client is not None:
            # Promote to most-recently-used. The request that just resolved this
            # client is now MRU, so eviction (which pops the LRU end) won't close
            # a client that's mid-request — except under a pathological burst of
            # >maxsize *distinct* tokens within one resolve→use window, which is
            # effectively unreachable on a single event loop. (A refcount would
            # make it airtight; not warranted at this scale.)
            self._data.move_to_end(key)
        return client

    async def put(self, key: str, client: CdeskClient) -> None:
        existing = self._data.get(key)
        if existing is not None and existing is not client:
            await _safe_close(existing)
        self._data[key] = client
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            # Evict the least-recently-used (see get() for why this is safe vs
            # in-flight clients) and free its connections.
            _evicted_key, evicted = self._data.popitem(last=False)
            await _safe_close(evicted)

    async def pop(self, key: str) -> None:
        client = self._data.pop(key, None)
        if client is not None:
            await _safe_close(client)

    async def aclose(self) -> None:
        for client in list(self._data.values()):
            await _safe_close(client)
        self._data.clear()
