"""Request-scoped CdeskClient proxy + per-server enum cache + the OAuth resolver.

These let all existing tools keep calling ``client.get(...)`` / the enum cache
unchanged while, in http mode, each call runs as the request's authenticated CDESK
user against the CDESK server that user chose at login.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cdesk_mcp.cdesk_client import CdeskClient
    from cdesk_mcp.enums import EnumCache, EnumEntry
    from cdesk_mcp.oauth.provider import CdeskOAuthProvider

# Raised from 32 when the enum-cache key went from per-SERVER to per-CALLER: the
# key space is now the number of active users, not the number of hosted CDESK
# servers, and evicting an active user just means re-fetching their enums. Each
# entry holds a few small enum lists, and there is one map per enum endpoint (4).
_DEFAULT_MAX_TENANTS = 128


class RequestScopedClient:
    """Drop-in for ``CdeskClient`` from the tools' / EnumCache's perspective.

    Exposes the same async ``get/post/put/delete/close`` surface, but resolves
    the *actual* client per call via the injected ``resolve`` callable. This is
    what lets all 37 existing tools keep calling ``client.get(...)`` unchanged
    while, in http mode, each call runs as the request's authenticated CDESK
    user.

    ``resolve`` is a zero-arg **async** callable returning a ``CdeskClient`` or
    raising a ``RuntimeError`` with an actionable message (e.g. not
    authenticated)."""

    def __init__(self, resolve: Any) -> None:
        self._resolve = resolve

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        client = await self._resolve()
        return await client.get(path, params=params)

    async def post(self, path: str, *, json: dict[str, Any] | None = None) -> Any:
        client = await self._resolve()
        return await client.post(path, json=json)

    async def put(self, path: str, *, json: dict[str, Any] | None = None) -> Any:
        client = await self._resolve()
        return await client.put(path, json=json)

    async def delete(self, path: str) -> Any:
        client = await self._resolve()
        return await client.delete(path)

    async def close(self) -> None:
        # Per-user clients are owned by the provider's pool (http) or by __main__
        # (stdio); the proxy itself owns nothing.
        return None


class RequestScopedEnumCache:
    """Drop-in for ``EnumCache`` that keeps a SEPARATE cache per CALLER.

    Enum ids (status/type/priority …) are tenant-specific, so a shared cache
    would serve one tenant's ids to another. The subtlety that matters: a CDESK
    *server* is not a tenant. One host holds many ENVIRONMENTS, each an
    ``admin_id`` (``User::getAdminId()`` = ``id_parent``, else own id), and the
    backend scopes enums by exactly that
    (``EnumEnum::where('admin_id', App::$user->getAdminId())``). Keying this map
    on the base URL alone therefore shared one cache across every environment on
    a host: whoever loaded first won, and the next caller got that environment's
    status names and ids — a cross-environment read, and a name→id resolution
    that yields a FOREIGN id for the write that follows.

    So the key is per-caller (``resolve_cache_key``), currently
    ``base_url + login``. That can over-partition — two users in the same
    environment get their own cache — but it can never under-partition, which is
    the direction that leaks. Getting the exact ``admin_id`` would need a lookup
    per credential; identity is free on the token.

    Every ``EnumCache`` method is only ever called inside an authenticated request
    handler, so ``resolve_cache_key`` (which reads the current access token) is
    always valid here — including for the sync methods.

    ``make_cache`` builds a cache bound to the shared request-scoped client, which
    already routes to the current request's server; only the *memoization* is
    split per caller, which is the whole point."""

    def __init__(
        self,
        *,
        resolve_cache_key: Callable[[], str],
        make_cache: Callable[[], EnumCache],
        max_tenants: int = _DEFAULT_MAX_TENANTS,
    ) -> None:
        self._resolve_cache_key = resolve_cache_key
        self._make_cache = make_cache
        self._max = max_tenants
        self._caches: OrderedDict[str, EnumCache] = OrderedDict()

    def _current(self) -> EnumCache:
        key = self._resolve_cache_key()
        cache = self._caches.get(key)
        if cache is not None:
            self._caches.move_to_end(key)
            return cache
        cache = self._make_cache()
        self._caches[key] = cache
        self._caches.move_to_end(key)
        while len(self._caches) > self._max:
            self._caches.popitem(last=False)  # evict least-recently-used caller
        return cache

    # --- async EnumCache surface ---
    async def load(self) -> None:
        await self._current().load()

    async def refresh(self) -> None:
        await self._current().refresh()

    async def resolve(
        self, bucket: str, name: str, *, parent_id: int | None = None,
        allow_refresh: bool = True,
    ) -> int | None:
        return await self._current().resolve(
            bucket, name, parent_id=parent_id, allow_refresh=allow_refresh
        )

    async def resolve_entry(
        self, bucket: str, name: str, *, parent_id: int | None = None,
        allow_refresh: bool = True,
    ) -> EnumEntry | None:
        return await self._current().resolve_entry(
            bucket, name, parent_id=parent_id, allow_refresh=allow_refresh
        )

    # --- sync EnumCache surface ---
    @property
    def settings(self) -> dict[str, Any]:
        # Per-server, like the enum buckets themselves: one tenant may have a
        # module switched off while another has it on.
        return self._current().settings

    def list_names(self, bucket: str) -> list[str]:
        return self._current().list_names(bucket)

    def find_candidates(
        self, bucket: str, name: str, max_count: int = 5, min_ratio: float | None = None,
    ) -> list[str]:
        cache = self._current()
        if min_ratio is None:  # preserve EnumCache's own default
            return cache.find_candidates(bucket, name, max_count=max_count)
        return cache.find_candidates(bucket, name, max_count=max_count, min_ratio=min_ratio)

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return self._current().snapshot()

    def id_name_map(self, bucket: str) -> dict[Any, str]:
        return self._current().id_name_map(bucket)

    def action_code_name_map(self, bucket: str) -> dict[Any, str]:
        return self._current().action_code_name_map(bucket)

    # --- properties ---
    @property
    def endpoint(self) -> str:
        return self._current().endpoint

    @property
    def bucket_names(self) -> list[str]:
        return self._current().bucket_names

    @property
    def loaded(self) -> bool:
        return self._current().loaded

    @property
    def is_stale(self) -> bool:
        return self._current().is_stale


def make_oauth_resolver(provider: CdeskOAuthProvider) -> Any:
    """Build the async resolve() callable for http mode: read the current
    request's access token and return its CdeskClient — pooled, or reconstructed
    from the credential carried inside the token on a miss — or raise a clear error."""
    from mcp.server.auth.middleware.auth_context import get_access_token

    async def resolve() -> CdeskClient:
        tok = get_access_token()
        if tok is None:
            raise RuntimeError(
                "Not authenticated to CDESK. Reconnect the CDESK connector and "
                "sign in with your CDESK credentials."
            )
        client = await provider.get_or_reconstruct_client(tok)
        if client is None:
            raise RuntimeError(
                "Your CDESK session is no longer active. Reconnect the CDESK "
                "connector to sign in again."
            )
        return client

    return resolve
