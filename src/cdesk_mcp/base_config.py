"""The calling user's CDESK base configuration (`GET /api/core/base-config`).

The CDESK web app loads this at startup for every logged-in user: module
settings (work orders, tasks, fills, customers, …), status enums with their
action codes and every language variant, the user's own record and rights. The
v3 API has no equivalent for several of these, and the per-module v1 settings
endpoints (`workorder/settings`, `task/settings`, …) are admin-only
(`Acl::canAccess('globalconfig')`), so a non-admin user could not read them.
This endpoint has no such check (Core/Module.php baseConfig) and is built per
user.

It is large (~460 KB, ~0.7 s on cmpp 2026-10-12), so it is fetched only where a
tool needs it, at most once per tool call, and never cached across calls
(tenant settings change at any time — same rule as the enums). A failed read
yields None and the caller degrades (no form rules, CDESK decides).

NB: the payload also carries the session's apitoken (`user.token`) — never
pass the `user` block, or the whole payload, through to a tool result.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Protocol

log = logging.getLogger(__name__)

PATH = "core/base-config"


class _ClientProto(Protocol):
    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any: ...


# This call's payload per client, like enums._CALL_SCOPE: each MCP tool call
# runs in its own asyncio task, so a ContextVar set here is seen only by the
# rest of that call.
_CALL_SCOPE: contextvars.ContextVar[dict[int, Any] | None] = contextvars.ContextVar(
    "cdesk_base_config_call_scope", default=None,
)
_UNREADABLE = object()


async def load(client: _ClientProto) -> dict[str, Any] | None:
    """The `data` block of core/base-config for the calling user, read at most
    once per tool call. None when it cannot be read."""
    scope = _CALL_SCOPE.get()
    if scope is None:
        scope = {}
        _CALL_SCOPE.set(scope)
    key = id(client)
    if key not in scope:
        try:
            response = await client.get(PATH)
            data = response.get("data") if isinstance(response, dict) else None
            scope[key] = data if isinstance(data, dict) else _UNREADABLE
        except Exception as e:  # noqa: BLE001 — callers degrade, never fail on it
            log.warning("core/base-config unreadable: %s: %s", type(e).__name__, e)
            scope[key] = _UNREADABLE
    data = scope[key]
    return None if data is _UNREADABLE else data


def section(data: Any, *path: str) -> Any:
    """data[path[0]][path[1]]…, or None where a level is missing."""
    node = data
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node
