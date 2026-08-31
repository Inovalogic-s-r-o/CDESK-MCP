"""De-prefixed OAuth discovery routes for subpath (reverse-proxy) hosting.

When the server is hosted under a path prefix — i.e. ``CDESK_PUBLIC_URL`` carries
a path such as ``https://host/mcp-server`` — OAuth discovery clients (claude.ai)
still fetch the RFC 9728 protected-resource metadata at the domain *root*
(``https://host/.well-known/oauth-protected-resource/...``). A root reverse proxy
forwards those root ``/.well-known/`` requests to us with the ``/mcp-server``
prefix stripped, so the request arrives as
``/.well-known/oauth-protected-resource/mcp``.

The stock MCP SDK, however, derives that route's path from the resource URL's
path and so registers it at ``/.well-known/oauth-protected-resource/mcp-server/mcp``
— which the proxy-stripped request never matches (→ 404). This module adds the
de-prefixed companion route. Its body still advertises the fully-prefixed
``resource`` + ``authorization_servers`` (so the token audience and OAuth
endpoints stay correct); only the route *path* is stripped to match the proxy.

No-op when ``CDESK_PUBLIC_URL`` has no path (root hosting, e.g. the dev-tunnel
``/host-cdesk`` flow) — the SDK's own root-level routes already serve everything.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from mcp.server.auth.handlers.metadata import ProtectedResourceMetadataHandler
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.shared.auth import ProtectedResourceMetadata
from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger(__name__)

_PRM_WELL_KNOWN = "/.well-known/oauth-protected-resource"


def subpath_protected_resource_route(auth_settings: AuthSettings) -> str | None:
    """The de-prefixed protected-resource metadata path to serve, or None when the
    public URL has no path prefix (root hosting) or has an unexpected shape.

    Example: issuer ``https://h/mcp-server`` + resource ``https://h/mcp-server/mcp``
    → ``/.well-known/oauth-protected-resource/mcp``."""
    if auth_settings.resource_server_url is None:
        return None
    issuer = str(auth_settings.issuer_url).rstrip("/")
    prefix = urlsplit(issuer).path.rstrip("/")
    if not prefix:
        return None  # root hosting — the SDK already serves at the root
    resource_path = urlsplit(str(auth_settings.resource_server_url)).path
    if not resource_path.startswith(prefix):
        return None  # unexpected shape; leave the SDK's routes untouched
    return _PRM_WELL_KNOWN + resource_path[len(prefix):]


def register_subpath_well_known_routes(mcp: FastMCP, auth_settings: AuthSettings) -> None:
    """Mount the de-prefixed protected-resource metadata route when hosted under a
    path prefix. Additive — the SDK's prefixed route still exists and is harmless."""
    stripped = subpath_protected_resource_route(auth_settings)
    if stripped is None:
        return

    metadata = ProtectedResourceMetadata(
        resource=AnyHttpUrl(str(auth_settings.resource_server_url)),
        authorization_servers=[AnyHttpUrl(str(auth_settings.issuer_url).rstrip("/"))],
        scopes_supported=auth_settings.required_scopes,
    )
    handler = ProtectedResourceMetadataHandler(metadata)

    @mcp.custom_route(stripped, methods=["GET", "OPTIONS"])  # type: ignore[untyped-decorator]
    async def protected_resource_metadata(request: Request) -> Response:  # pragma: no cover - exercised live
        return await handler.handle(request)

    log.info(
        "subpath hosting: also serving protected-resource metadata at %s "
        "(resource=%s)",
        stripped, auth_settings.resource_server_url,
    )
