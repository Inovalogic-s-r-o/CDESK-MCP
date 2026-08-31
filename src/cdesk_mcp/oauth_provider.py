"""Backwards-compatible re-export shim.

The OAuth authorization-server implementation moved into the ``cdesk_mcp.oauth``
package (provider / models / proxy / login). This module re-exports the public
names so existing imports keep working. Prefer importing from ``cdesk_mcp.oauth``
directly.
"""

from cdesk_mcp.oauth import (
    CdeskAccessToken,
    CdeskAuthCode,
    CdeskOAuthProvider,
    RequestScopedClient,
    make_oauth_resolver,
)

__all__ = [
    "CdeskAccessToken",
    "CdeskAuthCode",
    "CdeskOAuthProvider",
    "RequestScopedClient",
    "make_oauth_resolver",
]
