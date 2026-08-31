"""Remote (http) OAuth authorization-server package for cdesk-mcp.

Public surface (import from here, not the submodules):
- ``CdeskOAuthProvider`` — the OAuth 2.1 authorization server (oauth/provider.py)
- ``CdeskAuthCode`` / ``CdeskAccessToken`` — grant models (oauth/models.py)
- ``RequestScopedClient`` / ``RequestScopedEnumCache`` / ``make_oauth_resolver`` —
  per-request client + per-server enum cache (oauth/proxy.py)
- ``register_login_route`` — the /login consent page (oauth/login.py)
- ``register_subpath_well_known_routes`` — de-prefixed OAuth discovery routes for
  subpath/reverse-proxy hosting (oauth/well_known.py)
"""

from cdesk_mcp.oauth.azure_login import register_azure_login_routes
from cdesk_mcp.oauth.login import register_login_route
from cdesk_mcp.oauth.models import (
    CdeskAccessToken,
    CdeskAuthCode,
    CdeskCredential,
    CdeskRefreshToken,
)
from cdesk_mcp.oauth.provider import CdeskOAuthProvider
from cdesk_mcp.oauth.proxy import (
    RequestScopedClient,
    RequestScopedEnumCache,
    make_oauth_resolver,
)
from cdesk_mcp.oauth.well_known import register_subpath_well_known_routes

__all__ = [
    "CdeskAccessToken",
    "CdeskAuthCode",
    "CdeskCredential",
    "CdeskOAuthProvider",
    "CdeskRefreshToken",
    "RequestScopedClient",
    "RequestScopedEnumCache",
    "make_oauth_resolver",
    "register_azure_login_routes",
    "register_login_route",
    "register_subpath_well_known_routes",
]
