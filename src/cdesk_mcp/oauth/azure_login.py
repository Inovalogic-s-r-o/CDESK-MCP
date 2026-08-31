"""Office365 / Microsoft Entra "Sign in with Microsoft" routes (http mode).

Delegates the Microsoft round-trip entirely to CDESK's Azure SSO. This MCP server
runs no Microsoft OAuth and holds no Microsoft secrets — and (per the CDESK MCP
integration manual) needs no client_id/client_secret. One connector works across
ANY CDESK server the user selects:

  GET /login/azure/start
    → look up the chosen server's Azure connector id via the PUBLIC
      ``/api/auth/connector`` list (the manual's Step 2), then 302 the browser to
      ``<server>/api/auth/azure/redirect?azureId&cdesk_redirect_uri&state``.

  GET /login/azure/callback   (CDESK redirects the browser back here)
    → on success CDESK appends ``?token=<APITOKEN>&refresh_token=<REFRESH>&state=``
      (on failure ``?error=access_denied&state=``). We hand the tokens straight to
      the provider's existing ``complete_login`` (identical to the password path)
      and 302 back to the OAuth client (Claude). No code-exchange step — CDESK
      delivers the apitoken directly.

The opaque OAuth ``session`` rides out and back as ``state`` (CDESK round-trips it)
and doubles as the CSRF binding. The chosen CDESK server is carried in our own
``cdesk_redirect_uri`` query (``base_url``) so the callback knows which server the
apitoken belongs to.

SECURITY: the apitoken + refresh token arrive in the browser redirect URL. The
callback runs over HTTPS (the public URL) and must NEVER log the raw callback URL
or the tokens.
"""

from __future__ import annotations

import html
import logging
from urllib.parse import urlencode

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from cdesk_mcp.oauth._connector import ProbeFn, probe_connector
from cdesk_mcp.oauth._web import _client_ip, _RateLimiter, _secure_html
from cdesk_mcp.oauth.provider import CdeskOAuthProvider

log = logging.getLogger(__name__)

_AZURE_MAX_ATTEMPTS = 10
_AZURE_WINDOW_SECONDS = 60.0

_REDIRECT_PATH = "api/auth/azure/redirect"


def _error_page(
    session: str, message: str, status_code: int = 502, *, login_url: str = "/login"
) -> Response:
    """A friendly error page with a link back to the login page to retry (with
    the password form or another Microsoft attempt).

    ``login_url`` is built from CDESK_PUBLIC_URL by the caller rather than
    hardcoded to ``/login``: under path-prefix hosting (see the subpath
    well-known routes in server.py) a root-absolute link lands outside the app."""
    retry = (
        f'<p><a href="{html.escape(login_url, quote=True)}'
        f'?session={html.escape(session, quote=True)}">Back to sign-in</a></p>'
        if session else ""
    )
    return _secure_html(
        f"<h1>Microsoft sign-in</h1><p>{html.escape(message)}</p>{retry}",
        status_code=status_code,
    )


def register_azure_login_routes(
    mcp: FastMCP,
    provider: CdeskOAuthProvider,
    *,
    public_url: str,
    timeout_seconds: float = 30.0,
    trust_forwarded: bool = False,
    probe: ProbeFn = probe_connector,
) -> None:
    """Mount GET /login/azure/start and /login/azure/callback (unauthenticated by
    design — the user proves identity at Microsoft; CDESK returns the apitoken
    directly to the callback). The Azure connector id is discovered per CDESK
    server from the public /api/auth/connector list, so one mount serves every
    server the user may select."""
    limiter = _RateLimiter(_AZURE_MAX_ATTEMPTS, _AZURE_WINDOW_SECONDS)
    public = public_url.rstrip("/")
    login_url = f"{public}/login"

    def _err(session: str, message: str, status_code: int = 502) -> Response:
        return _error_page(session, message, status_code, login_url=login_url)

    @mcp.custom_route("/login/azure/start", methods=["GET"])  # type: ignore[untyped-decorator]
    async def azure_start(request: Request) -> Response:  # pragma: no cover - exercised live
        if not limiter.allow(_client_ip(request, trust_forwarded)):
            return _secure_html(
                "<h1>Too many attempts</h1><p>Please wait a minute and try again.</p>",
                status_code=429,
            )
        session = request.query_params.get("session", "")
        if not session or not await provider.peek_session(session):
            return _err(
                "", "Invalid or expired sign-in link. Please restart the connection.", 400
            )
        # The user supplies their CDESK server URL (no silent default). This path
        # skips the browser's form validation entirely — the button navigates via
        # JS — so check_base_url's specific reason is the only feedback there is.
        base_url, url_problem = provider.check_base_url(request.query_params.get("server", ""))
        if base_url is None:
            problem = url_problem or "That CDESK server address cannot be used."
            return _err(
                session,
                f"{problem} Then click Sign in with Microsoft again.",
                400,
            )

        # Step 2 — discover the chosen server's Azure connector id. Re-discovered
        # here, server-side, every time: the login page's probe may be stale or
        # bypassed, and an id round-tripped through the browser would be a
        # user-injected parameter in the third-party redirect built below.
        result = await probe(base_url, timeout_seconds=timeout_seconds)
        azure_id = result.azure_id
        if azure_id is None:
            return _err(
                session,
                "This CDESK server doesn't offer Microsoft sign-in. "
                "Please use your CDESK login instead.",
                400,
            )

        # Clean callback + `state` (carries the session, round-tripped). The chosen
        # server rides in `base_url` so the callback binds the token to it; CDESK
        # appends ?token=&refresh_token=&state= to whatever query we send.
        redirect_uri = f"{public}/login/azure/callback?" + urlencode({"base_url": base_url})
        query = urlencode(
            {"azureId": azure_id, "cdesk_redirect_uri": redirect_uri, "state": session}
        )
        # `&all-routes` (valueless flag) — backend workaround: getRedirect proxies
        # to azureverify via an internal sub-request that otherwise can't resolve
        # the route (the Slim route-loading optimization), so it bounced to
        # /login?non-sso. This flag forces full route loading. Appended raw (no
        # `=`) to match the form the backend specified.
        cdesk_redirect = f"{base_url.rstrip('/')}/{_REDIRECT_PATH}?{query}&all-routes"
        return RedirectResponse(cdesk_redirect, status_code=302)

    @mcp.custom_route("/login/azure/callback", methods=["GET"])  # type: ignore[untyped-decorator]
    async def azure_callback(request: Request) -> Response:  # pragma: no cover - exercised live
        if not limiter.allow(_client_ip(request, trust_forwarded)):
            return _secure_html(
                "<h1>Too many attempts</h1><p>Please wait a minute and try again.</p>",
                status_code=429,
            )
        # SECURITY: `token`/`refresh_token` arrive here in the URL — never log the
        # raw callback URL or these values. `state` carries our opaque session
        # (the CSRF binding); `base_url` is the server we started against.
        session = request.query_params.get("state", "")
        apitoken = request.query_params.get("token", "")
        refresh = request.query_params.get("refresh_token", "")
        error = request.query_params.get("error", "")
        base_url = provider.valid_base_url(request.query_params.get("base_url", ""))
        if not session or not await provider.peek_session(session):
            return _err(
                "", "Invalid or expired sign-in link. Please restart the connection.", 400
            )
        if base_url is None:
            return _err(session, "That CDESK server isn't allowed.", 400)
        if error or not apitoken:
            return _err(
                session,
                "Microsoft sign-in was declined, or your Microsoft account isn't "
                "linked to a CDESK user. Please try again or use your CDESK login.",
                400,
            )

        # refresh_token may be absent if the CDESK server has no token-signing
        # secret configured (manual: treat as optional).
        try:
            redirect_url = await provider.complete_login(
                session,
                login="microsoft-sso",
                apitoken=apitoken,
                refresh_token=refresh or None,
                base_url=base_url,
            )
        except KeyError:
            return _err(session, "Sign-in session expired. Please restart the connection.", 400)
        except ValueError:
            return _err(session, "That CDESK server isn't allowed.", 400)
        return RedirectResponse(redirect_url, status_code=302)
