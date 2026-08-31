"""Probe a CDESK server's PUBLIC login-methods list — ``GET /api/auth/connector``.

CDESK's own login screen discovers what a server offers by calling this endpoint
unauthenticated and reading the array it returns::

    {"data":[{"id":186,"code":"azure","button":true,
              "value":"Sign in with Microsoft",...},
             {"id":"","name":"CDESK account"}]}

Entra availability is *existence of an entry with* ``code == "azure"`` — never a
boolean flag. The tenant is selected by the request's own Host header (there is no
tenant parameter), so the probe must be made against the exact address the user
typed.

Two consumers, which is why this lives in its own module rather than in
``azure_login.py``: the login page's ``POST /login/probe`` (always mounted) and
``/login/azure/start`` (mounted only when Microsoft sign-in is enabled).

WHAT THIS MUST NOT BECOME. ``oauth/login.py`` carries a scar comment (see the
``authenticate()`` call there) about a ``GET /auth/me`` pre-flight probe that was
deleted: when that one endpoint began returning 500 on a tenant whose data
endpoints were fine, nobody could sign in. This endpoint is public and documented,
which makes it a better probe — not a safe gate. Hence the status taxonomy below
separates "reached it, it is definitively not a CDESK" from "could not get a clear
answer", and callers must only ever block on the former.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

from cdesk_mcp.cdesk_client import normalize_api_base

log = logging.getLogger(__name__)

_CONNECTOR_PATH = "auth/connector"

# Deliberately far below the 30s service timeout: a person is watching a spinner,
# and this route is unauthenticated, so a blackholed host must not pin a socket.
_PROBE_MAX_SECONDS = 5.0
_PROBE_CONNECT_SECONDS = 3.0
_MAX_REDIRECTS = 3
# Caps in-flight probes process-wide. Without it, N concurrent requests against a
# host that swallows packets hold N sockets and tasks for the full timeout.
_MAX_CONCURRENT_PROBES = 8
_probe_slots = asyncio.Semaphore(_MAX_CONCURRENT_PROBES)

# Identifies us to WAF operators who see the hit and want to allowlist it.
_USER_AGENT = "cdesk-mcp/1.0 (connector discovery)"

#: What the probe concluded.
#:
#: ``ok``           reached a CDESK; ``azure_id`` says whether it offers Entra
#: ``not_json``     2xx, but the body isn't JSON — a website, not a CDESK
#: ``not_cdesk``    2xx JSON without a ``data`` list, or a redirect off-allowlist
#: ``unreachable``  DNS failure or connection refused (one status for BOTH — see below)
#: ``tls``          the certificate could not be verified
#: ``timeout``      no answer in time
#: ``http_error``   answered with a non-2xx (401/403 behind a proxy, 404, 5xx)
#:
#: Only ``not_json`` / ``not_cdesk`` / ``unreachable`` / ``tls`` are confident
#: negatives. ``timeout`` and ``http_error`` are inconclusive and must not block.
ProbeStatus = Literal[
    "ok", "unreachable", "tls", "timeout", "http_error", "not_json", "not_cdesk"
]

#: Resolves a hostname to a list of IP-address strings. Injectable so the
#: private-range check is testable without DNS.
Resolver = Callable[[str], list[str]]


class ProbeFn(Protocol):
    """The probe as callers depend on it.

    Passed into ``register_*`` rather than reached for as a module attribute:
    two modules import this helper, so monkeypatching one module's binding would
    silently leave the other calling the real thing."""

    async def __call__(
        self, base_url: str, *, timeout_seconds: float = ...
    ) -> ConnectorProbe: ...


@dataclass(frozen=True, slots=True)
class ConnectorProbe:
    """What ``probe_connector`` found. ``detail`` is operator-facing (logs only) —
    the user-facing wording lives in ``oauth/login.py`` next to the other page
    strings, so all of it can be read and changed in one place."""

    status: ProbeStatus
    azure_id: str | None = None
    http_status: int | None = None
    detail: str = ""

    @property
    def has_azure(self) -> bool:
        return self.azure_id is not None


def redact_userinfo(url: str) -> str:
    """Replace ``user:pw@`` with ``***@``.

    ``normalize_base_url`` deliberately accepts (and preserves) userinfo — it is
    how you reach a CDESK behind a Basic-auth proxy. That means a base URL can
    carry a password, so it must never be echoed to the page, returned in JSON, or
    written to a log line without passing through here first."""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    _, _, hostport = parts.netloc.rpartition("@")
    return urlunsplit((parts.scheme, f"***@{hostport}", parts.path, parts.query, parts.fragment))


def connector_url(base_url: str) -> str:
    """``<tenant>/api/auth/connector`` for a tenant URL with or without ``/api``.

    Shares ``normalize_api_base`` with ``CdeskClient`` on purpose. ``check_base_url``
    deliberately does NOT strip a trailing ``/api`` (the client handles it), so
    naive concatenation here produced ``/api/api/auth/connector`` — a 404 — for
    ``cdesk.example.com/api``, which is an address that signs in perfectly well."""
    return normalize_api_base(base_url) + _CONNECTOR_PATH


def _is_tls_failure(exc: BaseException) -> bool:
    """True when an ``ssl.SSLError`` is anywhere in the cause chain.

    Keyed off the chain rather than a substring of ``str(exc)`` because the
    message text differs across OpenSSL builds and Python versions."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, ssl.SSLError):
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


def _default_resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return []
    return [str(info[4][0]) for info in infos]


def resolves_private(host: str, resolve: Resolver = _default_resolve) -> bool:
    """True if the host resolves to any loopback / private / link-local / reserved
    address — including the cloud metadata address ``169.254.169.254``.

    Used to decide how much detail the probe result may reveal, NOT to block: this
    connector explicitly supports on-prem LAN installs (``normalize_base_url``
    accepts single-label intranet names and plain ``http://``). Resolution failure
    returns False — we can't tell, and the connect will fail anyway into the shared
    "couldn't reach" answer.

    Known, accepted gap: resolve-then-connect is not atomic, so DNS rebinding can
    still slip a public answer here and a private one to the socket."""
    for raw in resolve(host):
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            return True
    return False


async def host_is_private(base_url: str, resolve: Resolver | None = None) -> bool:
    """Async wrapper for :func:`resolves_private` — the default resolver blocks."""
    host = urlsplit(base_url).hostname or ""
    if not host:
        return False
    return await asyncio.to_thread(resolves_private, host, resolve or _default_resolve)


async def probe_connector(
    base_url: str,
    *,
    timeout_seconds: float = _PROBE_MAX_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
    allow_url: Callable[[str], bool] | None = None,
) -> ConnectorProbe:
    """GET ``<base_url>/api/auth/connector`` and classify the answer.

    ``transport`` is injectable so every branch is testable with
    ``httpx.MockTransport`` and no network. ``allow_url`` (when given) is consulted
    for every redirect hop: without it, an open redirect on an allowlisted host
    would turn this into an SSRF to a host the allowlist excludes.

    Never raises — every failure becomes a status. Logs at debug, not warning: a
    host scan against this route would otherwise flood the log."""
    url = connector_url(base_url)
    timeout = httpx.Timeout(
        min(timeout_seconds, _PROBE_MAX_SECONDS),
        connect=min(timeout_seconds, _PROBE_CONNECT_SECONDS),
    )
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    async with _probe_slots:
        try:
            async with httpx.AsyncClient(
                timeout=timeout, transport=transport, follow_redirects=False
            ) as http:
                resp = await http.get(url, headers=headers)
                # Followed by hand rather than with follow_redirects=True so each
                # hop can be re-checked against the allowlist.
                for _ in range(_MAX_REDIRECTS):
                    if not resp.is_redirect:
                        break
                    nxt = resp.next_request
                    if nxt is None:
                        return ConnectorProbe(
                            "http_error", http_status=resp.status_code,
                            detail="redirect without a usable Location",
                        )
                    if allow_url is not None and not allow_url(str(nxt.url)):
                        return ConnectorProbe(
                            "not_cdesk", http_status=resp.status_code,
                            detail="redirected to a server outside the allowlist",
                        )
                    resp = await http.send(nxt)
                if resp.is_redirect:
                    return ConnectorProbe(
                        "http_error", http_status=resp.status_code, detail="too many redirects"
                    )
        except httpx.TimeoutException as e:
            log.debug("connector probe timed out for %s: %s", redact_userinfo(url), type(e).__name__)
            return ConnectorProbe("timeout", detail=type(e).__name__)
        except httpx.HTTPError as e:
            tls = _is_tls_failure(e)
            log.debug(
                "connector probe failed for %s: %s%s",
                redact_userinfo(url), type(e).__name__, " (TLS)" if tls else "",
            )
            return ConnectorProbe("tls" if tls else "unreachable", detail=type(e).__name__)

    if resp.status_code // 100 != 2:
        return ConnectorProbe(
            "http_error", http_status=resp.status_code, detail=f"HTTP {resp.status_code}"
        )
    try:
        body = resp.json()
    except ValueError:
        return ConnectorProbe(
            "not_json", http_status=resp.status_code, detail="2xx body is not JSON"
        )

    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return ConnectorProbe(
            "not_cdesk", http_status=resp.status_code, detail="JSON has no `data` list"
        )
    # An EMPTY list is a CDESK with no third-party login, not a non-CDESK. And the
    # plain-password entry is matched by NOTHING here on purpose: its name
    # ("CDESK account") is localised per tenant, so it is not a reliable marker.
    azure_id: str | None = None
    for entry in data:
        if isinstance(entry, dict) and entry.get("code") == "azure" and entry.get("id"):
            azure_id = str(entry["id"])
            break
    return ConnectorProbe("ok", azure_id=azure_id, http_status=resp.status_code)
