"""Shared web helpers for the unauthenticated OAuth HTML routes (login + SSO).

Extracted from oauth/login.py so both the password login page and the
"Sign in with Microsoft" routes (oauth/azure_login.py) reuse the same security
headers, client-IP resolution, and per-IP rate limiter.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

# Cap distinct IPs tracked by a limiter so a flood of distinct (incl. spoofed)
# source addresses can't grow the map without bound. LRU-evicted past this.
_MAX_TRACKED_IPS = 10_000

_SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


def _secure_html(body: str, status_code: int = 200) -> HTMLResponse:
    """HTMLResponse with clickjacking / MIME-sniffing protection headers."""
    return HTMLResponse(body, status_code=status_code, headers=dict(_SECURITY_HEADERS))


def _secure_json(payload: Mapping[str, object], status_code: int = 200) -> JSONResponse:
    """JSONResponse carrying the same header set as :func:`_secure_html`, plus
    ``no-store``. Kept here so the security headers stay defined in one place.

    Deliberately emits no ``Access-Control-Allow-Origin``: the login page's probe
    route is same-origin only, and its answer (reachable / not-a-CDESK / …) is
    exactly the sort of thing a cross-origin page should not be able to read."""
    headers = dict(_SECURITY_HEADERS)
    headers["Cache-Control"] = "no-store"
    return JSONResponse(dict(payload), status_code=status_code, headers=headers)


def _client_ip(request: Request, trust_forwarded: bool = False) -> str:
    """Client IP for the rate limiter. `CF-Connecting-IP` / `X-Forwarded-For` are
    honored **only** when `trust_forwarded` is set (i.e. behind a trusted proxy —
    `CDESK_TRUST_PROXY`); otherwise they're attacker-spoofable, so we fall back to
    the real socket peer."""
    if trust_forwarded:
        cf = request.headers.get("cf-connecting-ip")
        if cf:
            return cf.strip()
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    client = request.client
    return client.host if client is not None else "unknown"


class _RateLimiter:
    """In-memory per-replica sliding-window limiter. `now` is injectable for
    tests. Per-replica only (N replicas → N× the window) — defense-in-depth.

    Memory is bounded: at most `max_keys` IPs are tracked, LRU-evicted beyond
    that, so a flood of distinct (including spoofed-header) source IPs against
    the unauthenticated routes can't grow the map without bound."""

    def __init__(
        self,
        max_attempts: int,
        window_seconds: float,
        now: Callable[[], float] = time.monotonic,
        max_keys: int = _MAX_TRACKED_IPS,
    ) -> None:
        self._max = max_attempts
        self._window = window_seconds
        self._now = now
        self._max_keys = max_keys
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def allow(self, key: str) -> bool:
        now = self._now()
        cutoff = now - self._window
        bucket = self._hits.get(key)
        if bucket is None:
            bucket = deque()
            self._hits[key] = bucket
        else:
            self._hits.move_to_end(key)  # mark most-recently-used
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        allowed = len(bucket) < self._max
        if allowed:
            bucket.append(now)
        # Bound memory: evict the least-recently-seen IPs beyond the cap.
        while len(self._hits) > self._max_keys:
            self._hits.popitem(last=False)
        return allowed
