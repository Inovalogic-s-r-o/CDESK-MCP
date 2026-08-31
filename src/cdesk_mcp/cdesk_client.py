"""HTTP client for CDESK API v3.

Authentication model: credentials from env → auth exchange at first request →
cache apitoken in memory → use `Authorization: apitoken <hex>` on subsequent calls.
On 401 mid-session, the cached token is dropped and a fresh auth is attempted once.

Two credential kinds (m2m preferred when both are configured):
  * m2m (CDESK_CLIENT_ID + CDESK_CLIENT_SECRET): POST /auth/jwttoken mints a
    short-lived (5-min) HS256 JWT, which is then sent as `Authorization:
    Bearer <jwt>` to POST /auth/login with an empty body — same apitoken
    response as a password login. The pair can re-mint at any time, so no
    refresh token is needed.
  * password (CDESK_LOGIN + CDESK_PASSWORD): classic POST /auth/login body.

The password lives only in this process's memory; it is never logged or written.
Logs include the request method, path, status, and duration — never headers,
credentials, or response bodies.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, Self

import httpx

from cdesk_mcp.config import Config

log = logging.getLogger(__name__)

_LOGIN_PATH = "auth/login"
_JWTTOKEN_PATH = "auth/jwttoken"
_RENEW_PATH = "auth/renewtokens"
_LOGOUT_PATH = "logout"
_HEALTH_PATH = "v3/task/enums"
_MAX_ATTEMPTS = 3
# Methods safe to replay on a 5xx / mid-flight network error: a retry of one
# of these can't create a duplicate (GET/HEAD/OPTIONS have no effect; PUT and
# DELETE are idempotent — a re-PUT carries the same timestamp_check, a
# re-DELETE of an already-deleted id is a no-op/404). POST is NOT here: it is
# only retried when the failure is provably pre-send (ConnectError/Timeout),
# so a committed-but-unacknowledged create is never silently re-sent.
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
_BACKOFF_BASE_SECONDS = 1.0
# Wait schedule between login attempts on transient failure (network
# error or 5xx). Intentionally long because login runs at startup and a
# minute-scale wait gives CDESK time to recover from real outages
# (restart, deploy, brief DB hiccup) without giving up prematurely.
# Worst case total wait is 70s before raising. Indexed by (attempt - 1);
# has _MAX_ATTEMPTS - 1 entries because no wait happens after the final
# attempt (we raise instead).
_LOGIN_BACKOFF_SECONDS: tuple[float, ...] = (10.0, 60.0)


def normalize_api_base(base_url: str) -> str:
    """Canonicalize a CDESK tenant URL to ``<tenant>/api/`` (trailing slash).

    Accept the tenant root with or without an ``/api`` suffix; callers always talk
    to ``<tenant>/api/``. Strip a trailing ``/api`` so it can be re-added
    canonically — that way `.env` only carries the bare URL
    (https://cmpp.seal.sk) while older configs that still include /api keep
    working.

    Module-level (not just a CdeskClient method) because ``oauth/_connector.py``
    builds ``/api/auth/connector`` from a user-typed address and must apply exactly
    the same rule: concatenating blindly produced ``/api/api/auth/connector`` for
    an address that signs in fine."""
    normalized_base = base_url.rstrip("/")
    if normalized_base.endswith("/api"):
        normalized_base = normalized_base[: -len("/api")]
    return normalized_base + "/api/"


class CdeskAuthError(RuntimeError):
    """Raised when login fails terminally (bad creds, 2FA, malformed response, or
    upstream login endpoint unavailable after retries).

    ``status`` is the HTTP status that classified the failure when there was a
    single one (401 = bad credentials, 302 = 2FA challenge), else None. Callers
    that render a human-facing message — the /login consent page — branch on it
    rather than matching the message text, which is written for operators and
    names env vars.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class CdeskApiError(RuntimeError):
    """CDESK signaled an error inside the response body (`msg.error[]`) even
    though HTTP status was 2xx.

    CDESK frequently returns HTTP 200 + `{data: false, msg: {error: [...]}}`
    for things like optimistic-lock conflicts and validation failures. The
    error layer (`errors.translate_error`) reads `self.body` to produce the
    LLM-facing message — same pipeline as `httpx.HTTPStatusError`.
    """

    def __init__(self, body: dict[str, Any], http_status: int) -> None:
        self.body = body
        self.http_status = http_status
        msg_errors = []
        msg = body.get("msg") if isinstance(body, dict) else None
        if isinstance(msg, dict):
            entries = msg.get("error")
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        text = entry.get("message")
                        if isinstance(text, str):
                            msg_errors.append(text)
        summary = "; ".join(msg_errors) if msg_errors else "(no message extracted)"
        super().__init__(f"CDESK body-level error (HTTP {http_status}): {summary}")


class CdeskClient:
    """Async wrapper over httpx.AsyncClient with CDESK login + retries.

    Construction: prefer `CdeskClient.from_env()`. Missing base URL / login /
    password raises ValueError, so tools that depend on the client surface a
    clear error to the LLM rather than crashing the server.
    """

    def __init__(
        self,
        *,
        base_url: str,
        login: str,
        password: str = "",
        timeout_seconds: float,
        apitoken: str | None = None,
        refresh_token: str | None = None,
        client_id: int | None = None,
        client_secret: str = "",
        on_renew: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("CDESK_BASE_URL is required")
        if not login:
            raise ValueError("CDESK_LOGIN is required")
        # A client authenticates with a password (it logs in to obtain an
        # apitoken), with an already-issued apitoken (the token-based connect,
        # which holds no password), or with the m2m pair (client_id +
        # client_secret, which can mint a fresh apitoken at any time).
        has_m2m = client_id is not None and bool(client_secret)
        if not password and not apitoken and not has_m2m:
            raise ValueError(
                "CDESK_PASSWORD or CDESK_CLIENT_ID+CDESK_CLIENT_SECRET is required"
            )

        self._http = httpx.AsyncClient(
            base_url=self._normalize_base_url(base_url), timeout=timeout_seconds
        )
        self._login = login
        self._password = password
        # Seeded directly for token-based (no-password) construction; for a
        # password client it's populated lazily by the first login.
        self._token: str | None = apitoken
        self._refresh_token: str | None = refresh_token
        self._client_id = client_id
        self._client_secret = client_secret
        # Optional async hook invoked with the fresh apitoken after a successful
        # CDESK renew. A generic persistence/write-back seam; the stateless OAuth
        # provider leaves it unset (the renewed token lives only in this client).
        self._on_renew = on_renew
        self._token_lock = asyncio.Lock()

    @property
    def login(self) -> str:
        """The CDESK login this client authenticates as (no secret exposed)."""
        return self._login

    @property
    def token(self) -> str | None:
        """The current cached CDESK apitoken (None before the first auth). The
        /login handler reads this after validation to embed it in the issued
        OAuth token."""
        return self._token

    @property
    def refresh_token(self) -> str | None:
        """The CDESK refresh token, if captured (from login) or seeded
        (token-based construction). Used to renew apitokens and embedded in the
        issued OAuth token to carry the session."""
        return self._refresh_token

    @classmethod
    def from_tokens(
        cls,
        *,
        base_url: str,
        login: str,
        apitoken: str,
        refresh_token: str | None,
        timeout_seconds: float,
        on_renew: Callable[[str], Awaitable[None]] | None = None,
    ) -> CdeskClient:
        """Build a client from an already-issued CDESK apitoken (+ optional
        refresh token), skipping password login — no password is held. On 401
        the client renews via the refresh token (if present) rather than
        re-logging-in. ``on_renew`` is invoked with the fresh apitoken after a
        successful renew (used to persist the rotated token)."""
        return cls(
            base_url=base_url,
            login=login,
            password="",
            timeout_seconds=timeout_seconds,
            apitoken=apitoken,
            refresh_token=refresh_token,
            on_renew=on_renew,
        )

    @classmethod
    def from_m2m(
        cls,
        *,
        base_url: str,
        login: str,
        client_id: int,
        client_secret: str,
        timeout_seconds: float,
        on_renew: Callable[[str], Awaitable[None]] | None = None,
    ) -> CdeskClient:
        """Build a client from the machine-to-machine pair (CDESK user id +
        OAuth-connector signing key) — no password is held. The pair mints a
        fresh apitoken via POST /auth/jwttoken → Bearer /auth/login whenever
        needed, including on a mid-session 401, so no refresh token is required.
        ``login`` is only a display label (e.g. ``client:<id>``); the real
        identity is the JWT's subject."""
        return cls(
            base_url=base_url,
            login=login,
            password="",
            timeout_seconds=timeout_seconds,
            client_id=client_id,
            client_secret=client_secret,
            on_renew=on_renew,
        )

    @classmethod
    def from_env(cls) -> CdeskClient:
        config = Config.from_env()
        # Precedence: the m2m pair wins over login+password when both are
        # configured (Config already warns + drops a partial pair).
        if config.client_id is not None and config.client_secret:
            return cls.from_m2m(
                base_url=config.base_url,
                login=config.login or f"client:{config.client_id}",
                client_id=config.client_id,
                client_secret=config.client_secret,
                timeout_seconds=config.timeout_seconds,
            )
        return cls(
            base_url=config.base_url,
            login=config.login,
            password=config.password,
            timeout_seconds=config.timeout_seconds,
        )

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Canonicalize a CDESK tenant URL to ``<tenant>/api/``. See
        :func:`normalize_api_base` — shared with the connector probe."""
        return normalize_api_base(base_url)

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return await self._request("GET", path, params=params)

    async def post(self, path: str, *, json: dict[str, Any] | None = None) -> Any:
        return await self._request("POST", path, json=json)

    async def put(self, path: str, *, json: dict[str, Any] | None = None) -> Any:
        return await self._request("PUT", path, json=json)

    async def delete(self, path: str) -> Any:
        return await self._request("DELETE", path)

    async def health_check(self) -> None:
        """Hit a known-good endpoint to verify credentials. Raises on failure.

        Triggers the initial login as a side effect (the auth happens lazily
        on the first authenticated request)."""
        await self.get(_HEALTH_PATH)

    async def authenticate(self) -> str:
        """Force the lazy login now and return the apitoken. Raises
        CdeskAuthError if the credentials don't work.

        This is the cheapest way to validate credentials: it makes exactly one
        call, `POST /auth/login`, which is the documented endpoint for the job
        (200 = success + apitoken, 401 = bad credentials, 302 = 2FA challenge —
        all three are classified by `_login_now`). Nothing else is requested, so
        no module ACL is involved and no other endpoint can make it fail.

        Prefer this over calling some arbitrary GET just to trigger the login.
        Doing that couples credential validation to an unrelated endpoint's
        availability: `oauth/login.py` once used a `GET /auth/me` probe that way,
        and a 500 on that one undocumented endpoint took down connector login on
        a tenant whose data endpoints were all healthy. That helper has since
        been removed — do not reintroduce it.

        Honors the same credential precedence as any other request (m2m pair →
        CDESK refresh token → password), so it is valid for every client kind."""
        return await self._ensure_token()

    async def logout(self) -> None:
        """Best-effort: tell CDESK to invalidate the current apitoken
        (``POST /api/logout``). Bypasses ``_request`` so it does NOT renew on a
        401 (an already-invalid token is fine — we're logging out). No-op if no
        token is held; never raises (the caller treats this as best-effort)."""
        token = self._token
        if not token:
            return
        try:
            await self._http.post(
                _LOGOUT_PATH,
                json={"token": token},
                headers={"Authorization": f"apitoken {token}"},
            )
        except Exception as e:  # network / HTTP error — logout is best-effort
            log.warning("CDESK logout request failed: %s: %s", type(e).__name__, e)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def _ensure_token(self) -> str:
        async with self._token_lock:
            if self._token is None:
                self._token = await self._acquire_token()
            return self._token

    async def _acquire_token(self) -> str:
        """Obtain a fresh apitoken. Preference order: the m2m pair (can mint a
        token at any time, no other state needed) → the CDESK refresh token
        (cheaper than a login, and the only option for token-based clients that
        hold no password) → a password login.

        Must be called with ``_token_lock`` held."""
        if self._client_id is not None and self._client_secret:
            return await self._jwt_login_now()
        if self._refresh_token:
            try:
                return await self._renew_now()
            except CdeskAuthError:
                # Refresh token rejected/expired. A password client can still
                # recover by logging in again; a token-based client cannot.
                if not self._password:
                    raise
        if self._password:
            return await self._login_now()
        raise CdeskAuthError(
            "CDESK session expired and there is no refresh token or password to "
            "re-authenticate. Reconnect the CDESK connector to sign in again."
        )

    async def _invalidate_token(self, expected: str | None = None) -> None:
        """Clear the cached apitoken. Pass `expected` (the token that just got a
        401) so a concurrent coroutine that already re-logged-in isn't clobbered:
        only clear if the cached token is still the stale one we used."""
        async with self._token_lock:
            if expected is None or self._token == expected:
                self._token = None

    async def _login_now(self) -> str:
        """POST /auth/login with retries on transient failures. Converts every
        terminal failure to CdeskAuthError so callers can use one except clause."""
        response: httpx.Response | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            start = time.monotonic()
            try:
                response = await self._http.post(
                    _LOGIN_PATH,
                    json={"login": self._login, "password": self._password},
                )
            except httpx.RequestError as e:
                duration_ms = (time.monotonic() - start) * 1000
                log.warning(
                    "CDESK login -> network error after %.0fms (attempt %d/%d): %s",
                    duration_ms, attempt, _MAX_ATTEMPTS, type(e).__name__,
                )
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_LOGIN_BACKOFF_SECONDS[attempt - 1])
                    continue
                raise CdeskAuthError(
                    f"Could not connect to CDESK at {self._http.base_url} after "
                    f"{_MAX_ATTEMPTS} attempts ({type(e).__name__}). The connection "
                    f"failed — check that CDESK_BASE_URL is reachable and that your "
                    f"network is up."
                ) from e

            duration_ms = (time.monotonic() - start) * 1000
            log.info("CDESK login -> %d (%.0fms)", response.status_code, duration_ms)

            if 500 <= response.status_code < 600 and attempt < _MAX_ATTEMPTS:
                # Honor Retry-After when present; otherwise the longer login schedule.
                sleep_for = (
                    self._parse_retry_after(response)
                    if "Retry-After" in response.headers
                    else _LOGIN_BACKOFF_SECONDS[attempt - 1]
                )
                await asyncio.sleep(sleep_for)
                continue
            break

        assert response is not None  # loop body either continues, returns, or raises

        status = response.status_code
        if status == 302:
            raise CdeskAuthError(
                "CDESK /auth/login returned 302 (likely a 2FA challenge or interstitial "
                "redirect). cdesk-mcp does not yet support these auth flows. "
                "If you're sure no 2FA is configured, share the response Location "
                "header to extend support.",
                status=302,
            )
        if status == 401:
            raise CdeskAuthError(
                "CDESK login failed (401). Check CDESK_LOGIN / CDESK_PASSWORD.",
                status=401,
            )
        if not response.is_success:
            raise CdeskAuthError(
                f"CDESK login endpoint at {self._http.base_url} returned HTTP {status} "
                f"on all {_MAX_ATTEMPTS} attempts. The connection reached the server "
                f"but it appears to be unavailable; please try again later."
            )

        return self._parse_login_response(response)

    def _parse_login_response(self, response: httpx.Response) -> str:
        """Extract the apitoken from a successful /auth/login response (shared
        by the password and Bearer-JWT login paths). The login response is
        *unwrapped* — `token` and `refreshToken` sit at the top level."""
        try:
            data = response.json()
        except ValueError as e:
            raise CdeskAuthError(f"CDESK login response was not JSON: {e}") from e

        if not isinstance(data, dict):
            raise CdeskAuthError("CDESK login response was not a JSON object")

        token = data.get("token")
        if not isinstance(token, str) or not token:
            raise CdeskAuthError("CDESK login response missing 'token' field")
        # Capture the refresh token if present. Don't clobber an existing
        # token with a missing value. (The m2m Bearer login typically returns
        # none — the m2m pair re-mints instead of refreshing.)
        refresh = data.get("refreshToken")
        if isinstance(refresh, str) and refresh:
            self._refresh_token = refresh
        return token

    async def _jwt_login_now(self) -> str:
        """Machine-to-machine login: POST /auth/jwttoken (client_id +
        client_secret → short-lived HS256 JWT) followed by POST /auth/login
        with `Authorization: Bearer <jwt>` and an empty body — the same
        apitoken response as a password login.

        The JWT is valid for ~5 minutes, so a login-401 most likely means it
        expired between the two calls (e.g. the login POST sat in a long retry
        backoff) — the WHOLE exchange is retried once in that case; a second
        login-401 raises so a genuinely rejected JWT terminates."""
        for exchange_attempt in (1, 2):
            jwt = await self._mint_jwt()
            response = await self._auth_post_with_retries(
                _LOGIN_PATH,
                json={},
                headers={"Authorization": f"Bearer {jwt}"},
                label="m2m login",
            )
            status = response.status_code
            if status == 401 and exchange_attempt == 1:
                log.info("CDESK m2m login -> 401 (JWT likely expired); re-minting once")
                continue
            if status == 401:
                raise CdeskAuthError(
                    "CDESK rejected the Bearer JWT on /auth/login (401) even after "
                    "re-minting. Check that the OAuth connector for CDESK_CLIENT_ID "
                    "is active and CDESK_CLIENT_SECRET matches its signing key."
                )
            if not response.is_success:
                raise CdeskAuthError(
                    f"CDESK /auth/login (Bearer JWT) returned HTTP {status}; the "
                    f"m2m session could not be established."
                )
            return self._parse_login_response(response)
        raise RuntimeError("unreachable: m2m exchange loop must return or raise")

    async def _mint_jwt(self) -> str:
        """POST /auth/jwttoken to exchange the m2m pair for a short-lived login
        JWT. Maps the documented terminal statuses to distinct messages:
        400 = missing creds / no active OAuth connector, 401 = invalid
        client_secret, 404 = no user for client_id."""
        response = await self._auth_post_with_retries(
            _JWTTOKEN_PATH,
            json={"client_id": self._client_id, "client_secret": self._client_secret},
            headers=None,
            label="jwttoken",
        )
        status = response.status_code
        if status == 400:
            raise CdeskAuthError(
                "CDESK /auth/jwttoken returned 400 — the user for CDESK_CLIENT_ID "
                "has no active OAuth 2.0 connector (or the request was missing "
                "credentials). Create/enable the connector in CDESK."
            )
        if status == 401:
            raise CdeskAuthError(
                "CDESK /auth/jwttoken returned 401 — CDESK_CLIENT_SECRET does not "
                "match the OAuth connector's signing key."
            )
        if status == 404:
            raise CdeskAuthError(
                "CDESK /auth/jwttoken returned 404 — no CDESK user exists for "
                "CDESK_CLIENT_ID."
            )
        if not response.is_success:
            raise CdeskAuthError(
                f"CDESK /auth/jwttoken returned HTTP {status}; the m2m login JWT "
                f"could not be obtained."
            )
        try:
            data = response.json()
        except ValueError as e:
            raise CdeskAuthError(f"CDESK jwttoken response was not JSON: {e}") from e
        access_token = data.get("access_token") if isinstance(data, dict) else None
        if not isinstance(access_token, str) or not access_token:
            raise CdeskAuthError("CDESK jwttoken response missing 'access_token' field")
        return access_token

    async def _auth_post_with_retries(
        self,
        path: str,
        *,
        json: dict[str, Any] | None,
        headers: dict[str, str] | None,
        label: str,
    ) -> httpx.Response:
        """POST to an auth endpoint with the login retry schedule (network
        errors and 5xx retried on _LOGIN_BACKOFF_SECONDS, Retry-After honored).
        Returns the final response — terminal-status mapping is the caller's
        job. Same structure as _login_now/_renew_now's loops."""
        response: httpx.Response | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            start = time.monotonic()
            try:
                response = await self._http.post(path, json=json, headers=headers)
            except httpx.RequestError as e:
                duration_ms = (time.monotonic() - start) * 1000
                log.warning(
                    "CDESK %s -> network error after %.0fms (attempt %d/%d): %s",
                    label, duration_ms, attempt, _MAX_ATTEMPTS, type(e).__name__,
                )
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_LOGIN_BACKOFF_SECONDS[attempt - 1])
                    continue
                raise CdeskAuthError(
                    f"Could not connect to CDESK at {self._http.base_url} after "
                    f"{_MAX_ATTEMPTS} attempts ({type(e).__name__}) during {label}."
                ) from e

            duration_ms = (time.monotonic() - start) * 1000
            log.info("CDESK %s -> %d (%.0fms)", label, response.status_code, duration_ms)

            if 500 <= response.status_code < 600 and attempt < _MAX_ATTEMPTS:
                sleep_for = (
                    self._parse_retry_after(response)
                    if "Retry-After" in response.headers
                    else _LOGIN_BACKOFF_SECONDS[attempt - 1]
                )
                await asyncio.sleep(sleep_for)
                continue
            break

        assert response is not None  # loop body either continues, breaks, or raises
        return response

    async def _renew_now(self) -> str:
        """POST /auth/renewtokens to exchange the CDESK refresh token for a fresh
        apitoken. Retries transient failures like ``_login_now`` and converts
        terminal failures to ``CdeskAuthError``.

        NOTE: unlike /auth/login (unwrapped), the renew response is *wrapped* in
        a ``data`` envelope: ``{"data": {"apitoken": "...", ...}}``. CDESK does
        not rotate the refresh token, so ``self._refresh_token`` is left as-is."""
        if not self._refresh_token:
            raise CdeskAuthError("No CDESK refresh token available to renew.")

        response: httpx.Response | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            start = time.monotonic()
            try:
                response = await self._http.post(
                    _RENEW_PATH,
                    json={
                        "refreshToken": self._refresh_token,
                        "requestedObject": {"apitoken": ""},
                    },
                )
            except httpx.RequestError as e:
                duration_ms = (time.monotonic() - start) * 1000
                log.warning(
                    "CDESK renew -> network error after %.0fms (attempt %d/%d): %s",
                    duration_ms, attempt, _MAX_ATTEMPTS, type(e).__name__,
                )
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_LOGIN_BACKOFF_SECONDS[attempt - 1])
                    continue
                raise CdeskAuthError(
                    f"Could not reach CDESK to renew the session after "
                    f"{_MAX_ATTEMPTS} attempts ({type(e).__name__})."
                ) from e

            duration_ms = (time.monotonic() - start) * 1000
            log.info("CDESK renew -> %d (%.0fms)", response.status_code, duration_ms)

            if 500 <= response.status_code < 600 and attempt < _MAX_ATTEMPTS:
                sleep_for = (
                    self._parse_retry_after(response)
                    if "Retry-After" in response.headers
                    else _LOGIN_BACKOFF_SECONDS[attempt - 1]
                )
                await asyncio.sleep(sleep_for)
                continue
            break

        assert response is not None  # loop body either continues, returns, or raises

        status = response.status_code
        if status in (401, 403):
            raise CdeskAuthError(
                "CDESK rejected the refresh token (it may have expired). "
                "Reconnect the CDESK connector to sign in again."
            )
        if not response.is_success:
            raise CdeskAuthError(
                f"CDESK renew endpoint returned HTTP {status}; the session could "
                f"not be renewed."
            )

        try:
            body = response.json()
        except ValueError as e:
            raise CdeskAuthError(f"CDESK renew response was not JSON: {e}") from e

        data = body.get("data") if isinstance(body, dict) else None
        apitoken = data.get("apitoken") if isinstance(data, dict) else None
        if not isinstance(apitoken, str) or not apitoken:
            raise CdeskAuthError("CDESK renew response missing 'data.apitoken' field")
        if self._on_renew is not None:
            # Hand the rotated apitoken to the optional persistence hook.
            # Best-effort: a write-back failure must not break the in-process
            # session that just renewed successfully.
            try:
                await self._on_renew(apitoken)
            except Exception:  # pragma: no cover - best effort
                log.warning("on_renew write-back failed", exc_info=True)
        return apitoken

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        """Two-level retry: outer handles 401 (relogin once), inner handles
        5xx/429/network with exponential backoff."""
        rel_path = path.lstrip("/")

        # Outer pass: at most two iterations (original + post-relogin).
        for relogin_pass in range(2):
            token = await self._ensure_token()
            headers = {"Authorization": f"apitoken {token}"}
            response = await self._do_with_backoff(
                method, rel_path, headers, params=params, json=json,
            )

            if response.status_code == 401 and relogin_pass == 0:
                log.info("CDESK 401 received; invalidating cached token and retrying once")
                # Pass the token we used so a concurrent coroutine that already
                # re-authenticated isn't clobbered (avoids a relogin stampede).
                await self._invalidate_token(expected=token)
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                raise

            if not response.content:
                return None
            parsed = response.json()

            # CDESK signals many failures (optimistic-lock conflicts,
            # some validation errors, etc.) as HTTP 200 + msg.error[] in
            # the body. Surface those as exceptions so tools' try/except
            # routes them through translate_error like any HTTP failure.
            _raise_if_body_signals_error(parsed, response.status_code)
            return parsed

        raise RuntimeError("unreachable: outer relogin loop must return or raise")

    async def _do_with_backoff(
        self,
        method: str,
        rel_path: str,
        headers: dict[str, str],
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Inner retry loop for 5xx / 429 / network errors. Returns the final
        response.

        Retries are gated by method idempotency so a non-idempotent POST is
        never silently replayed after the request may already have committed
        server-side (would create duplicate records). For POST, only a
        provably pre-send failure (ConnectError/ConnectTimeout — the bytes
        never left) is retried; a ReadTimeout or 5xx is surfaced as-is."""
        idempotent = method.upper() in _IDEMPOTENT_METHODS
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            start = time.monotonic()
            try:
                response = await self._http.request(
                    method, rel_path, params=params, json=json, headers=headers,
                )
            except httpx.RequestError as e:
                duration_ms = (time.monotonic() - start) * 1000
                # A connect-phase failure means the request never reached the
                # server, so even a POST is safe to retry. Any other network
                # error (e.g. ReadTimeout) may have committed the write.
                pre_send = isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout))
                can_retry = idempotent or pre_send
                log.warning(
                    "CDESK %s %s -> network error after %.0fms (attempt %d/%d): %s%s",
                    method, rel_path, duration_ms, attempt, _MAX_ATTEMPTS,
                    type(e).__name__,
                    "" if can_retry else " (non-idempotent, not retried)",
                )
                if can_retry and attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise

            duration_ms = (time.monotonic() - start) * 1000
            status = response.status_code
            log.info("CDESK %s %s -> %d (%.0fms)", method, rel_path, status, duration_ms)

            # 429 is rate-limiting (the request was rejected before processing),
            # so it's safe to retry for any method. 5xx may mean the write
            # landed, so only replay it for idempotent methods.
            retryable = status == 429 or (idempotent and 500 <= status < 600)
            if retryable and attempt < _MAX_ATTEMPTS:
                await asyncio.sleep(self._sleep_seconds(response, attempt))
                continue
            return response

        raise RuntimeError("unreachable: backoff loop must return or raise")

    def _sleep_seconds(self, response: httpx.Response, attempt: int) -> float:
        """Honor Retry-After on both 429 and 5xx; otherwise exponential backoff."""
        if "Retry-After" in response.headers:
            return self._parse_retry_after(response)
        return self._backoff(attempt)

    @staticmethod
    def _backoff(attempt: int) -> float:
        return _BACKOFF_BASE_SECONDS * (2.0 ** (attempt - 1)) + random.uniform(0, 0.25)

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float:
        value = response.headers.get("Retry-After")
        if value is None:
            return _BACKOFF_BASE_SECONDS
        try:
            return max(0.0, float(value))
        except ValueError:
            return _BACKOFF_BASE_SECONDS


def _raise_if_body_signals_error(parsed: Any, http_status: int) -> None:
    """Raise CdeskApiError when the body carries `msg.error[]` non-empty.

    CDESK signals optimistic-lock conflicts, some validation failures, and
    feature-disabled rejections this way (HTTP 2xx + msg.error[] in body).
    Without this check, every tool would silently treat such responses as
    success."""
    if not isinstance(parsed, dict):
        return
    msg = parsed.get("msg")
    if not isinstance(msg, dict):
        return
    errors = msg.get("error")
    if isinstance(errors, list) and errors:
        raise CdeskApiError(parsed, http_status)
