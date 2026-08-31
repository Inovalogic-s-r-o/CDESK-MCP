"""OAuth 2.1 authorization-server implementation for the remote (http) transport.

The MCP server acts as its own OAuth authorization server so that Claude
(web app / Desktop) can connect as a *custom connector*. Each end user proves
their identity by entering their **CDESK** login + password on our consent page
(`/login`, wired in oauth/login.py); we validate those live against CDESK,
capture the issued apitoken + refresh token, **discard the password**, and bind
the session to that CDESK user.

**Stateless design (no datastore).** Every artifact the OAuth flow needs — the
DCR client registration, the login session, the one-time authorization code, and
the issued access/refresh tokens — is **Fernet-encrypted into the opaque string
the client holds**. There is no Redis and no in-process state map: the server
recovers everything by *decrypting* the string presented on each request. The
per-session CDESK credential (login + apitoken + refresh token) rides inside the
auth code and the access/refresh tokens (the ``cred`` field). Because Fernet uses
one server-wide key with a per-token random IV + HMAC, every token is unique,
opaque to the holder, and tamper-proof; rotating ``CDESK_ENCRYPTION_KEY`` logs
everyone out at once (so it must be set and kept stable).

Sessions therefore survive a restart and span replicas for free — any process
with the same key can decrypt any token. Token lifetimes are enforced storage-
free via Fernet's ``ttl`` (the embedded creation timestamp).

Two properties a self-contained token can't provide alone are backed by small
process-local sets (see oauth/memsets.py), not a datastore:

* **auth-code single-use** — a redeemed code's hash is remembered for its short
  lifetime so it can't be replayed.
* **revocation** — ``revoke_token`` records the grant id so the session stops
  working immediately on this process, and best-effort logs the apitoken out at
  CDESK. (Both sets are lost on restart, which is safe: codes expire on their
  own within minutes, and the CDESK logout is the durable kill.)

PKCE, redirect-uri match, and code expiry are enforced by the SDK's
``TokenHandler`` *before* ``exchange_authorization_code`` runs, so the decrypted
``AuthorizationCode`` carries ``code_challenge`` / ``redirect_uri`` / ``expires_at``
/ ``scopes`` / ``resource`` faithfully from ``AuthorizationParams``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from urllib.parse import urlsplit

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    TokenError,
    construct_redirect_uri,
)
from cryptography.fernet import InvalidToken
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import ValidationError

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.oauth.crypto import TokenCipher
from cdesk_mcp.oauth.memsets import ExpiringKeySet
from cdesk_mcp.oauth.models import (
    CdeskAccessToken,
    CdeskAuthCode,
    CdeskCredential,
    CdeskRefreshToken,
)
from cdesk_mcp.oauth.pool import ClientPool

log = logging.getLogger(__name__)

_AUTH_CODE_TTL_SECONDS = 300  # 5 min — generous for the redirect round-trip
_ACCESS_TOKEN_TTL_SECONDS = 8 * 3600  # 8h working session; refreshed silently after
_LOGIN_SESSION_TTL_SECONDS = 600  # abandon half-finished login flows after 10 min
# The OAuth refresh token lives this long from issue. Self-encoded tokens are
# immutable, so this is a fixed (not sliding) window; capped in practice by the
# CDESK refresh token's own lifetime (renewal fails → user reconnects).
_REFRESH_TOKEN_TTL_SECONDS = 30 * 24 * 3600  # 30 days

# One DNS label: alphanumeric, inner hyphens allowed, no leading/trailing hyphen.
_HOST_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")


_EXAMPLE = "for example cdesk.example.com"


def normalize_base_url_or_reason(raw: str) -> tuple[str | None, str | None]:
    """``(canonical url, None)`` or ``(None, reason)`` — see ``normalize_base_url``.

    Checks **syntax only** — whether the text can be a URL at all. It makes no
    judgement about which servers are reachable, trustworthy or sensible; that is
    the user's choice, and an address that resolves to nothing simply fails at
    login with "Could not reach CDESK".

    ``reason`` is a finished sentence for the person typing into the login page,
    naming the specific problem rather than "invalid URL", so a pasted line break
    and a bad port don't produce the same message. Plain text, no markup — the page
    escapes whatever it renders.
    """
    url = raw.strip()
    if not url:
        return None, f"Enter the address of your CDESK server, {_EXAMPLE}."
    if any(ch.isspace() for ch in url):
        return None, (
            "The address can't contain spaces — check for a stray space or a "
            f"pasted line break, {_EXAMPLE}."
        )
    if url.startswith("//"):
        return None, f"Remove the leading slashes — enter just the address, {_EXAMPLE}."
    if "://" not in url:
        url = f"https://{url}"

    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        return None, (
            "The server address must use the https:// or http:// protocol. Replace "
            "the prefix with https://, or omit it entirely and https:// will be "
            f"applied automatically — {_EXAMPLE}."
        )
    # Userinfo (`https://user:pw@host`) is accepted — it is valid URL syntax and the
    # way to reach a CDESK behind an HTTP Basic-auth proxy. The checks below apply to
    # the host, which urlsplit's `hostname` already gives us without the userinfo.
    try:
        parts.port  # raises ValueError on a non-numeric / out-of-range port
    except ValueError:
        return None, (
            "The port after the colon isn't a valid number. Either correct it or "
            f"leave the colon out entirely — {_EXAMPLE}."
        )
    host = parts.hostname
    if not host:
        return None, (
            "The address is incomplete — it contains only the protocol. Add the "
            f"server name after it, {_EXAMPLE}."
        )
    # An IPv6 literal arrives bracketed in netloc; hostname strips the brackets,
    # so let those through on the ':' that no hostname may contain.
    if ":" not in host:
        labels = host.rstrip(".").split(".")
        if not all(_HOST_LABEL_RE.match(label) for label in labels):
            return None, (
                "The server name contains a character that can't appear in an "
                "address, or has an empty part — check the spelling, "
                f"{_EXAMPLE}."
            )
        # A single-label host ("cdesk", "localhost") is valid syntax — an intranet
        # name or a dev box — so it is accepted. If it doesn't resolve, the login
        # attempt says so ("Could not reach CDESK").

    path = parts.path.rstrip("/")
    # Lowercase the host (case-insensitive) but NOT any userinfo before the '@' —
    # that is a Basic-auth credential and folding its case would silently break it.
    userinfo, at, hostport = parts.netloc.rpartition("@")
    return f"{parts.scheme.lower()}://{userinfo}{at}{hostport.lower()}{path}", None


def normalize_base_url(raw: str) -> str | None:
    """Canonicalize a user-typed CDESK server address, or None if it can't be one.

    Users paste what they type into a browser's address bar — usually bare
    ``cdesk.example.com``. This turns the realistic spellings of one server into a
    single canonical string, so the same tenant is never treated as two, and
    rejects anything that isn't plausibly a server address *before* we send
    someone's password to it.

    ``normalize_base_url_or_reason`` is the same check with a user-facing
    explanation of *why* something was refused; this wrapper is for callers that
    only need the value.

    Rules:
      * surrounding whitespace trimmed; any *inner* whitespace is a rejection
      * no scheme typed → ``https://`` is prepended (detected by the literal
        ``://``, not by urlsplit, because a bare ``host:8080`` parses as a scheme)
      * an explicit ``http://`` is preserved — on-prem installs on a LAN exist
      * userinfo (``user:pw@host``) is kept as typed; it is valid syntax and the way
        to reach a CDESK behind a Basic-auth proxy
      * query and fragment dropped, trailing slashes stripped, scheme and host
        lowercased — a base URL has no use for them and they defeat equality
      * the host must be syntactically valid (well-formed labels, or an IP literal,
        with a valid port if given). A single-label intranet name is fine

    A trailing ``/api`` is deliberately NOT stripped here — ``CdeskClient``
    already normalizes that (it accepts the tenant root with or without it).
    """
    return normalize_base_url_or_reason(raw)[0]


class CdeskOAuthProvider(
    OAuthAuthorizationServerProvider[CdeskAuthCode, CdeskRefreshToken, CdeskAccessToken]
):
    """OAuth AS that federates identity to CDESK using stateless, self-encoded
    tokens — no datastore, only a Fernet cipher plus two small hardening sets."""

    def __init__(
        self,
        *,
        base_url: str,
        public_url: str,
        timeout_seconds: float,
        cipher: TokenCipher,
        base_url_options: list[tuple[str, str]] | None = None,
        allow_custom_base_url: bool = True,
    ) -> None:
        self._base_url = base_url
        self._public_url = public_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._cipher = cipher
        # CDESK server selection (login-page dropdown). Options are (label, url);
        # the first is the default. Falls back to the single base_url when unset.
        self._base_url_options = list(
            base_url_options or ([(urlsplit(base_url).netloc or "CDESK", base_url)] if base_url else [])
        )
        self._allow_custom_base_url = allow_custom_base_url
        # Per-replica LRU of live CdeskClients keyed by access token. Pure
        # in-process performance cache: rebuilt on miss from the credential
        # decrypted out of the access token, so it has no correctness role.
        self._pool = ClientPool()
        # Serializes reconstruction so concurrent requests for the same token
        # don't build duplicate clients.
        self._reconstruct_lock = asyncio.Lock()
        # Process-local hardening (see memsets.py): single-use auth codes and
        # revoked grants. Not a datastore — lost on restart, which is safe.
        self._used_codes = ExpiringKeySet()
        self._revoked_grants = ExpiringKeySet()

    @property
    def public_url(self) -> str:
        """The externally reachable base URL of this server (OAuth issuer); used to
        build absolute callback URLs (e.g. the Office365 SSO redirect_uri)."""
        return self._public_url

    # ----- CDESK server (base_url) selection ----------------------------

    @property
    def base_url_options(self) -> list[tuple[str, str]]:
        """The (label, url) servers offered on the login dropdown (first = default)."""
        return list(self._base_url_options)

    @property
    def allow_custom_base_url(self) -> bool:
        return self._allow_custom_base_url

    @property
    def default_base_url(self) -> str:
        """Pre-selected server URL (first option), or the configured base_url."""
        return self._base_url_options[0][1] if self._base_url_options else self._base_url

    def valid_base_url(self, base_url: str) -> str | None:
        """Resolve a user-submitted server choice to a base_url we'll honor, or None.

        The input is canonicalized first (``normalize_base_url`` — adds a missing
        ``https://``, drops query/fragment/trailing slash, rejects anything that
        isn't plausibly a server), and the configured options are canonicalized the
        same way before comparison. So a user typing ``cdesk.example.com`` matches a
        configured ``https://cdesk.example.com/`` instead of being rejected over a
        scheme or a trailing slash.

        With custom URLs allowed, any address that survives normalization is
        accepted (NO SSRF check — by design; the login page warns the user they own
        that choice). With them disallowed, only the configured options are.

        ``check_base_url`` is the same decision plus the reason it went that way;
        this wrapper is for callers with nothing to show a user."""
        return self.check_base_url(base_url)[0]

    def check_base_url(self, base_url: str) -> tuple[str | None, str | None]:
        """``(canonical url, None)`` if we'll honor it, else ``(None, reason)``.

        The reason is a finished, user-facing sentence naming the actual problem, so
        the login page can say *why* instead of "that server isn't allowed or the
        URL is invalid". Plain text — the page escapes what it renders."""
        url, reason = normalize_base_url_or_reason(base_url or "")
        if url is None:
            return None, reason
        for _label, opt_url in self._base_url_options:
            if normalize_base_url(opt_url) == url:
                return url, None
        if self._allow_custom_base_url:
            return url, None
        # Restricted deployment: name what IS accepted, otherwise the user has no
        # way to find out (the labels aren't shown anywhere on the page).
        listed = sorted({normalize_base_url(u) or u for _label, u in self._base_url_options})
        if listed:
            return None, f"This connector only works with: {', '.join(listed)}."
        return None, "This connector has no CDESK servers configured — ask your administrator."

    def _cred_base_url(self, cred: CdeskCredential) -> str:
        """The server a credential targets, defaulting to the configured default for
        legacy credentials minted before base_url existed."""
        return cred.base_url or self._base_url

    # ----- Dynamic Client Registration (RFC 7591) -----------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        # The client_id IS the encrypted client record (see register_client) — no
        # lookup, just decrypt. A forged/garbage/tampered id fails the HMAC and
        # reads as an unknown client.
        try:
            plain = self._cipher.decrypt(client_id)
        except InvalidToken:
            return None
        try:
            info = OAuthClientInformationFull.model_validate_json(plain)
        except ValidationError:
            return None
        # The encoded blob predates the id assignment, so restore the presented
        # (ciphertext) id as the canonical client_id.
        info.client_id = client_id
        return info

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        # Make the registration self-encoding: replace the SDK-generated random
        # client_id with the ciphertext of the whole client record. The SDK's
        # RegistrationHandler returns this same client_info object to the caller,
        # so the client receives the self-describing id and presents it later;
        # get_client just decrypts it. Nothing is stored.
        client_info.client_id = self._cipher.encrypt(client_info.model_dump_json())

    # ----- Authorization-code grant -------------------------------------

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """Stash the request inside a self-encoded session token and bounce the
        browser to our own CDESK login page.

        Returns a redirect URL (the SDK's authorize handler 302s to it)."""
        session = self._cipher.encrypt(
            json.dumps(
                {"client_id": client.client_id or "", "params": params.model_dump_json()}
            )
        )
        return construct_redirect_uri(f"{self._public_url}/login", session=session)

    def _read_session(self, session: str) -> dict[str, object] | None:
        """Decrypt a login-session token (rejecting it past its TTL). Returns the
        ``{client_id, params}`` payload, or None if invalid/expired."""
        try:
            plain = self._cipher.decrypt(session, ttl_seconds=_LOGIN_SESSION_TTL_SECONDS)
        except InvalidToken:
            return None
        try:
            data: dict[str, object] = json.loads(plain)
        except json.JSONDecodeError:
            return None
        return data

    async def peek_session(self, session_id: str) -> bool:
        """True if the login session token is valid and unexpired (used by /login GET)."""
        return self._read_session(session_id) is not None

    async def login_context(self, session_id: str) -> dict[str, str] | None:
        """Display info for the consent page so the user can see who they're
        authorizing (anti-phishing): the requesting client's name + the redirect
        target host. Returns None if the session token is invalid/expired."""
        data = self._read_session(session_id)
        if data is None:
            return None
        params = AuthorizationParams.model_validate_json(str(data["params"]))
        client = await self.get_client(str(data["client_id"]))
        client_name = (
            (client.client_name if client and client.client_name else None)
            or str(data["client_id"])
            or "Unknown application"
        )
        return {
            "client_name": client_name,
            "redirect_host": urlsplit(str(params.redirect_uri)).netloc,
        }

    async def complete_login(
        self,
        session_id: str,
        *,
        login: str,
        apitoken: str,
        refresh_token: str | None,
        base_url: str,
    ) -> str:
        """Called by the /login POST after CDESK credentials are validated live.

        Mints a one-time authorization code (self-encoded, carrying the CDESK
        credential incl. the chosen server) bound to the requesting client. Returns
        the redirect URL back to the MCP client (carrying code + state). Raises
        KeyError if the session token is invalid/expired, or ValueError if base_url
        is not an allowed/valid server (defense-in-depth — the route validates too)."""
        data = self._read_session(session_id)
        if data is None:
            raise KeyError(session_id)
        resolved = self.valid_base_url(base_url)
        if resolved is None:
            raise ValueError(f"Disallowed CDESK server: {base_url!r}")
        client_id = str(data["client_id"])
        params = AuthorizationParams.model_validate_json(str(data["params"]))

        cred = CdeskCredential(
            login=login,
            apitoken=apitoken,
            cdesk_refresh_token=refresh_token,
            base_url=resolved,
        )
        auth_code = CdeskAuthCode(
            code="",  # set to the ciphertext below (can't self-reference before encrypt)
            scopes=params.scopes or [],
            expires_at=time.time() + _AUTH_CODE_TTL_SECONDS,
            client_id=client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
            cdesk_login=login,
            cred=cred,
        )
        code = self._cipher.encrypt(auth_code.model_dump_json())
        return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> CdeskAuthCode | None:
        try:
            plain = self._cipher.decrypt(
                authorization_code, ttl_seconds=_AUTH_CODE_TTL_SECONDS
            )
        except InvalidToken:
            return None
        try:
            code = CdeskAuthCode.model_validate_json(plain)
        except ValidationError:
            return None
        # Restore the opaque string the client presented (not stored in the
        # ciphertext to avoid self-reference) so the SDK + single-use gate key on it.
        code.code = authorization_code
        if code.expires_at < time.time():
            return None
        if code.client_id != (client.client_id or ""):
            return None
        return code

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: CdeskAuthCode
    ) -> OAuthToken:
        # PKCE / redirect / expiry already verified by the SDK TokenHandler.
        # Single-use gate: a self-encoded code stays decryptable until its TTL, so
        # we remember redeemed codes for their lifetime and reject replays.
        if authorization_code.code in self._used_codes:
            raise TokenError("invalid_grant", "Authorization code already used")
        self._used_codes.add(authorization_code.code, ttl_seconds=_AUTH_CODE_TTL_SECONDS)

        cred = authorization_code.cred
        if cred is None:
            raise TokenError(
                "invalid_grant", "Login session expired; please reconnect the connector."
            )

        grant_id = secrets.token_urlsafe(16)
        access = self._issue_access_token(
            client_id=authorization_code.client_id,
            scopes=authorization_code.scopes,
            resource=authorization_code.resource,
            cred=cred,
            grant_id=grant_id,
        )
        refresh = self._issue_refresh_token(
            client_id=authorization_code.client_id,
            scopes=authorization_code.scopes,
            resource=authorization_code.resource,
            cred=cred,
            grant_id=grant_id,
        )
        await self._pool.put(access, self._build_session_client(cred))
        log.info("Issued access token for CDESK user %r", authorization_code.cdesk_login)
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=_ACCESS_TOKEN_TTL_SECONDS,
            scope=" ".join(authorization_code.scopes) or None,
            refresh_token=refresh,
        )

    async def load_access_token(self, token: str) -> CdeskAccessToken | None:
        try:
            plain = self._cipher.decrypt(token, ttl_seconds=_ACCESS_TOKEN_TTL_SECONDS)
        except InvalidToken:
            return None
        try:
            at = CdeskAccessToken.model_validate_json(plain)
        except ValidationError:
            return None
        at.token = token  # restore the presented bearer string
        if at.expires_at is not None and at.expires_at < time.time():
            return None
        if at.grant_id and at.grant_id in self._revoked_grants:
            await self._pool.pop(token)
            return None
        return at

    # ----- Refresh-token grant ------------------------------------------

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> CdeskRefreshToken | None:
        try:
            plain = self._cipher.decrypt(
                refresh_token, ttl_seconds=_REFRESH_TOKEN_TTL_SECONDS
            )
        except InvalidToken:
            return None
        try:
            rt = CdeskRefreshToken.model_validate_json(plain)
        except ValidationError:
            return None
        rt.token = refresh_token  # restore the presented string
        if rt.grant_id and rt.grant_id in self._revoked_grants:
            return None
        # The SDK checks client_id match, expiry, and scope subset itself.
        return rt

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: CdeskRefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        cred = refresh_token.cred
        if cred is None or (
            refresh_token.grant_id and refresh_token.grant_id in self._revoked_grants
        ):
            raise TokenError(
                "invalid_grant", "CDESK session has ended; please reconnect the connector."
            )
        access = self._issue_access_token(
            client_id=refresh_token.client_id,
            scopes=scopes,
            resource=refresh_token.resource,  # keep the audience binding
            cred=cred,
            grant_id=refresh_token.grant_id,
        )
        await self._pool.put(access, self._build_session_client(cred))
        # Reuse the same (immutable) refresh-token string — no rotation, matching
        # the prior behavior; the credential it carries is unchanged.
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=_ACCESS_TOKEN_TTL_SECONDS,
            scope=" ".join(scopes) or None,
            refresh_token=refresh_token.token,
        )

    async def revoke_token(self, token: CdeskAccessToken | CdeskRefreshToken) -> None:
        # Self-encoded tokens can't be un-issued, so neutralize the whole session
        # by remembering the grant id (rejected everywhere it's checked on this
        # process) and best-effort logging the apitoken out at CDESK — the durable
        # kill that holds across replicas / a restart.
        grant_id = getattr(token, "grant_id", "")
        if grant_id:
            self._revoked_grants.add(grant_id, ttl_seconds=_REFRESH_TOKEN_TTL_SECONDS)
        tok = getattr(token, "token", "")
        if tok:
            await self._pool.pop(tok)
        cred = getattr(token, "cred", None)
        if cred is not None:
            await self._logout_cdesk(cred)

    # ----- Per-request client resolution --------------------------------

    def client_for(self, token: str) -> CdeskClient | None:
        """The pooled CdeskClient for an access token (same-replica hit only).

        Prefer ``get_or_reconstruct_client`` on the request path — it rebuilds
        from the credential carried in the access token on a miss."""
        return self._pool.get(token)

    async def get_or_reconstruct_client(
        self, access_token: AccessToken
    ) -> CdeskClient | None:
        """Resolve the live CdeskClient for an access token, reconstructing it
        from the credential carried inside the (already-decrypted) access token
        on a pool miss.

        This is what lets a tool call succeed after a restart or on a replica that
        didn't handle the login — the credential travels in the token, not a
        store. Returns None if the grant was revoked or the token carries no
        credential (→ the caller surfaces "reconnect")."""
        token = access_token.token
        grant_id = getattr(access_token, "grant_id", "")
        if grant_id and grant_id in self._revoked_grants:
            await self._pool.pop(token)
            return None
        cached = self._pool.get(token)
        if cached is not None:
            return cached
        cred = getattr(access_token, "cred", None)
        if cred is None:
            return None
        async with self._reconstruct_lock:
            cached = self._pool.get(token)  # double-check after acquiring the lock
            if cached is not None:
                return cached
            client = self._build_session_client(cred)
            await self._pool.put(token, client)
            return client

    # ----- token minting -------------------------------------------------

    def _issue_access_token(
        self,
        *,
        client_id: str,
        scopes: list[str],
        resource: str | None,
        cred: CdeskCredential,
        grant_id: str,
    ) -> str:
        """Build a self-encoded access token string carrying the credential."""
        now = int(time.time())
        model = CdeskAccessToken(
            token="",  # filled with the ciphertext on load (avoids self-reference)
            client_id=client_id,
            scopes=scopes,
            expires_at=now + _ACCESS_TOKEN_TTL_SECONDS,
            resource=resource,
            cdesk_login=cred.login,
            grant_id=grant_id,
            cred=cred,
        )
        return self._cipher.encrypt(model.model_dump_json())

    def _issue_refresh_token(
        self,
        *,
        client_id: str,
        scopes: list[str],
        resource: str | None,
        cred: CdeskCredential,
        grant_id: str,
    ) -> str:
        """Build a self-encoded refresh token string carrying the credential."""
        now = int(time.time())
        model = CdeskRefreshToken(
            token="",
            client_id=client_id,
            scopes=scopes,
            expires_at=now + _REFRESH_TOKEN_TTL_SECONDS,
            grant_id=grant_id,
            resource=resource,
            cred=cred,
        )
        return self._cipher.encrypt(model.model_dump_json())

    def _build_session_client(self, cred: CdeskCredential) -> CdeskClient:
        """A pooled, password-free session client for the credential's chosen server.
        The credential is frozen in the token, so there is no write-back: when CDESK
        renews the apitoken the fresh one lives only in this in-process client for the
        token's life; a later reconstruction starts again from the embedded apitoken
        and renews via the embedded (non-rotating) CDESK refresh token as needed."""
        return CdeskClient.from_tokens(
            base_url=self._cred_base_url(cred),
            login=cred.login,
            apitoken=cred.apitoken,
            refresh_token=cred.cdesk_refresh_token,
            timeout_seconds=self._timeout_seconds,
        )

    def build_cdesk_client(self, login: str, password: str, base_url: str) -> CdeskClient:
        """Construct a password-bearing CdeskClient against the chosen CDESK server.

        Used by the /login handler to validate credentials before we issue a token."""
        return CdeskClient(
            base_url=base_url,
            login=login,
            password=password,
            timeout_seconds=self._timeout_seconds,
        )

    def build_token_client(
        self, login: str, apitoken: str, refresh_token: str | None, base_url: str
    ) -> CdeskClient:
        """Construct a *password-free* CdeskClient from an already-issued CDESK
        apitoken (+ optional refresh token) against the given server. On 401 it
        renews via the refresh token."""
        return CdeskClient.from_tokens(
            base_url=base_url,
            login=login,
            apitoken=apitoken,
            refresh_token=refresh_token,
            timeout_seconds=self._timeout_seconds,
        )

    async def aclose(self) -> None:
        """Close every pooled CdeskClient (server shutdown)."""
        await self._pool.aclose()

    # ----- internals ----------------------------------------------------

    async def _logout_cdesk(self, cred: CdeskCredential) -> None:
        """Best-effort: ask CDESK to invalidate the apitoken so revocation holds
        beyond this process (across replicas / a restart)."""
        client = self.build_token_client(
            cred.login, cred.apitoken, cred.cdesk_refresh_token, self._cred_base_url(cred)
        )
        try:
            await client.logout()
        except Exception:  # pragma: no cover - best effort
            log.warning("CDESK logout failed for %r", cred.login, exc_info=True)
        finally:
            await client.close()
