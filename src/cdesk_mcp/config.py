"""Environment configuration. Validation deferred to logs in __main__ via Config.warnings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from dotenv import find_dotenv, load_dotenv

DEFAULT_TIMEOUT_SECONDS = 30.0
# Default minimum number of distinct records that must support a theme before
# the LLM may assert it as a pattern (used by verify_claims). Applies ONLY to
# pattern-kind claims; specific/absence claims ignore it. Overridable per call
# and via CDESK_EVIDENCE_THRESHOLD.
DEFAULT_EVIDENCE_THRESHOLD = 3


DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000
# Path the Streamable-HTTP endpoint is served on. "/mcp" is the MCP ecosystem's
# de-facto default; "" (or "/") mounts it at the origin root instead.
DEFAULT_MCP_PATH = "/mcp"
VALID_TRANSPORTS = ("stdio", "http")


def _normalize_mcp_path(raw: str) -> str:
    """Normalize CDESK_MCP_PATH to "" (root hosting) or "/segment[/segment...]".

    Accepts the value with or without the leading slash and ignores trailing
    slashes, so "mcp", "/mcp" and "/mcp/" are the same endpoint. "" and "/" both
    mean *root*: the whole origin is the endpoint and clients paste the bare URL."""
    path = raw.strip().rstrip("/")
    if not path:
        return ""
    return path if path.startswith("/") else "/" + path


def _split_csv(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated env value into a tuple, trimming + dropping empties."""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_base_urls(raw: str) -> tuple[tuple[str, str], ...]:
    """Parse CDESK_BASE_URLS — a comma-separated list of ``Label=URL`` pairs — into
    ordered (label, url) tuples for the login-page server dropdown. An entry with no
    ``=`` uses its host as the label. Empties are dropped; the first entry is the
    pre-selected default."""
    out: list[tuple[str, str]] = []
    for part in _split_csv(raw):
        label, sep, url = part.partition("=")
        if sep:
            label, url = label.strip(), url.strip()
        else:
            url, label = part.strip(), ""
        if not url:
            continue
        if not label:
            label = urlsplit(url).netloc or url
        out.append((label, url))
    return tuple(out)


@dataclass(frozen=True)
class Config:
    log_level: str
    base_url: str
    login: str
    password: str
    timeout_seconds: float
    evidence_threshold: int
    dotenv_path: str  # empty if no .env file was located
    # Machine-to-machine OAuth credentials (POST /auth/jwttoken). When BOTH are
    # set they take precedence over login+password: the client exchanges them
    # for a short-lived JWT, then Bearer-logs-in for an apitoken. client_id is
    # the CDESK user id the token authenticates as; client_secret is the
    # signing key from that user's OAuth 2.0 connector.
    client_id: int | None = None
    client_secret: str = ""
    # Transport / remote-hosting settings. transport="stdio" (default) keeps the
    # single-tenant local behaviour; transport="http" runs the Streamable-HTTP
    # OAuth server where each user authenticates with their own CDESK login.
    transport: str = "stdio"
    public_url: str = ""  # externally reachable base URL (OAuth issuer); http mode
    # Path the Streamable-HTTP endpoint is mounted on, appended to public_url to
    # form the address clients paste. Normalized to "" (root hosting — the whole
    # origin IS the endpoint, e.g. https://mcp.example.com) or a leading-slash
    # path with no trailing slash (the "/mcp" default). It also determines the
    # OAuth resource/audience and, through it, the RFC 9728 well-known route, so
    # changing it changes the URL clients must use.
    mcp_path: str = DEFAULT_MCP_PATH
    # http-mode CDESK server selection. The user picks which CDESK server their
    # session targets on the login page. base_urls is the ordered (label, url)
    # dropdown from CDESK_BASE_URLS (first = default); when empty the single
    # base_url below is the only option. allow_custom_base_url enables the
    # "Custom…" free-text option (any scheme; no SSRF guard — see docs).
    base_urls: tuple[tuple[str, str], ...] = ()
    allow_custom_base_url: bool = True
    # http-mode "Sign in with Microsoft" (Office365 SSO). When enabled, the login
    # page offers a Microsoft button that delegates to CDESK's Azure SSO (see
    # oauth/azure_login.py). CDESK returns the apitoken directly to the callback,
    # so no m2m service credential is needed; the per-server Azure connector id is
    # discovered at runtime from the public /api/auth/connector list (no config).
    azure_login_enabled: bool = True
    # Fernet key that encrypts the stateless, self-encoded OAuth tokens (http
    # mode). Set it once and keep it STABLE: sessions survive restarts and span
    # replicas only because any process with this key can decrypt the tokens
    # Claude holds; rotating it logs everyone out. Unset → ephemeral per-process
    # key (a warning is emitted) and sessions don't survive a restart.
    encryption_key: str = ""
    http_host: str = DEFAULT_HTTP_HOST
    http_port: int = DEFAULT_HTTP_PORT
    # Extra entries appended to the http transport's DNS-rebinding allowlists —
    # for fronting proxies / additional client origins (comma-separated env).
    allowed_hosts_extra: tuple[str, ...] = ()
    allowed_origins_extra: tuple[str, ...] = ()
    # Honor CF-Connecting-IP / X-Forwarded-For for the login rate limiter only
    # when behind a trusted proxy; otherwise those headers are attacker-spoofable.
    trust_forwarded_for: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> Config:
        path = find_dotenv(usecwd=True)
        if path:
            load_dotenv(path)

        warnings: list[str] = []

        timeout_raw = os.getenv("CDESK_TIMEOUT_SECONDS", "")
        if timeout_raw:
            try:
                timeout = float(timeout_raw)
                if timeout <= 0:
                    warnings.append(
                        f"CDESK_TIMEOUT_SECONDS={timeout_raw!r} must be > 0; "
                        f"defaulting to {DEFAULT_TIMEOUT_SECONDS}"
                    )
                    timeout = DEFAULT_TIMEOUT_SECONDS
            except ValueError:
                warnings.append(
                    f"CDESK_TIMEOUT_SECONDS={timeout_raw!r} is not a number; "
                    f"defaulting to {DEFAULT_TIMEOUT_SECONDS}"
                )
                timeout = DEFAULT_TIMEOUT_SECONDS
        else:
            timeout = DEFAULT_TIMEOUT_SECONDS

        threshold_raw = os.getenv("CDESK_EVIDENCE_THRESHOLD", "")
        evidence_threshold = DEFAULT_EVIDENCE_THRESHOLD
        if threshold_raw:
            try:
                evidence_threshold = int(threshold_raw)
                if evidence_threshold < 1:
                    warnings.append(
                        f"CDESK_EVIDENCE_THRESHOLD={threshold_raw!r} must be >= 1; "
                        f"defaulting to {DEFAULT_EVIDENCE_THRESHOLD}"
                    )
                    evidence_threshold = DEFAULT_EVIDENCE_THRESHOLD
            except ValueError:
                warnings.append(
                    f"CDESK_EVIDENCE_THRESHOLD={threshold_raw!r} is not an integer; "
                    f"defaulting to {DEFAULT_EVIDENCE_THRESHOLD}"
                )
                evidence_threshold = DEFAULT_EVIDENCE_THRESHOLD

        base_url = os.getenv("CDESK_BASE_URL", "")
        if base_url and not base_url.startswith(("http://", "https://")):
            warnings.append(
                f"CDESK_BASE_URL={base_url!r} should start with http:// or https://; "
                "requests will likely fail"
            )

        transport = os.getenv("CDESK_TRANSPORT", "stdio").strip().lower() or "stdio"
        if transport not in VALID_TRANSPORTS:
            warnings.append(
                f"CDESK_TRANSPORT={transport!r} is not one of {VALID_TRANSPORTS}; "
                "defaulting to 'stdio'"
            )
            transport = "stdio"

        public_url = os.getenv("CDESK_PUBLIC_URL", "").rstrip("/")
        if transport == "http" and not public_url:
            warnings.append(
                "CDESK_TRANSPORT=http but CDESK_PUBLIC_URL is unset. It must be the "
                "externally reachable HTTPS base URL (e.g. your ngrok URL) — it is "
                "used as the OAuth issuer and the clients' redirect target."
            )
        if public_url and not public_url.startswith(("http://", "https://")):
            warnings.append(
                f"CDESK_PUBLIC_URL={public_url!r} should start with http:// or https://"
            )

        mcp_path_raw = os.getenv("CDESK_MCP_PATH")
        mcp_path = (
            DEFAULT_MCP_PATH if mcp_path_raw is None
            else _normalize_mcp_path(mcp_path_raw)
        )
        if any(ch in mcp_path for ch in "?# "):
            warnings.append(
                f"CDESK_MCP_PATH={mcp_path_raw!r} contains a query, fragment or space; "
                f"it must be a plain path — defaulting to {DEFAULT_MCP_PATH!r}"
            )
            mcp_path = DEFAULT_MCP_PATH

        base_urls = _parse_base_urls(os.getenv("CDESK_BASE_URLS", ""))
        allow_custom_base_url = os.getenv(
            "CDESK_ALLOW_CUSTOM_BASE_URL", "true"
        ).strip().lower() in ("1", "true", "yes", "on")

        encryption_key = os.getenv("CDESK_ENCRYPTION_KEY", "").strip()
        if transport == "http" and not encryption_key:
            warnings.append(
                "CDESK_TRANSPORT=http but CDESK_ENCRYPTION_KEY is unset — OAuth tokens "
                "will be signed with an ephemeral per-process key, so every session is "
                "lost on restart and won't span replicas. Set a stable "
                "CDESK_ENCRYPTION_KEY (generate one with: python -c \"from "
                "cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\")."
            )

        allowed_hosts_extra = _split_csv(os.getenv("CDESK_ALLOWED_HOSTS", ""))
        allowed_origins_extra = _split_csv(os.getenv("CDESK_ALLOWED_ORIGINS", ""))
        trust_forwarded_for = os.getenv("CDESK_TRUST_PROXY", "").strip().lower() in (
            "1", "true", "yes", "on",
        )

        client_id_raw = os.getenv("CDESK_CLIENT_ID", "").strip()
        client_id: int | None = None
        if client_id_raw:
            try:
                client_id = int(client_id_raw)
            except ValueError:
                warnings.append(
                    f"CDESK_CLIENT_ID={client_id_raw!r} is not an integer (it is "
                    "the CDESK user id); machine-to-machine auth is disabled"
                )
        client_secret = os.getenv("CDESK_CLIENT_SECRET", "").strip()
        if (client_id is None) != (not client_secret):
            warnings.append(
                "CDESK_CLIENT_ID and CDESK_CLIENT_SECRET must BOTH be set for "
                "machine-to-machine auth; ignoring the partial pair and falling "
                "back to CDESK_LOGIN/CDESK_PASSWORD"
            )
            client_id = None
            client_secret = ""

        # On by default: SSO needs no extra config (the per-server Azure connector
        # id is discovered at runtime from the public /api/auth/connector list) and
        # the login page only offers the button for servers that actually report an
        # azure connector, so there is nothing to opt into. The env var remains as
        # an explicit kill switch for an operator who wants password-only.
        azure_login_enabled = os.getenv("CDESK_AZURE_LOGIN_ENABLED", "").strip().lower() not in (
            "0", "false", "no", "off",
        )

        http_host = os.getenv("CDESK_HTTP_HOST", "").strip() or DEFAULT_HTTP_HOST
        port_raw = os.getenv("CDESK_HTTP_PORT", "")
        http_port = DEFAULT_HTTP_PORT
        if port_raw:
            try:
                http_port = int(port_raw)
            except ValueError:
                warnings.append(
                    f"CDESK_HTTP_PORT={port_raw!r} is not an integer; "
                    f"defaulting to {DEFAULT_HTTP_PORT}"
                )

        return cls(
            log_level=os.getenv("CDESK_LOG_LEVEL", "INFO"),
            base_url=base_url,
            login=os.getenv("CDESK_LOGIN", ""),
            password=os.getenv("CDESK_PASSWORD", ""),
            timeout_seconds=timeout,
            evidence_threshold=evidence_threshold,
            dotenv_path=path,
            client_id=client_id,
            client_secret=client_secret,
            transport=transport,
            public_url=public_url,
            mcp_path=mcp_path,
            base_urls=base_urls,
            allow_custom_base_url=allow_custom_base_url,
            azure_login_enabled=azure_login_enabled,
            encryption_key=encryption_key,
            http_host=http_host,
            http_port=http_port,
            allowed_hosts_extra=allowed_hosts_extra,
            allowed_origins_extra=allowed_origins_extra,
            trust_forwarded_for=trust_forwarded_for,
            warnings=tuple(warnings),
        )

    def required_missing(self) -> tuple[str, ...]:
        """Names of required env vars that are unset/empty. Empty tuple == all present.

        In http (remote OAuth) mode CDESK_LOGIN / CDESK_PASSWORD are NOT required —
        each user supplies their own credentials at the OAuth login screen — but
        CDESK_BASE_URL and CDESK_PUBLIC_URL are.

        In stdio mode the credentials requirement is satisfied by EITHER the
        m2m pair (CDESK_CLIENT_ID + CDESK_CLIENT_SECRET) or login+password."""
        missing: list[str] = []
        if not self.base_url:
            missing.append("CDESK_BASE_URL")
        if self.transport == "http":
            if not self.public_url:
                missing.append("CDESK_PUBLIC_URL")
            # CDESK_ENCRYPTION_KEY is NOT required: without it the server still
            # starts with an ephemeral key (sessions just don't survive a
            # restart). A warning is emitted from from_env() in that case.
            return tuple(missing)
        if self.client_id is not None and self.client_secret:
            # m2m auth configured — login/password are optional (the identity
            # comes from the JWT minted for client_id).
            return tuple(missing)
        if not self.login:
            missing.append("CDESK_LOGIN")
        if not self.password:
            missing.append("CDESK_PASSWORD")
        return tuple(missing)

    def server_options(self) -> tuple[tuple[str, str], ...]:
        """The (label, url) CDESK servers offered on the login dropdown in http
        mode: CDESK_BASE_URLS when set, else the single CDESK_BASE_URL (labelled).
        The first entry is the pre-selected default."""
        if self.base_urls:
            return self.base_urls
        if self.base_url:
            return ((urlsplit(self.base_url).netloc or "CDESK", self.base_url),)
        return ()

    def default_base_url(self) -> str:
        """The pre-selected server URL (first option), or CDESK_BASE_URL."""
        options = self.server_options()
        return options[0][1] if options else self.base_url

    @staticmethod
    def _is_localhost(url: str) -> bool:
        return (urlsplit(url).hostname or "") in ("localhost", "127.0.0.1", "::1")

    def mcp_route(self) -> str:
        """The Starlette route the endpoint is mounted on. Same as mcp_path except
        for root hosting, where the route must be "/" rather than an empty string."""
        return self.mcp_path or "/"

    def endpoint_url(self) -> str:
        """The full address clients paste into their AI app — public_url + mcp_path.
        Also the OAuth resource (token audience), so the two can never drift."""
        return f"{self.public_url}{self.mcp_path}"

    def insecure_http_urls(self) -> tuple[str, ...]:
        """In http mode, the names of URL settings that use plaintext http://
        (so credentials/tokens would traverse the network in the clear).
        localhost is exempt so local testing still works. Empty tuple == OK.

        Covers the server's own URLs plus the *configured* CDESK servers
        (CDESK_BASE_URL + CDESK_BASE_URLS). Custom per-session URLs the user pastes
        at login are NOT checked here — they're a runtime choice the user owns."""
        if self.transport != "http":
            return ()
        bad: list[str] = []
        for name, url in (
            ("CDESK_BASE_URL", self.base_url),
            ("CDESK_PUBLIC_URL", self.public_url),
        ):
            if url.startswith("http://") and not self._is_localhost(url):
                bad.append(name)
        for label, url in self.base_urls:
            if url.startswith("http://") and not self._is_localhost(url):
                bad.append(f"CDESK_BASE_URLS[{label}]")
        return tuple(bad)
