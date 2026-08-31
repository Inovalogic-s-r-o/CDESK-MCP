"""Entry point. `uv run cdesk-mcp` lands here via the [project.scripts] mapping.

Async-throughout: one asyncio.run wraps the entire boot so the CdeskClient
lifecycle (and its connection pool) survives across startup probe, enum load,
and tool invocations under server.run_stdio_async()."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit

from mcp.server.auth.settings import (
    AuthSettings,
    ClientRegistrationOptions,
    RevocationOptions,
)
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.config import Config
from cdesk_mcp.enums import EnumCache
from cdesk_mcp.logging_setup import setup_logging
from cdesk_mcp.oauth import CdeskOAuthProvider
from cdesk_mcp.oauth.crypto import TokenCipher
from cdesk_mcp.server import build_server

log = logging.getLogger(__name__)


def main() -> None:
    asyncio.run(_main_async())


async def _main_async() -> None:
    config = Config.from_env()
    setup_logging(config.log_level)
    log.info("cdesk-mcp starting (transport=%s)", config.transport)
    log.info("dotenv: %s", config.dotenv_path or "<not found, using process env>")

    for warning in config.warnings:
        log.warning("%s", warning)

    if config.transport == "http":
        await _run_http(config)
        return

    await _run_stdio(config)


def _build_cipher(config: Config) -> TokenCipher:
    """Build the Fernet cipher that encrypts the stateless OAuth tokens.

    With CDESK_ENCRYPTION_KEY set, use it — sessions then survive restarts and
    span replicas, because any process with the same key can decrypt the tokens
    Claude holds. Without it, fall back to an ephemeral per-process key (a clear
    warning is emitted by Config): tokens become undecryptable on restart, so
    everyone has to reconnect."""
    if config.encryption_key:
        return TokenCipher(config.encryption_key)
    return TokenCipher.generate()


async def _run_http(config: Config) -> None:
    """Remote OAuth-server mode: each user authenticates with their own CDESK
    login at the consent page; tokens are mapped to per-user CdeskClients."""
    missing = config.required_missing()
    if missing:
        log.error(
            "Cannot start in http mode — missing required settings: %s. "
            "CDESK_BASE_URL is the tenant; CDESK_PUBLIC_URL is the externally "
            "reachable HTTPS base URL (e.g. your ngrok URL).",
            ", ".join(missing),
        )
        return

    insecure = config.insecure_http_urls()
    if insecure:
        log.error(
            "Refusing to start in http mode: %s must use https:// (localhost is "
            "exempt). Over http:// the CDESK password/apitoken and OAuth tokens "
            "would traverse the network in cleartext.",
            ", ".join(insecure),
        )
        return

    public_url = config.public_url
    cipher = _build_cipher(config)
    provider = CdeskOAuthProvider(
        base_url=config.default_base_url(),
        public_url=public_url,
        timeout_seconds=config.timeout_seconds,
        cipher=cipher,
        base_url_options=list(config.server_options()),
        allow_custom_base_url=config.allow_custom_base_url,
    )
    auth_settings = AuthSettings(
        issuer_url=AnyHttpUrl(public_url),
        resource_server_url=AnyHttpUrl(config.endpoint_url()),
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=["cdesk"],
            default_scopes=["cdesk"],
        ),
        revocation_options=RevocationOptions(enabled=True),
    )

    # The SDK's localhost default for DNS-rebinding protection rejects the public
    # (ngrok) Host header. Allow the public host explicitly plus the local bind.
    public_host = urlsplit(public_url).netloc
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            public_host,
            f"{config.http_host}:{config.http_port}",
            "127.0.0.1:*",
            "localhost:*",
            *config.allowed_hosts_extra,
        ],
        allowed_origins=[
            public_url,
            "https://claude.ai",
            "https://claude.com",
            *config.allowed_origins_extra,
        ],
    )

    server = build_server(
        oauth_provider=provider,
        auth_settings=auth_settings,
        host=config.http_host,
        port=config.http_port,
        transport_security=transport_security,
        streamable_http_path=config.mcp_route(),
        trust_forwarded=config.trust_forwarded_for,
        evidence_threshold=config.evidence_threshold,
        azure_login_enabled=config.azure_login_enabled,
        # Timeout for the runtime /api/auth/connector discovery call.
        service_timeout_seconds=config.timeout_seconds,
    )
    log.info(
        "cdesk-mcp listening on http://%s:%d%s (clients connect to: %s) — OAuth issuer %s",
        config.http_host, config.http_port, config.mcp_route(),
        config.endpoint_url(), public_url,
    )
    try:
        await server.run_streamable_http_async()
    finally:
        await provider.aclose()


async def _run_stdio(config: Config) -> None:
    client: CdeskClient | None = None
    task_cache: EnumCache | None = None
    request_cache: EnumCache | None = None
    catalog_cache: EnumCache | None = None
    deal_cache: EnumCache | None = None
    probe_error: str | None = None
    cache_warnings: dict[str, str] = {}

    missing = config.required_missing()
    if missing:
        log.warning(
            "Required CDESK env vars missing: %s. The server will still start, but "
            "any tool that calls CDESK will return an auth error until these are set.",
            ", ".join(missing),
        )
    else:
        (
            client, task_cache, request_cache, catalog_cache, deal_cache,
            probe_error, cache_warnings,
        ) = await _startup_probe(config)

    try:
        server = build_server(
            client=client,
            cache=task_cache,
            request_cache=request_cache,
            request_catalog_cache=catalog_cache,
            deal_cache=deal_cache,
            probe_error=probe_error,
            cache_warnings=cache_warnings,
            evidence_threshold=config.evidence_threshold,
        )
        await server.run_stdio_async()
    finally:
        if client is not None:
            await client.close()


async def _startup_probe(
    config: Config,
) -> tuple[
    CdeskClient | None,
    EnumCache | None,
    EnumCache | None,
    EnumCache | None,
    EnumCache | None,
    str | None,
    dict[str, str],
]:
    """Best-effort: build client and load the four enum caches (task,
    request, request-catalog, deal) in parallel using
    ``return_exceptions=True`` so one broken endpoint doesn't take the whole
    server down. The first GET against each endpoint also doubles as an
    auth / connectivity check via the lazy login.

    Returns a 7-tuple:
      (client, task_cache, request_cache, catalog_cache, deal_cache,
       probe_error, cache_warnings)

    Three failure modes:

    * **Total failure** — all caches errored, meaning auth or network is
      broken. Returns (None, None, None, None, None, error_str, {}) so
      build_server reports status='probe_failed' with the actual
      exception in the error message. (The tenant might also be down
      entirely; either way every tool would fail.)
    * **Partial failure** — at least one cache loaded successfully, so
      the client is known to work. The broken cache(s) stay alive but
      unloaded; they'll retry on demand (cache.resolve auto-loads on
      first use). cache_warnings carries the per-module error strings
      so server_info can show the user *which* module is degraded,
      while task / customer / user tools that don't depend on the
      broken endpoint keep working normally. (A tenant with the
      Deal module disabled degrades this way, by design.)
    * **All loaded** — returns the full client+caches set with empty
      cache_warnings."""
    client = CdeskClient.from_env()
    task_cache = EnumCache(client)
    request_cache = EnumCache(client, endpoint="v3/request/enums")
    catalog_cache = EnumCache(client, endpoint="v3/request-catalog/enums")
    deal_cache = EnumCache(client, endpoint="v3/contract/enums")

    results = await asyncio.gather(
        task_cache.load(),
        request_cache.load(),
        catalog_cache.load(),
        deal_cache.load(),
        return_exceptions=True,
    )

    cache_warnings: dict[str, str] = {}
    labels = ("task", "request", "request_catalog", "deal")
    for label, result in zip(labels, results):
        if isinstance(result, BaseException):
            cache_warnings[label] = f"{type(result).__name__}: {result}"

    # All failed → very likely auth/network. Treat as a hard probe
    # failure so the LLM sees one clear error rather than four.
    if len(cache_warnings) == len(labels):
        # The errors usually have the same root cause; surface the
        # first one with its label for context.
        first_label, first_err = next(iter(cache_warnings.items()))
        error_str = f"{first_label} enums: {first_err}"
        log.warning(
            "CDESK startup probe failed (all caches errored — likely auth or "
            "connectivity): %s",
            error_str,
        )
        await client.close()
        return None, None, None, None, None, error_str, {}

    if cache_warnings:
        log.warning(
            "CDESK partial startup — some enum caches failed: %s. The server "
            "is up; tools that don't need these enums work normally, and "
            "name-resolution on affected modules will retry on demand.",
            "; ".join(f"{k}: {v}" for k, v in cache_warnings.items()),
        )
    else:
        log.info(
            "CDESK ready against %s; task buckets: %s; request buckets: %s; "
            "catalog buckets: %s; deal buckets: %s",
            config.base_url,
            task_cache.bucket_names,
            request_cache.bucket_names,
            catalog_cache.bucket_names,
            deal_cache.bucket_names,
        )

    return (
        client, task_cache, request_cache, catalog_cache, deal_cache,
        None, cache_warnings,
    )


if __name__ == "__main__":
    main()
