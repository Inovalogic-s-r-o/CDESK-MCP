"""FastMCP server factory and tool registrations.

RELEASE-BRANCH SCOPE: five modules are not registered here.

Three are held back for KNOWN PROBLEMS: the **Knowledge base** has open backend
defects on this tenant (the article type never persists, an omitted visibility
publishes the article, and the whole surface is ungated by
`knowledgebase.enabled` — see docs/bugs.md); the **Schedule / calendar** has no
`sb` filter surface and `get_schedule` / `get_schedule_planning` answer HTTP 500;
and **Approval**'s decide paths cannot be exercised because no request catalog
carries an `approval_rule_id`, so no approval can be spawned to act on (all
verified 2026-07-31).

Two — **Project** and **Work order** — are withheld by RELEASE DECISION, not for
any defect: both were live-verified end to end on 2026-07-31 (full CRUD round
trips; projects found clean, work order's only finding was an undocumented sort
capability). They can be restored from `develop` whenever the release scope
allows, with no backend prerequisite.

All five modules' files are deleted on this branch.

The CNB CMDB export IS included: it only ever runs against the production CNB
tenant, and on any other server it fails with an explicit message naming the
CMDB custom fields that endpoint's column set needs — never with misleading
output.

When CDESK credentials are missing or the startup health probe fails, the
server still registers all 65 tools — backed by stand-in client/cache
stubs that raise a clear, *context-specific* error when called. This way
an MCP client (Claude Desktop, etc.) can call tools/list and see what's
available, and the LLM gets actionable text on its first tool-call
attempt instead of a mysteriously short tool list.

Two distinct degraded states the stubs surface:

* **unconfigured** — required .env vars are missing. The fix is to fill
  in .env and restart.
* **probe_failed** — env vars are set, but the startup probe to CDESK
  threw an exception (network unreachable, bad credentials, module
  disabled, …). The actual exception's type+message is preserved here
  AND exposed via server_info, so the LLM can guide the user without
  the user having to read the server's stderr."""

from __future__ import annotations

import inspect
import logging
from typing import Any, cast

from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from cdesk_mcp import __version__
from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.config import DEFAULT_EVIDENCE_THRESHOLD
from cdesk_mcp.enums import EnumCache
from cdesk_mcp.oauth import (
    CdeskOAuthProvider,
    RequestScopedClient,
    RequestScopedEnumCache,
    make_oauth_resolver,
    register_azure_login_routes,
    register_login_route,
    register_subpath_well_known_routes,
)
from cdesk_mcp.tools._export_cache import ExportSnapshotStore
from cdesk_mcp.tools._helpers import forbid_unknown_arguments
from cdesk_mcp.tools.cmdb import register_cmdb_tools
from cdesk_mcp.tools.cnb import register_cnb_tools
from cdesk_mcp.tools.fill import register_fill_tools
from cdesk_mcp.tools.customers import register_customer_tools
from cdesk_mcp.tools.deals import register_deal_tools
from cdesk_mcp.tools.grounding import build_domains, register_grounding_tools
from cdesk_mcp.tools.request_catalogs import register_request_catalog_tools
from cdesk_mcp.tools.requests import register_request_tools
from cdesk_mcp.tools.tasks import register_task_tools
from cdesk_mcp.tools.users import register_user_tools

log = logging.getLogger(__name__)

# List tools that accept filter parameters, guarded against unknown keyword
# arguments after registration. A misspelled filter is the one silent failure
# that yields WRONG data rather than a no-op: FastMCP drops the unrecognized
# key, CDESK runs unfiltered, and the full set comes back looking filtered.
# The Request module guards its own ten tools at registration; these are the
# rest. Deliberately only the list tools — flipping every tool would reject any
# harmless extra a host attaches, a far wider blast radius than the bug.
_FILTERABLE_LIST_TOOLS: tuple[str, ...] = (
    "list_tasks",
    "list_customers",
    "list_users",
    "list_fills",
    "list_deals",
    "list_configuration_items",
    "list_cmdb_types",
    "list_request_catalogs",
)

_SERVER_INFO_DESC = inspect.cleandoc(
    """
    Return identifying info about this MCP server (name, version, readiness).
    Useful as a liveness check and to confirm which build of cdesk-mcp is running.
    Does NOT verify CDESK API connectivity.
    """
)

# Surfaced to the client model on connect (MCP server `instructions`). It is the
# one place a directive lands in the model's context every session regardless of
# which tool it reaches for — so it's where we steer answers toward the
# truth-gate. It cannot *force* a tool call (the client model decides), but it
# flips the default from "model's discretion" to "ground it unless trivial".
_SERVER_INSTRUCTIONS = inspect.cleandoc(
    """
    This server exposes REAL data from a CDESK helpdesk/CRM tenant: requests
    (tickets), tasks, customers, users. Treat it as the source of truth — the
    user's questions are about their actual data, not hypotheticals.

    GROUND EVERY FACTUAL ANSWER. Whenever the user asks what exists, what is
    true, what they have/need/owe, what someone reported, or what is scheduled
    (e.g. "what do I take to my meeting tomorrow?", "did anyone report X?",
    "which customers are in Y?"), you MUST answer through the grounded-answer
    tools, not from memory or guesswork:
      1. Pick the module the answer lives in — requests, tasks, customers, or
         users. If you can't tell, ask the user, or search several by calling
         collect_records once per module.
      2. Call collect_records(domain, ...) to gather the evidence.
      3. Form claims, each with a record_id + a VERBATIM quote, and call
         verify_claims(domain, ...).
      4. State only what came back confirmed, citing the record id. Never assert
         a fact you did not verify, and never bend a quote's meaning.

    "I found nothing about that" is a correct, expected answer — give it plainly
    rather than inventing a plausible reply. You are an AI and can hallucinate;
    before inferring beyond the verified facts, ask the user whether they want
    facts-only or also inference, and label any inference as such next to the
    record id + quote it rests on. The `truthful_answer` prompt walks through
    this whole procedure.

    Plain list_*/get_* tools are fine for navigation and CRUD, but do NOT answer
    a user's factual question from them without running the verify step above.

    PARAMETER-COLLECTION PROTOCOL — applies to every create_*, update_*,
    delete_*, set_* and other write tool (the ones whose annotations carry
    destructiveHint). They write to the real tenant, so before calling one, walk
    the user through EVERY parameter — required and optional — and collect values:
      - Required: if the user skips one or answers unclearly, ask again. Do NOT
        invent, infer, or default the value.
      - Optional: offer each one as well. If the user declines or says "skip" /
        "leave blank", omit that parameter from the call entirely (CDESK then
        applies its own server-side default).
    Call only once you hold an explicit value or an explicit skip for every
    parameter.

    FIELD-SCOPE PROTOCOL — applies to every single-record get_* tool that takes
    `fieldset` / `fields`. Before such a detail-GET, ask the user how much of the
    record they want, then map their answer on:
      - "base"     → fieldset="base"   (minimal identity fields)
      - "extended" → omit fieldset     (CDESK default; everyday detail)
      - "all"      → fieldset="all"    (every field incl. custom fields)
      - "custom"   → ask WHICH specific fields they want, then pass their exact
        field names via `fields=[...]`. Do not guess names — if unsure, fetch
        fieldset="all" once and show the available keys. Note that
        fieldset="custom" is a different thing: it returns the record's
        custom-field (cfield_*) values, not a user-picked selection.
    If the user doesn't care, or it's an internal lookup, omit both parameters
    (the extended default applies).

    STRIPPED sb_raw CLAUSES. Each list tool naming its LIVE-VERIFIED working
    columns means this section. CDESK silently ignores unknown filter columns and
    returns UNFILTERED data, so this server strips clauses it cannot trust rather
    than let you believe a filter applied. Two shapes come back:
      - Pure-AND filter → the offending leaves are stripped individually and
        listed in `unsupported_filters`; the working remainder still filtered
        server-side. You MUST page through the COMPLETE result set and apply the
        stripped criteria to the items yourself before answering.
      - Any OR connector present → the ENTIRE filter is dropped (partial
        stripping would rewrite the boolean expression) and the list ran
        UNFILTERED. `unsupported_filters` then carries `entire_filter_dropped` +
        `original_filter`; evaluate that whole expression yourself, honoring its
        AND/OR connectors.
    Never report a filtered answer as complete while `unsupported_filters` is
    present. The documented CDQL shorthand is NOT accepted — the live server
    silently ignores it, so always use the structured tree form.
    """
)

_UNCONFIGURED_MSG = (
    "CDESK credentials are not configured for this cdesk-mcp instance. "
    "Set CDESK_BASE_URL, CDESK_LOGIN, and CDESK_PASSWORD in your .env file "
    "(or in the MCP client's env block for this server) and restart the "
    "server. See the README's 'Configure' section for details."
)


def _probe_failed_msg(probe_error: str) -> str:
    """Tool-call error text when the startup probe failed despite .env
    being filled in. Includes the actual exception (type + message) so
    the LLM can relay the real cause rather than echoing the generic
    'credentials not configured' line."""
    return (
        f"cdesk-mcp's startup probe to CDESK failed: {probe_error}. "
        f"The .env credentials are set, but the server couldn't reach "
        f"the tenant or authenticate. Common causes: CDESK_BASE_URL "
        f"unreachable (VPN / firewall / typo), bad CDESK_LOGIN or "
        f"CDESK_PASSWORD, account hit by 2FA (302 redirect), or the "
        f"Tasks / Requests / Customers module disabled in tenant "
        f"settings. Fix the cause and restart the server. The full "
        f"traceback is in the server's stderr."
    )


class _UnconfiguredCdeskClient:
    """Stand-in for CdeskClient when env vars are missing or the startup
    probe failed. Every async method raises a clear instructional error
    so the LLM can guide the user instead of surfacing a
    silently-shortened tool list.

    The constructor takes the error message verbatim so the same stub
    class serves both the 'unconfigured' and 'probe_failed' states with
    different copy."""

    def __init__(self, message: str = _UNCONFIGURED_MSG) -> None:
        self._message = message

    async def get(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(self._message)

    async def post(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(self._message)

    async def put(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(self._message)

    async def delete(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(self._message)

    async def close(self) -> None:
        pass


class _UnconfiguredEnumCache:
    """Stand-in for EnumCache. Sync read methods return empty values
    (so they're safe to call during tool registration / description
    rendering); async lookups raise the same instructional error."""

    bucket_names: list[str] = []
    loaded: bool = False
    is_stale: bool = False
    # Empty, never {"enabled": False} — an unconfigured server must not be
    # mistaken for "the tenant switched this module off".
    settings: dict[str, Any] = {}

    def __init__(self, message: str = _UNCONFIGURED_MSG) -> None:
        self._message = message

    async def load(self) -> None:
        raise RuntimeError(self._message)

    async def refresh(self) -> None:
        raise RuntimeError(self._message)

    async def resolve(
        self,
        bucket: str,
        name: str,
        *,
        parent_id: int | None = None,
        allow_refresh: bool = True,
    ) -> int | None:
        # Treat absent name as "no filter" (matches real EnumCache contract)
        # so tools that don't actually need enum resolution still work
        # against unrelated paths.
        if not name:
            return None
        raise RuntimeError(self._message)

    async def resolve_entry(
        self,
        bucket: str,
        name: str,
        *,
        parent_id: int | None = None,
        allow_refresh: bool = True,
    ) -> Any:
        # Mirror resolve(): absent name → no filter; otherwise the instructional
        # "not configured" error (used by resolve_enum_field_or_raise).
        if not name:
            return None
        raise RuntimeError(self._message)

    def list_names(self, _bucket: str) -> list[str]:
        return []

    def find_candidates(
        self, _bucket: str, _name: str, max_count: int = 5, min_ratio: float = 0.3,
    ) -> list[str]:
        return []

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {}

    def id_name_map(self, _bucket: str) -> dict[Any, str]:
        # collect_records calls this on the task/request domain caches to
        # render readable statuses. Returning {} keeps the degraded-mode
        # instructional error (raised by the client stub) the path the LLM
        # sees, instead of an AttributeError from a missing method.
        return {}

    def action_code_name_map(self, _bucket: str) -> dict[Any, str]:
        # Same as id_name_map, for the requests domain (status_by_action_code).
        return {}


def build_server(
    client: CdeskClient | None = None,
    cache: EnumCache | None = None,
    request_cache: EnumCache | None = None,
    request_catalog_cache: EnumCache | None = None,
    deal_cache: EnumCache | None = None,
    probe_error: str | None = None,
    cache_warnings: dict[str, str] | None = None,
    *,
    oauth_provider: CdeskOAuthProvider | None = None,
    auth_settings: AuthSettings | None = None,
    host: str | None = None,
    port: int | None = None,
    transport_security: TransportSecuritySettings | None = None,
    streamable_http_path: str | None = None,
    trust_forwarded: bool = False,
    evidence_threshold: int = DEFAULT_EVIDENCE_THRESHOLD,
    # Mirrors Config.azure_login_enabled, which is on unless an operator switched
    # it off. __main__ always passes the configured value; this default only
    # matters to other callers, and it must not disagree with the config.
    azure_login_enabled: bool = True,
    service_timeout_seconds: float = 30.0,
) -> FastMCP:
    """Build a FastMCP server instance.

    All client/cache args are optional — when omitted, stand-in stubs
    are used so all 65 tools register regardless. Calling any CDESK-
    dependent tool against the stubs raises a clear error explaining
    *why* the server is degraded:

    * client is None AND probe_error is None → state = "unconfigured"
      (env vars missing). Error text points at .env setup.
    * client is None AND probe_error is set → state = "probe_failed"
      (env vars present, but the startup probe blew up — every cache
      load errored). Error text embeds `probe_error` verbatim so the
      user sees the real cause (e.g. "ConnectError: All connection
      attempts failed", "CDESK login failed (401)") instead of a
      misleading generic "credentials not configured".
    * client is not None → state = "ready", or "degraded" if
      cache_warnings is non-empty (partial probe — some module's enums
      endpoint failed but the rest of the server is usable).

    cache_warnings maps module-key → error string for endpoints that
    failed during startup but didn't take the whole probe down.
    server_info exposes it so the LLM can tell the user *which* module
    is degraded without anyone reading the subprocess's stderr.

    The four caches target different /enums endpoints (task / request
    / request-catalog / deal) so each module can resolve names within its own
    tenant-specific enum space."""
    cache_warnings = cache_warnings or {}
    http_mode = oauth_provider is not None

    if http_mode:
        # Remote OAuth server: per-user credentials, no startup probe / stubs.
        mcp = FastMCP(
            "cdesk-mcp",
            instructions=_SERVER_INSTRUCTIONS,
            auth_server_provider=oauth_provider,
            auth=auth_settings,
            host=host or "127.0.0.1",
            port=port or 8000,
            transport_security=transport_security,
            # Route the endpoint is mounted on. "/" = root hosting, where the
            # bare public URL is the address clients paste; the SDK default and
            # ours is "/mcp". Must match the path in auth_settings'
            # resource_server_url or the token audience won't line up.
            streamable_http_path=streamable_http_path or "/mcp",
        )
    else:
        mcp = FastMCP("cdesk-mcp", instructions=_SERVER_INSTRUCTIONS)

    @mcp.tool(
        description=_SERVER_INFO_DESC,
        annotations=ToolAnnotations(title="Server info", readOnlyHint=True),
    )
    def server_info() -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": "cdesk-mcp",
            "version": __version__,
            "transport": "http" if http_mode else "stdio",
        }
        if http_mode:
            # Auth is per-request; the server itself is always ready.
            info["status"] = "ready"
            info["auth"] = "oauth2"
            return info
        if client is not None:
            info["status"] = "degraded" if cache_warnings else "ready"
        elif probe_error:
            info["status"] = "probe_failed"
            info["probe_error"] = probe_error
        else:
            info["status"] = "unconfigured"
        if cache_warnings:
            info["cache_warnings"] = dict(cache_warnings)
        return info

    if http_mode:
        # All tools and enum caches run against a request-scoped proxy that
        # resolves the calling user's CdeskClient from their OAuth token. Users
        # pick their CDESK server at login, so enum ids are NOT tenant-wide — the
        # caches below are split per server (see RequestScopedEnumCache).
        assert oauth_provider is not None  # narrowed by http_mode
        proxy = cast(
            CdeskClient, RequestScopedClient(make_oauth_resolver(oauth_provider))
        )
        effective_client = proxy
        # Enum ids are per-CDESK-server, and users pick their server at login, so
        # each endpoint's cache is split per server (keyed by the credential's
        # base_url carried in the request's token). See RequestScopedEnumCache.
        from mcp.server.auth.middleware.auth_context import get_access_token

        def _current_base_url() -> str:
            tok = get_access_token()
            cred = getattr(tok, "cred", None) if tok is not None else None
            base_url = getattr(cred, "base_url", "") if cred is not None else ""
            return base_url or oauth_provider.default_base_url

        def _current_cache_key() -> str:
            """Per-caller key for anything memoized across requests.

            NOT the base URL alone: one CDESK host holds many environments
            (``admin_id``), and enums are scoped per environment server-side, so
            a host-only key served one environment's enum ids to another. Adding
            the login can over-partition (same environment, two users → two
            caches) but never under-partition, which is the direction that
            leaks. \\x00 can't occur in either part, so the pair is unambiguous.
            """
            tok = get_access_token()
            cred = getattr(tok, "cred", None) if tok is not None else None
            login = getattr(cred, "login", "") if cred is not None else ""
            return f"{_current_base_url()}\x00{login}"

        def _scoped_cache(endpoint: str) -> EnumCache:
            return cast(
                EnumCache,
                RequestScopedEnumCache(
                    resolve_cache_key=_current_cache_key,
                    make_cache=lambda: EnumCache(proxy, endpoint=endpoint),
                ),
            )

        effective_cache = _scoped_cache("v3/task/enums")
        effective_request_cache = _scoped_cache("v3/request/enums")
        effective_catalog_cache = _scoped_cache("v3/request-catalog/enums")
        effective_deal_cache = _scoped_cache("v3/contract/enums")
        # CNB export snapshots are bucketed per CALLER, same key as the enum
        # caches — a host-only key let one environment's export overwrite
        # another's on the same CDESK server.
        export_store = ExportSnapshotStore(resolve_cache_key=_current_cache_key)
        register_login_route(
            mcp, oauth_provider,
            trust_forwarded=trust_forwarded,
            azure_enabled=azure_login_enabled,
        )
        # "Sign in with Microsoft" (Office365 SSO): 302s the browser into CDESK's
        # Azure SSO, which returns the apitoken directly to the callback. The
        # per-server Azure connector id is discovered at runtime from the public
        # /api/auth/connector list, so this works for any server the user picks.
        # Enabled unless an operator switched it off: the login page reveals the
        # button only for servers whose connector list actually has an azure entry,
        # so mounting it costs nothing on servers that don't.
        if azure_login_enabled:
            register_azure_login_routes(
                mcp, oauth_provider,
                public_url=oauth_provider.public_url,
                timeout_seconds=service_timeout_seconds,
                trust_forwarded=trust_forwarded,
            )
        # When hosted under a path prefix (CDESK_PUBLIC_URL carries a path, e.g.
        # https://host/mcp-server), OAuth clients fetch the protected-resource
        # metadata at the domain root and a root reverse proxy strips the prefix;
        # also serve that metadata at the de-prefixed path so it matches. No-op
        # for root hosting.
        if auth_settings is not None:
            register_subpath_well_known_routes(mcp, auth_settings)
    else:
        # Choose the message that the stubs will raise on every tool call.
        # When probe_error is set, the message embeds the actual exception
        # — that's the whole point of distinguishing the two states.
        stub_message = (
            _probe_failed_msg(probe_error) if probe_error else _UNCONFIGURED_MSG
        )
        effective_client = (
            client if client is not None
            else cast(CdeskClient, _UnconfiguredCdeskClient(stub_message))
        )
        effective_cache = (
            cache if cache is not None
            else cast(EnumCache, _UnconfiguredEnumCache(stub_message))
        )
        effective_request_cache = (
            request_cache if request_cache is not None
            else cast(EnumCache, _UnconfiguredEnumCache(stub_message))
        )
        effective_catalog_cache = (
            request_catalog_cache if request_catalog_cache is not None
            else cast(EnumCache, _UnconfiguredEnumCache(stub_message))
        )
        effective_deal_cache = (
            deal_cache if deal_cache is not None
            else cast(EnumCache, _UnconfiguredEnumCache(stub_message))
        )
        # stdio is single-tenant, so one constant snapshot bucket is enough.
        export_store = ExportSnapshotStore(resolve_cache_key=lambda: "local")

    register_task_tools(mcp, effective_client, effective_cache)
    register_customer_tools(mcp, effective_client)
    register_user_tools(mcp, effective_client)
    register_request_tools(mcp, effective_client, effective_request_cache)
    register_request_catalog_tools(mcp, effective_client, effective_catalog_cache)
    # CMDB is read-only and resolves no enum names client-side, so it takes
    # no cache — the CI status enums are fetched directly by get_ci_enums.
    register_cmdb_tools(mcp, effective_client)
    # Fill (fulfillments) is full CRUD but, like CMDB, resolves no enum names
    # client-side, so it takes no cache — get_fill_enums fetches directly.
    register_fill_tools(mcp, effective_client)
    # Deal statuses/phases/types ARE tenant enum data — 4th cached module.
    register_deal_tools(mcp, effective_client, effective_deal_cache)
    # CNB-only bulk CI export (fails with a clear message on non-CNB servers).
    register_cnb_tools(mcp, effective_client, export_store)

    # Domain-agnostic grounded-answer tools (collect_records / verify_claims /
    # the truthful_answer prompt) span requests, tasks, customers, users. The
    # registry reuses the already-built task + request enum caches; customers
    # and users have no enum endpoint and resolve with cache=None.
    grounding_domains = build_domains(
        effective_client, effective_cache, effective_request_cache
    )
    register_grounding_tools(
        mcp, effective_client, grounding_domains, evidence_threshold
    )

    forbid_unknown_arguments(mcp, _FILTERABLE_LIST_TOOLS)

    return mcp
