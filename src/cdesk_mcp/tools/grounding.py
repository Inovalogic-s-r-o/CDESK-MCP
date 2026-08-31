"""Grounded ("100% truth") answering, generalized across CDESK modules.

The truth-gate that used to live only in the Request module now works for any
domain the API can fetch — requests, tasks, customers, users, fills. Two tools
plus a prompt implement the grounding contract:

- ``collect_records(domain, ...)`` — page through ALL records of a domain that
  match a filter and return a compact, id-tagged, HTML-stripped corpus. No
  analysis; just the raw evidence the LLM reads before forming claims.
- ``verify_claims(domain, claims)`` — the truth-gate. Re-fetch each cited
  record, confirm the quote is a literal substring of it (or, for requests, its
  discussion thread), recount distinct supporters, apply the pattern threshold,
  and re-run ``absence`` searches server-side.
- ``truthful_answer`` prompt — ties the two together and encodes the grounding
  contract, including the FIRST step of asking the user which module holds the
  information.

A :class:`GroundingDomain` registry (see :func:`build_domains`) carries the
per-domain specifics — endpoints, which record fields hold the human text, the
status enum bucket, the optional discussion thread, the enum cache, and a filter
resolver — so the engine below stays domain-agnostic.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.config import DEFAULT_EVIDENCE_THRESHOLD
from cdesk_mcp.enums import EnumCache
from cdesk_mcp.filters import (
    USER_TYPE_LABELS,
    build_customer_filter,
    build_fill_filter,
    build_request_filter,
    build_task_filter,
    build_user_filter,
    encode_sb,
    ensure_fill_start_date_window,
)
from cdesk_mcp.tools._helpers import (
    resolve_enum_field_or_raise,
    resolve_enum_or_raise,
    to_llm_error,
    unwrap_list,
    unwrap_record,
)
from cdesk_mcp.tools._text import (
    MIN_QUOTE_CHARS,
    normalize_for_display,
    normalize_for_match,
)

# Upper bound on how many records collect_records will page through in one call,
# to keep the corpus within an LLM's context budget. Hitting it sets
# coverage.truncated=True so the answer can disclose partial coverage.
_MAX_COLLECT_RECORDS = 1000

# Pagination upper bound — mirrors the per-module list tools' 100 cap to keep
# response sizes within an LLM's context budget. CDESK accepts larger; defensive.
_MAX_PER_PAGE = 100

# Max concurrent CDESK fetches within one tool call. Discussion threads (collect)
# and claims (verify) are fetched in parallel under these caps so a broad query
# doesn't serialize into a client timeout, while still bounding load on CDESK.
_DISCUSSION_CONCURRENCY = 8
_VERIFY_CONCURRENCY = 8

_VALID_CLAIM_KINDS = ("specific", "pattern", "absence")

# (cache | None, filters_dict) -> sb dict | None. Does enum name→id resolution
# (where the domain has an enum cache) then calls the matching build_*_filter.
FilterResolver = Callable[
    [EnumCache | None, dict[str, Any]], Awaitable[dict[str, Any] | None]
]


# --- domain registry ----------------------------------------------------

@dataclass(frozen=True)
class DiscussionSpec:
    """A domain's threaded sub-resource (only requests have one)."""

    path_template: str            # "v3/request/{id}/discussion"
    text_fields: tuple[str, ...]  # per-message text fields, display priority


@dataclass(frozen=True)
class GroundingDomain:
    """Everything the grounding engine needs to collect + verify one domain."""

    name: str                          # "requests" | "tasks" | "customers" | "users"
    list_endpoint: str                 # "v3/request"
    detail_template: str               # "v3/request/{id}"
    id_fields: tuple[str, ...]         # first present key wins
    num_fields: tuple[str, ...]        # () -> num=None
    title_fields: tuple[str, ...]
    body_fields: tuple[str, ...]
    status_fields: tuple[str, ...]     # () -> no status (e.g. users)
    created_fields: tuple[str, ...]
    status_bucket: str | None          # enum bucket for id→name display; None if no enum
    cache: EnumCache | None            # task/request enum cache; None for customers/users
    discussion: DiscussionSpec | None  # None when the domain has no thread
    resolve_filter: FilterResolver
    allowed_filter_keys: frozenset[str]
    filter_help: str
    # Requests store status by enum action_code (not id); set True so status
    # display uses action_code_name_map instead of id_name_map. Tasks (which
    # carry the enum id) leave this False.
    status_by_action_code: bool = False


# --- per-domain filter resolvers ----------------------------------------
#
# Each resolves any enum *_name keys to ids (where the domain has a cache), then
# calls the matching pure builder from filters.py. Mirrors the resolution that
# the per-module list_* tools already do, so collect/verify and list behave
# identically. Customers/users have no enum cache — their resolvers never touch
# `cache` and read no *_name keys.

async def _resolve_request_filter(
    cache: EnumCache | None, f: dict[str, Any]
) -> dict[str, Any] | None:
    if cache is None:
        # Survives `python -O` (unlike assert) and gives a clear cause.
        raise RuntimeError("requests domain is missing its enum cache")
    # Requests filter status/priority by enum ACTION_CODE, not id (see
    # build_request_filter / resolve_enum_field_or_raise).
    status_id = await resolve_enum_field_or_raise(
        cache, "status", f.get("status_name"), kind="status", field="action_code"
    )
    priority_id = await resolve_enum_field_or_raise(
        cache, "priority", f.get("priority_name"), kind="priority", field="action_code"
    )
    place_id = await resolve_enum_or_raise(cache, "place", f.get("place_name"), kind="place")
    # Category type: sb `type` filters cat_type_id (backend JCD-32964); resolve
    # the NAME → id. base_type is the raw base-type letter ('H'/'C'/…).
    cat_type_id = await resolve_enum_or_raise(
        cache, "cat_type_id", f.get("cat_type_name"), kind="cat_type"
    )
    return build_request_filter(
        text_search=f.get("text_search"),
        status_id=status_id,
        priority_id=priority_id,
        cat_type_id=cat_type_id,
        base_type=f.get("base_type"),
        # `company` mirrors the list_requests/create_request param name; accept
        # the legacy `customer_id` key too as a transitional alias.
        customer_id=f.get("company", f.get("customer_id")),
        solver_id=f.get("solver_id"),
        solver_group_id=f.get("solver_group_id"),
        catalog_id=f.get("catalog_id"),
        deal_id=f.get("deal_id"),
        project_contract_id=f.get("project_contract_id"),
        branch_id=f.get("branch_id"),
        place_id=place_id,
        superior_request_id=f.get("superior_request_id"),
        created_after=f.get("created_after"),
        created_before=f.get("created_before"),
        sb_raw=f.get("sb_raw"),
    )


async def _resolve_task_filter(
    cache: EnumCache | None, f: dict[str, Any]
) -> dict[str, Any] | None:
    if cache is None:
        # Survives `python -O` (unlike assert) and gives a clear cause.
        raise RuntimeError("tasks domain is missing its enum cache")
    status_id = await resolve_enum_or_raise(cache, "status", f.get("status_name"), kind="status")
    type_id = await resolve_enum_or_raise(cache, "type", f.get("type_name"), kind="type")
    # NB: only_open/only_late remain removed — the sb TRUE/FALSE path can't
    # reach those whereFilter branches (string 'TRUE' vs boolean true), which
    # would poison absence verdicts with a false sense of scope.
    # created_after/before are back: they work via the whereFilter key
    # `created_at` with W3C-normalized values (build_task_filter handles the
    # normalization; verified live 2026-06-04).
    return build_task_filter(
        text_search=f.get("text_search"),
        status_id=status_id,
        type_id=type_id,
        solver_id=f.get("solver_id"),
        customer_id=f.get("customer_id"),
        valid_from_after=f.get("valid_from_after"),
        valid_from_before=f.get("valid_from_before"),
        deadline_after=f.get("deadline_after"),
        deadline_before=f.get("deadline_before"),
        created_after=f.get("created_after"),
        created_before=f.get("created_before"),
        sb_raw=f.get("sb_raw"),
    )


async def _resolve_customer_filter(
    cache: EnumCache | None, f: dict[str, Any]
) -> dict[str, Any] | None:
    # No enum cache — `cache` is None and never used. `status` is a raw
    # single-char flag (e.g. "A"); the customer sb honors `status` (verified).
    return build_customer_filter(
        text_search=f.get("text_search"),
        status=f.get("status"),
        sb_raw=f.get("sb_raw"),
    )


async def _resolve_user_filter(
    cache: EnumCache | None, f: dict[str, Any]
) -> dict[str, Any] | None:
    # No enum cache — `cache` is None and never used.
    user_type = f.get("user_type")
    if user_type is not None and user_type not in USER_TYPE_LABELS:
        # Mirror list_users' client-side guard so a bogus type can't silently
        # match zero rows and turn an absence claim into a false confirmed_absence.
        raise ValueError(
            f"user_type={user_type!r} is not a valid CDESK user type. "
            f"Allowed values: {', '.join(USER_TYPE_LABELS)}."
        )
    # NB: no only_deleted — the endpoint only honors it as a FLAT query param
    # (onlyDeleted=1), which the grounding engine's sb-only list call can't
    # carry; via sb it silently matched ACTIVE users (verified live), which
    # would poison absence verdicts. Removed rather than mislead.
    return build_user_filter(
        text_search=f.get("text_search"),
        company_id=f.get("company_id"),
        user_type=user_type,
        status=f.get("status"),
        sb_raw=f.get("sb_raw"),
    )


async def _resolve_fill_filter(
    cache: EnumCache | None, f: dict[str, Any]
) -> dict[str, Any] | None:
    # No enum cache — fills resolve no enum names. CRITICAL for absence
    # correctness: the v3 fill list defaults to the CURRENT CALENDAR MONTH when
    # no date window is sent, so an unbounded collect would silently miss older
    # fills and could turn a real record into a false confirmed_absence.
    #
    # sb_raw bypasses the typed worked_from/worked_to window (the two can't be
    # combined — build_fill_filter rejects it), so a raw filter that doesn't
    # itself bound start_date would still scan only the current month. AND-in a
    # wide start_date window when it doesn't (ensure_fill_start_date_window). The
    # typed path injects the same wide window when the caller gives no bound.
    sb_raw = f.get("sb_raw")
    if sb_raw is not None:
        sb_raw = ensure_fill_start_date_window(sb_raw)
    has_window = f.get("worked_from") or f.get("worked_to") or sb_raw
    return build_fill_filter(
        text_search=f.get("text_search"),
        solver_id=f.get("solver_id"),
        company_id=f.get("company_id"),
        request_id=f.get("request_id"),
        deal_id=f.get("deal_id"),
        project_contract_id=f.get("project_contract_id"),
        assign_id=f.get("assign_id"),
        invoiced=f.get("invoiced"),
        worked_from=f.get("worked_from") or (None if has_window else "2000-01-01"),
        worked_to=f.get("worked_to") or (None if has_window else "2099-12-31"),
        sb_raw=sb_raw,
    )


# Exactly the keys each resolver reads — anything else is rejected by
# _validate_filters before any enum/HTTP work, which is what cleanly stops a
# status_name/type_name slipping into a customer/user search.
_REQUEST_FILTER_KEYS = frozenset({
    "text_search", "status_name", "priority_name", "place_name",
    "cat_type_name", "base_type",
    # `company` matches the list_requests param; `customer_id` kept as a
    # transitional alias (see _resolve_request_filter).
    "company", "customer_id",
    "solver_id", "solver_group_id", "catalog_id", "deal_id",
    "project_contract_id", "branch_id", "superior_request_id",
    "created_after", "created_before", "sb_raw",
})
_TASK_FILTER_KEYS = frozenset({
    "text_search", "status_name", "type_name", "solver_id", "customer_id",
    "valid_from_after", "valid_from_before",
    "deadline_after", "deadline_before",
    "created_after", "created_before", "sb_raw",
})
_CUSTOMER_FILTER_KEYS = frozenset({"text_search", "status", "sb_raw"})
_USER_FILTER_KEYS = frozenset({
    "text_search", "company_id", "user_type", "status", "sb_raw",
})
_FILL_FILTER_KEYS = frozenset({
    "text_search", "solver_id", "company_id", "request_id", "deal_id",
    "project_contract_id", "assign_id", "invoiced", "worked_from", "worked_to",
    "sb_raw",
})

_REQUEST_FILTER_HELP = (
    "requests: text_search, status_name, priority_name, place_name, "
    "cat_type_name (category type, resolved to cat_type_id), base_type "
    "(base-type letter e.g. 'H'), company, solver_id, solver_group_id, "
    "catalog_id, deal_id, project_contract_id, branch_id, "
    "superior_request_id, created_after/before, sb_raw "
    "(sla/due-date/deleted filters are not honored by the v3 request list)"
)
_TASK_FILTER_HELP = (
    "tasks: text_search, status_name, type_name, solver_id, customer_id, "
    "valid_from_after/before, deadline_after/before, created_after/before, "
    "sb_raw (open/late filters are not honored by the v3 task list)"
)
_CUSTOMER_FILTER_HELP = "customers: text_search, status, sb_raw"
_USER_FILTER_HELP = "users: text_search, company_id, user_type, status, sb_raw"
_FILL_FILTER_HELP = (
    "fills: text_search, solver_id, company_id, request_id, deal_id, "
    "project_contract_id, assign_id, invoiced, worked_from/worked_to, sb_raw "
    "(without worked_from/worked_to the full history is scanned automatically; "
    "the live list otherwise defaults to the current calendar month)"
)


def build_domains(
    client: CdeskClient,
    task_cache: EnumCache,
    request_cache: EnumCache,
) -> dict[str, GroundingDomain]:
    """Build the grounding domain registry from the live enum caches.

    `client` is accepted for symmetry / future per-domain wiring; the engine is
    handed the client explicitly at call time. Customers and users carry
    cache=None — they have no /enums endpoint."""
    return {
        "requests": GroundingDomain(
            name="requests",
            list_endpoint="v3/request",
            detail_template="v3/request/{id}",
            id_fields=("id_req", "id"),
            # The v3 request record's human number is `number`; `req_num` is
            # absent on this API (kept as a defensive fallback).
            num_fields=("number", "req_num", "num"),
            title_fields=("req_title", "title", "name"),
            body_fields=(
                "req_text", "description", "solution", "solver_note",
                "first_conclusion", "note", "computer_code", "code",
            ),
            status_fields=("req_status", "status"),
            created_fields=("idatetime", "created_at"),
            status_bucket="status",
            cache=request_cache,
            discussion=DiscussionSpec(
                "v3/request/{id}/discussion",
                # The discussion GET returns each message body in `post_text`
                # (with a `type` field marking the channel) — NOT the
                # customer_text/technote_text keys the POST path writes to. Read
                # post_text first so a quote copied from the surfaced thread
                # verifies; the others are kept as defensive fallbacks for any
                # alternate response shape.
                ("post_text", "customer_text", "technote_text", "text"),
            ),
            resolve_filter=_resolve_request_filter,
            allowed_filter_keys=_REQUEST_FILTER_KEYS,
            filter_help=_REQUEST_FILTER_HELP,
            status_by_action_code=True,
        ),
        "tasks": GroundingDomain(
            name="tasks",
            list_endpoint="v3/task",
            detail_template="v3/task/{id}",
            id_fields=("id",),
            num_fields=("num",),
            title_fields=("name",),
            body_fields=("description",),
            status_fields=("status",),
            created_fields=("created_at",),
            status_bucket="status",
            cache=task_cache,
            discussion=None,
            resolve_filter=_resolve_task_filter,
            allowed_filter_keys=_TASK_FILTER_KEYS,
            filter_help=_TASK_FILTER_HELP,
        ),
        "customers": GroundingDomain(
            name="customers",
            list_endpoint="v3/company",
            detail_template="v3/company/{id}",
            id_fields=("id",),
            num_fields=("code",),
            title_fields=("name", "alias"),
            body_fields=(
                "code", "city", "street", "zip", "email_general", "phone_general",
            ),
            status_fields=("status",),
            created_fields=("created_at",),
            status_bucket=None,
            cache=None,
            discussion=None,
            resolve_filter=_resolve_customer_filter,
            allowed_filter_keys=_CUSTOMER_FILTER_KEYS,
            filter_help=_CUSTOMER_FILTER_HELP,
        ),
        "users": GroundingDomain(
            name="users",
            list_endpoint="v3/user",
            detail_template="v3/user/{id}",
            id_fields=("id",),
            # CDESK exposes the login as "user" on the record (find_user reads
            # r.get("login") or r.get("user")); keep both so either shape works.
            num_fields=("user", "login", "nick"),
            title_fields=("name",),
            body_fields=(
                # "mobil" is CDESK's cz spelling (see _USER_CREATE_FIELDS); "user"
                # is the wire login. Reading "mobile"/"login" alone misses them.
                "nick", "user", "login", "email", "phone", "mobil", "description",
            ),
            status_fields=(),
            created_fields=("created_at",),
            status_bucket=None,
            cache=None,
            discussion=None,
            resolve_filter=_resolve_user_filter,
            allowed_filter_keys=_USER_FILTER_KEYS,
            filter_help=_USER_FILTER_HELP,
        ),
        "fills": GroundingDomain(
            name="fills",
            list_endpoint="v3/fill",
            detail_template="v3/fill/{id}",
            # PK is `id` (the record's `fill_id` is a self/group ref and is null
            # on a plain fill — verified live 2026-06-24).
            id_fields=("id", "fill_id"),
            num_fields=("num",),
            title_fields=("description", "name"),
            body_fields=(
                "description", "private_description", "used_material",
                "task_name", "task_num",
            ),
            # Fills carry no enum status (invoiced/signature are flags), so no
            # status display.
            status_fields=(),
            created_fields=("created_at", "valid_from"),
            status_bucket=None,
            cache=None,
            discussion=None,
            resolve_filter=_resolve_fill_filter,
            allowed_filter_keys=_FILL_FILTER_KEYS,
            filter_help=_FILL_FILTER_HELP,
        ),
    }


_VALID_DOMAINS = ("requests", "tasks", "customers", "users", "fills")


def _require_domain(domains: dict[str, GroundingDomain], name: Any) -> GroundingDomain:
    dom = domains.get(name) if isinstance(name, str) else None
    if dom is None:
        raise ValueError(
            f"unknown domain {name!r}; valid domains: {list(_VALID_DOMAINS)}"
        )
    return dom


def _validate_filters(domain: GroundingDomain, filters: Any) -> dict[str, Any]:
    """Ensure `filters` is a dict whose keys are all valid for this domain.
    Rejecting unknown keys up front is what keeps a status_name/type_name from
    ever reaching a domain (customers/users) that has no enum to resolve it."""
    if not isinstance(filters, dict):
        raise ValueError("filters must be an object/dict of filter keys")
    unknown = set(filters) - domain.allowed_filter_keys
    if unknown:
        raise ValueError(
            f"unknown filter keys for domain {domain.name!r}: {sorted(unknown)}. "
            f"Valid keys: {sorted(domain.allowed_filter_keys)}"
        )
    return filters


# CDESK sb columns that DON'T scope a search to a topic: the filter builders'
# default "exclude soft-deleted" clause and the open/late/deleted bucket toggles.
_NON_SCOPING_COLS = frozenset({"deleted_at", "open", "late", "onlyDeleted"})


def _sb_is_scoped(sb: dict[str, Any]) -> bool:
    """Does the RESOLVED CDESK sb filter carry at least one clause that scopes
    the search to a topic — more than the default exclude-deleted clause, a
    bucket toggle, or an empty text search?

    Inspecting the resolved sb (not the raw absence_query) is what stops a
    match-all sb_raw like {"q": ""} from passing as 'scoped': the requests
    builder always emits a default 'exclude deleted' clause, so a non-None sb
    alone never proves the query was scoped. A leaf scopes when it targets a
    real column (not in _NON_SCOPING_COLS) or carries a non-empty text `q`.

    Recurses into nested `group` nodes: the structured sb_raw form documented
    to the LLM allows them (e.g. an OR of two columns needs a sub-group), so a
    genuinely scoped query nested one level down must not read as scopeless."""
    if not isinstance(sb, dict):
        return False
    # Primary-text form: a top-level non-empty q scopes the search.
    top_q = sb.get("q")
    if isinstance(top_q, str) and top_q.strip():
        return True
    query = sb.get("query")
    if not isinstance(query, list):
        return False
    for leaf in query:
        if not isinstance(leaf, dict):
            continue
        # Nested group node: scoped iff any descendant leaf scopes.
        if leaf.get("group") is True or isinstance(leaf.get("query"), list):
            if _sb_is_scoped(leaf):
                return True
            continue
        col = leaf.get("col")
        if col is None:
            # Text leaf (no col): scopes only if its q is a non-empty string.
            lq = leaf.get("q")
            if isinstance(lq, str) and lq.strip():
                return True
        elif col not in _NON_SCOPING_COLS:
            return True
    return False


def _is_list_envelope(response: Any) -> bool:
    """True if `response` actually carried a list at the data path (possibly an
    empty list). Distinguishes a genuine zero-match result from a malformed /
    `data: false` / empty body that unwrap_list also flattens to [] — so an
    absence claim is never 'confirmed' off a non-result."""
    inner = response
    if isinstance(inner, dict) and isinstance(inner.get("data"), dict):
        inner = inner["data"]
    if isinstance(inner, list):
        return True
    if isinstance(inner, dict):
        return isinstance(inner.get("data"), list)
    return False


# --- registration -------------------------------------------------------

def register_grounding_tools(
    mcp: FastMCP,
    client: CdeskClient,
    domains: dict[str, GroundingDomain],
    evidence_threshold: int = DEFAULT_EVIDENCE_THRESHOLD,
) -> None:
    """Register the domain-agnostic grounded-answer tools + prompt.

    `evidence_threshold` is the default minimum distinct supporting records for
    a *pattern*-kind claim in verify_claims (overridable per call)."""

    @mcp.tool(
        description=_COLLECT_RECORDS_DESC,
        annotations=ToolAnnotations(title="Collect records", readOnlyHint=True),
    )
    async def collect_records(
        domain: str,
        filters: dict[str, Any] | None = None,
        include_discussion: bool = False,
        max_records: int = _MAX_COLLECT_RECORDS,
        text_chars_per_record: int = 2000,
    ) -> dict[str, Any]:
        discussion_unsupported = False
        try:
            dom = _require_domain(domains, domain)
            if max_records < 1 or max_records > _MAX_COLLECT_RECORDS:
                raise ValueError(
                    f"max_records must be between 1 and {_MAX_COLLECT_RECORDS} "
                    f"(got {max_records})"
                )
            if text_chars_per_record < 0:
                raise ValueError("text_chars_per_record must be >= 0")
            filters = _validate_filters(dom, filters or {})
            sb = await dom.resolve_filter(dom.cache, filters)
            sb_encoded = encode_sb(sb) if sb is not None else None

            # Resolve status ids → names for readable evidence. Best-effort: if
            # the enum cache can't load, the map is empty and _collect_entry
            # falls back to the raw id. Skipped for domains with no enum cache.
            status_names: dict[Any, str] = {}
            if dom.cache is not None and dom.status_bucket is not None:
                try:
                    await dom.cache.load()
                except Exception:
                    pass
                status_names = (
                    dom.cache.action_code_name_map(dom.status_bucket)
                    if dom.status_by_action_code
                    else dom.cache.id_name_map(dom.status_bucket)
                )

            items: list[dict[str, Any]] = []
            total_reported: int | None = None
            truncated = False
            text_truncated_count = 0
            page = 1
            # Hard page ceiling so a misbehaving backend can't loop forever.
            max_pages = (max_records // _MAX_PER_PAGE) + 2
            while page <= max_pages:
                params: dict[str, Any] = {"pg": page, "pp": _MAX_PER_PAGE}
                if sb_encoded is not None:
                    params["sb"] = sb_encoded
                response = await client.get(dom.list_endpoint, params=params)
                records, meta = unwrap_list(response)
                if total_reported is None:
                    total_reported = _meta_total(meta)
                for rec in records:
                    if len(items) >= max_records:
                        truncated = True
                        break
                    entry, was_truncated = _collect_entry(
                        dom, rec, text_chars_per_record, status_names
                    )
                    if was_truncated:
                        text_truncated_count += 1
                    items.append(entry)
                if truncated or len(records) < _MAX_PER_PAGE:
                    break
                page += 1

            if include_discussion:
                if dom.discussion is None:
                    # The domain has no thread; signal it rather than erroring.
                    discussion_unsupported = True
                else:
                    # Fetch threads concurrently (bounded) so a broad result set
                    # doesn't serialize into a client timeout.
                    spec = dom.discussion
                    disc_sem = asyncio.Semaphore(_DISCUSSION_CONCURRENCY)

                    async def _attach_discussion(entry: dict[str, Any]) -> bool:
                        async with disc_sem:
                            disc = await _safe_discussion_text(client, spec, entry["id"])
                        if not disc:
                            return False
                        # Cap discussion text like the body text so the context
                        # budget the cap protects isn't blown (BUG 5a). No
                        # truthiness gate: 0 must cap to "" (ids/titles-only
                        # sweep), not mean "unlimited" — `if cap and ...` made
                        # 0 return FULL threads, the opposite of the contract.
                        capped = False
                        if len(disc) > text_chars_per_record:
                            disc = disc[:text_chars_per_record]
                            capped = True
                        entry["discussion"] = disc
                        return capped

                    eligible = [e for e in items if isinstance(e.get("id"), int)]
                    capped_flags = await asyncio.gather(
                        *(_attach_discussion(e) for e in eligible)
                    )
                    text_truncated_count += sum(1 for c in capped_flags if c)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"collect_records input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="collect_records") from e

        coverage: dict[str, Any] = {
            "scanned": len(items),
            "total_reported": total_reported,
            "truncated": truncated,
            "cap": max_records,
            "text_truncated_count": text_truncated_count,
            "include_discussion": include_discussion,
        }
        if discussion_unsupported:
            coverage["discussion_unsupported"] = True
        return {"domain": dom.name, "items": items, "coverage": coverage}

    @mcp.tool(
        description=_VERIFY_CLAIMS_DESC,
        annotations=ToolAnnotations(title="Verify claims", readOnlyHint=True),
    )
    async def verify_claims(
        domain: str,
        claims: list[dict[str, Any]],
        pattern_threshold: int | None = None,
    ) -> dict[str, Any]:
        try:
            dom = _require_domain(domains, domain)
            if not isinstance(claims, list):
                raise ValueError("claims must be a list of claim objects")
            threshold = (
                pattern_threshold
                if isinstance(pattern_threshold, int)
                and not isinstance(pattern_threshold, bool)  # bool is an int subclass
                and pattern_threshold >= 1
                else evidence_threshold
            )
            # Memoize fetches across all claims/evidence in this call so a record
            # cited more than once is fetched once (round-2 BUG 8); the caches
            # hold in-flight Tasks so this holds even though claims run
            # concurrently. A semaphore bounds load on CDESK; gather preserves
            # claim order.
            record_cache: dict[int, "asyncio.Task[dict[str, Any] | None]"] = {}
            disc_cache: dict[int, "asyncio.Task[str]"] = {}
            sem = asyncio.Semaphore(_VERIFY_CONCURRENCY)

            async def _run(index: int, claim: Any) -> dict[str, Any]:
                async with sem:
                    return await _verify_one_claim(
                        dom, client, claim, index, threshold, record_cache, disc_cache
                    )

            # return_exceptions=True so a RuntimeError from one claim (e.g.
            # unconfigured client) doesn't cancel the others mid-flight and
            # orphan their in-flight fetch Tasks ("Task exception never
            # retrieved"). Every _run still awaits its cached Task; then we
            # surface the first real error.
            results = await asyncio.gather(
                *(_run(i, c) for i, c in enumerate(claims)),
                return_exceptions=True,
            )
            verdicts: list[dict[str, Any]] = []
            for r in results:
                if isinstance(r, BaseException):
                    raise r
                verdicts.append(r)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"verify_claims input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="verify_claims") from e

        summary: dict[str, int] = {}
        for v in verdicts:
            summary[v["verdict"]] = summary.get(v["verdict"], 0) + 1
        return {
            "domain": dom.name,
            "verified": True,
            "threshold_used": threshold,
            "claims": verdicts,
            "summary": summary,
            "verdict_meanings": _VERDICT_MEANINGS,
        }

    @mcp.prompt(
        name="truthful_answer",
        description=(
            "Answer a question about CDESK data (requests, tasks, customers, or "
            "users) using only verifiable evidence — never invent or twist data, "
            "and treat 'nothing found' as a correct answer."
        ),
    )
    def truthful_answer(question: str) -> str:
        return _TRUTHFUL_PROMPT.format(
            question=question, domains=", ".join(_VALID_DOMAINS)
        )


# --- engine internals ---------------------------------------------------

def _first_field(rec: dict[str, Any], fields: tuple[str, ...]) -> Any:
    """First present key's value among `fields` (presence-based, so an explicit
    None still wins over a later field). None if none of the keys are present."""
    for f in fields:
        if f in rec:
            return rec[f]
    return None


def _meta_total(meta: dict[str, Any]) -> int | None:
    """Best-effort total-record count from list meta. CDESK's field name isn't
    contractually documented (see unwrap_list), so we probe the common keys.
    Display-only — the truth-gate never depends on this value."""
    for key in ("totalItems", "total", "totalCount", "recordsTotal", "count"):
        value = meta.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _record_text(
    domain: GroundingDomain,
    record: dict[str, Any],
    normalize: Callable[[object], str],
) -> tuple[str, str]:
    """The record's human text as (title, body), each cleaned by `normalize`.

    SINGLE source of truth for what collect_records DISPLAYS and what
    verify_claims MATCHES against — both derive from here so the two can never
    drift. The truth-gate depends on that: the LLM must be able to verify a
    quote copied from exactly what it was shown (rounds 1–2 BUG 3/7). Pass
    normalize_for_display to read, normalize_for_match to compare.

    title = first non-empty title field; body = all non-empty body fields joined
    with a space (so a quote spanning two body fields still verifies, while one
    crossing the title/body boundary — shown separately — does not)."""
    title = ""
    for f in domain.title_fields:
        val = record.get(f)
        if isinstance(val, str) and val.strip():
            title = normalize(val)
            break
    body_parts: list[str] = []
    for f in domain.body_fields:
        val = record.get(f)
        if isinstance(val, str) and val.strip():
            body_parts.append(normalize(val))
    body = " ".join(p for p in body_parts if p)
    return title, body


def _collect_entry(
    domain: GroundingDomain,
    rec: dict[str, Any],
    text_chars_per_record: int,
    status_names: dict[Any, str],
) -> tuple[dict[str, Any], bool]:
    """Project a raw record to a compact evidence entry. Returns
    (entry, text_was_truncated). `status_names` maps status enum id → name so
    the LLM gets a readable status instead of an opaque id (BUG 4)."""
    title, text = _record_text(domain, rec, normalize_for_display)
    truncated = False
    # No truthiness gate: text_chars_per_record=0 must cap to "" (a cheap
    # ids/titles-only sweep), not mean "unlimited" — `if cap and ...` made 0
    # return FULL bodies for up to 1000 records, the opposite of the contract.
    if len(text) > text_chars_per_record:
        text = text[:text_chars_per_record]
        truncated = True
    raw_status = _first_field(rec, domain.status_fields)
    coerced_status = _coerce_int(raw_status)
    # Only an int/str id can be a key in status_names; if a tenant returns
    # status as a nested object/list (unhashable), show it raw rather than
    # crashing the whole collect on dict.get(unhashable).
    status = (
        status_names.get(coerced_status, raw_status)
        if isinstance(coerced_status, (int, str))
        else raw_status
    )
    entry = {
        "id": _coerce_int(_first_field(rec, domain.id_fields)),
        "num": _first_field(rec, domain.num_fields),
        "title": title,
        "text": text,
        "status": status,
        "created": _first_field(rec, domain.created_fields),
    }
    return entry, truncated


def _coerce_int(value: Any) -> Any:
    """Return an int when the id arrives as a numeric string; else pass through."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


async def _fetch_discussion_records(
    client: CdeskClient, spec: DiscussionSpec, record_id: int
) -> list[Any]:
    """Fetch a record's discussion messages. Best-effort: returns [] if the
    thread can't be loaded (so the caller yields empty text); RuntimeError
    (unconfigured / instructional) propagates so the real cause surfaces."""
    try:
        response = await client.get(spec.path_template.format(id=record_id))
    except RuntimeError:
        raise
    except Exception:
        return []
    records, _ = unwrap_list(response)
    return records


def _discussion_text(
    spec: DiscussionSpec, records: list[Any], normalize: Callable[[object], str]
) -> str:
    """Flatten discussion message bodies into ONE string, each cleaned by
    `normalize`. SINGLE source of truth for the discussion text collect_records
    DISPLAYS and verify_claims MATCHES against — both derive from here so they
    can't drift (rounds 1–2 BUG 1/7)."""
    parts: list[str] = []
    for msg in records:
        if not isinstance(msg, dict):
            continue
        for f in spec.text_fields:
            val = msg.get(f)
            if isinstance(val, str) and val.strip():
                parts.append(normalize(val))
    return " ".join(parts)


async def _safe_discussion_text(
    client: CdeskClient, spec: DiscussionSpec, record_id: int
) -> str:
    """Display-normalized discussion text for collect_records. Best-effort: a
    failure on one record must not sink the whole collect call."""
    records = await _fetch_discussion_records(client, spec, record_id)
    return _discussion_text(spec, records, normalize_for_display)


def _record_match_units(domain: GroundingDomain, record: dict[str, Any]) -> list[str]:
    """The record's text in the SAME units collect_records displays — title as
    one unit, joined body as one unit — match-normalized via _record_text, the
    shared source of truth. A quote spanning two body fields still verifies
    (round-2 BUG 7); one crossing the title/body boundary (shown separately)
    still fails (round-1 BUG 3). Empty units are dropped so a markup-only field
    can't become a universal-match hole."""
    title, body = _record_text(domain, record, normalize_for_match)
    return [u for u in (title, body) if u]


async def _discussion_match_text(
    client: CdeskClient,
    spec: DiscussionSpec,
    record_id: int,
    cache: dict[int, "asyncio.Task[str]"],
) -> str:
    """Match-normalized discussion text as ONE unit, mirroring the display
    (_safe_discussion_text) via the shared _discussion_text so a quote copied
    from the shown thread verifies (round-1 BUG 1, round-2 BUG 7). Memoized per
    verify call (round-2 BUG 8) via an in-flight Task so concurrent claims share
    one fetch; a fetch failure yields "" (the quote then can't be confirmed from
    discussion), RuntimeError propagates."""
    task = cache.get(record_id)
    if task is None:
        task = asyncio.ensure_future(_load_discussion_match(client, spec, record_id))
        cache[record_id] = task
    return await task


async def _load_discussion_match(
    client: CdeskClient, spec: DiscussionSpec, record_id: int
) -> str:
    return _discussion_text(
        spec, await _fetch_discussion_records(client, spec, record_id), normalize_for_match
    )


async def _verify_one_claim(
    domain: GroundingDomain,
    client: CdeskClient,
    claim: Any,
    index: int,
    threshold: int,
    record_cache: dict[int, "asyncio.Task[dict[str, Any] | None]"],
    disc_cache: dict[int, "asyncio.Task[str]"],
) -> dict[str, Any]:
    """Test a single claim. specific/pattern → check each cited record exists
    and literally contains its quote, recount distinct ids, apply threshold for
    pattern. absence → re-run the search and confirm zero matches."""
    if not isinstance(claim, dict):
        return {
            "claim_id": index,
            "kind": None,
            "verdict": "error",
            "reason": "claim must be an object",
        }
    claim_id = claim.get("claim_id", index)
    statement = claim.get("statement", "")
    kind = claim.get("kind")
    if kind not in _VALID_CLAIM_KINDS:
        return {
            "claim_id": claim_id,
            "statement": statement,
            "kind": kind,
            "verdict": "error",
            "reason": f"kind must be one of {_VALID_CLAIM_KINDS}",
        }

    if kind == "absence":
        return await _verify_absence(domain, client, claim, claim_id, statement)

    # specific / pattern
    evidence = claim.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen: set[int] = set()
    # Match-units are derived by re-normalizing the whole record; cache them per
    # record so a claim citing the same record with several quotes normalizes it
    # once, not once per quote.
    units_cache: dict[int, list[str]] = {}
    no_match_reason = "quote is not a literal substring of the record text"
    if domain.discussion is not None:
        no_match_reason += " or its discussion"
    for ev in evidence:
        rid = ev.get("record_id") if isinstance(ev, dict) else None
        quote = ev.get("quote", "") if isinstance(ev, dict) else ""
        if not isinstance(rid, int) or isinstance(rid, bool):
            invalid.append({"record_id": rid, "quote": quote, "reason": "missing or non-integer record_id"})
            continue
        if not isinstance(quote, str):
            invalid.append({"record_id": rid, "quote": quote, "reason": "quote must be a string"})
            continue
        qn = normalize_for_match(quote)
        # Gate on the NORMALIZED length. A raw quote of only HTML/entities/
        # whitespace can be 15+ chars yet normalize to "" — and "" is a substring
        # of every string, which would forge the gate (round-2 BUG 6).
        if len(qn) < MIN_QUOTE_CHARS:
            invalid.append({"record_id": rid, "quote": quote, "reason": f"quote normalizes to fewer than {MIN_QUOTE_CHARS} chars"})
            continue
        record = await _fetch_record(client, domain, rid, record_cache)
        if record is None:
            invalid.append({"record_id": rid, "quote": quote, "reason": "record not found or inaccessible"})
            continue
        # Match against the SAME units collect_records displays — title as one
        # unit, the joined body as one unit — so a quote copied verbatim from what
        # the LLM was shown verifies even if it spans two body fields (round-2
        # BUG 7), while a span crossing the title/body boundary (shown as separate
        # fields) still fails (round-1 BUG 3).
        units = units_cache.get(rid)
        if units is None:
            units = _record_match_units(domain, record)
            units_cache[rid] = units
        if any(qn in unit for unit in units):
            valid.append({"record_id": rid, "quote": quote})
            seen.add(rid)
            continue
        # Not in the record's own fields — for domains with a discussion thread
        # (requests) it may have been quoted from the thread that
        # collect_records(include_discussion=True) surfaced (round-1 BUG 1).
        if domain.discussion is not None:
            disc = await _discussion_match_text(client, domain.discussion, rid, disc_cache)
            if disc and qn in disc:
                valid.append({"record_id": rid, "quote": quote})
                seen.add(rid)
                continue
        invalid.append({"record_id": rid, "quote": quote, "reason": no_match_reason})

    supported_count = len(seen)
    if kind == "specific":
        verdict = "confirmed" if supported_count >= 1 else "rejected"
    else:  # pattern
        if supported_count >= threshold:
            verdict = "confirmed"
        elif supported_count >= 1:
            verdict = "below_threshold"
        else:
            verdict = "rejected"
    return {
        "claim_id": claim_id,
        "statement": statement,
        "kind": kind,
        "supported_count": supported_count,
        "threshold_used": threshold if kind == "pattern" else None,
        "valid_evidence": valid,
        "invalid_evidence": invalid,
        "verdict": verdict,
    }


async def _verify_absence(
    domain: GroundingDomain,
    client: CdeskClient,
    claim: dict[str, Any],
    claim_id: Any,
    statement: str,
) -> dict[str, Any]:
    """Confirm an 'absence' claim by running the search server-side. A query
    error must NOT be reported as confirmed_absence — it propagates so the LLM
    never claims 'nothing found' off a failed search."""
    aq = claim.get("absence_query")
    if not isinstance(aq, dict):
        return {
            "claim_id": claim_id,
            "statement": statement,
            "kind": "absence",
            "verdict": "error",
            "reason": "absence claim requires an 'absence_query' object",
        }
    # Resolve enums + build the filter under a local guard so an unknown filter
    # key, a bad enum name, or a malformed date in THIS absence_query becomes a
    # per-claim `error` verdict instead of aborting verification of every other
    # claim in the batch (BUG 2). The full filter surface is forwarded so a
    # scoped "no record about X from group G" isn't silently broadened (BUG 5b).
    try:
        _validate_filters(domain, aq)
        sb = await domain.resolve_filter(domain.cache, aq)
    except (ValueError, RuntimeError) as e:
        return {
            "claim_id": claim_id,
            "statement": statement,
            "kind": "absence",
            "verdict": "error",
            "reason": f"invalid absence_query: {e}",
        }
    if sb is None or not _sb_is_scoped(sb):
        # A scopeless absence_query (e.g. {}, {"include_deleted": true}, or a
        # match-all sb_raw like {"q": ""}) would otherwise list ALL records and
        # report a spurious 'refuted' against unrelated rows — or, on an empty
        # tenant, a fake 'confirmed_absence'. We inspect the RESOLVED sb (the
        # requests builder always adds a default 'exclude deleted' clause, so a
        # non-None sb alone doesn't prove scope). A negative must be proven
        # against a SCOPED search, never a match-all.
        return {
            "claim_id": claim_id,
            "statement": statement,
            "kind": "absence",
            "verdict": "error",
            "reason": (
                "absence_query has no effective scope; provide at least one "
                "topic filter (e.g. text_search) so the negative is proven "
                "against a scoped search, not a match-all"
            ),
        }
    params: dict[str, Any] = {"pg": 1, "pp": 5}
    params["sb"] = encode_sb(sb)
    response = await client.get(domain.list_endpoint, params=params)
    records, _ = unwrap_list(response)
    if not records:
        if not _is_list_envelope(response):
            # unwrap_list flattens malformed / {data:false} / empty bodies to []
            # too; a confirmed absence must rest on a genuine (possibly empty)
            # list, not a non-result — propagate rather than fake 'nothing found'.
            raise RuntimeError(
                "absence search returned no usable list payload; refusing to "
                "report confirmed_absence off a non-result"
            )
        return {
            "claim_id": claim_id,
            "statement": statement,
            "kind": "absence",
            "verdict": "confirmed_absence",
            "matched_ids": [],
        }
    matched = [
        _coerce_int(_first_field(r, domain.id_fields))
        for r in records
        if isinstance(r, dict)
    ]
    return {
        "claim_id": claim_id,
        "statement": statement,
        "kind": "absence",
        "verdict": "refuted",
        "matched_ids": matched,
    }


async def _fetch_record(
    client: CdeskClient,
    domain: GroundingDomain,
    record_id: int,
    cache: dict[int, "asyncio.Task[dict[str, Any] | None]"],
) -> dict[str, Any] | None:
    """Fetch a record for quote verification, memoized per verify call so a
    record cited more than once is fetched once (round-2 BUG 8) — even when
    claims run concurrently, since the cache holds the in-flight Task and
    concurrent callers await the same one. Returns None if the record is
    missing/inaccessible (→ invalid citation). RuntimeError (unconfigured /
    instructional) propagates so the tool surfaces the cause.

    Uses a plain detail GET (not the update-fetch path), so optimistic-locking
    fields like timestamp_check are never involved."""
    task = cache.get(record_id)
    if task is None:
        task = asyncio.ensure_future(_do_fetch_record(client, domain, record_id))
        cache[record_id] = task
    return await task


async def _do_fetch_record(
    client: CdeskClient, domain: GroundingDomain, record_id: int
) -> dict[str, Any] | None:
    try:
        envelope = await client.get(domain.detail_template.format(id=record_id))
    except RuntimeError:
        raise
    except Exception:
        return None
    record = unwrap_record(envelope)
    return record if isinstance(record, dict) else None


# --- contract, tool descriptions & prompt -------------------------------

# ACCEPTED COMPLIANCE RISK (reviewed 2026-07-29, Anthropic connector directory).
# This block is embedded in the collect_records + verify_claims descriptions and
# is entirely behavioural — steps 0-4 tell the model how to answer. The directory
# policy says "Describe what the tool does. Do not tell Claude how to behave",
# and rejects descriptions that "tell Claude to behave in ways unrelated to the
# tool's function".
#
# Kept deliberately. It IS these tools' function: verify_claims enforces the
# quote-matching in code (re-fetches each record, rejects any quote that is not a
# literal substring, recounts distinct supporters against the evidence
# threshold). The text documents a protocol the server actually implements rather
# than steering unrelated behaviour, which is the distinction the rule draws.
# Removing it would gut the server's whole purpose, so if a reviewer objects,
# argue the point — don't silently delete it.
_GROUNDING_CONTRACT = inspect.cleandoc(
    """
    GROUNDED-ANSWER CONTRACT — when answering a question from CDESK data:
    0. PICK THE MODULE FIRST. CDESK data lives in separate modules: requests
       (customer-facing helpdesk tickets), tasks (internal work items),
       customers (companies), users (people). Decide which module holds the
       answer; if you're not sure, ASK the user, or search several by calling
       collect_records once per module. Never guess the module.
    1. State something ONLY if you can cite a specific record id AND quote the
       exact words from it that say so. No quote → don't say it.
    2. The claim must be what the quote LITERALLY says — never reinterpret it
       ("I lost my phone" supports "someone mentioned a lost phone", never
       "a phone is broken").
    3. "I found no such thing" is a correct, expected answer. Never manufacture
       a plausible-sounding answer to seem helpful.
    4. ASK BEFORE INFERRING (with a reliability warning). When the user wants to
       find something out, FIRST ask whether they want verified facts only,
       or whether you may also infer — and in that SAME question warn them that
       you are an AI that can hallucinate and may provide incorrect
       information, so they should verify anything important. Default to facts
       only if they don't choose. If inference is permitted, every inferred
       statement must be shown together with the verified fact (record id +
       quote) it was drawn from and labeled as an inference — never stated as a
       fact.
    Counting matters only for population-level (pattern) claims: a theme is a
    real pattern only if enough distinct records support it (the threshold).
    For specific look-ups and existence checks, one record — or zero — is the
    whole answer and the threshold does not apply.
    """
)

_COLLECT_RECORDS_DESC = inspect.cleandoc(
    """
    Gather the record evidence needed to answer a question truthfully. Pages
    through ALL records of one `domain` matching the filter (to full coverage or
    a cap) and returns a compact, id-tagged corpus with HTML stripped — the raw
    material you read before forming claims. This tool does NO analysis.

    domain (required): one of "requests", "tasks", "customers", "users",
    "fills". Selects which module is searched — one call covers exactly one
    module, so spanning several takes one call each. ("fills" = logged
    work/time records, the module holding "how many hours did X log".)

    filters (optional): an object whose keys depend on the domain (same names as
    that module's list_* tool). Use the narrowest filter that could still
    contain the answer:
      - requests → text_search, status_name, priority_name, place_name,
        cat_type_name (category type → cat_type_id), base_type (letter),
        company, solver_id, solver_group_id, catalog_id, deal_id,
        project_contract_id, branch_id, superior_request_id,
        created_after/before, sb_raw. (sla/due-date/deleted filters are
        not honored by the v3 request list.)
      - tasks → text_search, status_name, type_name, solver_id, customer_id,
        valid_from_after/before, deadline_after/before, created_after/before,
        sb_raw. (open/late filters are not honored by the v3 task list.)
      Date windows take ISO-8601; a bare date is inclusive (start-of-day for
      after, end-of-day for before); naive datetimes are treated as
      tenant-local time (CDESK_TIMEZONE, default Europe/Bratislava).
      - customers → text_search, status, sb_raw.
      - users → text_search, company_id, user_type, status, sb_raw.
    Unknown keys for the chosen domain are rejected with the valid list.

    include_discussion=True also pulls each record's discussion thread (slower;
    only the requests domain has one — for others it's reported as
    coverage.discussion_unsupported and otherwise ignored).

    Returns:
      - domain: the resolved domain name.
      - items: [{id, num, title, text, status, created}] (text = cleaned
        title+body, capped at text_chars_per_record; pass 0 for a cheap
        ids/titles-only sweep with no body text).
      - coverage: {scanned, total_reported, truncated, cap,
        text_truncated_count, include_discussion[, discussion_unsupported]}. If
        truncated=True or your filter was broad, say so — your counts are then a
        LOWER BOUND.

    Next step: form claims (each with record_id + a verbatim quote copied
    exactly from a record's text) and pass them to verify_claims with the SAME
    domain.
    """
) + "\n\n" + _GROUNDING_CONTRACT

_VERIFY_CLAIMS_DESC = inspect.cleandoc(
    """
    Verify draft claims against CDESK before you state them. This is the
    truth-gate: the server re-fetches each cited record, confirms the quote is a
    literal substring of it, recounts distinct supporting records, and (for
    'absence') re-runs the search itself. Build your final answer ONLY from the
    returned verdicts.

    domain (required): one of "requests", "tasks", "customers", "users",
    "fills" — the SAME domain you collected from.

    claims: a list of claim objects, each:
      {
        "claim_id": "c1",
        "statement": "what you want to assert",
        "kind": "specific" | "pattern" | "absence",
        "evidence": [{"record_id": 123, "quote": "exact text from #123"}],
        "absence_query": { ...collect_records-style filter for this domain... }
      }

    kinds:
      - "specific" — a fact from one (or few) records. Confirmed if >=1 cited
        record really contains its quote. Count/threshold do NOT apply.
      - "pattern" — a population-level claim ("phones need fixing"). Confirmed
        only if >= threshold distinct records are verified; 1..threshold-1 →
        'below_threshold' (report as an isolated mention WITH its count, not as
        a need); 0 → 'rejected'.
      - "absence" — "there is no record about X". The server runs absence_query
        and returns 'confirmed_absence' (0 matches) or 'refuted' (matched_ids).

    pattern_threshold: optional override of the configured default.

    Per-claim verdict includes valid_evidence, invalid_evidence (with reasons:
    bad id, quote-too-short, quote-not-found), supported_count and verdict.
    Relay only confirmed claims (cite the ids); report below_threshold as
    isolated mentions with counts; DROP rejected claims; for refuted absence,
    correct yourself. If nothing is confirmed, the truthful answer is the
    negative — state it plainly.
    """
) + "\n\n" + _GROUNDING_CONTRACT

# A legend for the verdict vocabulary this tool invents, shipped in the result
# because nothing else defines these tokens: `_SERVER_INSTRUCTIONS` covers "state
# only what came back confirmed" and inference labelling, but says nothing about
# below_threshold / rejected / confirmed_absence / refuted. Without the legend a
# caller seeing `"verdict": "below_threshold"` has to guess what it licenses, and
# guessing wrong either suppresses a real finding or inflates one mention into a
# trend.
#
# DELIBERATELY DECLARATIVE — each entry says what the verdict MEANS about the
# evidence, not what the caller must do about it. This text ships inside a tool
# RESULT, the channel hosts treat as untrusted content, so imperative framing
# here matches an Anthropic directory rejection pattern (and the shape of a
# prompt-injection payload). It previously read "Build the answer ONLY from these
# verdicts … rejected → do NOT state it … correct yourself" under a key named
# `instructions_to_relay`, which is exactly the construction removed from
# `unsupported_filter_directive` — see that docstring in tools/_helpers.py.
# The OBLIGATION to answer only from verified evidence is unchanged and lives in
# its legitimate home, server.py's `_SERVER_INSTRUCTIONS`. Keep this a legend.
_VERDICT_MEANINGS = inspect.cleandoc(
    """
    What each verdict says about the evidence:
      - confirmed — a cited record really contains the quote; the claim is
        supported, and the supporting record ids are in valid_evidence.
      - below_threshold — fewer than `threshold_used` distinct records support
        it. It is an isolated mention of that size, not an established pattern;
        supported_count carries the exact number.
      - rejected — no cited record contained the quote, so the claim has no
        evidence behind it. invalid_evidence gives the per-quote reason.
      - confirmed_absence — the absence search ran and matched zero records:
        there is no record showing the thing.
      - refuted — the absence search DID match records, listed in matched_ids,
        so the asserted absence is false.

    A response where nothing is confirmed is a complete result, not a failure:
    it means the evidence does not support the claims as written.

    Inference is gated on the user's explicit consent (see the ask-gate in this
    server's instructions). An inference is only as good as the confirmed claims
    it rests on — their record ids and quotes — and is a different kind of
    statement from a verified fact.
    """
)

_TRUTHFUL_PROMPT = inspect.cleandoc(
    """
    Answer this question about CDESK data using ONLY verifiable evidence:

        {question}

    Procedure:
    0a. PICK THE MODULE. CDESK data lives in separate modules: {domains}. Decide
        which one holds the answer. If you're not sure, ASK the user which module
        to search — or search several by calling collect_records once per module.
        Do NOT guess the module.
    0b. ASK FIRST (with a reliability warning). Before answering, ask the user
        which they want: verified FACTS ONLY, or MAY YOU ALSO INFER (you will
        show the facts you inferred from). In that SAME question you must also
        warn them that you are an AI that can hallucinate and may provide
        incorrect information, so they should verify anything important. If they
        don't choose, default to facts only.
    1. Call collect_records with the chosen domain and the narrowest filter that
       could contain the answer. Note the coverage block (if truncated, your
       counts are a lower bound and you must say so).
    2. Read the returned text. Form candidate claims. Tag each:
         - "specific"  — a fact grounded in one/few records (no threshold).
         - "pattern"   — a population-level statement (needs the threshold).
         - "absence"   — "there is no record about X" (give an absence_query).
       For specific/pattern, attach each supporting record_id with a VERBATIM
       quote copied exactly from that record's text.
    3. Call verify_claims with the SAME domain and all your claims.
    4. Write the answer using ONLY the returned verdicts:
         - confirmed → state it, cite the record ids.
         - pattern below_threshold → report as an isolated mention with its
           exact count; do NOT present it as a need or trend.
         - rejected → drop it; do not state it.
         - confirmed_absence → "I found no record showing X."
         - refuted → correct yourself; matching records exist.
    5. If nothing is confirmed, the truthful answer is the negative. Saying
       "there is no such thing" is correct and expected — never invent or twist
       data to produce a more satisfying answer.
    6. Inference (only if the user allowed it in step 0b): you may add an
       "Interpretation" note, but every inferred statement must be shown next
       to the verified fact — record id + quote — it was drawn from, and
       labeled clearly as inference, never as fact.
    """
)
