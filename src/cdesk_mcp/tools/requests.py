"""Request module MCP tools.

Ten tools that expose CDESK's Request v3 CRUD + intent helpers + the
discussion sub-resource + the custom-field catalog to the LLM. Wires together
CdeskClient, EnumCache, the filter builder, and the error translator.

Optimistic locking is hidden: update_request and set_request_status
internally do GET → mutate → PUT (re-using the `timestamp_check` from
the GET response). The LLM never sees the version token.

Discussion semantics: the POST endpoint accepts `status` as a CHANNEL
selector (1 = customer message → customer_text, 2 = internal note →
technote_text). The MCP tool surface exposes that as `channel:
"customer" | "internal"` so the LLM doesn't have to remember magic
numbers.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from datetime import datetime, timezone
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp import base_config
from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.enums import EnumCache
from cdesk_mcp.field_policy import (
    REQUEST_BASE_FIELDS,
    REQUEST_FIELDS,
    describe,
    refuse_if_violated,
    request_base_settings,
    violations,
)
from cdesk_mcp.filters import build_request_filter, encode_sb
from cdesk_mcp.sort_columns import resolve_sort, sort_note
from cdesk_mcp.tools._helpers import (
    without_notify_flags as _without_notify_flags,
    notify_block_value,
    request_status_group,
    with_status_groups,
    list_meta,
    TEXT_MATCH_NOTE,
    EnumLabel,
    add_enum_names,
    annotate_write_warnings,
    add_names,
    apply_field_scope,
    ask_for_names,
    forbid_unknown_arguments,
    resolve_enum_field_or_raise,
    resolve_enum_or_raise,
    to_llm_error,
    unsupported_filter_directive,
    unwrap_list,
    unwrap_record,
    wrap_collection,
    validate_custom_fields,
    validate_fields,
    validate_fieldset,
    with_warning_texts,
)
from cdesk_mcp.tz import localize_naive_datetime

# LLM-friendly param name → CDESK field name. Centralises the rename
# layer so tool signatures stay clean and only this file knows the
# CDESK column quirks (id_company, id_solver, id_rc, …).
_REQUEST_CREATE_FIELDS = {
    "customer_id": "id_company",
    "solver_id": "id_solver",
    "deal_id": "id_rc",
    # NB: the create/update BODY field is `title` (it passes through unchanged).
    # `req_title` is only the DB column / `sb` filter name — do NOT remap to it,
    # or CDESK ignores the title (auto-naming tenants) or rejects the call.
    # status_name / type_name / priority_name / urgency_name / impact_name /
    # place_name / cat_type_name / cat_area_name / cat_area_2nd_name /
    # close_reason_name / solver_method_name / root_cause_name are resolved
    # client-side to ids — see create_request / update_request.
}

_REQUEST_TOOL_NAMES = (
    "list_requests",
    "get_request",
    "list_request_solver_groups",
    "get_request_custom_fields",
    "get_request_enums",
    "create_request",
    "update_request",
    "delete_request",
    "set_request_status",
    "list_request_discussion",
    "post_request_discussion",
)

# Fields CDESK strips from a request write (create and update alike) when the
# tenant feature behind them is off or the account lacks the write right. The
# API returns 200; CDESK's own `warnings` entry for such a field does not say
# which setting or right is missing, and not every CDESK version sends one, so
# the savedData diff uses this table to report the drop and its cause.
#
# Keys are the WIRE names as they appear in the body and in savedData. Each note
# names the setting and right, mirroring the wording CDESK uses for `code`.
_REQUEST_GATED_FIELDS: dict[str, str] = {
    "code": (
        "requires the tenant request-code feature (`request.code.status`) and the "
        "`request/fields/code` write right"
    ),
    "solver_note": (
        "requires the `request/fields/solver_note` write right on this account"
    ),
    "solution": (
        "requires the tenant setting `request.solution.enabled` and the "
        "`request/fields/solution` write right"
    ),
    "percent_done_manual": (
        "requires the tenant setting `project_contract.percent_done` and the "
        "`request/fields/percent-done` write right"
    ),
    "desired_date": (
        "requires the tenant setting `request.desiredDate.enabled` and the "
        "`request/fields/desired_date` write right"
    ),
    "due_date": (
        "requires the tenant setting `request.dueDate.enabled` and the right to "
        "change the due date"
    ),
    "calendar_date_start": (
        "requires the tenant setting `request.calendar_date_start` and the "
        "`request/fields/calendar_date_start` write right"
    ),
    "invoice_date": (
        "requires the Fills module (`fill.enabled`) with `fill.invoicing.date` "
        "set to the request invoice date, and the `request/fields/invoice_date` "
        "write right"
    ),
    "catalog_id": (
        "requires the tenant setting `request.catalog.enabled`"
    ),
    "close_reason_id": (
        "requires the tenant setting `request.close_reason.enabled`"
    ),
    "solver_method_id": (
        "requires the tenant setting `request.solver_method.enabled`"
    ),
    "root_cause_id": (
        "requires the tenant setting `request.root_cause.enabled` and the "
        "`request/fields/root_cause` write right"
    ),
}

# Pagination upper bound — mirrors the task module's 100 cap to keep
# response sizes within an LLM's context budget. CDESK itself accepts
# larger values; this is a defensive guard.
_MAX_PER_PAGE = 100

# Source column for the derived `is_periodic` flag. A request generated from a
# periodical request carries the parent id here; ad-hoc requests have null.
_PERIODIC_SRC = "periodical_request_id"


def _ensure_periodic_field(
    fieldset: str | None, fields: list[str] | None
) -> tuple[str | None, list[str] | None]:
    """Guarantee periodical_request_id is in the projection so `is_periodic` is
    always derivable. The default (extended) fieldset omits it.
    returnFields[] ALONE is restrictive (the response is narrowed to just those
    keys), so when no `fields` whitelist is given we also pin an explicit
    fieldset for returnFields to compose as a UNION rather than a restriction.

    When the caller DID pass a `fields` whitelist, we must NOT pin a fieldset:
    doing so would turn their narrow selection into the full extended set
    (UNION), silently defeating the field-narrowing they asked for. We only
    append periodical_request_id to their whitelist, keeping returnFields[]
    restrictive.
    """
    had_fields = bool(fields)
    fields = list(fields) if fields else []
    if _PERIODIC_SRC not in fields:
        fields.append(_PERIODIC_SRC)
    if fieldset is None and not had_fields:
        fieldset = "extended"  # match CDESK's implicit default so the union holds
    return fieldset, fields


# Enum-valued request fields and the label written next to each. Status and
# priority are stored by enum ACTION CODE (e.g. status 10 = "prijaté"); `type`
# is the key of a key/value bucket ("H"); the rest are enum ids. A write
# response's savedData uses the raw column names (req_status / req_priority),
# hence the aliases. Bucket names are the ones the write paths resolve against.
_REQUEST_ENUM_LABELS = (
    EnumLabel("status", "status", "status_name", by="action_code"),
    EnumLabel("req_status", "status", "status_name", by="action_code"),
    EnumLabel("priority", "priority", "priority_name", by="action_code"),
    EnumLabel("req_priority", "priority", "priority_name", by="action_code"),
    EnumLabel("type", "type", "type_name", by="key"),
    EnumLabel("req_type", "type", "type_name", by="key"),
    EnumLabel("urgency_id", "urgency", "urgency_name"),
    EnumLabel("impact_id", "impact", "impact_name"),
    EnumLabel("place_id", "place", "place_name"),
    EnumLabel("cat_type_id", "cat_type_id", "cat_type_name"),
    EnumLabel("cat_area_id", "cat_area", "cat_area_name"),
    EnumLabel("cat_area_2nd_id", "cat_area_2nd", "cat_area_2nd_name"),
    EnumLabel("close_reason_id", "close_reasons", "close_reason_name"),
    EnumLabel("solver_method_id", "solver_method", "solver_method_name"),
    EnumLabel("root_cause_id", "root_causes", "root_cause_name"),
)


def _auto_title(settings: Any, description: str | None) -> str | None:
    """The title CDESK's web form generates when auto-naming is on and the
    title is left empty: the description without HTML tags, cut to
    `max_auto_name_characters` (CDESK default 40) plus "...". None when
    auto-naming is off or there is no description."""
    if not isinstance(settings, dict) or not settings.get("auto_name_enabled"):
        return None
    if not description:
        return None
    text = re.sub(r"<[^>]*>", "", description)
    try:
        limit = int(settings.get("max_auto_name_characters") or 40)
    except (TypeError, ValueError):
        limit = 40
    return text[:limit] + "..." if len(text) > limit else text


async def _request_settings(cache: EnumCache) -> Any:
    """The tenant's live request settings (`config` in v3/request/enums), or
    None when unreadable — the form rules then constrain nothing."""
    try:
        await cache.load()
        return cache.extra("config")
    except Exception:  # noqa: BLE001
        return None


# --- time fields ---------------------------------------------------------------
# `duration_open` and `duration_reaction` are SLA WORKING seconds, not calendar
# time: counted only inside the operating hours of the request's SLA (or the
# default SLA's), holidays left out, minus
# the time spent in statuses that stop the SLA clock and in closed statuses.
# `duration_open` is computed once, when the request is closed (0 while open,
# reset to 0 on reopen, always 0 without an SLA); `duration_reaction` at the
# first reaction (null until then). The v3 endpoints return both (with
# fieldset=all and in write responses) but the v3 spec documents no request
# response field at all. A status-change write echoes an unrounded float
# (e.g. 78.175556) where the record stores 78. So every request result gets
# explicit, unit-named values next to the raw ones.

_CREATED_KEYS = ("idatetime",)
_CLOSED_KEYS = ("close_date", "req_close_date")


def _first(rec: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if rec.get(key):
            return rec[key]
    return None


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _seconds(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(value)


def _add_time_fields(rec: Any, now: datetime | None = None) -> Any:
    """Add `age_hours_calendar` (creation → close, or → now while open),
    `sla_open_hours` (null unless the request is closed and has an SLA) and
    `sla_reaction_hours` (null until the first reaction); round the two raw
    duration fields to whole seconds."""
    if not isinstance(rec, dict):
        return rec
    closed = _parse_dt(_first(rec, _CLOSED_KEYS))
    created = _parse_dt(_first(rec, _CREATED_KEYS))
    if created is not None:
        end = closed or now or datetime.now(timezone.utc)
        rec["age_hours_calendar"] = round((end - created).total_seconds() / 3600, 2)
    if "duration_open" in rec:
        seconds = _seconds(rec["duration_open"])
        if seconds is not None:
            rec["duration_open"] = seconds
        no_sla = "sla_id" in rec and not rec.get("sla_id")
        rec["sla_open_hours"] = (
            round(seconds / 3600, 2)
            if seconds is not None and closed is not None and not no_sla
            else None
        )
    if "duration_reaction" in rec:
        seconds = _seconds(rec["duration_reaction"])
        if seconds is not None:
            rec["duration_reaction"] = seconds
        rec["sla_reaction_hours"] = round(seconds / 3600, 2) if seconds is not None else None
    return rec


# CDESK sets `close_date` / `duration_open` when a request reaches a completed
# status (80, 81, 85, 86) or is refused (95). Accepted completion (90) does not
# run that step, so a request moved to 90 straight from an open status
# (below 80) is left without a close date.
_ACCEPTED_COMPLETION = 90
_COMPLETED_FROM = 80


async def _check_status_allowed(
    client: CdeskClient, id: int, current: dict[str, Any], status: Any,
) -> None:
    """Refuse a status change CDESK's status rules do not allow, naming the
    allowed statuses (GET /v3/request/{id}/statuses — the current one
    included). Unreadable statuses leave the decision to CDESK."""
    if status is None or status == current.get("status"):
        return
    try:
        rows = unwrap_record(await client.get(f"v3/request/{id}/statuses"))
    except Exception:  # noqa: BLE001 — CDESK still refuses a disallowed change
        return
    allowed = _allowed_statuses(rows)
    if not allowed or any(s["action_code"] == status for s in allowed):
        return
    names = ", ".join(f"{s['name']} ({s['action_code']})" for s in allowed)
    raise ValueError(
        f"request {id} cannot move to status {status} from its current status "
        f"{current.get('status')}; the tenant's status rules allow: {names}"
    )


def _with_direct_close_warning(result: dict[str, Any], current: Any, new: Any) -> dict[str, Any]:
    if new != _ACCEPTED_COMPLETION or not isinstance(current, int) or current >= _COMPLETED_FROM:
        return result
    return with_warning_texts(result, [
        f"The request moved from status {current} straight to 90 (accepted "
        f"completion). CDESK sets no close_date and no duration_open on that move "
        f"— only a completed status (80, 81, 85, 86) or 95 does — so the request "
        f"has no close date. Complete it first (e.g. 80) when the close date "
        f"matters."
    ])


async def _label_write_result(cache: EnumCache, result: dict[str, Any]) -> dict[str, Any]:
    """Enum labels on a write response: the record is in `savedData` (or in
    `data` on endpoints that return the record there)."""
    _add_time_fields(result.get("savedData"))
    _add_time_fields(result.get("data"))
    await add_enum_names(
        cache,
        [result.get("savedData"), result.get("data")],
        _REQUEST_ENUM_LABELS,
    )
    return result


def _add_is_periodic(rec: Any) -> Any:
    """Derive the boolean is_periodic from periodical_request_id (>0 / non-null
    means the request was generated from a periodical request)."""
    if isinstance(rec, dict) and _PERIODIC_SRC in rec:
        rec["is_periodic"] = bool(rec.get(_PERIODIC_SRC))
    return rec


def register_request_tools(
    mcp: FastMCP,
    client: CdeskClient,
    cache: EnumCache,
) -> None:
    """Register all Request-module tools on the given FastMCP instance.
    Called by build_server when client + the request enum cache are both
    available."""

    @mcp.tool(
        description=_LIST_REQUESTS_DESC + "\n\n" + sort_note("requests") + "\n\n" + TEXT_MATCH_NOTE,
        annotations=ToolAnnotations(title="List requests", readOnlyHint=True),
    )
    async def list_requests(
        text_search: str | None = None,
        status_name: str | None = None,
        status_group: Literal[
            "open", "completed", "closed", "closed_and_completed", "waiting_for_response",
        ] | None = None,
        priority_name: str | None = None,
        cat_type_name: str | None = None,
        base_type: str | None = None,
        company: int | list[int] | None = None,
        solver_id: int | list[int] | None = None,
        solver_group_id: int | None = None,
        catalog_id: int | None = None,
        deal_id: int | None = None,
        project_contract_id: int | None = None,
        branch_id: int | None = None,
        place_name: str | None = None,
        superior_request_id: int | None = None,
        periodic: bool | None = None,
        due_after: str | None = None,
        due_before: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        include_deleted: bool = False,
        sb_raw: str | dict[str, Any] | None = None,
        page: int = 1,
        per_page: int = 20,
        sort: str | None = None,
        fieldset: str | None = None,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            if per_page < 1 or per_page > _MAX_PER_PAGE:
                raise ValueError(
                    f"per_page must be between 1 and {_MAX_PER_PAGE} (got {per_page})"
                )
            if page < 1:
                raise ValueError(f"page must be 1 or greater (got {page})")
            # Requests filter status/priority by enum ACTION_CODE, not id.
            status_id = await resolve_enum_field_or_raise(
                cache, "status", status_name, kind="status", field="action_code"
            )
            priority_id = await resolve_enum_field_or_raise(
                cache, "priority", priority_name, kind="priority", field="action_code"
            )
            place_id = await resolve_enum_or_raise(cache, "place", place_name, kind="place")
            # Category type: the sb `type` column filters cat_type_id. Resolve
            # the NAME → cat_type_id via the same enum bucket create_request
            # uses; base_type is the raw base-type letter.
            cat_type_id = await resolve_enum_or_raise(
                cache, "cat_type_id", cat_type_name, kind="cat_type"
            )
            # sb_raw clauses on columns the live endpoint doesn't honor are
            # stripped (the backend would silently ignore them and return the
            # unfiltered set anyway) and reported via `unsupported_filters` so
            # the agent filters the items itself.
            dropped_clauses: list[dict[str, Any]] = []
            sb = build_request_filter(
                text_search=text_search,
                status_id=status_id,
                status_group=status_group,
                priority_id=priority_id,
                cat_type_id=cat_type_id,
                base_type=base_type,
                customer_id=company,
                solver_id=solver_id,
                solver_group_id=solver_group_id,
                catalog_id=catalog_id,
                deal_id=deal_id,
                project_contract_id=project_contract_id,
                branch_id=branch_id,
                place_id=place_id,
                superior_request_id=superior_request_id,
                periodic=periodic,
                due_after=due_after,
                due_before=due_before,
                created_after=created_after,
                created_before=created_before,
                sb_raw=sb_raw,
                dropped_cols_out=dropped_clauses,
            )
            params: dict[str, Any] = {"pg": page, "pp": per_page}
            fieldset, fields = _ensure_periodic_field(fieldset, fields)
            apply_field_scope(params, fieldset, fields)
            only = ask_for_names(params, "request", detail=False)
            if sort:
                params["sort"] = resolve_sort("requests", sort)
            # Flat query param, NOT an sb column (an sb deleted_at clause is
            # silently ignored). includeDeleted=1 surfaces soft-deleted
            # requests and composes with sb filters.
            # Requests are the ONLY module where this works — task/catalog
            # deletes are unreachable through the API.
            if include_deleted:
                params["includeDeleted"] = 1
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response, paging = await client.get_page("v3/request", params=params)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"list_requests input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_requests") from e

        records, meta = unwrap_list(response)
        add_names(records, "request", detail=False, fieldset=fieldset, fields=fields, only=only)
        for rec in records:
            _add_is_periodic(rec)
            _add_time_fields(rec)
        await add_enum_names(cache, records, _REQUEST_ENUM_LABELS)
        result = {"items": records, "meta": list_meta(meta, paging, dropped=dropped_clauses), "page": page, "per_page": per_page}
        if dropped_clauses:
            result["unsupported_filters"] = unsupported_filter_directive(dropped_clauses)
        return result

    @mcp.tool(
        description=_GET_REQUEST_DESC,
        annotations=ToolAnnotations(title="Get request", readOnlyHint=True),
    )
    async def get_request(
        id: int,
        fieldset: str | None = None,
        fields: list[str] | None = None,
        include_deleted: bool = False,
        include_allowed_statuses: bool = False,
    ) -> Any:
        try:
            validate_fieldset(fieldset)
            validate_fields(fields)
            fieldset, fields = _ensure_periodic_field(fieldset, fields)
            params: dict[str, Any] = {}
            if fieldset:
                params["fieldset"] = fieldset
            if fields:
                params["returnFields[]"] = fields
            ask_for_names(params, "request", detail=True)
            # Soft-deleted requests 404 on a plain GET; includeDeleted=1
            # returns them (request module only).
            if include_deleted:
                params["includeDeleted"] = 1
            response = await client.get(f"v3/request/{id}", params=params or None)
        except ValueError as e:
            raise RuntimeError(f"get_request input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="get_request", record_id=id) from e
        record = _add_time_fields(_add_is_periodic(unwrap_record(response)))
        add_names([record], "request", detail=True, fieldset=fieldset, fields=fields)
        await add_enum_names(cache, [record], _REQUEST_ENUM_LABELS)
        if include_allowed_statuses and isinstance(record, dict):
            try:
                rows = unwrap_record(await client.get(f"v3/request/{id}/statuses"))
            except Exception as e:
                raise to_llm_error(e, operation="get_request", record_id=id) from e
            record["allowed_statuses"] = _allowed_statuses(rows)
        return record

    @mcp.tool(
        description=_LIST_REQUEST_SOLVER_GROUPS_DESC,
        annotations=ToolAnnotations(title="List request solver groups", readOnlyHint=True),
    )
    async def list_request_solver_groups(
        customer_id: int,
        cat_area_id: int | None = None,
        cat_area_2nd_id: int | None = None,
        catalog_id: int | None = None,
        project_contract_id: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"company_id": customer_id, "include_metadata": "true"}
        for key, value in (
            ("cat_area_id", cat_area_id), ("cat_area_2nd_id", cat_area_2nd_id),
            ("catalog_id", catalog_id), ("project_contract_id", project_contract_id),
        ):
            if value is not None:
                params[key] = value
        try:
            response = await client.get("v3/request/solver-groups", params=params)
        except Exception as e:
            raise to_llm_error(e, operation="list_request_solver_groups") from e
        rows = unwrap_record(response)
        items = [
            {k: row[k] for k in _SOLVER_GROUP_FIELDS if k in row}
            for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)
        ]
        result: dict[str, Any] = {"items": items, "count": len(items)}
        preselected = response.get("preselected_solver_group_id") if isinstance(response, dict) else None
        if preselected is not None:
            result["preselected_solver_group_id"] = preselected
        if not items:
            result["note"] = (
                "No solver groups: the tenant does not use solver groups on "
                "requests, or none fits this customer / service area / catalog."
            )
        return result

    @mcp.tool(
        description=_GET_REQUEST_CUSTOM_FIELDS_DESC,
        annotations=ToolAnnotations(title="Get request custom fields", readOnlyHint=True),
    )
    async def get_request_custom_fields(catalog_id: int | None = None) -> Any:
        try:
            params = {"catalog_id": catalog_id} if catalog_id is not None else None
            response = await client.get("v3/request/custom-fields", params=params)
        except Exception as e:
            raise to_llm_error(e, operation="get_request_custom_fields") from e
        wrapped = wrap_collection(
            unwrap_record(response),
            kind="custom-field definitions for the Request module",
        )
        if catalog_id is not None and not wrapped.get("count"):
            # The endpoint answers an empty list for a nonexistent catalog just
            # as it does for a real one with no definitions, so the generic
            # "none exist" note would assert something we did not verify.
            wrapped["note"] = (
                f"No custom-field definitions came back for catalog_id="
                f"{catalog_id}. That means EITHER the catalog has none, OR no "
                f"such catalog exists — this endpoint answers an empty list for "
                f"both. Confirm the id via list_request_catalogs if it matters."
            )
        return wrapped

    @mcp.tool(
        description=_GET_REQUEST_ENUMS_DESC,
        annotations=ToolAnnotations(title="Get request enums", readOnlyHint=True),
    )
    async def get_request_enums(refresh: bool = False) -> dict[str, Any]:
        try:
            # Ensure the cache is populated — on a cold cache `snapshot()` is
            # empty until something triggers a load, so always load() (idempotent)
            # and refresh() only when asked.
            if refresh:
                await cache.refresh()
            else:
                await cache.load()
        except Exception as e:
            raise to_llm_error(e, operation="get_request_enums") from e
        result: dict[str, Any] = dict(cache.snapshot())
        result["status"] = with_status_groups(result.get("status"), request_status_group)
        result["fields"] = {
            "always_required": ["company", "description (unless a catalog hides it)",
                                "type_code or type_name"],
            "by_tenant_setting": {
                **describe(REQUEST_FIELDS, cache.extra("config")),
                **describe(REQUEST_BASE_FIELDS,
                           request_base_settings(await base_config.load(client))),
            },
            "note": (
                "State of each configurable create_request/update_request "
                "parameter on this tenant, from its live request settings: off "
                "(refused when given), optional, or required (must be given on "
                "create, cannot be cleared on update). Per-record rules — the "
                "service area per customer/category, SLA-computed deadlines, "
                "required custom fields per catalog — are checked by CDESK."
            ),
        }
        return result

    @mcp.tool(
        description=_CREATE_REQUEST_DESC,
        annotations=ToolAnnotations(title="Create request", destructiveHint=True),
    )
    async def create_request(
        company: int,
        title: str | None = None,
        description: str | None = None,
        status_name: str | None = None,
        type_name: str | None = None,
        type_code: str | None = None,
        priority_name: str | None = None,
        urgency_name: str | None = None,
        impact_name: str | None = None,
        place_name: str | None = None,
        cat_type_name: str | None = None,
        cat_area_name: str | None = None,
        cat_area_2nd_name: str | None = None,
        close_reason_name: str | None = None,
        solver_method_name: str | None = None,
        root_cause_name: str | None = None,
        solver_id: int | None = None,
        solver_group_id: int | None = None,
        catalog_id: int | None = None,
        deal_id: int | None = None,
        project_contract_id: int | None = None,
        branch_id: int | None = None,
        sla_id: int | None = None,
        superior_request_id: int | None = None,
        desired_date: str | None = None,
        due_date: str | None = None,
        due_date_fixed: bool | None = None,
        calendar_date_start: str | None = None,
        invoice_date: str | None = None,
        solution: str | None = None,
        solver_note: str | None = None,
        first_conclusion: str | None = None,
        note: str | None = None,
        code: str | None = None,
        computer_code: str | None = None,
        for_customer: bool | None = None,
        user_entered_by: int | None = None,
        contact_entered_by: int | None = None,
        entered_by: str | None = None,
        planned_hours: float | None = None,
        hour_rate: float | None = None,
        request_weight: float | None = None,
        percent_done: int | None = None,
        scheduling_mode: int | None = None,
        change_manager_id: int | None = None,
        reaction_missed_id_solver: int | None = None,
        # Advanced / integration fields. Documented as a group in the
        # tool description; accepted as optional kwargs so power users
        # can still set them without us enumerating each one.
        notify_block: int | None = None,
        notify_customer_block: int | None = None,
        notify_sms_block: int | None = None,
        from_email: str | None = None,
        from_name: str | None = None,
        to_email: str | None = None,
        to_name: str | None = None,
        ec_login: str | None = None,
        ec_email: str | None = None,
        ec_phone: str | None = None,
        extsys_id: str | None = None,
        extsys_code: str | None = None,
        extsys_status: str | None = None,
        custom_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            if title is not None and not title.strip():
                raise ValueError("title must be a non-empty string when given")
            if description is not None and not description.strip():
                raise ValueError("description must be a non-empty string when given")
            # Fields with no setting behind them are always required.
            fixed: list[str] = []
            if description is None and catalog_id is None:  # a catalog may hide it
                fixed.append("description is always required (unless a catalog hides it)")
            if type_code is None and type_name is None:
                fixed.append("type_code or type_name (the request type) is always required")
            # Everything else follows the TENANT's form, read live on every call:
            # an off field is refused, a required one must be given
            # (field_policy.py). This names the parameter and the setting where
            # CDESK answers a bare "Vyberte riešiteľa". Priority has no such
            # rule — CDESK takes the SLA's default.
            settings = await _request_settings(cache)
            refuse_if_violated(fixed + violations(
                REQUEST_FIELDS, settings, {
                    "solver_id": solver_id, "solver_group_id": solver_group_id,
                    "title": title, "code": code, "cat_type_name": cat_type_name,
                    "cat_area_name": cat_area_name, "cat_area_2nd_name": cat_area_2nd_name,
                    "desired_date": desired_date, "due_date": due_date,
                    "due_date_fixed": due_date_fixed or None,  # False = not fixed, never "set"
                    "close_reason_name": close_reason_name,
                    "solver_method_name": solver_method_name,
                    "root_cause_name": root_cause_name, "solution": solution,
                    "urgency_name": urgency_name, "impact_name": impact_name,
                    "catalog_id": catalog_id, "place_name": place_name,
                    "branch_id": branch_id,
                },
            ) + violations(
                REQUEST_BASE_FIELDS, request_base_settings(await base_config.load(client)),
                {"invoice_date": invoice_date, "percent_done": percent_done},
            ))
            # With auto-naming on, the CDESK web form fills an empty title from
            # the description before saving; the server itself does not (a
            # title-less create stores ''). Do what the form does so the
            # request is not saved nameless.
            if title is None:
                title = _auto_title(settings, description)
            validate_custom_fields(custom_fields)
            # Naive datetimes are interpreted as TENANT-LOCAL wall-clock time
            # and sent with an explicit offset — CDESK reads naive values as
            # UTC. Bare dates pass through unchanged.
            desired_date = localize_naive_datetime("desired_date", desired_date)
            due_date = localize_naive_datetime("due_date", due_date)
            calendar_date_start = localize_naive_datetime(
                "calendar_date_start", calendar_date_start
            )
            invoice_date = localize_naive_datetime("invoice_date", invoice_date)
            # first_conclusion is a TIMESTAMP column, even though CDESK's own
            # OpenAPI annotation declares it type="string". Text sent to it is
            # coerced to "0000-00-00 00:00:00" — a successful-looking write that
            # corrupts the field. Validate it as a datetime so a non-date is
            # rejected up front.
            first_conclusion = localize_naive_datetime(
                "first_conclusion", first_conclusion
            )

            # type_code guard: a blank string would be forwarded as `type: ""`
            # (only None is dropped from the body) on a column the backend is
            # known to mangle silently; and when BOTH type_code and type_name
            # are set, type_code would silently win — reject both cases, like
            # the sb_raw-vs-typed conflict guard elsewhere.
            if type_code is not None and not type_code.strip():
                raise ValueError("type_code must be a non-empty string when provided")
            if type_code is not None and type_name is not None:
                raise ValueError(
                    "provide either type_code or type_name, not both "
                    "(type_code would silently override type_name)"
                )
            # CDESK stores req_status and req_priority by the enum ACTION_CODE,
            # NOT the enum id — sending the id overflows the tinyint column and
            # silently defaults/corrupts the value.
            status_id = await resolve_enum_field_or_raise(
                cache, "status", status_name, kind="status", field="action_code"
            )
            type_id = await resolve_enum_or_raise(cache, "type", type_name, kind="type")
            priority_id = await resolve_enum_field_or_raise(
                cache, "priority", priority_name, kind="priority", field="action_code"
            )
            urgency_id = await resolve_enum_or_raise(
                cache, "urgency", urgency_name, kind="urgency"
            )
            impact_id = await resolve_enum_or_raise(cache, "impact", impact_name, kind="impact")
            place_id = await resolve_enum_or_raise(cache, "place", place_name, kind="place")
            cat_type_id = await resolve_enum_or_raise(
                cache, "cat_type_id", cat_type_name, kind="cat_type"
            )
            cat_area_id = await resolve_enum_or_raise(
                cache, "cat_area", cat_area_name, kind="cat_area"
            )
            # cat_area_2nd is hierarchical: it needs its parent area to resolve.
            # If only cat_area_2nd_name is given (cat_area_id is None) it would
            # resolve unscoped (wrong parent / AmbiguousEnumNameError), so require
            # cat_area_name alongside it.
            if cat_area_2nd_name is not None and cat_area_id is None:
                raise ValueError(
                    "provide cat_area_name when setting cat_area_2nd_name so the "
                    "2nd-level area resolves under the right parent"
                )
            cat_area_2nd_id = await resolve_enum_or_raise(
                cache, "cat_area_2nd", cat_area_2nd_name, kind="cat_area_2nd",
                parent_id=cat_area_id,
            )
            close_reason_id = await resolve_enum_or_raise(
                cache, "close_reasons", close_reason_name, kind="close_reason"
            )
            solver_method_id = await resolve_enum_or_raise(
                cache, "solver_method", solver_method_name, kind="solver_method"
            )
            root_cause_id = await resolve_enum_or_raise(
                cache, "root_causes", root_cause_name, kind="root_cause"
            )

            body = _build_request_body(
                {
                    "customer_id": company,
                    "title": title,
                    "description": description,
                    "status": status_id,
                    # The request `type` ("druh") is a string code (e.g. "H");
                    # where the type enum is key/value-shaped, type_name can't
                    # resolve to an id. Prefer the explicit type_code.
                    "type": type_code if type_code is not None else type_id,
                    "priority": priority_id,
                    "urgency_id": urgency_id,
                    "impact_id": impact_id,
                    "place_id": place_id,
                    "cat_type_id": cat_type_id,
                    "cat_area_id": cat_area_id,
                    "cat_area_2nd_id": cat_area_2nd_id,
                    "close_reason_id": close_reason_id,
                    "solver_method_id": solver_method_id,
                    "root_cause_id": root_cause_id,
                    "solver_id": solver_id,
                    "solver_group_id": solver_group_id,
                    "catalog_id": catalog_id,
                    "deal_id": deal_id,
                    "project_contract_id": project_contract_id,
                    "branch_id": branch_id,
                    "sla_id": sla_id,
                    "superior_request_id": superior_request_id,
                    "desired_date": desired_date,
                    "due_date": due_date,
                    "due_date_fixed": None if due_date_fixed is None else int(due_date_fixed),
                    "calendar_date_start": calendar_date_start,
                    "invoice_date": invoice_date,
                    "solution": solution,
                    "solver_note": solver_note,
                    "first_conclusion": first_conclusion,
                    "note": note,
                    "code": code,
                    "computer_code": computer_code,
                    # CDESK's wire `for_customer` flag is INVERTED vs its name —
                    # wire True hides the request from the customer (True ->
                    # hidden_for_customer=1). Send the negation so this tool's
                    # for_customer=True means customer-VISIBLE, per the param
                    # name + OpenAPI.
                    "for_customer": (None if for_customer is None else not for_customer),
                    "user_entered_by": user_entered_by,
                    "contact_entered_by": contact_entered_by,
                    "entered_by": entered_by,
                    "planned_hours": planned_hours,
                    "hour_rate": hour_rate,
                    "request_weight": request_weight,
                    "percent_done_manual": percent_done,
                    "scheduling_mode": scheduling_mode,
                    "change_manager_id": change_manager_id,
                    "reaction_missed_id_solver": reaction_missed_id_solver,
                    "notify_block": notify_block_value(notify_block),
                    "notify_customer_block": notify_customer_block,
                    "notify_sms_block": notify_sms_block,
                    "from_email": from_email,
                    "from_name": from_name,
                    "to_email": to_email,
                    "to_name": to_name,
                    "ec_login": ec_login,
                    "ec_email": ec_email,
                    "ec_phone": ec_phone,
                    "extsys_id": extsys_id,
                    "extsys_code": extsys_code,
                    "extsys_status": extsys_status,
                }
            )
            # customFields keys are already wire-format (cfield_*) — set after
            # the body build so they bypass the LLM-name→CDESK-name rename map.
            if custom_fields:
                body["customFields"] = custom_fields
            response = await client.post("v3/request", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"create_request input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="create_request") from e
        # Gated fields such as code and solver_note can be dropped silently (a
        # PUT carrying solver_note is then a complete no-op — 200, not stored,
        # udatetime not even bumped), so diff them against savedData.
        # Labels go on AFTER the diff, so it compares only CDESK's own keys.
        return await _label_write_result(cache, annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=tuple(_REQUEST_GATED_FIELDS),
            field_notes=_REQUEST_GATED_FIELDS,
        ))

    @mcp.tool(
        description=_UPDATE_REQUEST_DESC,
        annotations=ToolAnnotations(title="Update request", destructiveHint=True),
    )
    async def update_request(
        id: int,
        title: str | None = None,
        description: str | None = None,
        status_name: str | None = None,
        type_name: str | None = None,
        type_code: str | None = None,
        priority_name: str | None = None,
        urgency_name: str | None = None,
        impact_name: str | None = None,
        place_name: str | None = None,
        cat_type_name: str | None = None,
        cat_area_name: str | None = None,
        cat_area_2nd_name: str | None = None,
        close_reason_name: str | None = None,
        solver_method_name: str | None = None,
        root_cause_name: str | None = None,
        company: int | None = None,
        solver_id: int | None = None,
        solver_group_id: int | None = None,
        catalog_id: int | None = None,
        deal_id: int | None = None,
        project_contract_id: int | None = None,
        branch_id: int | None = None,
        sla_id: int | None = None,
        superior_request_id: int | None = None,
        desired_date: str | None = None,
        due_date: str | None = None,
        due_date_fixed: bool | None = None,
        calendar_date_start: str | None = None,
        invoice_date: str | None = None,
        solution: str | None = None,
        solver_note: str | None = None,
        first_conclusion: str | None = None,
        note: str | None = None,
        code: str | None = None,
        for_customer: bool | None = None,
        planned_hours: float | None = None,
        hour_rate: float | None = None,
        percent_done: int | None = None,
        custom_fields: dict[str, Any] | None = None,
        notify_block: int | None = None,
        notify_customer_block: int | None = None,
        notify_sms_block: int | None = None,
    ) -> dict[str, Any]:
        try:
            if title is not None and not title.strip():
                raise ValueError("title must be a non-empty string")
            validate_custom_fields(custom_fields)
            # Naive datetimes are interpreted as TENANT-LOCAL wall-clock time
            # and sent with an explicit offset — CDESK reads naive values as
            # UTC. Bare dates pass through unchanged.
            desired_date = localize_naive_datetime("desired_date", desired_date)
            due_date = localize_naive_datetime("due_date", due_date)
            calendar_date_start = localize_naive_datetime(
                "calendar_date_start", calendar_date_start
            )
            invoice_date = localize_naive_datetime("invoice_date", invoice_date)
            # first_conclusion is a TIMESTAMP column, even though CDESK's own
            # OpenAPI annotation declares it type="string". Text sent to it is
            # coerced to "0000-00-00 00:00:00" — a successful-looking write that
            # corrupts the field. Validate it as a datetime so a non-date is
            # rejected up front.
            first_conclusion = localize_naive_datetime(
                "first_conclusion", first_conclusion
            )

            # type_code guard: a blank string would be forwarded as `type: ""`
            # (only None is dropped from the body) on a column the backend is
            # known to mangle silently; and when BOTH type_code and type_name
            # are set, type_code would silently win — reject both cases, like
            # the sb_raw-vs-typed conflict guard elsewhere.
            if type_code is not None and not type_code.strip():
                raise ValueError("type_code must be a non-empty string when provided")
            if type_code is not None and type_name is not None:
                raise ValueError(
                    "provide either type_code or type_name, not both "
                    "(type_code would silently override type_name)"
                )
            # The tenant's form rules apply to updates too: an off field may not
            # be set, a required one not cleared (left out = unchanged). Checked
            # before any name is resolved, so an off field is reported as off
            # rather than as an unresolvable name.
            refuse_if_violated(violations(
                REQUEST_FIELDS, await _request_settings(cache), {
                    "solver_id": solver_id, "solver_group_id": solver_group_id,
                    "title": title, "code": code, "cat_type_name": cat_type_name,
                    "cat_area_name": cat_area_name, "cat_area_2nd_name": cat_area_2nd_name,
                    "desired_date": desired_date, "due_date": due_date,
                    "due_date_fixed": due_date_fixed or None,  # False = not fixed, never "set"
                    "close_reason_name": close_reason_name,
                    "solver_method_name": solver_method_name,
                    "root_cause_name": root_cause_name, "solution": solution,
                    "urgency_name": urgency_name, "impact_name": impact_name,
                    "catalog_id": catalog_id, "place_name": place_name,
                    "branch_id": branch_id,
                },
                update=True,
            ) + violations(
                REQUEST_BASE_FIELDS, request_base_settings(await base_config.load(client)),
                {"invoice_date": invoice_date, "percent_done": percent_done},
                update=True,
            ))
            # CDESK stores req_status and req_priority by the enum ACTION_CODE,
            # NOT the enum id — sending the id overflows the tinyint column and
            # silently defaults/corrupts the value.
            status_id = await resolve_enum_field_or_raise(
                cache, "status", status_name, kind="status", field="action_code"
            )
            type_id = await resolve_enum_or_raise(cache, "type", type_name, kind="type")
            priority_id = await resolve_enum_field_or_raise(
                cache, "priority", priority_name, kind="priority", field="action_code"
            )
            urgency_id = await resolve_enum_or_raise(
                cache, "urgency", urgency_name, kind="urgency"
            )
            impact_id = await resolve_enum_or_raise(cache, "impact", impact_name, kind="impact")
            place_id = await resolve_enum_or_raise(cache, "place", place_name, kind="place")
            cat_type_id = await resolve_enum_or_raise(
                cache, "cat_type_id", cat_type_name, kind="cat_type"
            )
            cat_area_id = await resolve_enum_or_raise(
                cache, "cat_area", cat_area_name, kind="cat_area"
            )
            # cat_area_2nd is hierarchical: it needs its parent area to resolve.
            # If only cat_area_2nd_name is given (cat_area_id is None) it would
            # resolve unscoped (wrong parent / AmbiguousEnumNameError), so require
            # cat_area_name alongside it.
            if cat_area_2nd_name is not None and cat_area_id is None:
                raise ValueError(
                    "provide cat_area_name when setting cat_area_2nd_name so the "
                    "2nd-level area resolves under the right parent"
                )
            cat_area_2nd_id = await resolve_enum_or_raise(
                cache, "cat_area_2nd", cat_area_2nd_name, kind="cat_area_2nd",
                parent_id=cat_area_id,
            )
            close_reason_id = await resolve_enum_or_raise(
                cache, "close_reasons", close_reason_name, kind="close_reason"
            )
            solver_method_id = await resolve_enum_or_raise(
                cache, "solver_method", solver_method_name, kind="solver_method"
            )
            root_cause_id = await resolve_enum_or_raise(
                cache, "root_causes", root_cause_name, kind="root_cause"
            )

            current = await _fetch_for_update_request(client, id)
            await _check_status_allowed(client, id, current, status_id)

            body = _build_request_body(
                {
                    "title": title,
                    "description": description,
                    "status": status_id,
                    # The request `type` ("druh") is a string code (e.g. "H");
                    # where the type enum is key/value-shaped, type_name can't
                    # resolve to an id. Prefer the explicit type_code.
                    "type": type_code if type_code is not None else type_id,
                    "priority": priority_id,
                    "urgency_id": urgency_id,
                    "impact_id": impact_id,
                    "place_id": place_id,
                    "cat_type_id": cat_type_id,
                    "cat_area_id": cat_area_id,
                    "cat_area_2nd_id": cat_area_2nd_id,
                    "close_reason_id": close_reason_id,
                    "solver_method_id": solver_method_id,
                    "root_cause_id": root_cause_id,
                    "customer_id": company,
                    "solver_id": solver_id,
                    "solver_group_id": solver_group_id,
                    "catalog_id": catalog_id,
                    "deal_id": deal_id,
                    "project_contract_id": project_contract_id,
                    "branch_id": branch_id,
                    "sla_id": sla_id,
                    "superior_request_id": superior_request_id,
                    "desired_date": desired_date,
                    "due_date": due_date,
                    "due_date_fixed": None if due_date_fixed is None else int(due_date_fixed),
                    "calendar_date_start": calendar_date_start,
                    "invoice_date": invoice_date,
                    "solution": solution,
                    "solver_note": solver_note,
                    "first_conclusion": first_conclusion,
                    "note": note,
                    "code": code,
                    # CDESK's wire `for_customer` flag is INVERTED vs its name —
                    # wire True hides the request from the customer (True ->
                    # hidden_for_customer=1). Send the negation so this tool's
                    # for_customer=True means customer-VISIBLE, per the param
                    # name + OpenAPI.
                    "for_customer": (None if for_customer is None else not for_customer),
                    "planned_hours": planned_hours,
                    "hour_rate": hour_rate,
                    "percent_done_manual": percent_done,
                    "notify_block": notify_block_value(notify_block),
                    "notify_customer_block": notify_customer_block,
                    "notify_sms_block": notify_sms_block,
                }
            )
            # customFields keys are already wire-format (cfield_*) — set after
            # the body build so they bypass the LLM-name→CDESK-name rename map.
            if custom_fields:
                body["customFields"] = custom_fields
            if not _without_notify_flags(body):
                # An update with no field would still issue a PUT that changes
                # nothing and bumps `udatetime` on a destructive-hinted tool.
                # Fail before the write so "nothing happened" is explicit.
                raise ValueError(
                    "nothing to update — pass at least one field besides `id`. "
                    "An empty update would still write to the tenant and bump "
                    "the record's timestamp without changing anything."
                )
            body["timestamp_check"] = current["timestamp_check"]
            response = await client.put(f"v3/request/{id}", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"update_request input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="update_request", record_id=id) from e
        # solver_note: silently dropped like code (see create_request note).
        result = annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=tuple(_REQUEST_GATED_FIELDS),
            field_notes=_REQUEST_GATED_FIELDS,
        )
        result = _with_direct_close_warning(result, current.get("status"), body.get("status"))
        return await _label_write_result(cache, result)

    @mcp.tool(
        description=_DELETE_REQUEST_DESC,
        annotations=ToolAnnotations(title="Delete request", destructiveHint=True),
    )
    async def delete_request(id: int) -> dict[str, Any]:
        try:
            await client.delete(f"v3/request/{id}")
        except Exception as e:
            raise to_llm_error(e, operation="delete_request", record_id=id) from e
        return {"deleted": id}

    @mcp.tool(
        description=_SET_REQUEST_STATUS_DESC,
        annotations=ToolAnnotations(title="Set request status", destructiveHint=True),
    )
    async def set_request_status(
        id: int,
        status_name: str,
        notify_block: int | None = None,
        notify_customer_block: int | None = None,
        notify_sms_block: int | None = None,
    ) -> dict[str, Any]:
        try:
            # req_status is stored/matched by enum ACTION_CODE, not the enum id.
            status_id = await resolve_enum_field_or_raise(
                cache, "status", status_name, kind="status", field="action_code"
            )
            if status_id is None:
                raise RuntimeError("set_request_status requires a non-empty status_name")
            current = await _fetch_for_update_request(client, id)
            await _check_status_allowed(client, id, current, status_id)
            body: dict[str, Any] = {"status": status_id, "timestamp_check": current["timestamp_check"]}
            for key, value in (
                ("notify_block", notify_block_value(notify_block)),
                ("notify_customer_block", notify_customer_block),
                ("notify_sms_block", notify_sms_block),
            ):
                if value is not None:
                    body[key] = value
            response = await client.put(f"v3/request/{id}", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"set_request_status input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="set_request_status", record_id=id) from e
        result = _with_direct_close_warning(
            response if isinstance(response, dict) else {"data": response},
            current.get("status"), status_id,
        )
        return await _label_write_result(cache, result)

    @mcp.tool(
        description=_LIST_REQUEST_DISCUSSION_DESC,
        annotations=ToolAnnotations(title="List request discussion", readOnlyHint=True),
    )
    async def list_request_discussion(
        request_id: int,
        include_deleted: bool = False,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            validate_fields(fields)
            params: dict[str, Any] = {}
            if include_deleted:
                params["includeDeleted"] = "true"
            if fields:
                params["returnFields[]"] = fields
            response = await client.get(
                f"v3/request/{request_id}/discussion",
                params=params if params else None,
            )
        except ValueError as e:
            raise RuntimeError(f"list_request_discussion input error: {e}") from e
        except Exception as e:
            raise to_llm_error(
                e, operation="list_request_discussion", record_id=request_id,
            ) from e
        # Normalize to the same shape as the other list tools. Discussion isn't
        # paginated, so there's no page/per_page — but unwrap_list still lifts
        # the messages out of `data` and keeps discussionAcl/config/signature
        # metadata in `meta`.
        records, meta = unwrap_list(response)
        # With a `fields` whitelist, CDESK answers an empty LIST for any message
        # that carries none of the requested columns — so `items` could come back
        # as [[], [], {...}, {...}], where the bare [] entries are
        # indistinguishable from empty records. Entries differ in shape (a
        # status-change entry has different keys from a posted message), so this
        # is per-record: the mixed case is the common one, because any request
        # with a status change is mixed.
        kept = [r for r in records if not (isinstance(r, list) and not r)]
        dropped = len(records) - len(kept)
        result: dict[str, Any] = {"items": kept, "meta": meta}
        if dropped:
            result["note"] = (
                f"{dropped} of {len(records)} message(s) carried none of the "
                f"requested `fields` ({', '.join(fields or [])}) and came back "
                f"empty, so they are omitted from `items` — discussion entries "
                f"differ in shape (a status-change entry has different keys from "
                f"a posted message), and a field valid on one may not exist on "
                f"another. An unfiltered call shows every entry with its real "
                f"keys."
            )
        return result

    @mcp.tool(
        description=_POST_REQUEST_DISCUSSION_DESC,
        annotations=ToolAnnotations(title="Post request discussion", destructiveHint=True),
    )
    async def post_request_discussion(
        request_id: int,
        channel: Literal["customer", "internal"],
        text: str | None = None,
        attachments: list[str] | None = None,
        change_status_to_waiting_for_customer: bool = False,
        work_order_id: int | None = None,
        customer_signature: int | None = None,
        notify_user_ids: list[int] | None = None,
        discussion_receivers: list[str] | None = None,
        custom_notify: bool = False,
        notify_block: int | None = None,
        notify_customer_block: int | None = None,
        notify_sms_block: int | None = None,
    ) -> dict[str, Any]:
        try:
            receivers = _validate_receiver_emails(discussion_receivers)
            if notify_block == 1 and (receivers or notify_user_ids):
                raise ValueError(
                    "notify_block=1 notifies nobody, so it cannot be combined with "
                    "notify_user_ids or discussion_receivers — leave out either the "
                    "block or the receivers."
                )
            if custom_notify and not (receivers or notify_user_ids):
                raise ValueError(
                    "custom_notify=True replaces CDESK's default receivers, so it "
                    "needs at least one entry in notify_user_ids or "
                    "discussion_receivers; with none, nobody would be notified."
                )
        except ValueError as e:
            raise RuntimeError(f"post_request_discussion input error: {e}") from e
        # Resolved before the POST: a receiver that can't be resolved must fail
        # the call, not produce a post that looks notified and wasn't.
        users = await _resolve_notify_users(client, notify_user_ids or [])
        manual = _merge_receivers(users, receivers)
        try:
            body = _build_discussion_body(
                channel=channel,
                text=text,
                attachments=attachments,
                change_status_to_waiting_for_customer=(
                    change_status_to_waiting_for_customer
                ),
                work_order_id=work_order_id,
                customer_signature=customer_signature,
                custom_notify=custom_notify,
                manual_receivers=manual,
                notify_block=notify_block,
                notify_customer_block=notify_customer_block,
                notify_sms_block=notify_sms_block,
            )
            response = await client.post(
                f"v3/request/{request_id}/discussion", json=body,
            )
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"post_request_discussion input error: {e}") from e
        except Exception as e:
            raise to_llm_error(
                e, operation="post_request_discussion", record_id=request_id,
            ) from e
        result = response if isinstance(response, dict) else {"data": response}
        if manual:
            result["notification"] = _notification_report(result, manual)
        return result


    # FastMCP builds each tool's argument model with pydantic's default
    # extra="ignore", so an unrecognised keyword is dropped BEFORE the tool body
    # runs — our code never sees it. On update_request, a typo like titel= (for
    # title=), or a create-only field such as computer_code=, would otherwise
    # succeed while changing nothing and saying nothing. Forbid extras so the
    # caller gets a validation error naming the offending key instead of a
    # silent no-op.
    #
    # Scoped to this module deliberately: forbidding extras on every tool would
    # reject any harmless extra a host might attach.
    forbid_unknown_arguments(mcp, _REQUEST_TOOL_NAMES)


# --- internals ----------------------------------------------------------

async def _fetch_for_update_request(client: CdeskClient, id: int) -> dict[str, Any]:
    """Fetch a request and unwrap to the record dict, ready to feed the
    optimistic lock into a follow-up PUT.

    CDESK's Request module reads `timestamp_check` from the PUT body, same
    as Task/Customer/User — per the OpenAPI spec's PUT description
    ("required: timestamp_check, must equal the updated_at from last fetch").

    fieldset=all is required: the v3 fieldset default (`extended`) does not
    include `timestamp_check` — only `all` carries it."""
    envelope = await client.get(f"v3/request/{id}", params={"fieldset": "all"})
    record = unwrap_record(envelope)
    if not isinstance(record, dict):
        raise RuntimeError(
            f"unexpected response shape fetching request {id} for update: "
            f"{type(record).__name__}"
        )
    if "timestamp_check" not in record:
        raise RuntimeError(
            f"request {id} record has no timestamp_check field; cannot apply "
            f"optimistic lock"
        )
    return record



def _build_request_body(public_fields: dict[str, Any]) -> dict[str, Any]:
    """Map LLM-facing field names → CDESK field names, dropping Nones.

    Server-computed fields (req_num, creator, updatedby, close/reaction
    dates) are never sent — we don't expose them as tool params, so this
    layer just enforces the rename + None-filter deal."""
    body: dict[str, Any] = {}
    for public_name, value in public_fields.items():
        if value is None:
            continue
        cdesk_name = _REQUEST_CREATE_FIELDS.get(public_name, public_name)
        body[cdesk_name] = value
    return body


def _build_discussion_body(
    *,
    channel: Literal["customer", "internal"],
    text: str | None,
    attachments: list[str] | None,
    change_status_to_waiting_for_customer: bool,
    work_order_id: int | None,
    customer_signature: int | None,
    custom_notify: bool,
    manual_receivers: list[dict[str, Any]],
    notify_block: int | None,
    notify_customer_block: int | None,
    notify_sms_block: int | None,
) -> dict[str, Any]:
    """Build the POST body for adding a discussion message.

    Channel routing:
      - channel="customer" → status=1, text lands in customer_text,
        receivers in customer_to, customer-targeted attachments.
      - channel="internal" → status=2, text lands in technote_text,
        receivers in technote_to, internal attachments.

    Extra receivers go in `<channel>_to.manual` as `{notify, name, type: 0}`
    entries — the shape the CDESK UI sends; CDESK refuses the spec's
    top-level `discussion_receivers` key. Without `customNotify` the manual
    entries are added to the default receivers; with it they replace them.

    Either text OR attachments must be present (CDESK rejects an empty
    message)."""
    if channel not in ("customer", "internal"):
        raise ValueError(
            f"channel must be 'customer' or 'internal', got {channel!r}"
        )

    if not text and not attachments:
        raise ValueError(
            "post_request_discussion requires either `text` or `attachments` "
            "(or both); CDESK rejects empty messages."
        )

    body: dict[str, Any] = {"status": 1 if channel == "customer" else 2}

    if channel == "customer":
        if text:
            body["customer_text"] = text
        if attachments:
            body["attachment"] = attachments
        if customer_signature is not None:
            body["customer_signature"] = customer_signature
    else:
        if text:
            body["technote_text"] = text
        if attachments:
            body["attachment_internal"] = attachments

    to_key = "customer_to" if channel == "customer" else "technote_to"
    if manual_receivers:
        body[to_key] = {"auto": [], "manual": manual_receivers}
        if custom_notify:
            body["customNotify"] = True

    if change_status_to_waiting_for_customer:
        body["changeStatusToWaitingForCustomer"] = True
    if work_order_id is not None:
        body["work_order_id"] = work_order_id
    # `notify_block` (snake case) silences the post itself and the status
    # change it causes; the camel-case `notifyBlock` would stop only the latter.
    block = notify_block_value(notify_block)
    if block is not None:
        body["notify_block"] = block
    if notify_customer_block is not None:
        body["notifyCustomerBlock"] = notify_customer_block
    if notify_sms_block is not None:
        body["notifySmsBlock"] = notify_sms_block

    return body


# Deliberately loose: CDESK stores any string as a receiver (even `not-an-email`
# is accepted and reported as notified), so this only has to catch what is
# plainly not an address.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s.]+$")


def _validate_receiver_emails(emails: list[str] | None) -> list[str]:
    """Strip and format-check the raw receiver addresses; raise ValueError
    naming every bad one."""
    if not emails:
        return []
    cleaned = [e.strip() if isinstance(e, str) else e for e in emails]
    bad = [e for e in cleaned if not isinstance(e, str) or not _EMAIL_RE.match(e)]
    if bad:
        raise ValueError(
            f"discussion_receivers contains values that are not email "
            f"addresses: {bad!r}"
        )
    return cleaned


async def _resolve_notify_users(
    client: CdeskClient, user_ids: list[int],
) -> list[tuple[int, str, str]]:
    """Look up each user's (id, email, name) via GET v3/user/{id}.

    Raises RuntimeError naming the id when a user can't be read or has no
    email, so the post is never sent without a receiver that was asked for."""
    async def one(uid: int) -> tuple[int, str, str]:
        try:
            envelope = await client.get(
                f"v3/user/{uid}", params={"returnFields[]": ["id", "email", "name"]},
            )
        except Exception as e:
            raise to_llm_error(
                e, operation="post_request_discussion (notify_user_ids lookup)",
                record_id=uid,
            ) from e
        record = unwrap_record(envelope)
        email = record.get("email") if isinstance(record, dict) else None
        if not isinstance(email, str) or not email.strip():
            raise RuntimeError(
                f"post_request_discussion input error: user {uid} has no email "
                f"address in CDESK, so it cannot be notified; nothing was posted."
            )
        name = record.get("name") if isinstance(record, dict) else None
        return uid, email.strip(), name if isinstance(name, str) and name else email.strip()

    return list(await asyncio.gather(*(one(uid) for uid in dict.fromkeys(user_ids))))


def _merge_receivers(
    users: list[tuple[int, str, str]], emails: list[str],
) -> list[dict[str, Any]]:
    """The `manual` entries, deduplicated case-insensitively by address. Each
    carries `name` because an entry without one is stored but shows as an
    empty receiver list in CDESK.

    CDESK users also carry `id` + `user_src: "user"`, the shape the UI's
    receiver picker sends: the backend then adds the in-app (bell) copy itself
    and routes the person as a user rather than an outside address, so their
    own notification settings apply. Plain addresses have no CDESK user and
    stay email-only. Users go first, so an address given both ways keeps the
    user entry."""
    merged: dict[str, dict[str, Any]] = {}
    for uid, email, name in users:
        merged.setdefault(email.lower(), {
            "notify": email, "name": name, "type": 0, "id": uid, "user_src": "user",
        })
    for email in emails:
        merged.setdefault(email.lower(), {"notify": email, "name": email, "type": 0})
    return list(merged.values())


def _notification_report(
    response: dict[str, Any], manual: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare the receivers sent with the ones CDESK stored on the post.

    `notify_email_to` comes back as a JSON-encoded string on the POST (a list
    on the discussion GET); `msg.success` carries CDESK's "notification sent"
    text, and is absent when nothing was sent. The list holds one entry per
    channel, so only email entries (type 0) count as recorded receivers — a
    user's in-app copy (type 4) can carry the same address."""
    data = response.get("data")
    raw = data.get("notify_email_to") if isinstance(data, dict) else None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else []
        except json.JSONDecodeError:
            raw = []
    entries = [r for r in (raw if isinstance(raw, list) else []) if isinstance(r, dict)]
    recorded = list(dict.fromkeys(
        str(r["notify"]) for r in entries
        if str(r.get("type", 0)) == "0"
        and isinstance(r.get("notify"), str) and "@" in r["notify"]
    ))
    in_app_ids = sorted({
        int(r["id"]) for r in entries
        if str(r.get("type")) == "4" and isinstance(r.get("id"), int)
    })
    recorded_lower = {r.lower() for r in recorded}
    requested = [str(m["notify"]) for m in manual]
    missing = [r for r in requested if r.lower() not in recorded_lower]

    msg = response.get("msg")
    success = msg.get("success") if isinstance(msg, dict) else None
    confirmations = [
        str(s.get("message") if isinstance(s, dict) else s)
        for s in (success if isinstance(success, list) else [])
    ]

    report: dict[str, Any] = {
        "requested": requested,
        "recorded_receivers": recorded,
        "in_app_recorded_for_user_ids": in_app_ids,
        "cdesk_confirmation": confirmations,
    }
    if missing or not confirmations:
        report["warning"] = (
            "The post was saved, but CDESK did not confirm the notification as "
            "requested: "
            + (f"these receivers are not in the post's notified list: {missing}. "
               if missing else "")
            + ("CDESK returned no notification-sent message. " if not confirmations else "")
            + "These receivers have not been shown to be notified."
        )
    return report


# --- Tool descriptions --------------------------------------------------

_LIST_REQUESTS_DESC = inspect.cleandoc(
    """
    List CDESK requests (helpdesk tickets) visible to your account. A
    request is the customer-facing ticket entity — distinct from internal
    tasks. Returns a paginated `items` array; `meta.total` is the number of records matching the filter across all pages (CDESK's own count) and `meta.last_page` tells whether this is the last page. A count question
    ("how many …") is one call with per_page=1, read meta.total.

    Common filters:
      - text_search: free-text across the request's default columns.
      - status_name / priority_name: resolved via get_request_enums to
        tenant-specific ids (diacritic-insensitive, Slovak/Czech/English).
      - cat_type_name: the request CATEGORY type by name — resolved via
        get_request_enums to its cat_type_id (do NOT pass the base-type
        letter here). base_type: the base-type LETTER itself (e.g. "H").
      - company / solver_id: int or list[int].
      - catalog_id: filter by the request-catalog template the ticket
        was created from (0 / null = ad-hoc).
      - solver_group_id / branch_id / place_name /
        superior_request_id: further scoping.
      - periodic: True → only periodic requests (those generated from a
        periodical request), False → only ad-hoc requests, omitted → both.

    Every returned item carries a derived `is_periodic` boolean (True when
    the request was generated from a periodical request) alongside the raw
    `periodical_request_id` it is derived from.

    Each record carries `customer_name` / `solver_name` next to the
    customer and solver ids.
    Enum fields carry their label alongside: `status_name`, `priority_name`,
    `type_name`, `cat_type_name`, … sit next to the raw value. The raw
    `status` and `priority` are enum ACTION CODES, not names or ids —
    `priority: 2` is whatever entry has action code 2 (on some tenants
    "dlhodobá"), not "2 - vysoká". The `*_name` values are the names the
    status_name / priority_name / … parameters accept. A value with no
    label (empty, or a category switched off on the tenant) has no
    `*_name` key.

    Working filters: text_search, status_name, priority_name, cat_type_name,
    base_type, company, solver_id, solver_group_id, catalog_id, deal_id,
    project_contract_id, branch_id, place_name, superior_request_id, periodic,
    due_after/due_before, created_after/created_before, include_deleted.
      - due_after / due_before: deadline (req_due_date) window, ISO-8601.
        CAVEAT: due_before alone also returns every request that has NO due
        date — the server's upper bound lets undated records through. To mean
        "has a deadline before X", pass due_after as well (e.g. a far-past
        date) or drop the undated items client-side.
      - created_after / created_before: creation-date window (ISO-8601; a
        bare date is inclusive — start-of-day for after, end-of-day for
        before; a naive datetime is treated as tenant-local time).
      - include_deleted: also return soft-deleted requests (composes with
        the other filters). Requests are the ONLY module where deleted
        records are reachable; deleted tasks/catalogs are gone for good.
    NOT offered (the CDESK v3 request list silently ignores or
    mis-evaluates them): request type ("druh"), sla_id.
    Use client-side filtering for those.

    For advanced filters not exposed here, pass sb_raw as a CDESK sb object
    (JSON string or object) in the structured tree form. Only the
    working columns are applied server-side (id_req, req_num,
    code, urgency_id, impact_id, id_company, catalog_id, id_solver,
    solver_group_id, id_rc, project_contract_id, cat_area_id, branch_id,
    place_id, change_manager_id, superior_request_id, periodical_request_id,
    request_weight, percent_done, status, priority, type (INTEGER cat-type id),
    baseType (base-type letter), dateFrom/dateTo (raw due-date bounds; dateTo
    also passes undated requests), created_at, assign_date, udatetime,
    due_date, close_date (W3C datetime values), plus the no-col text leaf).
    Clauses on any other column are STRIPPED, and the response carries an
    `unsupported_filters` block naming them.

    Response shaping (same semantics as get_request): fieldset selects the
    field group per record ("base"/"extended"/"all"/"custom"); fields is a
    whitelist of field names (returnFields; union with fieldset);
    `customer_name` / `solver_name` are added to it.
    Use fieldset="base" or fields=[...] to keep large lists compact.

    status_group filters by CDESK's own status groups (what its UI quick filters
    use): "open" (otvorené — every status below 80), "completed" (vyriešené —
    80, 81, 85, 86), "closed" (uzavreté — ONLY 90, 91, 95: accepted, without
    acceptance, refused), "closed_and_completed" (vyriešené, uzavreté — both;
    i.e. everything no longer open) and "waiting_for_response" (čaká na
    odozvu). A request in "ukončené" (80) is completed, not closed; "no longer
    open" is closed_and_completed. Combine with per_page=1 and meta.total to
    count. get_request_enums lists each status's group.

    Time fields: `age_hours_calendar` is the calendar time from creation to
    close (to now while the request is open), in hours. CDESK's own
    `duration_open` / `duration_reaction` (seconds; with fieldset "all" and in
    write results) are SLA WORKING time — only inside the SLA's operating
    hours, without holidays and without time in statuses that stop the SLA
    clock — not calendar time. `duration_open` is set when the request is
    closed: it is 0 while open, after a reopen, and without an SLA, so
    `sla_open_hours` is null then; `sla_reaction_hours` is null until the
    first reaction.
    """
)

# A solver-group row of GET /v3/request/solver-groups is a group's user record;
# these keys identify it and say why the form lists it first.
_SOLVER_GROUP_FIELDS = (
    "id", "name", "email", "assigned_group_to_catarea", "assigned_group_to_company",
)


def _allowed_statuses(rows: Any) -> list[dict[str, Any]]:
    """The statuses GET /v3/request/{id}/statuses returns, as action code +
    name (the value set_request_status / update_request take as status_name)."""
    return [
        {"action_code": row.get("action_code"), "name": row.get("name")}
        for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)
    ]


_GET_REQUEST_DESC = inspect.cleandoc(
    """
    Fetch a single request by id. Returns the request record including
    SLA metadata. (The optimistic-lock token for updates is fetched
    internally by update_request — you never need it.)

    Time fields: `age_hours_calendar` is the calendar time from creation to
    close (to now while the request is open), in hours. CDESK's own
    `duration_open` / `duration_reaction` (seconds; with fieldset "all" and in
    write results) are SLA WORKING time — only inside the SLA's operating
    hours, without holidays and without time in statuses that stop the SLA
    clock — not calendar time. `duration_open` is set when the request is
    closed: it is 0 while open, after a reopen, and without an SLA, so
    `sla_open_hours` is null then; `sla_reaction_hours` is null until the
    first reaction.

    fieldset selects the field group returned: "base", "extended"
    (CDESK's default when omitted), "all" (base+extended+custom), or
    "custom" (custom fields only). Pass "all" or "custom" to read the
    stored custom-field (cfield_*) values.

    fields: a whitelist of field names to return (CDESK's returnFields);
    `customer_name` / `solver_name` are added to it. Composes with fieldset
    as a union; unknown names are silently ignored by CDESK.

    include_deleted: a soft-deleted request 404s on a plain GET; pass
    True to retrieve it anyway (requests are the only module where
    deleted records remain reachable).

    include_allowed_statuses: True adds `allowed_statuses` — the statuses
    the request may be moved to now (action code + name, the current one
    included), per the tenant's status rules and the request's catalog
    status flow. CDESK refuses any other status on set_request_status /
    update_request.

    The response carries a derived `is_periodic` boolean (True when the
    request was generated from a periodical request) plus the raw
    `periodical_request_id` it rests on — both are included regardless of
    fieldset/fields.

    Each record carries `customer_name` / `solver_name` next to the
    customer and solver ids.
    Enum fields carry their label alongside: `status_name`, `priority_name`,
    `type_name`, `cat_type_name`, … sit next to the raw value. The raw
    `status` and `priority` are enum ACTION CODES, not names or ids —
    `priority: 2` is whatever entry has action code 2 (on some tenants
    "dlhodobá"), not "2 - vysoká". The `*_name` values are the names the
    status_name / priority_name / … parameters accept. A value with no
    label (empty, or a category switched off on the tenant) has no
    `*_name` key.
    """
)

_LIST_REQUEST_SOLVER_GROUPS_DESC = inspect.cleandoc(
    """
    The solver groups CDESK's request form offers for `solver_group_id` on a
    request of this customer, optionally narrowed by service area
    (cat_area_id / cat_area_2nd_id), request-catalog item (catalog_id) and
    project contract (project_contract_id). Groups assigned to the service
    area come first, then the customer's groups, then the rest; a project
    contract with its own solver group adds its project-team group first.
    Each item: id, name, email, and the flags assigned_group_to_catarea /
    assigned_group_to_company marking the preferred groups.
    `preselected_solver_group_id`, when present, is the group the form would
    preselect for a new request.

    An empty list means the tenant does not use solver groups on requests
    (get_request_enums → fields shows use_solver_group), or no group fits.
    Service-area ids come from get_request_enums, catalog ids from
    list_request_catalogs.
    """
)

_GET_REQUEST_CUSTOM_FIELDS_DESC = inspect.cleandoc(
    """
    List the custom-field definitions configured for the Request module.
    Returns {items, count} — plus a `note` when count is 0, meaning the module
    has no custom fields defined (an empty result, not a failure). Each
    baseproperty in `items` carries a ready-to-use key in the form
    `cfield_<propertyId>_<basepropertyId>`.

    Workflow: (1) call this tool; (2) find the field by name; (3) pass
    `custom_fields={"<key>": <value>}` to create_request/update_request
    (scalar for single-value fields; for select/relation fields send the
    option id); (4) read stored values via get_request(id, fieldset="all").

    Optional catalog_id scopes the catalog to a request-catalog variant.

    KNOWN LIMITATION: the tenant currently ignores
    customFields writes silently — values do not persist. After any
    write, a get_request(id, fieldset="custom") read shows whether the
    value actually persisted; a value that did not stick was not
    stored despite the 200.
    """
)

_GET_REQUEST_ENUMS_DESC = inspect.cleandoc(
    """
    Return the cached lookup tables for the Request module. Buckets
    include status, type, priority, urgency, impact, cat_type, cat_area,
    cat_area_2nd, place, close_reason, solver_method, root_cause, and
    other tenant-specific enums.

    Use this to discover which names are valid for create_request /
    update_request / set_request_status. Pass refresh=True to force a
    fresh fetch.

    Also returns `fields`: the state of each configurable create_request /
    update_request parameter on this tenant — off, optional or required —
    derived from its live request settings, each with the setting behind
    it, plus the parameters that are always required.
    """
)

_CREATE_REQUEST_DESC = inspect.cleandoc(
    """
    Create a new CDESK request (helpdesk ticket).

    Always required: company (the company id the ticket belongs to),
    description (HTML allowed; a catalog may hide it), and a request type
    (type_code, e.g. "H" — or type_name). Every other configurable field
    follows the tenant's CDESK form, read live on every call: a field the
    tenant has switched off is refused when given, a required one must be
    given, an optional one may be left out. Examples: use_solver_group
    decides whether solver_id or solver_group_id is the required one (the
    other is off or optional); the title is required unless the tenant
    has auto-naming on — then a missing title is generated from the
    description exactly as the CDESK web form does (HTML stripped, cut to
    the tenant's max_auto_name_characters + "..."); code, category type,
    desired date, close reason,
    solution, urgency/impact, place, branch and catalog can each be off.
    get_request_enums lists every field's state under `fields`; a call
    breaking a rule is rejected naming each parameter and its setting.
    Priority is optional — CDESK takes the SLA's default.
    When the tenant flags any company tag as "requests may be added"
    (CDESK's default customer-type tag does), only companies carrying such
    a tag accept requests; any other company is refused with "Zadajte
    aktívneho zákazníka" even though it is active.

    Dropdown / enum fields — do NOT guess a value. type, status,
    priority, urgency, impact, place, and the category fields
    (cat_type / cat_area / cat_area_2nd) are pick-lists: fetch the
    allowed options with get_request_enums and PRESENT THEM TO THE
    USER AS A DROPDOWN to choose from, then pass the chosen value.
    Likewise resolve a solver by offering the matches from
    list_users / find_user rather than inventing an id.

    Catalog flow — when the user wants to create FROM a catalog, the
    FIRST question is always which catalog (template). Steps:
      1. Pick the catalog from list_request_catalogs:
         - When the request's subject, problem or kind of work is already
           known, that usually identifies a single best-matching catalog.
           The choice is consequential: the catalog fixes which fields are
           prefilled and which are value-restricted (step 2).
         - Otherwise (no info to match on yet), offer the catalogs as a
           dropdown and let the user pick one.
      2. Read that catalog with get_request_catalog and look at its
         base_params: those fields (type, cat_type, priority, solver/
         solver_group, etc.) are PREFILLED defaults. Show the user the
         prefilled values so they can keep or change them — for a field
         whose base_params carries restrict="limited", only offer the
         values in its `allowed` list.
      3. Then ask only for the fields still missing (typically company,
         title, description) and submit with catalog_id set.
    NB: passing catalog_id records the catalog link but does NOT itself
    apply the template defaults server-side — you must read base_params
    and pass the resolved values explicitly.

    Common fields:
      - status_name / priority_name / urgency_name / impact_name /
        place_name (all resolved via get_request_enums).
      - type_code: the request type / "druh" — a short string code (e.g. "H"
        for Helpdesk), NOT an enum id. CDESK requires it when the tenant has a
        base type configured. (type_name is only usable on tenants whose type
        enum is id-based; where it is key/value, pass type_code.)
      - solver_id, solver_group_id: who handles the request (CDESK requires a
        solver or solver group). Solver group ids are user-group ids
        (list_request_solver_groups lists the ones the request form offers;
        a root group, root_group=true in list_user_groups, is not accepted);
        whether the group is required follows use_solver_group
        (get_request_enums → fields).
      - catalog_id: instantiate from a request-catalog template.
      - deal_id (id_rc) / project_contract_id: deal linkage.
      - desired_date, calendar_date_start, invoice_date, first_conclusion
        (ISO-8601 datetimes; a datetime without an offset is interpreted as
        TENANT-LOCAL time — CDESK_TIMEZONE, default Europe/Bratislava — and
        sent with the proper offset; bare dates pass through unchanged).
        `first_conclusion` belongs here even though the CDESK spec declares it
        a string: the column is a TIMESTAMP, so prose sent to it would store
        as "0000-00-00 00:00:00". Non-dates are rejected up front.
      - due_date: the deadline ("Termín splnenia do"), ISO-8601 like the dates
        above. CDESK computes it from the SLA when it is left out; a value sent
        replaces it. Off while the tenant has the due-date field off (dueDate).
        due_date_fixed=true marks it as fixed (not recomputed); available only
        with the tenant's fixedDueDate setting.
      - solution, solver_note, note, code, computer_code: free-text
        bodies. NB: `code` requires the
        request/fields/code write right — without it CDESK drops the
        value with a 200; this tool detects the drop and names it in
        the result's `warnings`.
      - for_customer: visibility flag (customer can see the ticket).
      - user_entered_by / contact_entered_by / entered_by: applicant
        identifier (one of: existing user id, customer-contact id, or
        free-text name).

    Categories (each resolved via get_request_enums):
      - cat_type_name / cat_area_name / cat_area_2nd_name.

    Resolution / closure (typically used on update, not create):
      - close_reason_name / solver_method_name / root_cause_name.

    Advanced / integration fields (accepted but not documented per-field;
    safe to omit): notify_block, notify_customer_block, notify_sms_block,
    from_email / from_name, to_email / to_name, ec_login / ec_email /
    ec_phone, extsys_id / extsys_code / extsys_status,
    change_manager_id, reaction_missed_id_solver, request_weight,
    scheduling_mode, planned_hours, hour_rate, percent_done.

    Custom fields:
      - custom_fields: flat dict of `cfield_<propertyId>_<basepropertyId>`
        keys → values (discover keys via get_request_custom_fields;
        select/relation fields take the option id). Unrecognized keys
        are NOT rejected — CDESK ignores them with a 200 and reports
        them in the result's `warnings` key.

    Tenant settings may make additional fields required (custom category
    fields, approval rules, etc.) — CDESK's 400/422 response surfaces
    those clearly via translate_error.

    Returns: {data: <new id>, savedData: {...full record...}}; check the
    optional `warnings` list for silently-dropped/ignored fields.

    Side effects (by CDESK): with the deal module and the tenant's automatic deal
    selection on, a request saved without deal_id is linked to the customer's deal
    when exactly one deal fits (taking the service area into account).
    """
)

_UPDATE_REQUEST_DESC = inspect.cleandoc(
    """
    Partial update of a request. Pass only the fields you want to
    change — everything else stays as-is. Optimistic locking is handled
    internally (GET → grab timestamp_check → PUT).

    Same name-based enum resolution as create_request:
    status_name / type_name / priority_name / urgency_name / impact_name /
    place_name / cat_type_name / cat_area_name / cat_area_2nd_name /
    close_reason_name / solver_method_name / root_cause_name. Several of these
    enums are gated per tenant and are simply absent on some servers; the error
    says which, and names the setting behind it.

    due_date sets the deadline (CDESK computes it from the SLA otherwise), and
    due_date_fixed=true keeps it from being recomputed — the latter only with
    the tenant's fixedDueDate setting.

    desired_date, due_date, calendar_date_start, invoice_date and first_conclusion are
    ISO-8601 datetimes — a value without an offset is read as tenant-local time
    (CDESK_TIMEZONE, default Europe/Bratislava). `first_conclusion` is one of
    them despite the CDESK spec declaring it a string: the column is a
    TIMESTAMP, and prose sent to it would store as "0000-00-00 00:00:00".

    Fields whose tenant feature is switched off are dropped by CDESK with a 200
    — the result's `warnings` list names each one and the setting it needs, so
    an unstored value is visible rather than silent.

    custom_fields: flat dict of `cfield_<propertyId>_<basepropertyId>`
    keys → values (discover keys via get_request_custom_fields; read
    stored values via get_request(id, fieldset="all")). Unrecognized
    keys are ignored with a 200 and reported in the result's `warnings`
    key (not a 400). `code` is dropped with a 200 without the
    request/fields/code write right — this tool detects the drop and
    names it in `warnings`.

    Reparenting (changing company) may be rejected once the request
    is completed or has bound devices/approvals — CDESK's error message
    will explain.

    Returns: {data: <id>, savedData: {...full record...}}; check the
    optional `warnings` list for silently-dropped/ignored fields.
        notify_block=1 sends no notification for this change; notify_customer_block
    / notify_sms_block suppress only the customer e-mails / the SMS.

    Side effects (by CDESK): moving a request to a closing status (80 ukončené,
    81, 85, 86, 95 zamietnuté) sets its close_date and close user, computes
    duration_open, sets first_conclusion on the first close to 80 and, on some
    tenants, invoice_date (not for 95); moving it back below 80 clears the close
    date, close reason, solver method and duration_open. A move straight to 90 /
    91 does not run this.
    A request without a deal can get one: see create_request.
    """
)

_DELETE_REQUEST_DESC = inspect.cleandoc(
    """
    Delete a request by id. Returns {deleted: <id>} on success. Fails
    with 409 if the request has dependencies. The 409 names them; they have to
    be broken before the delete can succeed.
    """
)

_SET_REQUEST_STATUS_DESC = inspect.cleandoc(
    """
    Intent helper: change a request's status by name. Resolves the
    status name via the Request enum cache, handles the optimistic-lock
    fetch internally, and PUTs only the {status, timestamp_check} pair.
    A status the tenant's status rules do not allow from the current one is
    refused before anything is written, naming the allowed statuses
    (get_request(include_allowed_statuses=True) lists them too).

    notify_block=1 sends no notification for this change (notify_customer_block
    / notify_sms_block: none to the customer / no SMS).

    Returns the updated request record. Closing a request makes CDESK compute
    `duration_open` (SLA working seconds, see get_request); the result carries
    it as `sla_open_hours` beside `age_hours_calendar`.

    Side effects (by CDESK): moving a request to a closing status (80 ukončené,
    81, 85, 86, 95 zamietnuté) sets its close_date and close user, computes
    duration_open, sets first_conclusion on the first close to 80 and, on some
    tenants, invoice_date (not for 95); moving it back below 80 clears the close
    date, close reason, solver method and duration_open. A move straight to 90 /
    91 does not run this.
    """
)

_LIST_REQUEST_DISCUSSION_DESC = inspect.cleandoc(
    """
    List the discussion messages on a request, newest first. Each
    message includes the channel (customer-visible or internal),
    author, timestamp, body, and per-message ACL metadata.

    include_deleted=True surfaces soft-deleted messages as well.
    fields: an exact whitelist of field names to keep per message
    (returnFields; unknown names silently ignored).
    """
)

_POST_REQUEST_DISCUSSION_DESC = inspect.cleandoc(
    """
    Add a discussion message to an existing request.

    Required:
      - request_id (path) — the parent request.
      - channel — "customer" (visible to the requester; uses
        customer_text + customer_to) OR "internal" (technicians only;
        uses technote_text + technote_to). CDESK encodes this as
        status=1 / status=2 in the wire format; this tool hides that.
      - text — message body (HTML allowed). At least one of `text` OR
        `attachments` must be provided; CDESK rejects empty messages.

    Optional:
      - attachments — list of CDESK file hash strings (from a prior
        upload). Routed to attachment / attachment_internal based on
        channel.
      - change_status_to_waiting_for_customer — when true, posting the
        message also moves the request to "waiting for customer"
        status (only meaningful on channel="customer").
      - work_order_id — associate the message with a work order.
      - customer_signature — 1 = primary signature, 2 = secondary;
        only honored on channel="customer".
      - notify_user_ids — CDESK user ids to email about this post. Each
        user's address is read from their CDESK user record (the ids
        find_user returns); an id that can't be read, or a user without
        an email, fails the call before anything is posted. They are
        added as CDESK users: they also get the in-app notification, and
        their own notification settings apply.
      - discussion_receivers — additional email addresses to notify,
        e.g. people who are not CDESK users. These get the email only.
        Values that are not email addresses fail the call before
        anything is posted.
      - custom_notify — what the receivers above do to CDESK's default
        notifications. False (default): they are notified in addition
        to the receivers CDESK picks by its own rules. True: only they
        are notified, and CDESK's default receivers are not. Requires at
        least one receiver.
      - notify_block — 1 = nobody is notified about this post: neither
        its receivers nor the notification a status change can trigger
        (with work_order_id, the work order's receivers still are).
        Cannot be combined with notify_user_ids / discussion_receivers.
      - notify_customer_block / notify_sms_block — 0 or 1 flags that
        suppress the customer / SMS notification of a status change the
        post triggers; they do not affect the post's own receivers.

    With no receivers given, CDESK notifies by its own rules.

    Returns the created message record. When receivers were given, a
    `notification` block lists the requested receivers, the email
    receivers CDESK recorded on the post, the user ids it recorded an
    in-app notification for, and CDESK's confirmation message; it
    carries a `warning` when a requested receiver is not in the recorded
    list or CDESK sent no confirmation.

    Side effects (by CDESK): with the tenant's automatic status change on, a post
    by the request's solver (or a member of its solver group) moves a request in
    "prijaté" / "priradené" to "v riešení", and a customer's post moves a request
    in "čakanie na zákazníka" back to "v riešení"; change_status_to_waiting_for_customer
    moves it to "čakanie na zákazníka".
    """
)
