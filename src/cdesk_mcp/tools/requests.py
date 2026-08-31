"""Request module MCP tools (M12).

Ten tools that expose CDESK's Request v3 CRUD + intent helpers + the
discussion sub-resource + the custom-field catalog to the LLM. Wires together CdeskClient (M3),
EnumCache (M5, per-instance endpoint M12), filter builder (M6, M12), and
error translator (M4).

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

import inspect
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.enums import EnumCache
from cdesk_mcp.filters import build_request_filter, encode_sb
from cdesk_mcp.tools._helpers import (
    annotate_write_warnings,
    apply_field_scope,
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

# Fields CDESK strips from a request write when the tenant feature behind them is
# off or the account lacks the write right. Confirmed in the backend source
# 2026-07-30: RequestModule::postIndex (which serves BOTH create and update) calls
# CdRequest::filterUnavailableFields, which unsets every column listed by
# RequestColumnsTrait::getUnavailableTableCols. The API returns 200 and emits NO
# message of its own for any of these except `code`, so without this table the
# value vanishes and the tool reports success.
#
# Keys are the WIRE names as they appear in the body and in savedData. Each note
# names the setting and right, mirroring the wording CDESK uses for `code`.
_REQUEST_TOOL_NAMES = (
    "list_requests",
    "get_request",
    "get_request_custom_fields",
    "get_request_enums",
    "create_request",
    "update_request",
    "delete_request",
    "set_request_status",
    "list_request_discussion",
    "post_request_discussion",
)

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
        "requires the tenant setting `project_deal.percent_done` and the "
        "`request/fields/percent-done` write right"
    ),
    "desired_date": (
        "requires the tenant setting `request.desiredDate.enabled` and the "
        "`request/fields/desired_date` write right"
    ),
    "calendar_date_start": (
        "requires the tenant setting `request.calendar_date_start` and the "
        "`request/fields/calendar_date_start` write right"
    ),
    "invoice_date": (
        "requires the Fills module (`fill.enabled`) and the "
        "`request/fields/invoice_date` write right"
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
    always derivable. The default (extended) fieldset omits it (verified live).
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
        description=_LIST_REQUESTS_DESC,
        annotations=ToolAnnotations(title="List requests", readOnlyHint=True),
    )
    async def list_requests(
        text_search: str | None = None,
        status_name: str | None = None,
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
            # Category type: the sb `type` column filters cat_type_id (backend
            # JCD-32964). Resolve the NAME → cat_type_id via the same enum bucket
            # create_request uses; base_type is the raw base-type letter.
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
            if sort:
                params["sort"] = sort
            # Flat query param, NOT an sb column (an sb deleted_at clause is
            # silently ignored). Verified live 2026-06-05: includeDeleted=1
            # surfaces soft-deleted requests and composes with sb filters.
            # Requests are the ONLY module where this works — task/catalog
            # deletes are unreachable through the API.
            if include_deleted:
                params["includeDeleted"] = 1
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response = await client.get("v3/request", params=params)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"list_requests input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_requests") from e

        records, meta = unwrap_list(response)
        for rec in records:
            _add_is_periodic(rec)
        result = {"items": records, "meta": meta, "page": page, "per_page": per_page}
        if dropped_clauses:
            result["unsupported_filters"] = unsupported_filter_directive(dropped_clauses)
        if sort:
            result["unsupported_sort"] = _sort_not_honored(sort)
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
            # Soft-deleted requests 404 on a plain GET; includeDeleted=1
            # returns them (verified live 2026-06-05; request module only).
            if include_deleted:
                params["includeDeleted"] = 1
            response = await client.get(f"v3/request/{id}", params=params or None)
        except ValueError as e:
            raise RuntimeError(f"get_request input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="get_request", record_id=id) from e
        return _add_is_periodic(unwrap_record(response))

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
    async def get_request_enums(refresh: bool = False) -> dict[str, list[dict[str, Any]]]:
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
        return cache.snapshot()

    @mcp.tool(
        description=_CREATE_REQUEST_DESC,
        annotations=ToolAnnotations(title="Create request", destructiveHint=True),
    )
    async def create_request(
        company: int,
        title: str,
        description: str,
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
            # title and description are mandatory on this tenant (CDESK rejects
            # the create with "Zadajte názov/popis požiadavky" otherwise) — they
            # are required params, so just guard against blank/whitespace.
            if not title.strip():
                raise ValueError("title must be a non-empty string")
            if not description.strip():
                raise ValueError("description must be a non-empty string")
            # A request type and a solver are mandatory on this tenant — CDESK
            # rejects the create with "Vyberte druh požiadavky" / "Vyberte
            # riešiteľa" (code 14) otherwise. Enforce here so the failure is a
            # clear, early error instead of a relayed Slovak backend message.
            if type_code is None and type_name is None:
                raise ValueError(
                    'a request type is required: provide type_code (e.g. "H" for '
                    "Helpdesk) or type_name — discover the allowed types via "
                    "get_request_enums and let the user pick from a dropdown"
                )
            if solver_id is None and solver_group_id is None:
                raise ValueError(
                    "a solver is required: provide solver_id or solver_group_id"
                )
            # Priority is a tool-level requirement (CDESK itself would default
            # it). Like the other enums, discover options via get_request_enums
            # and let the user pick from a dropdown — don't guess.
            if priority_name is None:
                raise ValueError(
                    "a priority is required: provide priority_name — discover the "
                    "allowed priorities via get_request_enums and let the user "
                    "pick from a dropdown"
                )
            validate_custom_fields(custom_fields)
            # Naive datetimes are interpreted as TENANT-LOCAL wall-clock time
            # and sent with an explicit offset — CDESK reads naive values as
            # UTC (verified live 2026-06-05 on task valid_from; same backend
            # datetime path). Bare dates pass through unchanged.
            desired_date = localize_naive_datetime("desired_date", desired_date)
            calendar_date_start = localize_naive_datetime(
                "calendar_date_start", calendar_date_start
            )
            invoice_date = localize_naive_datetime("invoice_date", invoice_date)
            # first_conclusion is a TIMESTAMP column (apiportal db/cd_req.yml:313,
            # migration 20190801161146), even though CDESK's own OpenAPI
            # annotation declares it type="string" and our description used to
            # repeat that. Text sent here was coerced to "0000-00-00 00:00:00"
            # — a successful-looking write that corrupted the field. Validate it
            # as a datetime so a non-date is rejected up front.
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
            # CDESK stores req_status and req_priority by the enum ACTION_CODE
            # (verified live), NOT the enum id — sending the id overflows the
            # tinyint column and silently defaults/corrupts the value.
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
                    # this tenant's type enum is key/value-shaped so type_name
                    # can't resolve to an id. Prefer the explicit type_code.
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
                    "calendar_date_start": calendar_date_start,
                    "invoice_date": invoice_date,
                    "solution": solution,
                    "solver_note": solver_note,
                    "first_conclusion": first_conclusion,
                    "note": note,
                    "code": code,
                    "computer_code": computer_code,
                    # CDESK's wire `for_customer` flag is INVERTED vs its name —
                    # wire True hides the request from the customer (verified
                    # live: True -> hidden_for_customer=1). Send the negation so
                    # this tool's for_customer=True means customer-VISIBLE, per
                    # the param name + OpenAPI.
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
                    "notify_block": notify_block,
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
        # solver_note: silently dropped like code (verified live 2026-06-05:
        # PUT with solver_note is a complete no-op — 200, not stored,
        # udatetime not even bumped). Diff both against savedData.
        return annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=tuple(_REQUEST_GATED_FIELDS),
            field_notes=_REQUEST_GATED_FIELDS,
        )

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
    ) -> dict[str, Any]:
        try:
            if title is not None and not title.strip():
                raise ValueError("title must be a non-empty string")
            validate_custom_fields(custom_fields)
            # Naive datetimes are interpreted as TENANT-LOCAL wall-clock time
            # and sent with an explicit offset — CDESK reads naive values as
            # UTC (verified live 2026-06-05 on task valid_from; same backend
            # datetime path). Bare dates pass through unchanged.
            desired_date = localize_naive_datetime("desired_date", desired_date)
            calendar_date_start = localize_naive_datetime(
                "calendar_date_start", calendar_date_start
            )
            invoice_date = localize_naive_datetime("invoice_date", invoice_date)
            # first_conclusion is a TIMESTAMP column (apiportal db/cd_req.yml:313,
            # migration 20190801161146), even though CDESK's own OpenAPI
            # annotation declares it type="string" and our description used to
            # repeat that. Text sent here was coerced to "0000-00-00 00:00:00"
            # — a successful-looking write that corrupted the field. Validate it
            # as a datetime so a non-date is rejected up front.
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
            # CDESK stores req_status and req_priority by the enum ACTION_CODE
            # (verified live), NOT the enum id — sending the id overflows the
            # tinyint column and silently defaults/corrupts the value.
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

            body = _build_request_body(
                {
                    "title": title,
                    "description": description,
                    "status": status_id,
                    # The request `type` ("druh") is a string code (e.g. "H");
                    # this tenant's type enum is key/value-shaped so type_name
                    # can't resolve to an id. Prefer the explicit type_code.
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
                    "calendar_date_start": calendar_date_start,
                    "invoice_date": invoice_date,
                    "solution": solution,
                    "solver_note": solver_note,
                    "first_conclusion": first_conclusion,
                    "note": note,
                    "code": code,
                    # CDESK's wire `for_customer` flag is INVERTED vs its name —
                    # wire True hides the request from the customer (verified
                    # live: True -> hidden_for_customer=1). Send the negation so
                    # this tool's for_customer=True means customer-VISIBLE, per
                    # the param name + OpenAPI.
                    "for_customer": (None if for_customer is None else not for_customer),
                    "planned_hours": planned_hours,
                    "hour_rate": hour_rate,
                    "percent_done_manual": percent_done,
                }
            )
            # customFields keys are already wire-format (cfield_*) — set after
            # the body build so they bypass the LLM-name→CDESK-name rename map.
            if custom_fields:
                body["customFields"] = custom_fields
            if not body:
                # An update with no field still issued a PUT, which changed
                # nothing and bumped `udatetime` on a destructive-hinted tool.
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
        return annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=tuple(_REQUEST_GATED_FIELDS),
            field_notes=_REQUEST_GATED_FIELDS,
        )

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
    async def set_request_status(id: int, status_name: str) -> dict[str, Any]:
        try:
            # req_status is stored/matched by enum ACTION_CODE, not the enum id.
            status_id = await resolve_enum_field_or_raise(
                cache, "status", status_name, kind="status", field="action_code"
            )
            if status_id is None:
                raise RuntimeError("set_request_status requires a non-empty status_name")
            current = await _fetch_for_update_request(client, id)
            body = {"status": status_id, "timestamp_check": current["timestamp_check"]}
            response = await client.put(f"v3/request/{id}", json=body)
        except RuntimeError:
            raise
        except Exception as e:
            raise to_llm_error(e, operation="set_request_status", record_id=id) from e
        return response if isinstance(response, dict) else {"data": response}

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
        # is per-record: an all-or-nothing check missed the mixed case, which is
        # the common one because any request with a status change is mixed.
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
        custom_notify: bool = False,
        discussion_receivers: list[str] | None = None,
        customer_to: dict[str, Any] | None = None,
        technote_to: dict[str, Any] | None = None,
        notify_block: int | None = None,
        notify_customer_block: int | None = None,
        notify_sms_block: int | None = None,
    ) -> dict[str, Any]:
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
                discussion_receivers=discussion_receivers,
                customer_to=customer_to,
                technote_to=technote_to,
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
        return response if isinstance(response, dict) else {"data": response}


    # FastMCP builds each tool's argument model with pydantic's default
    # extra="ignore", so an unrecognised keyword is dropped BEFORE the tool body
    # runs — our code never sees it. Live consequence on update_request: a typo
    # like titel= (for title=), or a create-only field such as computer_code=,
    # returned success, issued a PUT with an empty body and bumped udatetime,
    # changing nothing and saying nothing. Forbid extras so the caller gets a
    # validation error naming the offending key instead of a silent no-op.
    #
    # Scoped to this module deliberately: flipping it for all 96 tools would
    # reject any harmless extra a host might attach, which is a much wider blast
    # radius than the bug justifies.
    forbid_unknown_arguments(mcp, _REQUEST_TOOL_NAMES)


# --- internals ----------------------------------------------------------

async def _fetch_for_update_request(client: CdeskClient, id: int) -> dict[str, Any]:
    """Fetch a request and unwrap to the record dict, ready to feed the
    optimistic lock into a follow-up PUT.

    CDESK's Request module reads `timestamp_check` from the PUT body, same
    as Task/Customer/User — verified against the OpenAPI spec's PUT
    description ("required: timestamp_check, must equal the updated_at
    from last fetch").

    fieldset=all is required: the v3 fieldset default (`extended`) no longer
    includes `timestamp_check` — only `all` carries it (verified live)."""
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


def _sort_not_honored(sort: str) -> dict[str, Any]:
    """The `unsupported_sort` block attached whenever `sort` was passed.

    The v3 request list accepts `sort` and does not honor it: "id", "-id",
    "id desc" and an unknown column all return the same order (verified live
    2026-06-05, re-confirmed 2026-07-30). Passing any value only flips to
    oldest-first. Unknown FILTER columns already get this treatment via
    `unsupported_filters`; sort had no runtime signal at all, so a caller that
    asked for "the 5 highest-priority requests" received an arbitrary page in id
    order with nothing indicating the ordering never applied.

    Declarative on purpose — it states what the response is, not what to do
    about it (the same reason `unsupported_filter_directive` reads the way it
    does).
    """
    return {
        "requested_sort": sort,
        "applied": False,
        "actual_order": "id ascending (oldest first)",
        "what_this_means": (
            f"The CDESK v3 request list does not honor `sort`: the column name "
            f"and direction in {sort!r} had no effect, and these items are in id "
            f"ascending (oldest-first) order — passing any sort value at all "
            f"produces that, while omitting sort gives newest-first. The items "
            f"are therefore NOT ordered by the requested field. Ordering by a "
            f"field is only possible client-side, and only over the complete "
            f"result set: a single page sorted locally gives the top of THAT "
            f"page, not of the whole set, and `meta` carries no total to tell "
            f"the two apart. The complete set is this call repeated with `page` "
            f"incremented until a short or empty page comes back."
        ),
    }


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
    discussion_receivers: list[str] | None,
    customer_to: dict[str, Any] | None,
    technote_to: dict[str, Any] | None,
    notify_block: int | None,
    notify_customer_block: int | None,
    notify_sms_block: int | None,
) -> dict[str, Any]:
    """Build the POST body for adding a discussion message.

    Channel routing:
      - channel="customer" → status=1, text lands in customer_text,
        recipient in customer_to, customer-targeted attachments.
      - channel="internal" → status=2, text lands in technote_text,
        recipient in technote_to, internal attachments.

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
        if customer_to is not None:
            body["customer_to"] = customer_to
        if customer_signature is not None:
            body["customer_signature"] = customer_signature
    else:
        if text:
            body["technote_text"] = text
        if attachments:
            body["attachment_internal"] = attachments
        if technote_to is not None:
            body["technote_to"] = technote_to

    if change_status_to_waiting_for_customer:
        body["changeStatusToWaitingForCustomer"] = True
    if work_order_id is not None:
        body["work_order_id"] = work_order_id
    if custom_notify:
        body["customNotify"] = True
    if discussion_receivers is not None:
        body["discussion_receivers"] = discussion_receivers
    if notify_block is not None:
        body["notifyBlock"] = notify_block
    if notify_customer_block is not None:
        body["notifyCustomerBlock"] = notify_customer_block
    if notify_sms_block is not None:
        body["notifySmsBlock"] = notify_sms_block

    return body


# --- Tool descriptions --------------------------------------------------

_LIST_REQUESTS_DESC = inspect.cleandoc(
    """
    List CDESK requests (helpdesk tickets) visible to your account. A
    request is the customer-facing ticket entity — distinct from internal
    tasks. Returns a paginated `items` array; `meta` is usually empty on
    this backend (no total count is sent — detect the last page by a
    short/empty page, not by a total).

    Ordering: omitting `sort` gives newest-first (usually what you want).
    The sort COLUMN and direction are both IGNORED by the backend
    (verified live 2026-06-05: "id", "-id", "id desc" and an unknown
    column all produce the same order) — passing ANY sort value merely
    flips the ordering to oldest-first (id ascending). So: omit sort for
    newest-first, any value gives oldest-first, and a column name has no
    effect — ordering by a specific field only works client-side.

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
    mis-evaluates them — verified live): request type ("druh"), sla_id.
    Use client-side filtering for those.

    For advanced filters not exposed here, pass sb_raw as a CDESK sb object
    (JSON string or object) in the structured tree form. Only the
    LIVE-VERIFIED working columns are applied server-side (id_req, req_num,
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
    field group per record ("base"/"extended"/"all"/"custom"); fields is an
    exact whitelist of field names (returnFields; union with fieldset).
    Use fieldset="base" or fields=[...] to keep large lists compact.
    """
)

_GET_REQUEST_DESC = inspect.cleandoc(
    """
    Fetch a single request by id. Returns the request record including
    SLA metadata. (The optimistic-lock token for updates is fetched
    internally by update_request — you never need it.)

    fieldset selects the field group returned: "base", "extended"
    (CDESK's default when omitted), "all" (base+extended+custom), or
    "custom" (custom fields only). Pass "all" or "custom" to read the
    stored custom-field (cfield_*) values.

    fields: an exact whitelist of field names to return (CDESK's
    returnFields). Composes with fieldset as a union; unknown names are
    silently ignored by CDESK.

    include_deleted: a soft-deleted request 404s on a plain GET; pass
    True to retrieve it anyway (requests are the only module where
    deleted records remain reachable).

    The response carries a derived `is_periodic` boolean (True when the
    request was generated from a periodical request) plus the raw
    `periodical_request_id` it rests on — both are included regardless of
    fieldset/fields.
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

    KNOWN LIMITATION (2026-06-04): the tenant currently ignores
    customFields writes silently — values do not persist. After any
    write, a get_request(id, fieldset="custom") read shows whether the
    value actually persisted; a value that did not stick was not
    stored despite the 200 (backend fix pending).
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
    """
)

# ACCEPTED COMPLIANCE RISK (reviewed 2026-07-29, Anthropic connector directory).
# The "Catalog flow" section below is a behavioural workflow, not a description of
# the tool: "PRESENT THEM TO THE USER AS A DROPDOWN", "Show the user the prefilled
# values", "Then ask only for the fields still missing". The directory policy says
# "Describe what the tool does. Do not tell Claude how to behave."
#
# This is the WEAKEST of the two risks we accepted — unlike the grounding contract
# in tools/grounding.py, none of it is enforced server-side, so it is pure UI
# instruction. Kept because it encodes hard-won tenant behaviour (base_params
# prefills, restrict="limited" allowed lists) and rewriting a ~4.7k-char
# description risks losing tuned behaviour.
#
# If a reviewer flags anything in this repo, expect it to be this. The fix is a
# faithful rewrite, not deletion: turn each imperative into the API fact beneath
# it — e.g. "a value outside the `allowed` list is rejected" instead of "only
# offer the values in its `allowed` list".
_CREATE_REQUEST_DESC = inspect.cleandoc(
    """
    Create a new CDESK request (helpdesk ticket).

    Required: company (the company id the ticket belongs to),
    title (req_title), description (HTML allowed), a request type
    (type_code, e.g. "H" — or type_name), a priority (priority_name),
    and a solver (solver_id or solver_group_id). The tool enforces all
    of these up front with a clear error.

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
        enum is id-based; this tenant's is key/value, so pass type_code.)
      - solver_id, solver_group_id: who handles the request (CDESK requires a
        solver or solver group).
      - catalog_id: instantiate from a request-catalog template.
      - deal_id (id_rc) / project_contract_id: deal linkage.
      - desired_date, calendar_date_start, invoice_date, first_conclusion
        (ISO-8601 datetimes; a datetime without an offset is interpreted as
        TENANT-LOCAL time — CDESK_TIMEZONE, default Europe/Bratislava — and
        sent with the proper offset; bare dates pass through unchanged).
        `first_conclusion` belongs here even though the CDESK spec declares it
        a string: the column is a TIMESTAMP, so prose sent to it used to store
        as "0000-00-00 00:00:00". Non-dates are now rejected up front.
        (The deadline `req_due_date` is SLA-computed and not directly settable.)
      - solution, solver_note, note, code, computer_code: free-text
        bodies. NB: `code` requires the
        request/fields/code write right — without it CDESK silently
        drops the value with a 200 (verified live); this tool detects
        the drop and adds a `warnings` entry to the result.
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

    desired_date, calendar_date_start, invoice_date and first_conclusion are
    ISO-8601 datetimes — a value without an offset is read as tenant-local time
    (CDESK_TIMEZONE, default Europe/Bratislava). `first_conclusion` is one of
    them despite the CDESK spec declaring it a string: the column is a
    TIMESTAMP, and prose sent to it used to store as "0000-00-00 00:00:00".

    Fields whose tenant feature is switched off are dropped by CDESK with a 200
    — the result's `warnings` list names each one and the setting it needs, so
    an unstored value is visible rather than silent.

    custom_fields: flat dict of `cfield_<propertyId>_<basepropertyId>`
    keys → values (discover keys via get_request_custom_fields; read
    stored values via get_request(id, fieldset="all")). Unrecognized
    keys are ignored with a 200 and reported in the result's `warnings`
    key (not a 400). `code` is silently dropped without the
    request/fields/code write right — this tool detects the drop and
    adds a `warnings` entry.

    Reparenting (changing company) may be rejected once the request
    is completed or has bound devices/approvals — CDESK's error message
    will explain.

    Returns: {data: <id>, savedData: {...full record...}}; check the
    optional `warnings` list for silently-dropped/ignored fields.
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

    Returns the updated request record.
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
      - custom_notify + discussion_receivers — override the auto-
        resolved receiver set with explicit emails.
      - customer_to / technote_to — opaque receiver descriptors. Pass
        verbatim from CDESK UI if needed; this tool doesn't validate
        their shape.
      - notify_block / notify_customer_block / notify_sms_block — 0 or
        1 flags that suppress notifications.

    Returns the created message record.
    """
)
