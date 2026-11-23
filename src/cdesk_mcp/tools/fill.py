"""Fill module MCP tools — CDESK "fulfillments" (time / work records).

Six tools exposing the Fill v3 CRUD surface: list / get / get_enums / create /
update / delete. A fill is a unit of logged work (duration + description) hung
off a parent entity chosen by `assign_id` (1=company, 2=deal, 3=request,
4=work order, 5=task, 6=project contract, 7=days-off).

Like CMDB this module resolves NO enum names client-side (every field is an FK
id, flag, or free text), so it takes no EnumCache — `get_fill_enums` fetches the
enum groups directly.

Optimistic locking is hidden exactly as in the Task module: update_fill does
GET (fieldset=all) → mutate → PUT, echoing the `timestamp_check` token; the LLM
never sees it. The backend blocks writes to already-invoiced fills (HTTP 403
without the invoiced-write right; a delete answers 424) and to fills on a closed
request (424) — that surfaces through the normal error translator.

The date window filters via a `start_date` BETWEEN leaf (built by
build_fill_filter). A windowless list has no implicit month bound.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp import base_config
from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.field_policy import (
    FILL_FIELDS,
    describe,
    fill_settings,
    refuse_if_violated,
    violations,
)
from cdesk_mcp.filters import build_fill_filter, encode_sb
from cdesk_mcp.sort_columns import resolve_sort, sort_note
from cdesk_mcp.tools._helpers import (
    list_meta,
    TEXT_MATCH_NOTE,
    add_names,
    apply_field_scope,
    ask_for_names,
    canon_name,
    closest_row_names,
    to_llm_error,
    unsupported_filter_directive,
    unwrap_list,
    unwrap_record,
    validate_fields,
    validate_fieldset,
    with_warning_texts,
)
from cdesk_mcp.tz import localize_naive_datetime, tenant_timezone

# Pagination upper bound — mirrors the task/request modules' 100 cap to keep
# response sizes within an LLM's context budget.
_MAX_PER_PAGE = 100

# assign_id → the body field that MUST accompany it. Used for a clear
# client-side error instead of the backend's opaque 400. 7 = days-off is NOT a
# valid fill parent: CDESK refuses it every time with "Vyberte väzbu" —
# days-off fills are made by the days-off module, not here.
_ASSIGN_PARENT: dict[int, str] = {
    1: "company_id",
    2: "deal_id",
    3: "request_id",
    4: "work_order_id",
    5: "task_id",
    6: "project_contract_id",
}
_PARENT_ASSIGN: dict[str, int] = {field: kind for kind, field in _ASSIGN_PARENT.items()}
_ASSIGN_NAMES: dict[int, str] = {
    1: "company", 2: "deal", 3: "request", 4: "work order", 5: "task",
    6: "project contract",
}


def register_fill_tools(
    mcp: FastMCP,
    client: CdeskClient,
) -> None:
    """Register all Fill-module tools on the given FastMCP instance. Takes no
    EnumCache — Fill resolves no enum names client-side (see get_fill_enums)."""

    @mcp.tool(
        description=_LIST_FILLS_DESC + "\n\n" + sort_note("fills") + "\n\n" + TEXT_MATCH_NOTE,
        annotations=ToolAnnotations(title="List fills", readOnlyHint=True),
    )
    async def list_fills(
        text_search: str | None = None,
        solver_id: int | list[int] | None = None,
        company_id: int | list[int] | None = None,
        request_id: int | list[int] | None = None,
        deal_id: int | list[int] | None = None,
        project_contract_id: int | None = None,
        assign_id: int | None = None,
        place_id: int | None = None,
        invoiced: bool | None = None,
        signed: bool | None = None,
        rma: bool | None = None,
        worked_from: str | None = None,
        worked_to: str | None = None,
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
            # sb_raw clauses on columns the live endpoint doesn't honor are
            # stripped (the backend silently ignores them and would return the
            # unfiltered set) and reported via `unsupported_filters`.
            dropped_clauses: list[dict[str, Any]] = []
            sb = build_fill_filter(
                text_search=text_search,
                solver_id=solver_id,
                company_id=company_id,
                request_id=request_id,
                deal_id=deal_id,
                project_contract_id=project_contract_id,
                assign_id=assign_id,
                place_id=place_id,
                invoiced=invoiced,
                signed=signed,
                rma=rma,
                worked_from=worked_from,
                worked_to=worked_to,
                sb_raw=sb_raw,
                dropped_cols_out=dropped_clauses,
            )
            params: dict[str, Any] = {"pg": page, "pp": per_page}
            apply_field_scope(params, fieldset, fields)
            added_times = _ask_for_list_times(params, fields)
            ask_for_names(params, "fill", detail=False)
            if sort:
                params["sort"] = resolve_sort("fills", sort)
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response, paging = await client.get_page("v3/fill", params=params)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"list_fills input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_fills") from e

        records, meta = unwrap_list(response)
        warning = _add_list_minutes(records)
        add_names(records, "fill", detail=False, fieldset=fieldset, fields=fields)
        if added_times:
            records = [
                {k: v for k, v in r.items() if k not in added_times} if isinstance(r, dict) else r
                for r in records
            ]
        result: dict[str, Any] = {"items": records, "meta": list_meta(meta, paging, dropped=dropped_clauses), "page": page, "per_page": per_page}
        if warning:
            result["warnings"] = [warning]
        if dropped_clauses:
            result["unsupported_filters"] = unsupported_filter_directive(dropped_clauses)
        return result

    @mcp.tool(
        description=_GET_FILL_DESC,
        annotations=ToolAnnotations(title="Get fill", readOnlyHint=True),
    )
    async def get_fill(
        id: int,
        fieldset: str | None = None,
        fields: list[str] | None = None,
    ) -> Any:
        try:
            validate_fieldset(fieldset)
            validate_fields(fields)
            params: dict[str, Any] = {}
            if fieldset:
                params["fieldset"] = fieldset
            if fields:
                params["returnFields[]"] = fields
            ask_for_names(params, "fill", detail=True)
            response = await client.get(f"v3/fill/{id}", params=params)
        except ValueError as e:
            raise RuntimeError(f"get_fill input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="get_fill", record_id=id) from e
        record = unwrap_record(response)
        _add_detail_minutes(record)
        add_names([record], "fill", detail=True, fieldset=fieldset, fields=fields)
        return record

    @mcp.tool(
        description=_GET_FILL_ENUMS_DESC,
        annotations=ToolAnnotations(title="Get fill enums", readOnlyHint=True),
    )
    async def get_fill_enums() -> Any:
        try:
            response = await client.get("v3/fill/enums")
        except Exception as e:
            raise to_llm_error(e, operation="get_fill_enums") from e
        # The fill enums endpoint returns {data: false, enums: {...}} — the real
        # payload (maingroups, invoicing statuses/dates, deal & request
        # category types, ...) lives under `enums`, not `data`.
        result = response["enums"] if isinstance(response, dict) and "enums" in response else unwrap_record(response)
        if isinstance(result, dict):
            result = {**result, "fields": {
                "by_tenant_setting": describe(
                    FILL_FIELDS, fill_settings(await base_config.load(client))
                ),
                "note": (
                    "State of each configurable create_fill/update_fill parameter "
                    "on this tenant, from its live fill form settings: off (refused "
                    "when given), optional, or required (must be given on create, "
                    "cannot be cleared on update)."
                ),
            }}
        return result

    @mcp.tool(
        description=_CREATE_FILL_DESC,
        annotations=ToolAnnotations(
            title="Create fill", readOnlyHint=False, destructiveHint=False
        ),
    )
    async def create_fill(
        assign_id: int,
        valid_from: str,
        valid_to: str,
        solver_id: int | None = None,
        company_id: int | None = None,
        deal_id: int | None = None,
        request_id: int | None = None,
        work_order_id: int | None = None,
        task_id: int | None = None,
        project_contract_id: int | None = None,
        description: str | None = None,
        private_description: str | None = None,
        used_material: str | None = None,
        other_time: int | None = None,
        quantity: float | None = None,
        simple_use: bool | None = None,
        place_id: int | None = None,
        place_custom: str | None = None,
        place_travel_id: int | None = None,
        billing_item_id: int | None = None,
        billing_item_name: str | None = None,
        billing_travel_id: int | None = None,
        billing_kopfis_id: int | None = None,
        travel: bool | None = None,
        travel_distance: float | None = None,
        travel_fixed: float | None = None,
        travel_period: float | None = None,
        parking_fee: float | None = None,
        rma: int | None = None,
    ) -> dict[str, Any]:
        try:
            # Via locals: localize_naive_datetime is str | None -> str | None (it
            # maps "" to None), so it can't be assigned straight back onto these
            # required str params. The guard below narrows them for the reassign.
            localized_from = localize_naive_datetime("valid_from", valid_from)
            localized_to = localize_naive_datetime("valid_to", valid_to)
            if not localized_from or not localized_to:
                raise ValueError("valid_from and valid_to are required (ISO-8601)")
            valid_from, valid_to = localized_from, localized_to
            _ensure_chronological(valid_from, valid_to)
            # The worker is always required — by the CDESK form and the API
            # ("Zadajte riešiteľa").
            if solver_id is None:
                raise ValueError("solver_id (the worker) is required")
            # The tenant's fill FORM rules, read live from core/base-config (fill
            # settings are not in v3; v1 fill/settings is admin-only).
            _one_billing_item_param(billing_item_id, billing_item_name)
            refuse_if_violated(violations(
                FILL_FIELDS, fill_settings(await base_config.load(client)),
                {"description": description, "place_id": place_id,
                 "place_custom": place_custom,
                 "billing_item_id": billing_item_id if billing_item_name is None else billing_item_name},
            ))
            parents = {
                "company_id": company_id,
                "deal_id": deal_id,
                "request_id": request_id,
                "work_order_id": work_order_id,
                "task_id": task_id,
                "project_contract_id": project_contract_id,
            }
            _validate_assign(assign_id, parents)
            if billing_item_name is not None:
                scope_company, scope_deal = await _fill_scope(client, assign_id, parents)
                billing_item_id = await _billing_item_by_name(
                    client, billing_item_name, company_id=scope_company, deal_id=scope_deal,
                )

            body = _build_fill_body(
                {
                    "assign_id": assign_id,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "solver_id": solver_id,
                    **parents,
                    "description": description,
                    "private_description": private_description,
                    "used_material": used_material,
                    "other_time": other_time,
                    "quantity": quantity,
                    "simple_use": simple_use,
                    "place_id": place_id,
                    "place_custom": place_custom,
                    "place_travel_id": place_travel_id,
                    "billing_item_id": billing_item_id,
                    "billing_travel_id": billing_travel_id,
                    "billing_kopfis_id": billing_kopfis_id,
                    "travel": travel,
                    "travel_distance": travel_distance,
                    "travel_fixed": travel_fixed,
                    "travel_period": travel_period,
                    "parking_fee": parking_fee,
                    "rma": rma,
                }
            )
            response = await client.post("v3/fill", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"create_fill input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="create_fill") from e
        return await _with_billing_check(client, _with_write_minutes(response))

    @mcp.tool(
        description=_UPDATE_FILL_DESC,
        annotations=ToolAnnotations(title="Update fill", destructiveHint=True),
    )
    async def update_fill(
        id: int,
        assign_id: int | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        solver_id: int | None = None,
        company_id: int | None = None,
        deal_id: int | None = None,
        request_id: int | None = None,
        work_order_id: int | None = None,
        task_id: int | None = None,
        project_contract_id: int | None = None,
        description: str | None = None,
        private_description: str | None = None,
        used_material: str | None = None,
        other_time: int | None = None,
        quantity: float | None = None,
        simple_use: bool | None = None,
        place_id: int | None = None,
        place_custom: str | None = None,
        place_travel_id: int | None = None,
        billing_item_id: int | None = None,
        billing_item_name: str | None = None,
        billing_travel_id: int | None = None,
        billing_kopfis_id: int | None = None,
        travel: bool | None = None,
        travel_distance: float | None = None,
        travel_fixed: float | None = None,
        travel_period: float | None = None,
        parking_fee: float | None = None,
        rma: int | None = None,
    ) -> dict[str, Any]:
        try:
            # Same fill form rules: an off field may not be set, a required one
            # not cleared (left out = unchanged).
            _one_billing_item_param(billing_item_id, billing_item_name)
            refuse_if_violated(violations(
                FILL_FIELDS, fill_settings(await base_config.load(client)),
                {"description": description, "place_id": place_id,
                 "place_custom": place_custom,
                 "billing_item_id": billing_item_id if billing_item_name is None else billing_item_name},
                update=True,
            ))
            valid_from = localize_naive_datetime("valid_from", valid_from)
            valid_to = localize_naive_datetime("valid_to", valid_to)
            if valid_from is not None and valid_to is not None:
                _ensure_chronological(valid_from, valid_to)
            parents = {
                "company_id": company_id,
                "deal_id": deal_id,
                "request_id": request_id,
                "work_order_id": work_order_id,
                "task_id": task_id,
                "project_contract_id": project_contract_id,
            }
            # Changing assign_id requires sending the new matching parent id.
            if assign_id is not None:
                _validate_assign(assign_id, parents)

            current = await _fetch_for_update(client, id)
            # Without a new assign_id the stored one decides which parent
            # CDESK keeps.
            stored_assign = current.get("assign_id")
            if assign_id is None and stored_assign in _ASSIGN_PARENT:
                _refuse_foreign_parents(stored_assign, parents)
            if valid_from is not None and valid_to is None:
                current_to = current.get("valid_to")
                if isinstance(current_to, str):
                    _ensure_chronological(valid_from, current_to)
            elif valid_to is not None and valid_from is None:
                current_from = current.get("valid_from")
                if isinstance(current_from, str):
                    _ensure_chronological(current_from, valid_to)
            if billing_item_name is not None:
                scope_company, scope_deal = await _fill_scope(
                    client, assign_id if assign_id is not None else stored_assign,
                    parents, current,
                )
                billing_item_id = await _billing_item_by_name(
                    client, billing_item_name, company_id=scope_company, deal_id=scope_deal,
                )

            body = _build_fill_body(
                {
                    "assign_id": assign_id,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "solver_id": solver_id,
                    "company_id": company_id,
                    "deal_id": deal_id,
                    "request_id": request_id,
                    "work_order_id": work_order_id,
                    "task_id": task_id,
                    "project_contract_id": project_contract_id,
                    "description": description,
                    "private_description": private_description,
                    "used_material": used_material,
                    "other_time": other_time,
                    "quantity": quantity,
                    "simple_use": simple_use,
                    "place_id": place_id,
                    "place_custom": place_custom,
                    "place_travel_id": place_travel_id,
                    "billing_item_id": billing_item_id,
                    "billing_travel_id": billing_travel_id,
                    "billing_kopfis_id": billing_kopfis_id,
                    "travel": travel,
                    "travel_distance": travel_distance,
                    "travel_fixed": travel_fixed,
                    "travel_period": travel_period,
                    "parking_fee": parking_fee,
                    "rma": rma,
                }
            )
            _apply_lock_token(body, current)
            response = await client.put(f"v3/fill/{id}", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"update_fill input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="update_fill", record_id=id) from e
        return await _with_billing_check(client, _with_write_minutes(response))

    @mcp.tool(
        description=_DELETE_FILL_DESC,
        annotations=ToolAnnotations(title="Delete fill", destructiveHint=True),
    )
    async def delete_fill(id: int) -> dict[str, Any]:
        try:
            await client.delete(f"v3/fill/{id}")
        except Exception as e:
            raise to_llm_error(e, operation="delete_fill", record_id=id) from e
        return {"deleted": id}


# --- internals ----------------------------------------------------------

def _validate_assign(assign_id: int, parents: dict[str, Any]) -> None:
    """assign_id must be 1..6 and carry its matching parent id (clearer than
    the backend's opaque 400)."""
    if assign_id not in _ASSIGN_PARENT:
        raise ValueError(
            f"assign_id must be one of 1..6 (1=company, 2=deal, 3=request, "
            f"4=work order, 5=task, 6=project contract); got {assign_id}"
        )
    required = _ASSIGN_PARENT[assign_id]
    if parents.get(required) is None:
        raise ValueError(
            f"assign_id={assign_id} requires {required} to identify the parent"
        )
    _refuse_foreign_parents(assign_id, parents)


def _refuse_foreign_parents(assign_id: int, parents: dict[str, Any]) -> None:
    """A fill is linked to its company plus the ONE parent its assign_id names;
    CDESK answers 200 and drops any other parent id, so the link the caller
    meant would be lost without a word."""
    required = _ASSIGN_PARENT.get(assign_id)
    for field, value in parents.items():
        if value is None or field in ("company_id", required):
            continue
        raise ValueError(
            f"assign_id={assign_id} ({_ASSIGN_NAMES[assign_id]}) links the fill only "
            f"to {required}; CDESK does not store {field} on it. To log the work on "
            f"that record, use assign_id={_PARENT_ASSIGN[field]} "
            f"({_ASSIGN_NAMES[_PARENT_ASSIGN[field]]}); otherwise leave {field} out."
        )


def _one_billing_item_param(item_id: int | None, item_name: str | None) -> None:
    if item_id is not None and item_name is not None:
        raise ValueError("pass billing_item_id or billing_item_name, not both")


async def _with_billing_check(client: CdeskClient, result: dict[str, Any]) -> dict[str, Any]:
    """Add a warning when the written fill's billing item belongs to another
    customer's or another deal (see _billing_item_mismatch)."""
    warning = await _billing_item_mismatch(client, result)
    return with_warning_texts(result, [warning]) if warning else result


def _first_id(record: Any, fields: tuple[str, ...]) -> int | None:
    if not isinstance(record, dict):
        return None
    for field in fields:
        value = record.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        if isinstance(value, str) and value.strip().isdigit() and int(value) > 0:
            return int(value)
    return None


# --- billing items -----------------------------------------------------------
# A billing item ("účtovná položka", cd_billing_item) is defined on a deal, in
# the deal's billing tab. A fill takes the items of the deal it is filed under
# (its own deal, or its request's / task's / work order's deal) plus the items
# flagged global on the customer's other deals; "global" never reaches past the
# customer. CDESK's web form offers only those, but the API stores any item
# without checking. CDESK offers no list of them: the
# only place an item's id and name appear is a fill that already uses it (the
# list's billing_item_id / billing_item_title; a fill read with returnFields
# `billing_item` carries the whole item, including its deal in contract_id).
# So a name is looked up among the fills of the fill's customer, newest first.

# How many of the customer's newest fills a name lookup reads.
_ITEM_SCAN_PAGE = 100
_ITEM_SCAN_PAGES = 20
_ITEM_ID_HINT = (
    "Its id is shown in CDESK under Zákazky → Účtovné položky (column Id), or on "
    "the deal's Účtovanie tab; pass it as billing_item_id."
)


async def _fill_scope(
    client: CdeskClient,
    assign_id: Any,
    parents: dict[str, Any],
    current: dict[str, Any] | None = None,
) -> tuple[int | None, int | None]:
    """(customer id, deal id) the fill is filed under: the ones given, else the
    stored fill's, else a request parent's customer and deal."""
    company = parents.get("company_id")
    deal = parents.get("deal_id")
    request_id = parents.get("request_id")
    if current is not None:
        company = company or _first_id(current, ("company_id",))
        deal = deal or _first_id(current, ("contract_id",))
        request_id = request_id or _first_id(current, ("request_id",))
    if (company is None or deal is None) and request_id is not None and assign_id in (3, None):
        request = unwrap_record(await client.get(
            f"v3/request/{request_id}", params={"fieldset": "base"},
        ))
        company = company or _first_id(request, ("id_company",))
        deal = deal or _first_id(request, ("id_rc",))
    return company, deal


async def _billing_item_of_fill(client: CdeskClient, fill_id: Any) -> dict[str, Any] | None:
    """The billing item object of a fill (id, title, contract_id, global, …),
    or None when it has none or the read fails."""
    if not isinstance(fill_id, int) or isinstance(fill_id, bool):
        return None
    try:
        record = unwrap_record(await client.get(
            f"v3/fill/{fill_id}",
            params={"returnFields[]": ["billing_item", "company_id", "contract_id"]},
        ))
    except Exception:  # noqa: BLE001
        return None
    item = record.get("billing_item") if isinstance(record, dict) else None
    return item if isinstance(item, dict) else None


async def _billing_item_by_name(
    client: CdeskClient, name: str, *, company_id: int | None, deal_id: int | None,
) -> int:
    """The id of the billing item called `name` (case- and
    diacritic-insensitive) among those used on the customer's newest fills;
    with several items of that name, the one on the fill's deal (or a global
    one). Raises ValueError naming the candidates or the place to find the id."""
    wanted = canon_name(name)
    if not wanted:
        raise ValueError("billing_item_name must not be empty")
    if company_id is None:
        raise ValueError(
            "billing_item_name is looked up among the fills of the fill's customer, "
            "and this fill's customer is not known here; pass company_id, or "
            "billing_item_id. " + _ITEM_ID_HINT
        )
    sb = build_fill_filter(company_id=company_id)
    titles: dict[int, str] = {}
    used_on: dict[int, int] = {}  # item id → a fill that uses it
    scanned = 0
    complete = False
    for page in range(1, _ITEM_SCAN_PAGES + 1):
        params: dict[str, Any] = {
            "pg": page, "pp": _ITEM_SCAN_PAGE, "fieldset": "base",
            "returnFields[]": ["billing_item_id", "billing_item_title"],
            "sort": resolve_sort("fills", "valid_from.desc"),
        }
        if sb is not None:
            params["sb"] = encode_sb(sb)
        response, _ = await client.get_page("v3/fill", params=params)
        rows, _ = unwrap_list(response)
        scanned += len(rows)
        for row in rows:
            if not isinstance(row, dict):
                continue
            item_id = _first_id(row, ("billing_item_id",))
            title = row.get("billing_item_title")
            if item_id is None or not isinstance(title, str):
                continue
            titles.setdefault(item_id, title)
            fill_id = _first_id(row, ("id",))
            if fill_id is not None:
                used_on.setdefault(item_id, fill_id)
        if len(rows) < _ITEM_SCAN_PAGE:
            complete = True
            break
    matches = [i for i, t in titles.items() if canon_name(t) == wanted]
    searched = (
        f"all {scanned} fills of customer {company_id}" if complete
        else f"the {scanned} newest fills of customer {company_id}"
    )
    # A fill history can hold an item of another customer's deal (CDESK accepts
    # one), so each match is checked against the customer before it is used.
    items: dict[int, dict[str, Any]] = {}
    deal_owner: dict[int, int | None] = {}
    foreign: list[str] = []
    for item_id in list(matches):
        item = await _billing_item_of_fill(client, used_on.get(item_id)) or {}
        items[item_id] = item
        item_deal = _first_id(item, ("contract_id",))
        if item_deal is None:
            continue
        if item_deal not in deal_owner:
            deal_owner[item_deal] = await _deal_customer(client, item_deal)
        owner = deal_owner[item_deal]
        if owner is not None and owner != company_id:
            matches.remove(item_id)
            foreign.append(f"{item_id} (deal {item_deal} of customer {owner})")
    if len(matches) > 1 and deal_id is not None:
        on_deal = [
            i for i in matches
            if items[i].get("contract_id") == deal_id or items[i].get("global")
        ]
        if on_deal:
            matches = on_deal
    if len(matches) == 1:
        return matches[0]
    if not matches and foreign:
        raise ValueError(
            f"billing_item_name {name!r} on {searched} is only item "
            f"{', '.join(foreign)}, which belongs to another customer; it was used "
            f"there by mistake, so it is not used again. " + _ITEM_ID_HINT
        )
    if matches:
        listed = ", ".join(f"{i} ({titles[i]!r})" for i in sorted(matches))
        raise ValueError(
            f"billing_item_name {name!r} matches more than one billing item used on "
            f"{searched}: {listed}; pass the right one as billing_item_id."
        )
    near = closest_row_names([{"name": t} for t in titles.values()], name)
    hint = f" Similar names used there: {', '.join(near)}." if near else ""
    raise ValueError(
        f"no billing item named {name!r} is used on {searched}. CDESK offers no list "
        f"of billing items, so only items already used on the customer's fills can "
        f"be found by name.{hint} " + _ITEM_ID_HINT
    )


async def _deal_customer(client: CdeskClient, deal_id: int) -> int | None:
    """The customer id of a deal, or None when it cannot be read."""
    try:
        deal = unwrap_record(await client.get(
            f"v3/contract/{deal_id}", params={"fieldset": "base"},
        ))
    except Exception:  # noqa: BLE001
        return None
    return _first_id(deal, ("id_company",))


async def _billing_item_mismatch(client: CdeskClient, result: dict[str, Any]) -> str | None:
    """A warning when the fill's billing item belongs to another customer's deal,
    or — unless the item is global — to another deal than the fill's own; CDESK
    stores such a fill without an error. None when the item has no deal or
    cannot be read."""
    saved = result.get("savedData")
    if not isinstance(saved, dict) or not saved.get("billing_item_id"):
        return None
    item = await _billing_item_of_fill(client, _first_id(saved, ("id",)) or result.get("data"))
    if item is None:
        return None
    item_deal = _first_id(item, ("contract_id",))
    if item_deal is None:
        return None
    fill_deal = _first_id(saved, ("contract_id",))
    fill_company = _first_id(saved, ("company_id",))
    label = f"billing item {item.get('id')} ({item.get('title')!r})"
    deal_company = await _deal_customer(client, item_deal)
    if deal_company is not None and fill_company is not None and deal_company != fill_company:
        return (
            f"The fill's {label} belongs to deal {item_deal} of customer "
            f"{deal_company}, while the fill is filed under customer {fill_company}; "
            f"CDESK stored it anyway, so the work is billed on the other customer's "
            f"deal."
        )
    if fill_deal is not None and fill_deal != item_deal and not item.get("global"):
        return (
            f"The fill's {label} belongs to deal {item_deal}, while the fill is linked "
            f"to deal {fill_deal}; CDESK stored it anyway."
        )
    return None


async def _fetch_for_update(client: CdeskClient, id: int) -> dict[str, Any]:
    """Fetch a fill and unwrap to the record dict for the optimistic-lock PUT.

    fieldset=all is required: the v3 default (`extended`) omits `timestamp_check`
    (which for fills carries the record's `updated_at`)."""
    envelope = await client.get(f"v3/fill/{id}", params={"fieldset": "all"})
    record = unwrap_record(envelope)
    if not isinstance(record, dict):
        raise RuntimeError(
            f"unexpected response shape fetching fill {id} for update: "
            f"{type(record).__name__}"
        )
    if "timestamp_check" not in record:
        raise RuntimeError(
            f"fill {id} record has no timestamp_check field; cannot apply optimistic lock"
        )
    return record


def _apply_lock_token(body: dict[str, Any], current: dict[str, Any]) -> None:
    """Set body['timestamp_check'] from the fetched record ONLY when truthy. A
    never-modified fill has `updated_at`/`timestamp_check` null — sending null
    FAILS the PUT, while OMITTING the key makes CDESK skip the lock check."""
    token = current.get("timestamp_check")
    if token:
        body["timestamp_check"] = token


_FILL_CREATE_FIELDS: dict[str, str] = {
    # The deal link: the tool param is `deal_id` (CDESK calls the module
    # Zákazky), but the CDESK body field is still `contract_id`. Sending
    # `deal_id` is silently ignored and the create is then rejected with
    # 'Vyberte zákazku' (code 4).
    "deal_id": "contract_id",
}


# --- worked time -------------------------------------------------------------
# CDESK sends a fill's STORED worked time — valid_to − valid_from plus
# `other_time` (the correction), set on every save, what CDESK bills and sums in
# reports — as `duration_seconds`, and the correction as `other_time_seconds`,
# on the list and the detail alike. `duration` / `other_time` differ between
# them: seconds on a detail or write response, text on a list row ("1h 30min",
# the from→to span only). Every fill result therefore gets `duration_minutes` /
# `other_time_minutes` (int), computed from the seconds fields.

_TIME_COLUMNS = ("duration_seconds", "other_time_seconds")


def _ask_for_list_times(params: dict[str, Any], fields: list[str] | None) -> list[str]:
    """Add the time columns to a list request that names its `fields` (every
    fieldset carries them already). Returns the columns the caller did not ask
    for, to be removed from the rows again."""
    if not fields:
        return []
    added = [c for c in _TIME_COLUMNS if c not in fields]
    params["returnFields[]"] = [*fields, *added]
    return added



def _minutes(seconds: Any) -> int | None:
    if isinstance(seconds, bool):
        return None
    if isinstance(seconds, (int, float)):
        return round(seconds / 60)
    return None


def _add_minutes(record: Any) -> bool:
    """Add `duration_minutes` / `other_time_minutes` to a fill record from its
    seconds fields; False when the record carries no worked time."""
    if not isinstance(record, dict):
        return False
    duration = _minutes(record.get("duration_seconds"))
    if duration is not None:
        record["duration_minutes"] = duration
    other = _minutes(record.get("other_time_seconds"))
    if other is not None:
        record["other_time_minutes"] = other
    return duration is not None


def _add_detail_minutes(record: Any) -> None:
    _add_minutes(record)


def _add_list_minutes(rows: list[Any]) -> str | None:
    """Give every list row its minutes. Returns a warning when a row carries
    no worked time."""
    unmatched = sum(1 for r in rows if isinstance(r, dict) and not _add_minutes(r))
    if unmatched:
        return (
            f"duration_minutes could not be read for {unmatched} fill(s) on this "
            f"page; get_fill gives a fill's worked time (duration_seconds)."
        )
    return None


def _with_write_minutes(response: Any) -> dict[str, Any]:
    result = response if isinstance(response, dict) else {"data": response}
    _add_detail_minutes(result.get("data"))
    _add_detail_minutes(result.get("savedData"))
    return result


def _build_fill_body(public_fields: dict[str, Any]) -> dict[str, Any]:
    """Drop None values and rename the few params whose CDESK field name
    differs (see _FILL_CREATE_FIELDS). Everything else maps 1:1."""
    return {
        _FILL_CREATE_FIELDS.get(k, k): v
        for k, v in public_fields.items()
        if v is not None
    }


def _ensure_chronological(valid_from: str, valid_to: str) -> None:
    """Reject valid_to < valid_from across timezone offsets (a naive value is
    interpreted in the tenant timezone, matching localize_naive_datetime)."""
    try:
        vf = datetime.fromisoformat(valid_from)
        vt = datetime.fromisoformat(valid_to)
    except ValueError:
        return

    def _as_aware(dt: datetime) -> datetime:
        return dt.replace(tzinfo=tenant_timezone()) if dt.tzinfo is None else dt

    if _as_aware(vt) < _as_aware(vf):
        raise ValueError(
            f"valid_to ({valid_to!r}) must be on or after valid_from ({valid_from!r})"
        )


# --- tool descriptions --------------------------------------------------

_LIST_FILLS_DESC = inspect.cleandoc(
    """
    List CDESK fills (fulfillments — logged work/time records). Returns a
    paginated `items` array; `meta.total` is the number of records matching the filter across all pages (CDESK's own count) and `meta.last_page` tells whether this is the last page.

    Each record carries `customer_name` / `solver_name` next to the
    customer and solver ids.

    Worked time: every item carries `duration_minutes` (int) — the fill's
    stored worked time, i.e. valid_to − valid_from plus `other_time`, the value
    CDESK bills and sums in its reports — and `other_time_minutes`, the
    correction part (CDESK's `duration_seconds` / `other_time_seconds` in
    minutes). CDESK's own `duration` field in this list is TEXT of the from→to
    span only ("1h 24min", without the correction), and `other_time` is text
    too ("30min").

    Date window: pass worked_from / worked_to (ISO-8601) to scope the search to
    a work-date range. There is NO implicit month bound: a windowless call
    is the whole visible set, not just this month.

    Common filters:
      - worked_from / worked_to: work-date window (bounds fills.valid_from via a
        start_date BETWEEN; a one-sided bound is extended to far past/future).
      - solver_id / company_id / request_id / deal_id: int or list[int].
      - project_contract_id / assign_id / place_id: further scoping
        (assign_id: 1=company, 2=deal, 3=request, 4=work order, 5=task,
        6=project contract, 7=days-off).
      - invoiced / signed / rma: booleans (invoiced=already billed,
        signed=signature present).

    For advanced filters pass sb_raw as an sb filter object for the CDESK API v3
    fill list (JSON string or object, structured tree form; see
    docs/cdesk-api-v3.json). Working
    columns: fill_id, assign_id, company_id, solver_id, request_id, deal_id,
    project_contract_id, place_id, description (CONTAINS), used_material
    (CONTAINS), task, deal, work_order (ISNULL/ISNOTNULL link filters),
    duration (the stored worked time incl. other_time, in HOURS, numeric
    comparators — {"col": "duration", "c": ">=", "q": 1.5}; sorting by
    `duration` orders by the from→to span instead), invoiced_status, rma, is_signed,
    start_date (BETWEEN or =, strict W3C bounds), plus the no-col text leaf. The
    documented `request` key and the whole `end_date` column are NOT honored and
    are stripped (`end_date` is not in the allowlist at all — bound the window
    through `start_date` instead, which accepts `=` as well as BETWEEN) —
    clauses on any other column are reported in `unsupported_filters`
    for you to apply client-side after paging the full set.
    """
)

_GET_FILL_DESC = inspect.cleandoc(
    """
    Fetch a single fill (fulfillment) by id.
    Each record carries `customer_name` / `solver_name` next to the
    customer and solver ids.

    fieldset selects the field group returned: "base", "extended" (CDESK's
    default when omitted), "all" (base+extended+custom), or "custom". fields is
    a whitelist of field names (returnFields; union with fieldset);
    `customer_name` / `solver_name` are added to it.

    `duration_seconds` is the stored worked time (valid_to − valid_from plus
    `other_time_seconds`, the correction); `duration_minutes` /
    `other_time_minutes` give the same in minutes, as list_fills does.
    """
)

_GET_FILL_ENUMS_DESC = inspect.cleandoc(
    """
    Return the Fill module enum/metadata groups (maingroups, form fields,
    invoicing statuses/dates, deal & request category types, request
    status). Use these ids when building create/update payloads and sb filters.
    Billing items are not among them: CDESK offers no list of billing items
    (see create_fill).
    """
)

_CREATE_FILL_DESC = inspect.cleandoc(
    """
    Create a fill (log work/time). `assign_id` selects the parent kind and
    dictates which parent id is REQUIRED: 1=company→company_id, 2=deal→
    deal_id, 3=request→request_id, 4=work order→work_order_id, 5=task→
    task_id, 6=project contract→project_contract_id. (Days-off fills are not
    created here — CDESK refuses assign_id 7.) A fill is linked to its
    company_id plus that one parent; any other parent id is refused, because
    CDESK does not store it.

    valid_from / valid_to (ISO-8601; an explicit offset like +02:00 is
    recommended — a naive datetime is read as tenant-local) are required and
    valid_to must be ≥ valid_from. CDESK sets the worked time from them on
    every save: valid_to − valid_from plus `other_time` (seconds, may be
    negative), the way to correct it — there is no duration to set directly.
    The result carries `duration_minutes` / `other_time_minutes`.
    `solver_id` (the worker) is always required. Tenant billing settings may
    require a billing item; CDESK then refuses a fill without one (HTTP 400). Ids of
    places and other parents come from get_fill_enums and the related module
    list tools.

    Billing item ("účtovná položka"): billing_item_id, or billing_item_name.
    A billing item is defined on a deal (its Účtovanie tab). A fill may use
    the items of the deal it is filed under (its own deal, or its request's,
    task's or work order's deal) and the items flagged global on the same
    customer's other deals; a global item is never meant for another
    customer. CDESK offers no list of billing items — neither get_fill_enums
    nor any other tool lists them. An item's id and name appear only on fills
    that already use it: list_fills items carry billing_item_id and
    billing_item_title. billing_item_name looks the name up (case- and
    diacritic-insensitive) among the items used on the fill's customer's
    newest fills (up to 2000), preferring the fill's deal when several items
    share the name and leaving out an item of another customer's deal; an
    item never used on that customer's fills is not found by name, and the
    error then names where CDESK shows its id. A nonexistent
    billing_item_id is refused by CDESK. CDESK does accept an item that
    belongs to another customer's deal, or a non-global item of another deal;
    the result then carries a warning naming the item's deal and customer.

    `description`, the place (`place_id` / `place_custom`) and
    `billing_item_id` follow the tenant's CDESK fill form, read live on
    every call (also on update_fill): a field switched off is refused when
    given, a required one must be given. get_fill_enums lists their state
    under `fields`. Fills dated in the past and fills on a closed request
    can also be refused by tenant settings.

    Side effects (by CDESK): a fill on a request in "prijaté" or "priradené" moves
    that request to "v riešení" (action code 30) on tenants with the automatic
    status change after a fill.
    """
)

_UPDATE_FILL_DESC = inspect.cleandoc(
    """
    Update a fill (partial — send only the fields to change). Optimistic locking
    and the version token are handled internally (GET → mutate → PUT). Changing
    `assign_id` requires sending the new matching parent id; a parent id that
    does not match the fill's assign_id (the new one, else the stored one) is
    refused, because CDESK does not store it. An already-invoiced
    fill is write-blocked (HTTP 403) unless you hold the invoiced-write right,
    and a fill on a closed request is refused (HTTP 424).
    The worked time follows valid_from / valid_to plus `other_time` (seconds),
    recomputed by CDESK on every save; the result carries `duration_minutes`.

    Billing item ("účtovná položka"): billing_item_id, or billing_item_name.
    A billing item is defined on a deal (its Účtovanie tab). A fill may use
    the items of the deal it is filed under (its own deal, or its request's,
    task's or work order's deal) and the items flagged global on the same
    customer's other deals; a global item is never meant for another
    customer. CDESK offers no list of billing items — neither get_fill_enums
    nor any other tool lists them. An item's id and name appear only on fills
    that already use it: list_fills items carry billing_item_id and
    billing_item_title. billing_item_name looks the name up (case- and
    diacritic-insensitive) among the items used on the fill's customer's
    newest fills (up to 2000), preferring the fill's deal when several items
    share the name and leaving out an item of another customer's deal; an
    item never used on that customer's fills is not found by name, and the
    error then names where CDESK shows its id. A nonexistent
    billing_item_id is refused by CDESK. CDESK does accept an item that
    belongs to another customer's deal, or a non-global item of another deal;
    the result then carries a warning naming the item's deal and customer.
    """
)

_DELETE_FILL_DESC = inspect.cleandoc(
    """
    Delete (soft-delete) a fill by id. A fill on a CDESK invoice, an
    otherwise invoiced fill (without the invoiced-write right) and a fill on a
    closed request cannot be deleted (HTTP 424). Returns {"deleted": <id>} on
    success.
    """
)
