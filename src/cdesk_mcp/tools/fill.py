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
never sees it. The backend hard-blocks writes to already-invoiced fills (HTTP
409) — that surfaces through the normal error translator.

Live-audited 2026-06-24 (scripts/probe_fill_filters.py): the date window filters
via a `start_date` BETWEEN leaf (built by build_fill_filter). That audit also
found the list defaulting to the CURRENT CALENDAR MONTH with no window, but the
backend has since CHANGED: re-verified 2026-07-31, a windowless list returned
fills dated four months back. Do not rely on an implicit month bound.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.filters import build_fill_filter, encode_sb
from cdesk_mcp.tools._helpers import (
    apply_field_scope,
    to_llm_error,
    unsupported_filter_directive,
    unwrap_list,
    unwrap_record,
    validate_fields,
    validate_fieldset,
)
from cdesk_mcp.tz import localize_naive_datetime, tenant_timezone

# Pagination upper bound — mirrors the task/request modules' 100 cap to keep
# response sizes within an LLM's context budget.
_MAX_PER_PAGE = 100

# assign_id → the body field that MUST accompany it (7 = days-off needs none).
# Used for a clear client-side error instead of the backend's opaque 400.
_ASSIGN_PARENT: dict[int, str | None] = {
    1: "company_id",
    2: "deal_id",
    3: "request_id",
    4: "work_order_id",
    5: "task_id",
    6: "project_contract_id",
    7: None,
}


def register_fill_tools(
    mcp: FastMCP,
    client: CdeskClient,
) -> None:
    """Register all Fill-module tools on the given FastMCP instance. Takes no
    EnumCache — Fill resolves no enum names client-side (see get_fill_enums)."""

    @mcp.tool(
        description=_LIST_FILLS_DESC,
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
            if sort:
                params["sort"] = sort
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response = await client.get("v3/fill", params=params)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"list_fills input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_fills") from e

        records, meta = unwrap_list(response)
        result = {"items": records, "meta": meta, "page": page, "per_page": per_page}
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
            response = await client.get(f"v3/fill/{id}", params=params or None)
        except ValueError as e:
            raise RuntimeError(f"get_fill input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="get_fill", record_id=id) from e
        return unwrap_record(response)

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
        if isinstance(response, dict) and "enums" in response:
            return response["enums"]
        return unwrap_record(response)

    @mcp.tool(
        description=_CREATE_FILL_DESC,
        annotations=ToolAnnotations(title="Create fill", destructiveHint=True),
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
        duration: int | None = None,
        other_time: int | None = None,
        quantity: float | None = None,
        simple_use: bool | None = None,
        place_id: int | None = None,
        place_custom: str | None = None,
        place_travel_id: int | None = None,
        billing_item_id: int | None = None,
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
            parents = {
                "company_id": company_id,
                "deal_id": deal_id,
                "request_id": request_id,
                "work_order_id": work_order_id,
                "task_id": task_id,
                "project_contract_id": project_contract_id,
            }
            _validate_assign(assign_id, parents)

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
                    "duration": duration,
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
        return response if isinstance(response, dict) else {"data": response}

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
        duration: int | None = None,
        other_time: int | None = None,
        quantity: float | None = None,
        simple_use: bool | None = None,
        place_id: int | None = None,
        place_custom: str | None = None,
        place_travel_id: int | None = None,
        billing_item_id: int | None = None,
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
            valid_from = localize_naive_datetime("valid_from", valid_from)
            valid_to = localize_naive_datetime("valid_to", valid_to)
            if valid_from is not None and valid_to is not None:
                _ensure_chronological(valid_from, valid_to)
            # Changing assign_id requires sending the new matching parent id.
            if assign_id is not None:
                _validate_assign(
                    assign_id,
                    {
                        "company_id": company_id,
                        "deal_id": deal_id,
                        "request_id": request_id,
                        "work_order_id": work_order_id,
                        "task_id": task_id,
                        "project_contract_id": project_contract_id,
                    },
                )

            current = await _fetch_for_update(client, id)
            if valid_from is not None and valid_to is None:
                current_to = current.get("valid_to")
                if isinstance(current_to, str):
                    _ensure_chronological(valid_from, current_to)
            elif valid_to is not None and valid_from is None:
                current_from = current.get("valid_from")
                if isinstance(current_from, str):
                    _ensure_chronological(current_from, valid_to)

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
                    "duration": duration,
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
        return response if isinstance(response, dict) else {"data": response}

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
    """assign_id must be 1..7 and carry its matching parent id (clearer than
    the backend's opaque 400)."""
    if assign_id not in _ASSIGN_PARENT:
        raise ValueError(
            f"assign_id must be one of 1..7 (1=company, 2=deal, 3=request, "
            f"4=work order, 5=task, 6=project contract, 7=days-off); got {assign_id}"
        )
    required = _ASSIGN_PARENT[assign_id]
    if required is not None and parents.get(required) is None:
        raise ValueError(
            f"assign_id={assign_id} requires {required} to identify the parent"
        )


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
    # 'Vyberte zákazku' (code 4) — verified live 2026-07-31.
    "deal_id": "contract_id",
}


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
    paginated `items` array; `meta` is usually empty (no total count — detect
    the last page by a short/empty page).

    Date window: pass worked_from / worked_to (ISO-8601) to scope the search to
    a work-date range. There is NO implicit month bound — re-verified live
    2026-07-31 with fills dated one and four months back, both of which a
    windowless call returned. (An earlier audit on 2026-06-24 found the v1 list
    limited to the current calendar month; the backend changed, so older notes
    and any code written against that assumption are stale.) A windowless call
    is therefore the whole visible set, not just this month.

    Common filters (all live-verified 2026-06-24):
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
    docs/cdesk-api-v3.json). LIVE-VERIFIED working
    columns: fill_id, assign_id, company_id, solver_id, request_id, deal_id,
    project_contract_id, place_id, description (CONTAINS), used_material
    (CONTAINS), task, deal, work_order (ISNULL/ISNOTNULL link filters),
    duration (HOURS, numeric comparators), invoiced_status, rma, is_signed,
    start_date (BETWEEN or =, strict W3C bounds), plus the no-col text leaf. The
    documented `request` key and the whole `end_date` column are NOT honored and
    are stripped (`end_date` is not in the allowlist at all — bound the window
    through `start_date` instead, which accepts `=` as well as BETWEEN;
    verified live 2026-07-31) — clauses on any other column are reported in `unsupported_filters`
    for you to apply client-side after paging the full set.
    """
)

_GET_FILL_DESC = inspect.cleandoc(
    """
    Fetch a single fill (fulfillment) by id.

    fieldset selects the field group returned: "base", "extended" (CDESK's
    default when omitted), "all" (base+extended+custom), or "custom". fields is
    an exact whitelist of field names (returnFields; union with fieldset).
    """
)

_GET_FILL_ENUMS_DESC = inspect.cleandoc(
    """
    Return the Fill module enum/metadata groups (maingroups, form fields,
    invoicing statuses/dates, deal & request category types, request
    status). Use these ids when building create/update payloads and sb filters.
    """
)

_CREATE_FILL_DESC = inspect.cleandoc(
    """
    Create a fill (log work/time). `assign_id` selects the parent kind and
    dictates which parent id is REQUIRED: 1=company→company_id, 2=deal→
    deal_id, 3=request→request_id, 4=work order→work_order_id, 5=task→
    task_id, 6=project contract→project_contract_id, 7=days-off→none.

    valid_from / valid_to (ISO-8601; an explicit offset like +02:00 is
    recommended — a naive datetime is read as tenant-local) are required and
    valid_to must be ≥ valid_from; `duration` is normally derived from them.
    `solver_id` (the worker) is typically required by the tenant — pass it
    explicitly. An already-invoiced parent or tenant billing settings may
    require billing_item_id and will reject the write (HTTP 409). Fetch
    tenant-specific ids via get_fill_enums and the related module list tools.
    """
)

_UPDATE_FILL_DESC = inspect.cleandoc(
    """
    Update a fill (partial — send only the fields to change). Optimistic locking
    and the version token are handled internally (GET → mutate → PUT). Changing
    `assign_id` requires sending the new matching parent id. An already-invoiced
    fill is write-blocked (HTTP 409) unless you hold the invoiced-write right.
    """
)

_DELETE_FILL_DESC = inspect.cleandoc(
    """
    Delete (soft-delete) a fill by id. An already-invoiced fill cannot be
    deleted (HTTP 409). Returns {"deleted": <id>} on success.
    """
)
