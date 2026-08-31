"""Deal module MCP tools — CDESK deals ("zákazky", table
cd_real_contracts).

Six tools exposing the Deal v3 CRUD surface: list / get / get_enums /
create / update / delete. Deal statuses, phases and category types are
TENANT enum-table rows (with action codes and lang variants), so this module
takes an EnumCache on `v3/contract/enums` — the 4th cached module after
tasks/requests/catalogs.

Status model (apiportal Model/Contract.php): `rc_status` carries the standard
ACTION CODE (10 received, 20 in progress, 25 delivered, 30 refused,
80 completed), `rc_status_id` the tenant enum id; the server derives one from
the other on writes. "Open" means action_code < 30. The sb `status` filter key
additionally accepts the literal keywords 'open' / 'closed'.

Writes use RAW column names (`rc_title`, `id_company`, `rc_note`, …); the v1
save whitelist silently drops anything else with a 200.

READS return raw columns too. The Deal model declares an `id_rc`→`id` /
`rc_title`→`name` keymap, but the v1 list path never applies it (it selects
the columns unaliased and neither `getStdList` nor `getStdMany` remaps), so
BOTH the list and the detail endpoint return `id_rc`/`rc_title` — verified
live 2026-07-31. Read the id defensively (`id_rc` first, `id` as fallback) so
this stays correct if the keymap is ever wired up. NOTE `docs/cdesk-api-v3.json`
is stale here: its `/v3/contract` GET description still advertises the remap,
while the backend's current annotation (ContractV3Controller class docblock,
apiportal) explicitly says raw columns.

Optimistic locking is
hidden like the fill/task modules (GET fieldset=all → mutate → PUT echoing
`timestamp_check`) — for deals the token is REQUIRED on update (a missing
or stale one is a 409 recordUpdated conflict).

Source-verified against the apiportal v1 handlers 2026-07-20; the live v3
layer exists on the deployed backend only (no local checkout). Live E2E is
gated on the `deal` ACL for the API user — see .ai/scripts/e2e_test_deal.py.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.enums import EnumCache
from cdesk_mcp.filters import build_deal_filter, encode_sb
from cdesk_mcp.tools._helpers import (
    annotate_write_warnings,
    apply_field_scope,
    resolve_enum_or_raise,
    to_llm_error,
    unsupported_filter_directive,
    unwrap_list,
    unwrap_record,
    validate_fields,
    validate_fieldset,
    write_values_match,
)
from cdesk_mcp.tz import localize_naive_datetime, tenant_timezone

# Pagination upper bound — mirrors the other modules' 100 cap.
_MAX_PER_PAGE = 100

# The backend's `status` filter keywords (Model/Contract.php
# STATUS_GROUP_OPEN/CLOSED): open = action_code < 30, closed = >= 30.
_STATUS_KEYWORDS = ("open", "closed")

# Compact per-entry projection for get_deal_enums — the raw rows carry
# ~30 presentation fields (icons, colors, per-locale names) that just burn
# the LLM's context.
_ENUM_ENTRY_KEYS = ("id", "name", "action_code", "parent_id", "lang_en", "lang_sk")


def register_deal_tools(
    mcp: FastMCP,
    client: CdeskClient,
    cache: EnumCache,
) -> None:
    """Register all Deal-module tools on the given FastMCP instance.
    `cache` targets v3/contract/enums (buckets: status / phase / catType)."""

    @mcp.tool(
        description=_LIST_DEALS_DESC,
        annotations=ToolAnnotations(title="List deals", readOnlyHint=True),
    )
    async def list_deals(
        text_search: str | None = None,
        title: str | None = None,
        code: str | None = None,
        customer_id: int | list[int] | None = None,
        customer_name: str | None = None,
        status: str | int | None = None,
        phase_name: str | None = None,
        cat_type_name: str | None = None,
        responsible_person_id: int | None = None,
        for_invoicing: bool | None = None,
        due_after: str | None = None,
        due_before: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        updated_after: str | None = None,
        updated_before: str | None = None,
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
            status_value = await _resolve_status_filter(cache, status)
            phase_id = await resolve_enum_or_raise(
                cache, "phase", phase_name, kind="deal phase"
            )
            cat_type_id = await resolve_enum_or_raise(
                cache, "catType", cat_type_name, kind="deal category type"
            )
            dropped_clauses: list[dict[str, Any]] = []
            sb = build_deal_filter(
                text_search=text_search,
                title=title,
                code=code,
                customer_id=customer_id,
                customer_name=customer_name,
                status=status_value,
                cat_type_id=cat_type_id,
                phase_id=phase_id,
                responsible_person_id=responsible_person_id,
                for_invoicing=for_invoicing,
                due_after=due_after,
                due_before=due_before,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                sb_raw=sb_raw,
                dropped_cols_out=dropped_clauses,
            )
            params: dict[str, Any] = {"pg": page, "pp": per_page}
            apply_field_scope(params, fieldset, fields)
            if sort:
                params["sort"] = sort
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response = await client.get("v3/contract", params=params)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"list_deals input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_deals") from e

        records, meta = unwrap_list(response)
        meta.pop("enums", None)
        result = {"items": records, "meta": meta, "page": page, "per_page": per_page}
        if dropped_clauses:
            result["unsupported_filters"] = unsupported_filter_directive(dropped_clauses)
        return result

    @mcp.tool(
        description=_GET_DEAL_DESC,
        annotations=ToolAnnotations(title="Get deal", readOnlyHint=True),
    )
    async def get_deal(
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
            response = await client.get(f"v3/contract/{id}", params=params or None)
        except ValueError as e:
            raise RuntimeError(f"get_deal input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="get_deal", record_id=id) from e
        return unwrap_record(response)

    @mcp.tool(
        description=_GET_DEAL_ENUMS_DESC,
        annotations=ToolAnnotations(title="Get deal enums", readOnlyHint=True),
    )
    async def get_deal_enums() -> dict[str, Any]:
        try:
            response = await client.get("v3/contract/enums")
        except Exception as e:
            raise to_llm_error(e, operation="get_deal_enums") from e
        # Envelope: {data: false, enums: {...}, settings: {...}, tags: [...]}.
        # List buckets are slimmed to the decision-relevant fields; `settings`
        # carries the tenant gates (types.status == 2 → cat_type required,
        # phases.enabled → phase required) so it's passed through whole.
        enums = response.get("enums") if isinstance(response, dict) else None
        slim: dict[str, Any] = {}
        if isinstance(enums, dict):
            for bucket, entries in enums.items():
                if isinstance(entries, list):
                    slim[bucket] = [
                        {k: e[k] for k in _ENUM_ENTRY_KEYS if e.get(k) is not None}
                        for e in entries
                        if isinstance(e, dict)
                    ]
        result: dict[str, Any] = {"enums": slim}
        if isinstance(response, dict):
            if "settings" in response:
                result["settings"] = response["settings"]
            if "tags" in response:
                result["tags"] = response["tags"]
        # This endpoint is NOT gated by `contract.enabled`, so it answers 200
        # with the full enum lists even when the module is off — the only
        # signal is settings.enabled, which is easy to miss among ~17 keys.
        # Say it outright, or the enum lists read as "module is usable".
        settings = result.get("settings")
        if isinstance(settings, dict) and settings.get("enabled") is False:
            result["module_disabled"] = (
                "The Zákazky (deal) module is DISABLED for this tenant "
                "(settings.enabled=false). The status/phase/type lists below "
                "are returned anyway because this endpoint is not gated, but "
                "list_deals / get_deal / update_deal fail with 'Modul vypnutý' "
                "and the write tools refuse. Do not present the module as "
                "available; an administrator must enable it first."
            )
        return result

    @mcp.tool(
        description=_CREATE_DEAL_DESC,
        annotations=ToolAnnotations(title="Create deal", destructiveHint=True),
    )
    async def create_deal(
        title: str,
        customer_id: int,
        status_name: str | None = None,
        status_id: int | None = None,
        code: str | None = None,
        cat_type_name: str | None = None,
        phase_name: str | None = None,
        order_number: str | None = None,
        est_opening_date: str | None = None,
        opening_date: str | None = None,
        est_due_date: str | None = None,
        due_date: str | None = None,
        in_hours_plan: bool | None = None,
        in_bill_plan: bool | None = None,
        in_cost_plan: bool | None = None,
        bookkeeping_entity_id: int | None = None,
        note: str | None = None,
        responsible_person_ids: list[int] | None = None,
        coordinator_contact_ids: list[int] | None = None,
        tag_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        try:
            await _require_module_enabled(cache, "create_deal")
            resolved_status_id = await _require_status(cache, status_name, status_id)
            body = await _build_deal_body(
                cache,
                title=title,
                customer_id=customer_id,
                status_id=resolved_status_id,
                code=code,
                cat_type_name=cat_type_name,
                phase_name=phase_name,
                order_number=order_number,
                est_opening_date=est_opening_date,
                opening_date=opening_date,
                est_due_date=est_due_date,
                due_date=due_date,
                in_hours_plan=in_hours_plan,
                in_bill_plan=in_bill_plan,
                in_cost_plan=in_cost_plan,
                bookkeeping_entity_id=bookkeeping_entity_id,
                note=note,
                responsible_person_ids=responsible_person_ids,
                coordinator_contact_ids=coordinator_contact_ids,
                tag_ids=tag_ids,
            )
            response = await client.post("v3/contract", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"create_deal input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="create_deal") from e
        annotated = annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=("rc_code",),
        )
        return await _verify_write(
            client, annotated.get("data"), body, annotated, operation="create_deal"
        )

    @mcp.tool(
        description=_UPDATE_DEAL_DESC,
        annotations=ToolAnnotations(title="Update deal", destructiveHint=True),
    )
    async def update_deal(
        id: int,
        title: str | None = None,
        customer_id: int | None = None,
        status_name: str | None = None,
        status_id: int | None = None,
        code: str | None = None,
        cat_type_name: str | None = None,
        phase_name: str | None = None,
        order_number: str | None = None,
        est_opening_date: str | None = None,
        opening_date: str | None = None,
        est_due_date: str | None = None,
        due_date: str | None = None,
        in_hours_plan: bool | None = None,
        in_bill_plan: bool | None = None,
        in_cost_plan: bool | None = None,
        bookkeeping_entity_id: int | None = None,
        note: str | None = None,
        responsible_person_ids: list[int] | None = None,
        coordinator_contact_ids: list[int] | None = None,
        tag_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        try:
            await _require_module_enabled(cache, "update_deal")
            if status_name is not None and status_id is not None:
                raise ValueError("pass either status_name or status_id, not both")
            resolved_status_id = status_id
            if status_name is not None:
                resolved_status_id = await resolve_enum_or_raise(
                    cache, "status", status_name, kind="deal status"
                )
            body = await _build_deal_body(
                cache,
                title=title,
                customer_id=customer_id,
                status_id=resolved_status_id,
                code=code,
                cat_type_name=cat_type_name,
                phase_name=phase_name,
                order_number=order_number,
                est_opening_date=est_opening_date,
                opening_date=opening_date,
                est_due_date=est_due_date,
                due_date=due_date,
                in_hours_plan=in_hours_plan,
                in_bill_plan=in_bill_plan,
                in_cost_plan=in_cost_plan,
                bookkeeping_entity_id=bookkeeping_entity_id,
                note=note,
                responsible_person_ids=responsible_person_ids,
                coordinator_contact_ids=coordinator_contact_ids,
                tag_ids=tag_ids,
            )
            current = await _fetch_for_update(client, id)
            _check_dates_against_current(body, current)
            _apply_lock_token(body, current)
            response = await client.put(f"v3/contract/{id}", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"update_deal input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="update_deal", record_id=id) from e
        annotated = annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=("rc_code",),
        )
        # `timestamp_check` is a lock token we echoed, not a user value.
        verifiable = {k: v for k, v in body.items() if k != "timestamp_check"}
        return await _verify_write(
            client, id, verifiable, annotated, operation="update_deal"
        )

    @mcp.tool(
        description=_DELETE_DEAL_DESC,
        annotations=ToolAnnotations(title="Delete deal", destructiveHint=True),
    )
    async def delete_deal(id: int) -> dict[str, Any]:
        try:
            await _require_module_enabled(cache, "delete_deal")
        except RuntimeError:
            raise
        try:
            await client.delete(f"v3/contract/{id}")
        except Exception as e:
            raise to_llm_error(e, operation="delete_deal", record_id=id) from e
        return {"deleted": id}


# --- internals ----------------------------------------------------------

# Body column -> stored column for the write read-back. Every pair here has
# been round-tripped live (2026-07-31) so the normalizers below know the exact
# stored shape; anything unverified is deliberately absent rather than guessed,
# because a FALSE "not stored" warning is worse than a missing one.
_READ_BACK_FIELDS: tuple[str, ...] = (
    "rc_title", "rc_code", "id_company", "rc_status_id", "cat_type_id",
    "rc_phase_id", "order_number", "rc_note",
    "est_opening_date", "opening_date", "est_due_date", "due_date",
    "in_hours_plan", "in_bill_plan", "in_cost_plan",
    "bookkeeping_entity_id", "responsible_person_id", "coordinator_contact_id",
)

# LLM-facing names for the raw columns, so a warning reads like the tool's own
# parameters rather than the DB schema.
_PARAM_NAMES: dict[str, str] = {
    "rc_title": "title", "rc_code": "code", "id_company": "customer_id",
    "rc_status_id": "status", "cat_type_id": "cat_type_name",
    "rc_phase_id": "phase_name", "rc_note": "note",
    "responsible_person_id": "responsible_person_ids",
    "coordinator_contact_id": "coordinator_contact_ids",
}


async def _verify_write(
    client: CdeskClient,
    deal_id: Any,
    sent_body: dict[str, Any],
    response: dict[str, Any],
    *,
    operation: str,
) -> dict[str, Any]:
    """Re-read the record and warn about anything sent but not stored.

    Deal writes answer `{data: <id>}` with NO `savedData`, so
    annotate_write_warnings has nothing to diff and a silent drop would be
    invisible — the tool would report success for a value CDESK discarded. A
    read-back is the only way to honour "never claim a value was stored when it
    wasn't", so it costs one extra GET per write.
    """
    if not isinstance(deal_id, int):
        return response
    try:
        envelope = await client.get(f"v3/contract/{deal_id}", params={"fieldset": "all"})
        stored = unwrap_record(envelope)
    except Exception:  # noqa: BLE001 - verification must never fail the write
        return {
            **response,
            "write_verification": (
                f"The write succeeded, but {operation} could not re-read deal "
                f"{deal_id} afterwards, so field persistence was NOT verified. "
                f"CDESK returns no saved record for deal writes, so do not "
                f"state that a specific value was stored — confirm with "
                f"get_deal."
            ),
        }
    if not isinstance(stored, dict):
        return response
    dropped: list[str] = []
    for column in _READ_BACK_FIELDS:
        if column not in sent_body:
            continue
        sent = sent_body[column]
        if sent is None or sent == [] or sent == "":
            continue
        if not write_values_match(sent, stored.get(column)):
            label = _PARAM_NAMES.get(column, column)
            dropped.append(
                f"{label} (sent {sent!r}, stored {stored.get(column)!r})"
            )
    if not dropped:
        return response
    warnings = list(response.get("warnings") or [])
    warnings.append(
        "CDESK accepted the write but did NOT store: " + "; ".join(dropped)
        + ". Verified by re-reading the record. Do not tell the user these "
        "values were saved — the most common cause is a tenant setting or a "
        "field write right this account lacks."
    )
    return {**response, "warnings": warnings}


async def _require_module_enabled(cache: EnumCache, operation: str) -> None:
    """Refuse a WRITE when the tenant has the Zákazky module switched off.

    CDESK gates the module inconsistently: with `contract.enabled` false the
    GET list/detail and the PUT return 403 'Modul vypnutý', but **POST and
    DELETE return 200 and really do mutate** (verified live 2026-07-31 —
    created CDD-1 twice on a disabled tenant, then deleted it, all reported as
    success while GET on the same id answered 'Modul vypnutý'). So a user was
    told "Zákazka … uložená" for a record they then could not read back.

    The state is knowable client-side: `v3/contract/enums` is itself ungated
    and reports `settings.enabled`. This guard is a MITIGATION, not the fix —
    the backend should gate its own writes, since any other API client bypasses
    us. See docs/bugs.md.
    """
    if _module_disabled(cache):
        # The cached settings may predate an admin re-enabling the module, and
        # a false refusal is worse than a missed warning — re-read once before
        # refusing.
        try:
            await cache.refresh()
        except Exception:  # noqa: BLE001 - keep the original verdict on failure
            pass
        if not _module_disabled(cache):
            return
        raise RuntimeError(
            f"{operation} refused: the CDESK Zákazky (deal) module is DISABLED "
            f"for this tenant (`contract.enabled` is off, reported as "
            f"settings.enabled=false by get_deal_enums). Reads already fail "
            f"with 'Modul vypnutý', but this write would be ACCEPTED by CDESK "
            f"and create a record nobody can read back, so it was not sent. "
            f"An administrator enables the module under the tenant's global "
            f"settings ('Zapnúť modul Zákazky')."
        )


def _module_disabled(cache: EnumCache) -> bool:
    """True only when the tenant explicitly reports the module off. Absent or
    unreadable settings mean "unknown" — never block on a guess."""
    settings = getattr(cache, "settings", None)
    if not isinstance(settings, dict):
        return False
    return settings.get("enabled") is False


async def _resolve_status_filter(
    cache: EnumCache, status: str | int | None
) -> int | str | None:
    """List-filter status: pass ints (enum id or action code) through, accept
    the backend keywords 'open'/'closed', resolve any other string as a tenant
    status NAME → enum id (tenant enum ids are always > 80, so the backend's
    id-vs-action-code dispatch picks rc_status_id matching)."""
    if status is None or isinstance(status, int):
        return status
    lowered = status.strip().lower()
    if lowered in _STATUS_KEYWORDS:
        return lowered
    if lowered.isdigit():
        return int(lowered)
    return await resolve_enum_or_raise(cache, "status", status, kind="deal status")


async def _require_status(
    cache: EnumCache, status_name: str | None, status_id: int | None
) -> int:
    """Create requires a status (the backend 400s without rc_status_id or
    rc_status) — enforce it client-side with a clearer message."""
    if status_name is not None and status_id is not None:
        raise ValueError("pass either status_name or status_id, not both")
    if status_id is not None:
        return status_id
    resolved = await resolve_enum_or_raise(
        cache, "status", status_name, kind="deal status"
    )
    if resolved is None:
        raise ValueError(
            "a deal status is required — pass status_name (see "
            "get_deal_enums 'status' bucket) or status_id"
        )
    return resolved


async def _build_deal_body(
    cache: EnumCache,
    *,
    title: str | None,
    customer_id: int | None,
    status_id: int | None,
    code: str | None,
    cat_type_name: str | None,
    phase_name: str | None,
    order_number: str | None,
    est_opening_date: str | None,
    opening_date: str | None,
    est_due_date: str | None,
    due_date: str | None,
    in_hours_plan: bool | None,
    in_bill_plan: bool | None,
    in_cost_plan: bool | None,
    bookkeeping_entity_id: int | None,
    note: str | None,
    responsible_person_ids: list[int] | None,
    coordinator_contact_ids: list[int] | None,
    tag_ids: list[int] | None,
) -> dict[str, Any]:
    """Translate LLM params → the raw-column body the v1 save whitelist
    accepts. Dates are tenant-tz localized; opening/due chronology is checked
    for pairs present in THIS call (cross-checks against the stored record
    happen in update via _check_dates_against_current)."""
    cat_type_id = await resolve_enum_or_raise(
        cache, "catType", cat_type_name, kind="deal category type"
    )
    phase_id = await resolve_enum_or_raise(
        cache, "phase", phase_name, kind="deal phase"
    )
    dates = {
        "est_opening_date": localize_naive_datetime("est_opening_date", est_opening_date),
        "opening_date": localize_naive_datetime("opening_date", opening_date),
        "est_due_date": localize_naive_datetime("est_due_date", est_due_date),
        "due_date": localize_naive_datetime("due_date", due_date),
    }
    _ensure_chronological("est_opening_date", dates["est_opening_date"],
                          "est_due_date", dates["est_due_date"])
    _ensure_chronological("opening_date", dates["opening_date"],
                          "due_date", dates["due_date"])
    body: dict[str, Any] = {
        "rc_title": title,
        "id_company": customer_id,
        "rc_status_id": status_id,
        "rc_code": code,
        "cat_type_id": cat_type_id,
        "rc_phase_id": phase_id,
        "order_number": order_number,
        **dates,
        "in_hours_plan": in_hours_plan,
        "in_bill_plan": in_bill_plan,
        "in_cost_plan": in_cost_plan,
        "bookkeeping_entity_id": bookkeeping_entity_id,
        "rc_note": note,
        "responsible_person_id": responsible_person_ids,
        "coordinator_contact_id": coordinator_contact_ids,
        "selectedTags": tag_ids,
    }
    return {k: v for k, v in body.items() if v is not None}


def _ensure_chronological(
    from_name: str, from_value: str | None, to_name: str, to_value: str | None
) -> None:
    """Reject a due date before its opening date (the backend 400s with a
    Slovak message; fail earlier and clearer). Only checked when both bounds
    are in the same call."""
    if not from_value or not to_value:
        return
    try:
        frm = datetime.fromisoformat(from_value)
        to = datetime.fromisoformat(to_value)
    except ValueError:
        return

    def _aware(dt: datetime) -> datetime:
        return dt.replace(tzinfo=tenant_timezone()) if dt.tzinfo is None else dt

    if _aware(to) < _aware(frm):
        raise ValueError(
            f"{to_name} ({to_value!r}) must be on or after {from_name} ({from_value!r})"
        )


def _check_dates_against_current(
    body: dict[str, Any], current: dict[str, Any]
) -> None:
    """On update, cross-check a one-sided date change against the stored
    counterpart (mirrors the fill module's guard)."""
    for from_key, to_key in (("opening_date", "due_date"),
                             ("est_opening_date", "est_due_date")):
        sent_from, sent_to = body.get(from_key), body.get(to_key)
        if sent_from and not sent_to:
            stored_to = current.get(to_key)
            if isinstance(stored_to, str) and stored_to:
                _ensure_chronological(from_key, sent_from, to_key, stored_to)
        elif sent_to and not sent_from:
            stored_from = current.get(from_key)
            if isinstance(stored_from, str) and stored_from:
                _ensure_chronological(from_key, stored_from, to_key, sent_to)


async def _fetch_for_update(client: CdeskClient, id: int) -> dict[str, Any]:
    """Fetch a deal (fieldset=all, which includes `timestamp_check` — the
    record's update_date) for the optimistic-lock PUT. For deals the token
    is REQUIRED on update: omitting it is treated as a stale write (409)."""
    envelope = await client.get(f"v3/contract/{id}", params={"fieldset": "all"})
    record = unwrap_record(envelope)
    if not isinstance(record, dict):
        raise RuntimeError(
            f"unexpected response shape fetching deal {id} for update: "
            f"{type(record).__name__}"
        )
    return record


def _apply_lock_token(body: dict[str, Any], current: dict[str, Any]) -> None:
    """Echo timestamp_check when the record carries one. A never-updated
    deal has update_date null — then the backend's hasBeenUpdated()
    reports no conflict and the token can be safely omitted."""
    token = current.get("timestamp_check")
    if token:
        body["timestamp_check"] = token


# --- tool descriptions --------------------------------------------------

_LIST_DEALS_DESC = inspect.cleandoc(
    """
    List CDESK deals ("zákazky"). Returns a paginated `items` array whose
    keys are the RAW DB columns: the deal id arrives as `id_rc` and the
    title as `rc_title` — there are NO `id`/`name` keys on a deal row
    (unlike most other modules). Other keys are likewise raw (rc_num, rc_code,
    rc_status, rc_status_id, due_date, ...). In list rows `rc_title` is
    prefixed with the deal number ("CDD-1: Alpha"); get_deal returns
    it unprefixed.

    NOTE: unless the account has the 'deal/from-others' right, the server
    silently narrows the list to deals where the caller is responsible,
    the author, or in a visible group — a short list may mean limited rights,
    not few deals.

    Common filters:
      - status: a tenant status NAME (see get_deal_enums), a status id,
        or the literal keywords 'open' (not yet refused/completed) / 'closed'.
      - customer_id / customer_name: owning customer (id or name search).
      - title / code / text_search: text matching (text_search also spans
        rc_num and the customer name).
      - phase_name / cat_type_name: resolved against the tenant enums.
      - responsible_person_id, for_invoicing (has amounts ready to invoice),
      - due_after / due_before, created_*/updated_* windows (ISO-8601).

    For advanced filters pass sb_raw as an sb filter object for the CDESK API v3
    deal list (JSON string or object, structured tree form; see
    docs/cdesk-api-v3.json). Allowed columns
    (source-verified 2026-07-20; NOTE the FILTER keys are named differently
    from the response keys above — to filter, the id key is `id` (not
    `id_rc`), title is `title` (not `rc_title`), customer is `id_customer`
    (not `id_company`)): id, title, code, rc_num,
    id_customer, company.name, status, rc_status (strictly-less-than action
    code), type, phase, responsible_person_id, tag, subtree, for_invoicing,
    dateFrom/dateTo (due-date bounds), due_date, created_at, updated_at,
    last_invoice_issue_date, updated_by, plus the no-col text leaf. Clauses on
    other columns are stripped and reported in `unsupported_filters`.
    """
)

_GET_DEAL_DESC = inspect.cleandoc(
    """
    Fetch a single deal by id. Response keys are the RAW DB columns — the
    id arrives as `id_rc` and the title as `rc_title`, NOT `id`/`name`.
    fieldset selects the field group: "base",
    "extended" (default), "all" (incl. custom fields), or "custom"; fields is
    an exact whitelist (returnFields; union with fieldset) and must name raw
    columns too (e.g. fields=["rc_title"], not ["name"]). Empty dates are
    returned as null (stored zero-dates are normalized out).
    """
)

_GET_DEAL_ENUMS_DESC = inspect.cleandoc(
    """
    Return the Deal module enums and tenant settings: `enums.status`
    (tenant statuses with action_code — 10 received, 20 in progress,
    25 delivered, 30 refused, 80 completed; 'open' = action_code < 30),
    `enums.phase`, `enums.catType` (category types), plus `settings` (which
    fields this tenant REQUIRES: types.status == 2 → category type required,
    phases.enabled → phase required) and `tags`. Use these for
    create/update/list name resolution.
    """
)

_CREATE_DEAL_DESC = inspect.cleandoc(
    """
    Create a deal. Required: title, customer_id, and a status (status_name
    resolved against the tenant enums, or status_id). Depending on tenant
    settings (see get_deal_enums `settings`), cat_type_name and phase_name
    may also be required — the server rejects the create with a clear message
    if so.

    The deal number (CDD-<num>) is auto-assigned; `code` (rc_code) is
    auto-generated when omitted and the tenant has automatic codes enabled.
    responsible_person_ids is strongly recommended (the standard UI form
    requires it; the API stores it as a relation). Dates accept ISO-8601 — a
    naive datetime is read as tenant-local time; due dates must not precede
    their opening dates. tag_ids attaches existing tags (see
    get_deal_enums `tags`).
    """
)

_UPDATE_DEAL_DESC = inspect.cleandoc(
    """
    Update a deal (partial — send only the fields to change). Optimistic
    locking is handled internally (GET → mutate → PUT echoing the version
    token); on a 409 conflict simply retry — the fresh token is fetched
    automatically on each call.

    Status/phase lifecycle transitions happen here: pass status_name/status_id
    (and phase_name) to move the deal (e.g. to 'completed'/'refused' —
    the closed states). NOTE: closing a deal with uninvoiced fills is
    rejected by the server unless the account may ignore them; reopening a
    completed deal of an inactive customer is rejected too.
    responsible_person_ids / coordinator_contact_ids / tag_ids REPLACE the
    stored relation lists.
    """
)

_DELETE_DEAL_DESC = inspect.cleandoc(
    """
    Delete (soft-delete) a deal by id. Rejected while the deal has
    dependent records — fills, requests (incl. periodical), tasks, used
    materials, or billing items — with a validation error naming the blocker;
    those must be removed/relinked first. Returns {"deleted": <id>}.
    """
)
