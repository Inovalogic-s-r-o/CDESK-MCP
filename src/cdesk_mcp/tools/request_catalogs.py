"""Request Catalog module MCP tools.

Six tools that expose CDESK's Request Catalog v3 CRUD + the catalog
enum lookup. A request catalog is a *template* used by create_request
via the `catalog_id` field — it pre-populates form fields, controls
visibility, and ties the resulting request to category / approval
rules.

Mirrors the Customer-module pattern: small surface, optimistic-lock
fetch is recommended-but-not-enforced server-side (per the OpenAPI
spec), but we still grab timestamp_check on update for safety.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp import base_config
from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.enums import EnumCache
from cdesk_mcp.field_policy import (
    CATALOG_FIELDS,
    CHANGE_MANAGER_CUSTOM,
    base_params_given,
    catalog_settings,
    describe,
    refuse_if_violated,
    violations,
)
from cdesk_mcp.filters import build_request_catalog_filter, encode_sb
from cdesk_mcp.sort_columns import resolve_sort, sort_note
from cdesk_mcp.tools._helpers import (
    list_meta,
    TEXT_MATCH_NOTE,
    EnumLabel,
    add_enum_names,
    annotate_write_warnings,
    apply_clear_fields,
    resolve_enum_or_raise,
    apply_field_scope,
    to_llm_error,
    unsupported_filter_directive,
    unwrap_list,
    unwrap_record,
)

# LLM-friendly name → CDESK field name. Catalog's wire format is mostly
# straight-through — only `description` is a Python-friendly alias for
# CDESK's `desc` (which would otherwise shadow the builtin name).
_CATALOG_CREATE_FIELDS = {
    "description": "desc",
    "job_title_ids": "job_titles",
    "visible_group_ids": "visible_groups",
    "company_ids": "assigned_company_id",
    "contract_ids": "assigned_contract_id",
    "category_ids": "categories",
    "tile_ids": "request_catalog_category_tag_id",
}

# The parameters create and update share, in body order; both tools build their
# body from these through _catalog_body.
_CATALOG_PARAMS = (
    "name", "code", "description", "use_catalog_item_name", "hide_name",
    "hide_description", "change_management", "change_manager_id",
    "change_manager_type", "internal", "auto_close_request", "approval_rule_id",
    "simple_approval_rules", "approval_rules", "note", "search_keywords",
    "show_contact_info", "planned_work_enabled", "parallel_approvals_enabled",
    "include_fill_hours", "visible_group_ids", "company_ids", "contract_ids",
    "category_ids", "tile_ids", "custom_fields",
)

# update_request_catalog parameter → wire field for the fields its PUT clears
# with an explicit null (both nullable in the API spec).
_CATALOG_CLEARABLE = {
    "change_manager_id": "change_manager_id",
    "approval_rule_id": "approval_rule_id",
}

_MAX_PER_PAGE = 100

# Labels next to a catalog's raw enum values, from the live catalog enums:
# `type` is the KEY of the key/value `catalog_types` bucket (e.g. "H" =
# "Helpdesk"), `cat_type_id` the enum id in `cat_types`.
_CATALOG_ENUM_LABELS = (
    EnumLabel("type", "catalog_types", "type_name", by="key"),
    EnumLabel("cat_type_id", "cat_types", "cat_type_name"),
)


def register_request_catalog_tools(
    mcp: FastMCP,
    client: CdeskClient,
    cache: EnumCache,
) -> None:
    """Register all Request-Catalog tools on the given FastMCP instance.
    Called by build_server when client + the catalog enum cache are both
    available."""

    @mcp.tool(
        description=_LIST_CATALOGS_DESC + "\n\n" + sort_note("request_catalogs") + "\n\n" + TEXT_MATCH_NOTE,
        annotations=ToolAnnotations(title="List request catalogs", readOnlyHint=True),
    )
    async def list_request_catalogs(
        text_search: str | None = None,
        type_code: str | None = None,
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
            # Resolve type_code the same way create/update do — accept the code
            # ("H") OR the display name ("Helpdesk"), and error on an unknown
            # value — so a name the sibling tools accept doesn't silently
            # zero-match here (the sb compares against the stored code).
            resolved_type = await _resolve_catalog_type_code(cache, type_code)
            # Unsupported sb_raw columns are stripped and reported via
            # `unsupported_filters` — the agent must filter the items itself.
            dropped_clauses: list[dict[str, Any]] = []
            sb = build_request_catalog_filter(
                text_search=text_search,
                type_code=resolved_type,
                sb_raw=sb_raw,
                dropped_cols_out=dropped_clauses,
            )
            params: dict[str, Any] = {"pg": page, "pp": per_page}
            apply_field_scope(params, fieldset, fields)
            if sort:
                params["sort"] = resolve_sort("request_catalogs", sort)
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response, paging = await client.get_page("v3/request-catalog", params=params)
        except RuntimeError:
            raise  # _resolve_catalog_type_code's unknown-type message is friendly
        except ValueError as e:
            raise RuntimeError(f"list_request_catalogs input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_request_catalogs") from e

        records, meta = unwrap_list(response)
        await add_enum_names(cache, records, _CATALOG_ENUM_LABELS)
        result = {"items": records, "meta": list_meta(meta, paging, dropped=dropped_clauses), "page": page, "per_page": per_page}
        if dropped_clauses:
            result["unsupported_filters"] = unsupported_filter_directive(dropped_clauses)
        return result

    @mcp.tool(
        description=_GET_CATALOG_DESC,
        annotations=ToolAnnotations(title="Get request catalog", readOnlyHint=True),
    )
    async def get_request_catalog(
        id: int,
        fieldset: str | None = None,
        fields: list[str] | None = None,
    ) -> Any:
        try:
            params: dict[str, Any] = {}
            apply_field_scope(params, fieldset, fields)
            response = await client.get(
                f"v3/request-catalog/{id}", params=params or None
            )
        except ValueError as e:
            raise RuntimeError(f"get_request_catalog input error: {e}") from e
        except Exception as e:
            raise to_llm_error(
                e, operation="get_request_catalog", record_id=id,
            ) from e
        record = unwrap_record(response)
        await add_enum_names(cache, [record], _CATALOG_ENUM_LABELS)
        return record

    @mcp.tool(
        description=_GET_CATALOG_ENUMS_DESC,
        annotations=ToolAnnotations(title="Get request catalog enums", readOnlyHint=True),
    )
    async def get_request_catalog_enums(
        refresh: bool = False,
    ) -> dict[str, Any]:
        try:
            if refresh:
                await cache.refresh()
            else:
                await cache.load()  # populate a cold cache (snapshot is empty otherwise)
        except Exception as e:
            raise to_llm_error(e, operation="get_request_catalog_enums") from e
        result: dict[str, Any] = dict(cache.snapshot())
        options = cache.extra("approval_rule_options")
        if isinstance(options, dict):
            result["approval_rule_options"] = {
                field: [
                    {k: row.get(k) for k in ("id", "name", "category_id")}
                    for row in rows if isinstance(row, dict)
                ]
                for field, rows in options.items() if isinstance(rows, list)
            }
        result["fields"] = {
            "by_tenant_setting": describe(CATALOG_FIELDS, catalog_settings(
                cache.extra("config"), await base_config.load(client), {},
            )),
            "note": (
                "State of each configurable create_request_catalog / "
                "update_request_catalog parameter on this tenant, from its live "
                "request settings: off (refused when given), optional, or "
                "required (must be given on create, cannot be cleared on "
                "update). `base_params.<key>` is that base_params entry's value. "
                "Fields that depend on the catalog's own values (hyperlink type, "
                "change management, the request type's priority matrix) are "
                "decided per write."
            ),
        }
        return result

    @mcp.tool(
        description=_CREATE_CATALOG_DESC,
        annotations=ToolAnnotations(title="Create request catalog", destructiveHint=True),
    )
    async def create_request_catalog(
        name: str,
        code: str | None = None,
        description: str | None = None,
        type_code: str | None = None,
        use_catalog_item_name: bool | None = None,
        hide_name: bool | None = None,
        hide_description: bool | None = None,
        change_management: bool | None = None,
        change_manager_id: int | None = None,
        change_manager_type: Literal[1, 2] | None = None,
        internal: bool | None = None,
        auto_close_request: bool | None = None,
        cat_type: int | str | None = None,
        cat_area: int | str | None = None,
        cat_area_2nd: int | str | None = None,
        approval_rule_id: int | None = None,
        simple_approval_rules: list[int] | None = None,
        approval_rules: list[int] | None = None,
        job_title_ids: list[int] | None = None,
        note: str | None = None,
        search_keywords: list[str] | None = None,
        show_contact_info: bool | None = None,
        planned_work_enabled: bool | None = None,
        parallel_approvals_enabled: bool | None = None,
        include_fill_hours: bool | None = None,
        visible_group_ids: list[int] | None = None,
        company_ids: list[int] | None = None,
        contract_ids: list[int] | None = None,
        category_ids: list[int] | None = None,
        tile_ids: list[int] | None = None,
        custom_fields: list[dict[str, Any]] | None = None,
        base_params: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        scope = locals()
        values = {k: scope[k] for k in _CATALOG_PARAMS}
        try:
            _validate_name(name)
            _validate_code(code)
            resolved_type = await _resolve_catalog_type_code(cache, type_code)
            await _require_user(client, change_manager_id)
            template = await _resolve_template_fields(
                cache, cat_type, cat_area, cat_area_2nd, job_title_ids,
            )
            body = _catalog_body(values, resolved_type=resolved_type, template=template)
            # The tenant's catalog FORM rules, read live (field_policy). A new
            # catalog starts as the form starts it: change management off, and
            # the change-manager type preselected to 1, a named manager
            # (CDESK stores 1 too).
            await cache.load()
            form_manager_type = (
                change_manager_type if change_manager_type is not None else CHANGE_MANAGER_CUSTOM
            )
            refuse_if_violated(violations(
                CATALOG_FIELDS,
                catalog_settings(cache.extra("config"), await base_config.load(client), {
                    "type": resolved_type,
                    "change_management": bool(change_management),
                    "change_manager_type": form_manager_type,
                    "use_catalog_item_name": bool(use_catalog_item_name),
                    "cat_type_id": _filing_value(template, base_params, None),
                }),
                {**_form_given(values, cat_type, cat_area, cat_area_2nd, base_params),
                 # the preselected type counts only where the form shows it
                 **({"change_manager_type": form_manager_type} if change_management else {})},
            ))
            # The POST merges a partial base_params_decoded into the defaults it
            # fills in, per key and per setting, so the filing values go on in
            # the same write.
            patch = _base_params_patch(template, base_params)
            if patch:
                body[_BASE_PARAMS] = patch
            response = await client.post("v3/request-catalog", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"create_request_catalog input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="create_request_catalog") from e
        result = annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=_CHECK_FIELDS,
        )
        await add_enum_names(cache, [result.get("data")], _CATALOG_ENUM_LABELS)
        return result

    @mcp.tool(
        description=_UPDATE_CATALOG_DESC,
        annotations=ToolAnnotations(title="Update request catalog", destructiveHint=True),
    )
    async def update_request_catalog(
        id: int,
        name: str | None = None,
        code: str | None = None,
        description: str | None = None,
        type_code: str | None = None,
        use_catalog_item_name: bool | None = None,
        hide_name: bool | None = None,
        hide_description: bool | None = None,
        change_management: bool | None = None,
        change_manager_id: int | None = None,
        change_manager_type: Literal[1, 2] | None = None,
        internal: bool | None = None,
        auto_close_request: bool | None = None,
        cat_type: int | str | None = None,
        cat_area: int | str | None = None,
        cat_area_2nd: int | str | None = None,
        approval_rule_id: int | None = None,
        simple_approval_rules: list[int] | None = None,
        approval_rules: list[int] | None = None,
        job_title_ids: list[int] | None = None,
        note: str | None = None,
        search_keywords: list[str] | None = None,
        show_contact_info: bool | None = None,
        planned_work_enabled: bool | None = None,
        parallel_approvals_enabled: bool | None = None,
        include_fill_hours: bool | None = None,
        visible_group_ids: list[int] | None = None,
        company_ids: list[int] | None = None,
        contract_ids: list[int] | None = None,
        category_ids: list[int] | None = None,
        tile_ids: list[int] | None = None,
        custom_fields: list[dict[str, Any]] | None = None,
        base_params: dict[str, dict[str, Any]] | None = None,
        clear_fields: list[Literal["change_manager_id", "approval_rule_id"]] | None = None,
    ) -> dict[str, Any]:
        scope = locals()
        values = {k: scope[k] for k in _CATALOG_PARAMS}
        try:
            if name is not None:
                _validate_name(name)
            _validate_code(code)
            resolved_type = await _resolve_catalog_type_code(cache, type_code)
            await _require_user(client, change_manager_id)
            template = await _resolve_template_fields(
                cache, cat_type, cat_area, cat_area_2nd, job_title_ids,
            )
            body = _catalog_body(values, resolved_type=resolved_type, template=template)
            patch = _base_params_patch(template, base_params)
            apply_clear_fields(body, clear_fields, _CATALOG_CLEARABLE)
            if not body and not patch:
                raise ValueError("nothing to update — pass at least one field")
            current = await _fetch_for_update_catalog(client, id)
            await cache.load()
            refuse_if_violated(violations(
                CATALOG_FIELDS,
                catalog_settings(cache.extra("config"), await base_config.load(client), {
                    "type": resolved_type or current.get("type"),
                    "change_management": _given_or(change_management, current.get("change_management")),
                    "change_manager_type": _given_or(change_manager_type, current.get("change_manager_type")),
                    "use_catalog_item_name": _given_or(
                        use_catalog_item_name, current.get("use_catalog_item_name"),
                    ),
                    "cat_type_id": _filing_value(template, base_params, current.get(_BASE_PARAMS)),
                }),
                _form_given(values, cat_type, cat_area, cat_area_2nd, base_params),
                update=True, cleared=clear_fields or (),
            ))
            if cache.extra("config"):  # like every form rule: only with the settings read
                _require_change_manager(
                    change_management, change_manager_type, change_manager_id, current,
                )
            # The GET carries timestamp_check; when it is null the token is
            # rebuilt from updated_at (the PUT refuses a missing one with a
            # misleading conflict).
            token = current.get("timestamp_check") or _lock_token_from_updated_at(
                current.get("updated_at")
            )
            if not token:
                raise RuntimeError(
                    f"request-catalog {id} returned no timestamp_check or "
                    f"parseable updated_at, so the required optimistic-lock "
                    f"token can't be built — cannot safely update it. This is "
                    f"a backend data issue with the catalog record."
                )
            body["timestamp_check"] = token
            # A PUT without base_params_decoded resets every prefill value and
            # rule (see _BASE_PARAMS), so the stored object always goes along,
            # with this call's changes merged in.
            stored = current.get(_BASE_PARAMS)
            unpreservable = not (isinstance(stored, dict) and stored)
            if isinstance(stored, dict) and stored:
                body[_BASE_PARAMS] = _merge_base_params(stored, patch)
            elif patch:
                body[_BASE_PARAMS] = patch
            response = await client.put(f"v3/request-catalog/{id}", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"update_request_catalog input error: {e}") from e
        except Exception as e:
            raise to_llm_error(
                e, operation="update_request_catalog", record_id=id,
            ) from e
        result = annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=_CHECK_FIELDS,
        )
        if unpreservable:
            result = _with_warning(result, (
                "CDESK resets a catalog's prefill rules (priority, urgency, "
                "solver, ... and who may edit them) on every update that does not "
                "send them back, and this catalog's stored rules could not be "
                "read to send back — they may now be empty; check them with "
                "get_request_catalog(fieldset='all')."
            ))
        await add_enum_names(cache, [result.get("data")], _CATALOG_ENUM_LABELS)
        return result

    @mcp.tool(
        description=_DELETE_CATALOG_DESC,
        annotations=ToolAnnotations(title="Delete request catalog", destructiveHint=True),
    )
    async def delete_request_catalog(id: int) -> dict[str, Any]:
        try:
            await client.delete(f"v3/request-catalog/{id}")
        except Exception as e:
            raise to_llm_error(
                e, operation="delete_request_catalog", record_id=id,
            ) from e
        return {"deleted": id}


# --- internals ----------------------------------------------------------

async def _resolve_catalog_type_code(cache: EnumCache, type_code: str | None) -> str | None:
    """Resolve a base-type input to its string code for the catalog `type`
    body field. The catalog base types live in the key/value `catalog_types`
    bucket (e.g. {"key": "H", "value": "Helpdesk"}) — NOT an int-id bucket —
    so this matches the caller's value against either the code ("H") or the
    display name ("Helpdesk"), case-insensitively, and returns the code.

    (The OpenAPI doc names a `createOnlyBaseTypes` bucket that the live API
    does not ship; `catalog_types` is the real source.)"""
    if not type_code or not type_code.strip():
        return None
    await cache.load()
    types = cache.snapshot().get("catalog_types", [])
    valid: dict[str, Any] = {
        str(t["key"]): t.get("value")
        for t in types
        if isinstance(t, dict) and t.get("key") is not None
    }
    needle = type_code.strip().lower()
    for key, value in valid.items():
        if key.lower() == needle or (isinstance(value, str) and value.lower() == needle):
            return key
    options = ", ".join(f"{k} ({v})" for k, v in valid.items()) or "(none configured)"
    raise RuntimeError(
        f"Unknown catalog base type {type_code!r}. Valid types (code → name): "
        f"{options}. See get_request_catalog_enums → catalog_types."
    )


def _form_given(
    values: dict[str, Any],
    cat_type: Any, cat_area: Any, cat_area_2nd: Any,
    base_params: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """This write's values as CATALOG_FIELDS parameters."""
    return {
        **values,
        "cat_type": cat_type, "cat_area": cat_area, "cat_area_2nd": cat_area_2nd,
        **base_params_given(base_params),
    }


def _require_change_manager(
    change_management: bool | None, manager_type: int | None, manager_id: int | None,
    current: dict[str, Any],
) -> None:
    """An update that turns change management on (or switches to a named
    manager) must complete what the form then requires, when the catalog has
    no stored value for it — the update rule "left out = unchanged" has
    nothing to keep there."""
    on = _given_or(change_management, current.get("change_management"))
    if not on:
        return
    kind = _given_or(manager_type, current.get("change_manager_type"))
    if kind is None:
        raise ValueError(
            "change_manager_type is required with change management (1 = the "
            "user in change_manager_id, 2 = the request type's manager) and "
            "this catalog has none"
        )
    if kind == CHANGE_MANAGER_CUSTOM and _given_or(manager_id, current.get("change_manager_id")) is None:
        raise ValueError(
            "change_manager_id is required with change_manager_type 1 and this "
            "catalog has none"
        )


def _given_or(given: Any, stored: Any) -> Any:
    return given if given is not None else stored


def _filing_value(
    template: dict[str, Any], base_params: dict[str, dict[str, Any]] | None, stored: Any,
) -> Any:
    """The catalog's request type after this write — the form keys the
    priority matrix on it."""
    if template.get("cat_type_id") is not None:
        return template["cat_type_id"]
    entry = (base_params or {}).get("cat_type_id")
    if isinstance(entry, dict) and entry.get("value") is not None:
        return entry["value"]
    stored_entry = stored.get("cat_type_id") if isinstance(stored, dict) else None
    return stored_entry.get("value") if isinstance(stored_entry, dict) else None


def _lock_token_from_updated_at(updated_at: Any) -> str | None:
    """Convert the GET response's ISO-8601 `updated_at`
    (e.g. '2026-06-04T08:23:00+00:00') into the 'YYYY-MM-DD HH:MM:SS'
    form the catalog PUT accepts as `timestamp_check` (the ISO form and
    null are both rejected with a conflict error)."""
    if not isinstance(updated_at, str) or not updated_at:
        return None
    try:
        return datetime.fromisoformat(updated_at).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


async def _fetch_for_update_catalog(client: CdeskClient, id: int) -> dict[str, Any]:
    """Fetch a catalog and unwrap to the record dict, returning the
    timestamp_check (if present) for the follow-up PUT.

    Uses fieldset="all" because the DEFAULT catalog GET omits
    `base_params_decoded`, which every update has to send back. See
    `_BASE_PARAMS`."""
    envelope = await client.get(
        f"v3/request-catalog/{id}", params={"fieldset": "all"}
    )
    record = unwrap_record(envelope)
    if not isinstance(record, dict):
        raise RuntimeError(
            f"unexpected response shape fetching request-catalog {id} for update: "
            f"{type(record).__name__}"
        )
    return record


# `base_params_decoded` is the parsed form of a catalog's `base_params` — the
# per-field {value, insert, update, restrict, ...} rules that make a catalog a
# template (prefilled values and value restrictions). The write reads THIS key,
# not the `base_params` string it returns.
#
# Backend behaviour:
#   - a body that carries it, even partially, is MERGED per key and per setting
#     — omitted keys keep their rules, an entry with only `value` keeps its
#     insert/update/restrict. This holds on the POST too, so a create can file
#     the catalog in its single write.
#   - a PUT that does NOT carry it resets every value and setting to the
#     defaults — a name-only, note-only or job-titles-only update clears a
#     prefilled priority/urgency and an `update: 3` rule.
# So every update sends the stored object back with its changes merged in.
# (The other fields — visibility, planned work, use_catalog_item_name, the
# approval rules, job titles — are kept by the PUT when omitted, so they are
# not echoed.)
_BASE_PARAMS = "base_params_decoded"


# `for_customer` is left out: the write response echoes it as 1/0 while the
# record stores "hidden"/"", so a diff would always fire.
_CHECK_FIELDS = (
    "code", "change_management", "change_manager_id", "auto_close_request", "note",
)


def _for_customer_value(internal: bool | None) -> str | None:
    """The two `for_customers` enum keys: "hidden" (Interní) and "" (Obyčejná).
    Any other string is stored as "" without a message, so the tool takes a
    bool instead of the raw key."""
    if internal is None:
        return None
    return "hidden" if internal else ""


# Filing fields: the catalog stores them only inside `base_params` as
# {value, insert, update, restrict}; a top-level `cat_type_id` in the PUT body is
# ignored.
_BASE_PARAM_BUCKETS = {
    "cat_type_id": ("cat_types", "request type"),
    "cat_area_id": ("cat_areas", "catalog area"),
    "cat_area_2nd_id": ("cat_areas_2nd", "2nd-level catalog area"),
}


async def _resolve_catalog_enum(
    cache: EnumCache, bucket: str, value: int | str, kind: str,
) -> int:
    """An id is checked against the bucket, a name resolved through it. The
    server takes ids unchecked, so a wrong one would file the catalog nowhere."""
    await cache.load()
    known = cache.id_name_map(bucket)
    if not known:
        raise ValueError(
            f"this tenant has no {kind} defined (the {bucket!r} bucket of "
            f"get_request_catalog_enums is empty), so none can be set"
        )
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, int) and not isinstance(value, bool):
        if value not in known:
            options = ", ".join(f"{i} ({n})" for i, n in list(known.items())[:15])
            raise ValueError(f"{kind} id {value} does not exist; valid: {options}")
        return value
    resolved = await resolve_enum_or_raise(cache, bucket, str(value), kind=kind)
    assert resolved is not None  # a non-empty name either resolves or raises
    return resolved


async def _resolve_template_fields(
    cache: EnumCache,
    cat_type: int | str | None,
    cat_area: int | str | None,
    cat_area_2nd: int | str | None,
    job_title_ids: list[int] | None,
) -> dict[str, Any]:
    """Validate the filing fields and job titles before anything is written.
    Returns the base_params values keyed by field, plus `job_titles`."""
    out: dict[str, Any] = {}
    for key, value in (
        ("cat_type_id", cat_type), ("cat_area_id", cat_area),
        ("cat_area_2nd_id", cat_area_2nd),
    ):
        if value is not None:
            bucket, kind = _BASE_PARAM_BUCKETS[key]
            out[key] = await _resolve_catalog_enum(cache, bucket, value, kind)
    if job_title_ids is not None:
        await cache.load()
        known = cache.id_name_map("job_titles")
        unknown = [i for i in job_title_ids if i not in known]
        if unknown:
            options = ", ".join(f"{i} ({n})" for i, n in known.items())
            raise ValueError(
                f"job_title_ids {unknown} are not job titles; valid: {options}"
            )
        out["job_titles"] = list(job_title_ids)
    return out


# The base_params keys the backend keeps (five more than the API spec documents;
# the tenant-specific ones are left out), and the settings an entry may carry.
# The server ignores anything else without a word, so a typo would look stored.
_BASE_PARAM_KEYS = frozenset({
    "cat_type_id", "cat_area_id", "cat_area_2nd_id", "priority",
    "solver_group_id", "id_solver", "reaction_missed_id_solver", "other_solvers",
    "request_watchers", "responsible_person", "urgency_id", "impact_id",
    "cmdb_objects_select", "cmdb_filter_mode", "cmdb_filter",
    "cat_status_id", "cat_close_reason_id", "cat_solver_method_id",
    "cat_root_cause_id", "cat_branch_id", "cat_place_id", "cmdb_scanner_mobile",
    "computer_remote_access_by_email", "id_company",
    "project_contract_id", "user_entered_by", "applicant", "hyperlink",
    "hyperlink_target", "tags", "status_flow_transitions",
})
_BASE_PARAM_SETTINGS = frozenset({
    "value", "insert", "update", "show_when_creating_request", "restrict",
    "allowed", "prioritize",
})


def _base_params_patch(
    template: dict[str, Any], base_params: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """The partial base_params_decoded for this write: the resolved filing
    values (cat_type / cat_area / cat_area_2nd) plus the caller's own entries.
    Raises ValueError for an unknown key or setting, or a filing field given
    both ways."""
    patch: dict[str, Any] = {
        key: {"value": value} for key, value in template.items()
        if key in _BASE_PARAM_BUCKETS
    }
    for key, entry in (base_params or {}).items():
        if key not in _BASE_PARAM_KEYS:
            raise ValueError(
                f"base_params key {key!r} is not a catalog field; valid: "
                f"{', '.join(sorted(_BASE_PARAM_KEYS))}"
            )
        if not isinstance(entry, dict) or not entry:
            raise ValueError(f"base_params[{key!r}] must be a non-empty object")
        unknown = sorted(set(entry) - _BASE_PARAM_SETTINGS)
        if unknown:
            raise ValueError(
                f"base_params[{key!r}] has unknown settings {unknown}; valid: "
                f"{', '.join(sorted(_BASE_PARAM_SETTINGS))}"
            )
        if key in patch:
            raise ValueError(
                f"{key} is set both through its own parameter and base_params; "
                f"pass it once"
            )
        entry = dict(entry)
        # "No restriction" is the empty string; only "limited" restricts.
        # Accept "none" as the same thing.
        if entry.get("restrict") == "none":
            entry["restrict"] = ""
        patch[key] = entry
    return patch


def _merge_base_params(stored: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """The stored object with the patch merged in per key and per setting."""
    merged = dict(stored)
    for key, entry in patch.items():
        current = merged.get(key)
        merged[key] = {**current, **entry} if isinstance(current, dict) else dict(entry)
    return merged


# The binding options the spec documents for a catalog custom field.
_CUSTOM_FIELD_KEYS = frozenset({
    "id", "required", "required_status_config", "param_request",
    "param_request_detail", "tab_id",
})


def _validate_custom_field_bindings(custom_fields: Any) -> None:
    """Shape only: the server ignores an unknown property id with a 200,
    so an id cannot be confirmed here."""
    if custom_fields is None:
        return
    if not isinstance(custom_fields, list):
        raise ValueError("custom_fields must be a list of {id, ...} objects")
    for item in custom_fields:
        if not isinstance(item, dict):
            raise ValueError(f"custom_fields entries must be objects (got {item!r})")
        cf_id = item.get("id")
        if isinstance(cf_id, bool) or not isinstance(cf_id, int):
            raise ValueError(f"custom_fields entry needs an integer `id` (got {item!r})")
        unknown = sorted(set(item) - _CUSTOM_FIELD_KEYS)
        if unknown:
            raise ValueError(
                f"custom_fields entry {cf_id} has unknown keys {unknown}; valid: "
                f"{', '.join(sorted(_CUSTOM_FIELD_KEYS))}"
            )


def _validate_name(name: str) -> None:
    # No length limit: neither the CDESK form nor the API has one. The 30-char
    # limit is on `code`.
    if not name or not name.strip():
        raise ValueError("name must be a non-empty string")


def _validate_code(code: str | None) -> None:
    # The CDESK form caps the catalog code at 30 characters. The API stores
    # longer codes; the tool follows the form.
    if code is not None and len(code) > 30:
        raise ValueError(f"code must be ≤30 characters (the CDESK form's limit); got {len(code)}")


def _catalog_body(
    values: dict[str, Any], *, resolved_type: str | None, template: dict[str, Any],
) -> dict[str, Any]:
    """The write body create and update share. `values` holds the tool
    parameters by name (see _CATALOG_PARAMS); None means "not sent"."""
    keywords = values.get("search_keywords")
    if keywords is not None and (
        not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords)
    ):
        # A bare string is answered with an empty-bodied 500.
        raise ValueError("search_keywords must be a list of strings")
    _validate_custom_field_bindings(values.get("custom_fields"))
    public = dict(values)
    public["type"] = resolved_type
    public["for_customer"] = _for_customer_value(public.pop("internal"))
    public["job_title_ids"] = template.get("job_titles")
    groups = public.get("visible_group_ids")
    if groups is not None:
        public["visible_group_ids"] = [{"id": g} for g in groups]
    return _build_catalog_body(public)


def _with_warning(response: dict[str, Any], warning: str) -> dict[str, Any]:
    return {**response, "warnings": [*(response.get("warnings") or []), warning]}


async def _require_user(client: CdeskClient, user_id: int | None) -> None:
    """The catalog write stores any integer as `change_manager_id`, a
    nonexistent user included, so check it first."""
    if user_id is None:
        return
    try:
        await client.get(f"v3/user/{user_id}", params={"returnFields[]": "id"})
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise ValueError(
                f"change_manager_id {user_id} is not an existing user; look it "
                f"up with find_user or list_users"
            ) from e
        raise


def _build_catalog_body(public_fields: dict[str, Any]) -> dict[str, Any]:
    """Drop Nones, rename LLM-friendly names → CDESK field names."""
    body: dict[str, Any] = {}
    for public_name, value in public_fields.items():
        if value is None:
            continue
        cdesk_name = _CATALOG_CREATE_FIELDS.get(public_name, public_name)
        body[cdesk_name] = value
    return body


# --- Tool descriptions --------------------------------------------------

_LIST_CATALOGS_DESC = inspect.cleandoc(
    """
    List request-catalog templates. Each catalog is a reusable form
    template — `create_request(catalog_id=...)` instantiates a request
    from it, pre-populating type, category, and other defaults.

    Filters:
      - text_search: matches name, code, and description.
      - type_code: base request type — the string code (e.g. "H") OR the
        display name (e.g. "Helpdesk") from get_request_catalog_enums →
        catalog_types (resolved the same way create/update accept it; an
        unknown value errors with the valid options rather than matching
        nothing).

    NOT offered as typed params (the live CDESK catalog list silently ignores
    the plain column forms): type_id, auto_close_request,
    category/change-manager/approval columns, date ranges, and soft-deleted.
    Some of these ARE reachable through sb_raw's DOTTED forms below (type_id
    and the category type/area in particular) — filter client-side only for
    the rest.

    sb_raw — a CDESK sb filter object for the CDESK API v3 request-catalog list
    (JSON string or object, structured tree form; see docs/cdesk-api-v3.json).
    Advanced; mutually exclusive with the typed filters above. Only the
    working columns are applied server-side — id, type,
    companyId (a real company id returns the catalogs available to it; a bogus
    one returns none), job_title_id, and the DOTTED forms
    request_catalog.name / .type_id / .cat_type_id / .cat_area_id, plus the
    no-col text leaf. The dotted prefix is REQUIRED: the plain `name`, `code`,
    `desc`, `type_id` and `cat_type_id` are silently ignored by CDESK and are
    STRIPPED here, as is any other column — each one is named in the response's
    `unsupported_filters` block.

    Response shaping: fieldset selects the field group per record
    ("base"/"extended"/"all"/"custom"); fields is an exact whitelist of
    field names (returnFields; union with fieldset).

    Returns paginated items + meta.
    """
)

_GET_CATALOG_DESC = inspect.cleandoc(
    """
    Fetch a single request-catalog by id. Returns the catalog record
    including change-manager rules, status flows, and visibility
    configuration.

    fieldset selects the field group returned ("base"/"extended"/
    "all"/"custom"; CDESK default extended); fields is an exact
    whitelist of field names (returnFields; union with fieldset).
    """
)

_GET_CATALOG_ENUMS_DESC = inspect.cleandoc(
    """
    Return the cached lookup tables for the Request Catalog module.
    Most useful bucket: `catalog_types` — the base request types a new
    catalog can be tied to (key/value, e.g. {"key": "H", "value":
    "Helpdesk"}); pass the key OR the name as `type_code` in
    create_request_catalog. The others, as the tenant serves them (a bucket
    for a switched-off feature comes back empty):
      - cat_types / cat_areas / cat_areas_2nd: the request types and the
        1st- / 2nd-level catalog areas a catalog is filed under;
      - cat_status, flow_status_initial / flow_status_targets: request
        statuses, and the start and target statuses of a status flow;
      - cat_types_restricts / cat_status_restricts: the restriction modes
        ("" = unrestricted, "limited" = only the listed values);
      - job_titles, priorities, for_customers, updates (who may edit a
        request), prioritize, description_edit, cmdbSelect, dateSelect,
        settings_enabled, slaUnitTypes, cmdb_filter_modes, tags.
      - approval_rule_options: the approval rules each catalog approval field
        takes — `approval_rule_id`, `approval_rules` and
        `simple_approval_rules`, each a list of {id, name, category_id}.
        Present only while catalog approvals are on for the tenant (and the
        account may read approval rules).
      - categories: the request categories, present while they are enabled.
      - fields: the state of each configurable create/update parameter on this
        tenant — off, optional or required — with the setting behind it.
    There is no change-manager bucket; a change manager is a user id
    (find_user).

    Pass refresh=True to force a fresh fetch.
    """
)

_CREATE_CATALOG_DESC = inspect.cleandoc(
    """
    Create a new request-catalog template.

    Required:
      - name.

    Optional:
      - code: short identifier (max 30 characters, the CDESK form's
        limit). CDESK stores it only while request codes are on for the
        tenant (`code.status` > 0) — off, it is refused here; if CDESK
        still drops it, the result carries a `warnings` entry.
      - description: HTML body shown as the default request description
        when instantiated. note: an internal note shown in the catalog list.
      - type_code: base request type the catalog produces — the string
        code (e.g. "H") OR display name (e.g. "Helpdesk") from
        get_request_catalog_enums → catalog_types. Recommended when the
        tenant has more than one base type; the API derives type_id from it.
      - use_catalog_item_name: True → instantiated requests use the
        catalog's name as the title.
      - hide_name / hide_description: hide the corresponding fields on
        the request creation form (visual only; doesn't drop the data).
      - show_contact_info: show the contact block on the form (CDESK default
        True).
      - search_keywords: extra words the catalog is found by (list of strings).
      - change_management: True turns on change management for requests
        made from this catalog. change_manager_type picks the change manager:
        1 = the user in change_manager_id (checked to be an existing user),
        2 = the manager set for the request type. Turning change_management
        off keeps the stored manager.
      - internal: True sets the catalog's visibility to CDESK's "Interní"
        (internal) option, False to "Obyčejná" (ordinary).
      - auto_close_request: automatic closing of requests made from this
        catalog. CDESK defaults it to True on create.
      - planned_work_enabled / parallel_approvals_enabled /
        include_fill_hours: allow planned works, parallel approval processes,
        and fill hours on requests made from this catalog.
      - cat_type / cat_area / cat_area_2nd: where the catalog is filed — the
        request type and the 1st/2nd-level catalog area. Each takes an id or a
        name from get_request_catalog_enums (cat_types / cat_areas /
        cat_areas_2nd) and is checked against it.
      - approval_rule_id: the approval rule applied before solving (used only
        when catalog approvals are on for the tenant). simple_approval_rules:
        extra rules applied before solving (cleared whenever approval_rule_id
        is empty). approval_rules: the rules available while solving. The
        valid ids per field are in get_request_catalog_enums →
        approval_rule_options; the server stores any id unchecked.
      - job_title_ids: the job titles the catalog is visible to (ids from
        get_request_catalog_enums → job_titles, checked); empty = everyone.
      - visible_group_ids: the user groups the catalog is visible to.
      - company_ids / contract_ids: the companies and contracts the catalog
        is assigned to.
      - category_ids: request categories (ids from get_request_catalog_enums →
        categories; only when categories are enabled for the tenant). The
        server stores any id unchecked. REQUIRED on create when the tenant's
        request settings require categories (get_request_enums
        `create_requirements` lists "request categories" then) and the tenant
        has any — CDESK answers "Vyberte kategóriu požiadavky" otherwise
        (for accounts with access to the category field).
      - tile_ids: the request-catalog tiles the catalog appears under; a tile
        must be a leaf and not a divider, otherwise the server answers 400.
      - custom_fields: the request custom fields bound to the catalog, as
        [{id, required?, required_status_config?, param_request?,
        param_request_detail?, tab_id?}] — `id` is the property id from
        get_request_custom_fields (propertyId). The list REPLACES the current
        bindings.
      - base_params: the template's per-field prefill and restriction rules,
        keyed by field (priority, urgency_id, impact_id, id_solver,
        solver_group_id, other_solvers, request_watchers, tags, cat_status_id,
        id_company, applicant, hyperlink, status_flow_transitions, ...); each
        entry may carry value, insert, update (edit-right codes from
        get_request_catalog_enums → updates), show_when_creating_request,
        restrict ("" = unrestricted, "limited" = only the ids in allowed),
        and prioritize. Only the keys and settings sent change; the server
        fills in the rest. Also accepted: reaction_missed_id_solver,
        cmdb_objects_select, cmdb_filter_mode, cmdb_filter,
        cmdb_scanner_mobile, computer_remote_access_by_email.

    The fields follow the tenant's CDESK catalog form, read live from its
    request settings: a field the form does not offer is refused when given,
    and a field it requires must be given — nothing is written then, and the
    refusal names each field and its setting. The priority prefill
    (base_params.priority) is required unless an urgency × impact matrix
    sets it for the catalog's request type (then base_params.impact_id is);
    base_params.solver_group_id is required in solver-group mode; other
    solvers / responsible person follow their 0/1/2 setting; change_manager_type
    is required with change management, change_manager_id with type 1; a
    hyperlink template (type X) takes base_params.hyperlink and
    hyperlink_target and no request prefills. get_request_catalog_enums
    `fields` lists each state on this tenant.

    The server stores the job titles' visibility, groups, companies and custom
    field bindings without checking unknown ids (an unknown one is dropped
    with a 200).

    CDESK's own 400/422 messages (e.g. a missing category) come back
    verbatim.

    Returns: {data: {...full record...}}; `for_customer` is echoed there as
    1/0 and read back by get_request_catalog as "hidden"/"".
    """
)

_UPDATE_CATALOG_DESC = inspect.cleandoc(
    """
    Partial update of a request-catalog: only the fields passed change, every
    other field and relation keeps its value. Optimistic locking is handled
    internally (the PUT requires the record's version token).

    Same fields as create_request_catalog, under the same tenant form rules:
    an off field is refused, a required one cannot be cleared (left out, it
    keeps its value). `code` keeps its 30-character limit.

    Lists (job_title_ids, visible_group_ids, company_ids, contract_ids,
    category_ids, tile_ids, search_keywords, simple_approval_rules,
    approval_rules, custom_fields) REPLACE the stored relation; an empty list
    clears it. base_params changes only the keys and settings passed.
    clear_fields empties change_manager_id or approval_rule_id (clearing the
    approval rule also clears simple_approval_rules).

    CDESK resets a catalog's base_params (prefill values and edit rules) on any
    update that does not send them, so this tool reads the stored ones first
    and sends them back with the changes merged in; if they cannot be read, the
    result carries a `warnings` entry.

    Returns: {data: {...full record...}}; check the optional `warnings`
    list for silently-dropped fields.
    """
)

_DELETE_CATALOG_DESC = inspect.cleandoc(
    """
    Delete a request-catalog by id (soft-delete; the row gets deleted_at
    set). Recovery is via the CDESK UI — list_request_catalogs cannot
    surface soft-deleted catalogs (the backend ignores a deleted_at
    filter, so there is no include_deleted option).

    Returns {deleted: <id>} on success.
    """
)
