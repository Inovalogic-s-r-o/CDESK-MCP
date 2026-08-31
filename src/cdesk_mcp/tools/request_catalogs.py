"""Request Catalog module MCP tools (M12).

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
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.enums import EnumCache
from cdesk_mcp.filters import build_request_catalog_filter, encode_sb
from cdesk_mcp.tools._helpers import (
    annotate_write_warnings,
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
}

_MAX_PER_PAGE = 100


def register_request_catalog_tools(
    mcp: FastMCP,
    client: CdeskClient,
    cache: EnumCache,
) -> None:
    """Register all Request-Catalog tools on the given FastMCP instance.
    Called by build_server when client + the catalog enum cache are both
    available."""

    @mcp.tool(
        description=_LIST_CATALOGS_DESC,
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
                params["sort"] = sort
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response = await client.get("v3/request-catalog", params=params)
        except RuntimeError:
            raise  # _resolve_catalog_type_code's unknown-type message is friendly
        except ValueError as e:
            raise RuntimeError(f"list_request_catalogs input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_request_catalogs") from e

        records, meta = unwrap_list(response)
        result = {"items": records, "meta": meta, "page": page, "per_page": per_page}
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
        return unwrap_record(response)

    @mcp.tool(
        description=_GET_CATALOG_ENUMS_DESC,
        annotations=ToolAnnotations(title="Get request catalog enums", readOnlyHint=True),
    )
    async def get_request_catalog_enums(
        refresh: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        try:
            if refresh:
                await cache.refresh()
            else:
                await cache.load()  # populate a cold cache (snapshot is empty otherwise)
        except Exception as e:
            raise to_llm_error(e, operation="get_request_catalog_enums") from e
        return cache.snapshot()

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
    ) -> dict[str, Any]:
        try:
            if not name or not name.strip():
                raise ValueError("name must be a non-empty string")
            if len(name) > 30:
                raise ValueError(
                    f"name must be ≤30 characters (CDESK limit); got {len(name)}"
                )
            resolved_type = await _resolve_catalog_type_code(cache, type_code)

            body = _build_catalog_body(
                {
                    "name": name,
                    "code": code,
                    "description": description,
                    "type": resolved_type,
                    "use_catalog_item_name": use_catalog_item_name,
                    "hide_name": hide_name,
                    "hide_description": hide_description,
                }
            )
            response = await client.post("v3/request-catalog", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"create_request_catalog input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="create_request_catalog") from e
        return annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
        )

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
    ) -> dict[str, Any]:
        try:
            if name is not None and not name.strip():
                raise ValueError("name must be a non-empty string")
            if name is not None and len(name) > 30:
                raise ValueError(
                    f"name must be ≤30 characters (CDESK limit); got {len(name)}"
                )
            resolved_type = await _resolve_catalog_type_code(cache, type_code)

            current = await _fetch_for_update_catalog(client, id)

            body = _build_catalog_body(
                {
                    "name": name,
                    "code": code,
                    "description": description,
                    "type": resolved_type,
                    "use_catalog_item_name": use_catalog_item_name,
                    "hide_name": hide_name,
                    "hide_description": hide_description,
                }
            )
            # The catalog PUT REQUIRES a valid timestamp_check (omitting it —
            # or sending null — always fails with "Konflikt: záznam bol
            # medzičasom aktualizovaný"; verified live 2026-06-04). The GET
            # returns `timestamp_check: null`, so derive the token from
            # `updated_at` reformatted to CDESK's 'YYYY-MM-DD HH:MM:SS'.
            token = current.get("timestamp_check") or _lock_token_from_updated_at(
                current.get("updated_at")
            )
            if not token:
                # The PUT requires a valid timestamp_check; without one it
                # always fails with a (misleading) "modified by someone else"
                # conflict that no retry can clear. Fail fast with the real
                # cause instead of sending a known-doomed request.
                raise RuntimeError(
                    f"request-catalog {id} returned no timestamp_check or "
                    f"parseable updated_at, so the required optimistic-lock "
                    f"token can't be built — cannot safely update it. This is "
                    f"a backend data issue with the catalog record."
                )
            body["timestamp_check"] = token
            # Without this the PUT erases the catalog's prefill/restriction
            # rules — see _preserve_base_params.
            _preserve_base_params(body, current)
            response = await client.put(f"v3/request-catalog/{id}", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"update_request_catalog input error: {e}") from e
        except Exception as e:
            raise to_llm_error(
                e, operation="update_request_catalog", record_id=id,
            ) from e
        return _warn_if_base_params_unpreservable(
            annotate_write_warnings(
                response if isinstance(response, dict) else {"data": response},
                sent_body=body,
            ),
            current,
            body,
        )

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
    does not ship; `catalog_types` is the real source — verified live.)"""
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


def _lock_token_from_updated_at(updated_at: Any) -> str | None:
    """Convert the GET response's ISO-8601 `updated_at`
    (e.g. '2026-06-04T08:23:00+00:00') into the 'YYYY-MM-DD HH:MM:SS'
    form the catalog PUT accepts as `timestamp_check` (verified live —
    the ISO form and null are both rejected with a conflict error)."""
    if not isinstance(updated_at, str) or not updated_at:
        return None
    try:
        return datetime.fromisoformat(updated_at).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


async def _fetch_for_update_catalog(client: CdeskClient, id: int) -> dict[str, Any]:
    """Fetch a catalog and unwrap to the record dict, returning the
    timestamp_check (if present) for the follow-up PUT.

    Uses fieldset="all" because the DEFAULT catalog GET omits `base_params`
    (18 keys vs 63 — verified live 2026-07-29) and the PUT destroys any field
    it does not echo. See `_PRESERVE_ON_UPDATE`."""
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


# The catalog PUT rewrites `base_params` — the per-field
# {value, insert, update, restrict} rules that make a catalog a template at all
# (~1.7 KB of JSON) — from the body on EVERY update, so a body that doesn't
# carry it erases the catalog's whole configuration on a plain rename.
#
# The subtlety that matters: the PUT does NOT read the `base_params` key it
# returns. It reads **`base_params_decoded`**, the parsed-object form that
# `fieldset="all"` also returns. Echoing `base_params` back preserves nothing —
# not as the stored JSON string, not as a parsed object — which is why an
# earlier pass here wrongly concluded the loss was an unfixable backend defect
# and settled for warning about it. It is fixable: echoing
# `base_params_decoded` alone (with `base_params` omitted entirely) restores the
# rules byte-identically. Verified live 2026-07-29 by bisecting all 58 record
# keys down to a minimal preserving set of exactly ["base_params_decoded"].
#
# Reading it requires fieldset="all" — the default catalog GET omits both keys
# (18 keys vs 63), which is why _fetch_for_update_catalog pins that fieldset.
_PRESERVE_ON_UPDATE = "base_params_decoded"


def _preserve_base_params(body: dict[str, Any], current: dict[str, Any]) -> None:
    """Echo `base_params_decoded` into the PUT body so the update doesn't erase
    the catalog's prefill/restriction rules. Mutates `body` in place.

    Skips the echo when the caller set the key themselves, and when the fetched
    record has no value to preserve."""
    if _PRESERVE_ON_UPDATE in body:
        return  # caller is setting it explicitly — don't override
    decoded = current.get(_PRESERVE_ON_UPDATE)
    if decoded in (None, "", [], {}):
        return
    body[_PRESERVE_ON_UPDATE] = decoded


def _warn_if_base_params_unpreservable(
    response: dict[str, Any], current: dict[str, Any], body: dict[str, Any]
) -> dict[str, Any]:
    """Warn only when the preservation above could NOT be applied.

    The normal path loses nothing, so this stays silent. It fires when the
    fetched record had non-empty `base_params` but no `base_params_decoded` to
    echo — meaning the rules were about to be erased and this tool had nothing
    to send. Better a loud warning than a silent wipe if CDESK ever changes the
    shape of that response."""
    previous = current.get("base_params")
    if not previous or _PRESERVE_ON_UPDATE in body:
        return response
    size = len(previous) if isinstance(previous, str) else len(str(previous))
    warning = (
        f"DATA LOSS: this catalog's `base_params` ({size} chars of prefill and "
        "value-restriction rules) was erased by the update. The catalog PUT "
        "rewrites that configuration from the request body, and the record "
        f"carried no `{_PRESERVE_ON_UPDATE}` value for this tool to send back. "
        "The catalog's other fields are intact, but requests instantiated from "
        "it are no longer prefilled or value-restricted until the rules are "
        "re-entered in the CDESK UI."
    )
    existing = response.get("warnings")
    if isinstance(existing, list):
        response["warnings"] = [*existing, warning]
    else:
        response["warnings"] = [warning]
    return response


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
    the plain column forms — verified): type_id, auto_close_request,
    category/change-manager/approval columns, date ranges, and soft-deleted.
    Some of these ARE reachable through sb_raw's DOTTED forms below (type_id
    and the category type/area in particular) — filter client-side only for
    the rest. `sort` genuinely has NO observable effect on this endpoint
    (verified live 2026-06-05, re-confirmed 2026-07-31: `sort=name` still
    returns the default id-descending order on a set whose name order differs
    from its id order) — sort client-side instead. NB this is one of the few
    endpoints where that is still true; task / request / company / user all
    began honoring a bare column name.

    sb_raw — a CDESK sb filter object for the CDESK API v3 request-catalog list
    (JSON string or object, structured tree form; see docs/cdesk-api-v3.json).
    Advanced; mutually exclusive with the typed filters above. Only the
    LIVE-VERIFIED working columns are applied server-side — id, type,
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
    create_request_catalog. Other buckets include change-manager options,
    status flows, tags, and approval rules.

    Pass refresh=True to force a fresh fetch.
    """
)

_CREATE_CATALOG_DESC = inspect.cleandoc(
    """
    Create a new request-catalog template.

    Required:
      - name (max 30 characters; CDESK limit).

    Optional:
      - code: short identifier (max 30). NB: without the catalog
        code-field write right CDESK silently drops the value with a
        200 (verified live — savedData.code stays empty); this tool
        detects the drop and adds a `warnings` entry to the result.
      - description: HTML body shown as the default request description
        when instantiated.
      - type_code: base request type the catalog produces — the string
        code (e.g. "H") OR display name (e.g. "Helpdesk") from
        get_request_catalog_enums → catalog_types. Recommended when the
        tenant has more than one base type; the API derives type_id from it.
      - use_catalog_item_name: True → instantiated requests use the
        catalog's name as the title.
      - hide_name / hide_description: hide the corresponding fields on
        the request creation form (visual only; doesn't drop the data).

    Tenant settings may require additional fields (change-manager,
    status-flow, custom approvals, tags) — those are accepted server-
    side but not exposed individually here. If CDESK returns 400/422,
    its message will explain what's missing.

    Returns: {data: <new id>, savedData: {...full record...}}.
    """
)

_UPDATE_CATALOG_DESC = inspect.cleandoc(
    """
    Partial update of a request-catalog. Pass only the fields you want
    to change. Optimistic locking is handled internally (the lock token
    is derived from the record's updated_at; the PUT requires it).

    Same fields as create_request_catalog. Note: name still has the
    30-character limit on update, and `code` may be silently dropped
    without the write right (a `warnings` entry flags the drop).

    The catalog's `base_params` (its prefill defaults and value restrictions)
    are preserved across the update: the underlying PUT rewrites them from the
    request body, so this tool reads the stored configuration first and sends it
    back unchanged. In the rare case it cannot be read, the result carries a
    `warnings` entry saying the rules were erased and need re-entering in the
    CDESK UI.

    Returns: {data: <id>, savedData: {...full record...}}; check the
    optional `warnings` list for silently-dropped fields.
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
