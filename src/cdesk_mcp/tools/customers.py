"""Customer module MCP tools (M8).

Seven tools that expose CDESK's Customer v3 CRUD + an intent helper +
the custom-field catalog.
Mirrors the M7 Task tools pattern (CdeskClient + filter builder + error
translator), minus the enum cache (no /v3/company/enums endpoint).

Optimistic locking is hidden: update_customer internally does GET → grab
timestamp_check → PUT. The LLM never sees the version token. CDESK reads
`timestamp_check` from the PUT body despite the OpenAPI spec claiming
`updated_at` — verified against BaseModel::hasBeenUpdated in M7.
"""

from __future__ import annotations

import inspect
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.filters import build_customer_filter, encode_sb
from cdesk_mcp.tools._helpers import (
    annotate_write_warnings,
    apply_field_scope,
    redact_secrets,
    to_llm_error,
    unsupported_filter_directive,
    unwrap_list,
    unwrap_record,
    wrap_collection,
    validate_custom_fields,
    validate_fields,
    validate_fieldset,
    yn as _yn,
)
from cdesk_mcp.tz import localize_naive_datetime

# LLM-friendly name → CDESK field name. Centralised so the LLM vocabulary
# stays clean while CDESK quirks live in one place.
_CUSTOMER_CREATE_FIELDS = {
    "tag_ids": "tag_id",  # CDESK uses singular `tag_id` for an array of int ids
    "postal_code": "zip",  # M8.11: avoid shadowing the Python `zip()` builtin
}

_MAX_PER_PAGE = 100


def register_customer_tools(mcp: FastMCP, client: CdeskClient) -> None:
    """Register all Customer-module tools on the given FastMCP instance.
    Called by build_server when client is available.

    Note: Customer tools do NOT take an EnumCache — CDESK has no
    `/v3/company/enums` endpoint. Tag-id name resolution is therefore
    deferred to the LLM (which must discover tag ids via the CDESK UI
    or `sb_raw` queries). If CDESK ever exposes a tag enum endpoint,
    this signature can grow a `cache` parameter for symmetry with
    register_task_tools."""

    @mcp.tool(
        description=_LIST_CUSTOMERS_DESC,
        annotations=ToolAnnotations(title="List customers", readOnlyHint=True),
    )
    async def list_customers(
        text_search: str | None = None,
        status: str | None = None,
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
            # Unsupported sb_raw columns are stripped and reported via
            # `unsupported_filters` — the agent must filter the items itself.
            dropped_clauses: list[dict[str, Any]] = []
            sb = build_customer_filter(
                text_search=text_search,
                status=status,
                sb_raw=sb_raw,
                dropped_cols_out=dropped_clauses,
            )
            params: dict[str, Any] = {"pg": page, "pp": per_page}
            apply_field_scope(params, fieldset, fields)
            if sort:
                params["sort"] = sort
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response = await client.get("v3/company", params=params)
        except ValueError as e:
            raise RuntimeError(f"list_customers input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_customers") from e

        records, meta = unwrap_list(response)
        result = {"items": records, "meta": meta, "page": page, "per_page": per_page}
        if dropped_clauses:
            result["unsupported_filters"] = unsupported_filter_directive(dropped_clauses)
        return result

    @mcp.tool(
        description=_GET_CUSTOMER_DESC,
        annotations=ToolAnnotations(title="Get customer", readOnlyHint=True),
    )
    async def get_customer(
        id: int,
        fieldset: str | None = None,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            validate_fieldset(fieldset)
            validate_fields(fields)
            params: dict[str, Any] = {}
            if fieldset:
                params["fieldset"] = fieldset
            if fields:
                params["returnFields[]"] = fields
            response = await client.get(f"v3/company/{id}", params=params or None)
        except ValueError as e:
            raise RuntimeError(f"get_customer input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="get_customer", record_id=id) from e
        record = redact_secrets(unwrap_record(response))
        if isinstance(record, dict):
            return record
        return {"data": record}

    @mcp.tool(
        description=_GET_CUSTOMER_CUSTOM_FIELDS_DESC,
        annotations=ToolAnnotations(title="Get customer custom fields", readOnlyHint=True),
    )
    async def get_customer_custom_fields() -> Any:
        try:
            # The custom-field catalog is only documented on the company alias
            # of this module (`/v3/company/custom-fields`).
            response = await client.get("v3/company/custom-fields")
        except Exception as e:
            raise to_llm_error(e, operation="get_customer_custom_fields") from e
        return wrap_collection(
            unwrap_record(response),
            kind="custom-field definitions for the Customer (company) module",
        )

    @mcp.tool(
        description=_FIND_CUSTOMER_BY_NAME_DESC,
        annotations=ToolAnnotations(title="Find customer by name", readOnlyHint=True),
    )
    async def find_customer_by_name(name: str, max_results: int = 10) -> dict[str, Any]:
        """Intent helper — text-search shortcut for the common 'find a customer
        I know by name' workflow. Implemented on top of list_customers."""
        if max_results < 1 or max_results > _MAX_PER_PAGE:
            raise RuntimeError(
                f"find_customer_by_name input error: max_results must be 1..{_MAX_PER_PAGE}"
            )
        try:
            sb = build_customer_filter(text_search=name)
            params: dict[str, Any] = {"pg": 1, "pp": max_results}
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response = await client.get("v3/company", params=params)
        except Exception as e:
            raise to_llm_error(e, operation="find_customer_by_name") from e

        records, _meta = unwrap_list(response)
        # Trim to id + name + a few useful fields so the LLM gets a compact
        # disambiguation surface without burning context on full records.
        compact: list[dict[str, Any]] = []
        for r in records:
            if not isinstance(r, dict):
                continue
            compact.append(
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "alias": r.get("alias"),
                    "city": r.get("city"),
                    "country": r.get("country"),
                    "ico": r.get("ico"),
                }
            )
        return {"matches": compact, "count": len(compact)}

    @mcp.tool(
        description=_CREATE_CUSTOMER_DESC,
        annotations=ToolAnnotations(title="Create customer", destructiveHint=True),
    )
    async def create_customer(
        name: str,
        *,
        alias: str | None = None,
        code: str | None = None,
        ico: str | None = None,
        dic: str | None = None,
        icdph: str | None = None,
        street: str | None = None,
        city: str | None = None,
        postal_code: str | None = None,
        country: str | None = None,
        web_page: str | None = None,
        email_general: str | None = None,
        phone_general: str | None = None,
        mobile_general: str | None = None,
        email_invoice: str | None = None,
        contract_number: str | None = None,
        company_owner: int | None = None,
        id_maint_user: int | None = None,
        sla_id: int | None = None,
        expiration: str | None = None,
        tag_ids: list[int] | None = None,
        cdesk_allowed: bool | None = None,
        custom_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            validate_custom_fields(custom_fields)
            # Validates ISO shape; a naive datetime gets the tenant offset
            # (CDESK reads naive as UTC); bare dates pass through unchanged.
            expiration = localize_naive_datetime("expiration", expiration)
            body = _build_customer_body(
                {
                    "name": name,
                    "alias": alias,
                    "code": code,
                    "ico": ico,
                    "dic": dic,
                    "icdph": icdph,
                    "street": street,
                    "city": city,
                    "postal_code": postal_code,
                    "country": country,
                    "web_page": web_page,
                    "email_general": email_general,
                    "phone_general": phone_general,
                    "mobile_general": mobile_general,
                    "email_invoice": email_invoice,
                    "contract_number": contract_number,
                    "company_owner": company_owner,
                    "id_maint_user": id_maint_user,
                    "sla_id": sla_id,
                    "expiration": expiration,
                    "tag_ids": tag_ids,
                    "cdesk_allowed": _yn(cdesk_allowed),
                }
            )
            # customFields keys are already wire-format (cfield_*) — set after
            # the body build so they bypass the LLM-name→CDESK-name rename map.
            if custom_fields:
                body["customFields"] = custom_fields
            response = await client.post("v3/company", json=body)
        except ValueError as e:
            raise RuntimeError(f"create_customer input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="create_customer") from e
        return redact_secrets(annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
        ))

    @mcp.tool(
        description=_UPDATE_CUSTOMER_DESC,
        annotations=ToolAnnotations(title="Update customer", destructiveHint=True),
    )
    async def update_customer(
        id: int,
        *,
        name: str | None = None,
        alias: str | None = None,
        code: str | None = None,
        ico: str | None = None,
        dic: str | None = None,
        icdph: str | None = None,
        street: str | None = None,
        city: str | None = None,
        postal_code: str | None = None,
        country: str | None = None,
        web_page: str | None = None,
        email_general: str | None = None,
        phone_general: str | None = None,
        mobile_general: str | None = None,
        email_invoice: str | None = None,
        contract_number: str | None = None,
        company_owner: int | None = None,
        id_maint_user: int | None = None,
        sla_id: int | None = None,
        expiration: str | None = None,
        tag_ids: list[int] | None = None,
        cdesk_allowed: bool | None = None,
        custom_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            if name is not None and not name.strip():
                raise ValueError("name must be a non-empty string")
            validate_custom_fields(custom_fields)
            # Validates ISO shape; a naive datetime gets the tenant offset
            # (CDESK reads naive as UTC); bare dates pass through unchanged.
            expiration = localize_naive_datetime("expiration", expiration)

            current = await _fetch_for_update(client, id)
            body = _build_customer_body(
                {
                    "name": name,
                    "alias": alias,
                    "code": code,
                    "ico": ico,
                    "dic": dic,
                    "icdph": icdph,
                    "street": street,
                    "city": city,
                    "postal_code": postal_code,
                    "country": country,
                    "web_page": web_page,
                    "email_general": email_general,
                    "phone_general": phone_general,
                    "mobile_general": mobile_general,
                    "email_invoice": email_invoice,
                    "contract_number": contract_number,
                    "company_owner": company_owner,
                    "id_maint_user": id_maint_user,
                    "sla_id": sla_id,
                    "expiration": expiration,
                    "tag_ids": tag_ids,
                    "cdesk_allowed": _yn(cdesk_allowed),
                }
            )
            # customFields keys are already wire-format (cfield_*) — set after
            # the body build so they bypass the LLM-name→CDESK-name rename map.
            if custom_fields:
                body["customFields"] = custom_fields
            body["timestamp_check"] = current["timestamp_check"]
            response = await client.put(f"v3/company/{id}", json=body)
        except ValueError as e:
            raise RuntimeError(f"update_customer input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="update_customer", record_id=id) from e
        return redact_secrets(annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
        ))

    @mcp.tool(
        description=_DELETE_CUSTOMER_DESC,
        annotations=ToolAnnotations(title="Delete customer", destructiveHint=True),
    )
    async def delete_customer(id: int) -> dict[str, Any]:
        try:
            await client.delete(f"v3/company/{id}")
        except Exception as e:
            raise to_llm_error(e, operation="delete_customer", record_id=id) from e
        # M8.10: match delete_task's `{deleted: <id>}` shape for consistency.
        # CDESK's delete response shape varies (None for 204, envelope for 200);
        # the LLM doesn't need that — it just needs confirmation the id is gone.
        return {"deleted": id}


# ---------- helpers (module-private) -------------------------------------


def _build_customer_body(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop None values, rename LLM-friendly param names to CDESK field
    names per _CUSTOMER_CREATE_FIELDS."""
    body: dict[str, Any] = {}
    for k, v in raw.items():
        if v is None:
            continue
        cdesk_key = _CUSTOMER_CREATE_FIELDS.get(k, k)
        body[cdesk_key] = v
    return body


async def _fetch_for_update(client: CdeskClient, id: int) -> dict[str, Any]:
    """Fetch a customer and unwrap to the record dict, ready to feed the
    optimistic lock into a follow-up PUT.

    Per the M7 finding, CDESK reads `timestamp_check` from the PUT body
    (not `updated_at`). We require it to be present in the GET response.

    fieldset=all is required: the v3 fieldset default (`extended`) no longer
    includes `timestamp_check` — only `all` carries it (verified live)."""
    envelope = await client.get(f"v3/company/{id}", params={"fieldset": "all"})
    record = unwrap_record(envelope)
    if not isinstance(record, dict):
        raise RuntimeError(
            f"unexpected response shape fetching customer {id} for update: "
            f"{type(record).__name__}"
        )
    if "timestamp_check" not in record:
        raise RuntimeError(
            f"customer {id} record has no timestamp_check field; "
            f"cannot apply optimistic lock"
        )
    return record


# ---------- tool descriptions --------------------------------------------

_LIST_CUSTOMERS_DESC = inspect.cleandoc(
    """
    List customers (companies / partners) visible to your account. Returns a
    paginated list with the minimal fields useful for the LLM to triage and
    follow up with get_customer or find_customer_by_name.

    Filters:
    - text_search: full-text across name, alias, customer code, country,
      city, street, zip, ico, dic, icdph — the quickest way to search any of
      them at once (e.g. text_search="Bratislava").
    - sb_raw: an sb filter object for the CDESK API v3 company list (JSON
      string or object, structured tree form; see docs/cdesk-api-v3.json).
      Advanced; mutually exclusive with the typed filters. Only the
      LIVE-VERIFIED working columns are applied server-side — id, name,
      status, hour_rate, ico, dic, icdph, email, phone, created_at,
      updated_at (strict W3C datetimes), the dotted address columns
      company.city / company.street / company.zip / company.country, plus the
      no-col text leaf. So an exact address or tax-id filter IS available
      server-side — prefer it over paging the full set. Clauses on any other
      column (e.g. `code`, `company_num`, `type`, `sla_id`) are STRIPPED, and
      the response carries an `unsupported_filters` block naming them.

    Pagination: page (1-based), per_page (default 20, max 100).

    Ordering (re-verified live 2026-07-31 — the backend behaviour CHANGED):
    omitting `sort` gives id ascending (oldest first). A BARE COLUMN NAME now
    genuinely sorts ASCENDING by that column ("name" confirmed against a
    record deliberately given the highest id and an alphabetically-first
    name). A direction modifier is NOT honored — "-name" fell back to id
    ascending, i.e. the default order. So: for ascending order by one column
    pass the bare name; DESCENDING is not available server-side — page the
    set and sort client-side, and never assume a "-" prefix was applied.
    `meta` is usually empty (no total count) — detect the last page by a
    short/empty page.

    Response shaping (same semantics as get_customer): fieldset selects the
    field group per record ("base"/"extended"/"all"/"custom"); fields is an
    exact whitelist of field names (returnFields; union with fieldset).

    Returns: {items: [...], meta: {...}, page, per_page}.
    """
)

_GET_CUSTOMER_DESC = inspect.cleandoc(
    """
    Fetch one customer by id. Returns the record (typically including
    address, contacts, ico/dic/icdph, owner, sla, tags, etc.). Errors with
    a not-found message if the id doesn't exist or isn't in your admin scope.

    fieldset selects the field group returned: "base", "extended" (CDESK's
    default when omitted), "all" (base+extended+custom), or "custom". Pass
    "all" or "custom" to read the stored custom-field (cfield_*) values.

    Despite the name, "custom" is NOT custom-fields-only: alongside
    `customFields` it returns ~40 joined and computed extras, including the
    tenant `settings` matrix, `easyclickSettings`, `solvers`, `sla` and the
    shipping/invoicing address blocks (verified live 2026-07-31). It is a
    LARGE payload — when you only need the stored cfield_* values, pass their
    exact keys via `fields` instead.

    fields: an exact whitelist of field names to return (CDESK's
    returnFields). Composes with fieldset as a union; unknown names are
    silently ignored by CDESK.
    """
)

_GET_CUSTOMER_CUSTOM_FIELDS_DESC = inspect.cleandoc(
    """
    List the custom-field definitions configured for the Customer (company)
    module. Returns {items, count} — plus a `note` when count is 0, meaning the
    module has no custom fields defined (an empty result, not a failure). Each
    baseproperty in `items` carries a ready-to-use key in the form
    `cfield_<propertyId>_<basepropertyId>`.

    Workflow: (1) call this tool; (2) find the field by name; (3) pass
    `custom_fields={"<key>": <value>}` to create_customer/update_customer
    (scalar for single-value fields; for select/relation fields send the
    option id); (4) read stored values via get_customer(id, fieldset="all").

    KNOWN LIMITATION (2026-06-04): the tenant currently ignores
    customFields writes silently — values do not persist. After any
    write, a get_customer(id, fieldset="custom") read shows whether the
    value actually persisted; a value that did not stick was not
    stored despite the 200 (backend fix pending).
    """
)

_FIND_CUSTOMER_BY_NAME_DESC = inspect.cleandoc(
    """
    Intent helper for the common "find customer by name" workflow. Runs a
    full-text search across name, alias, code, address, and tax IDs, then
    returns a compact list of matches with id, name, alias, city, country,
    and ico — enough for the LLM to disambiguate without burning context
    on full records.

    Resolves a customer name to the id that create/update/delete require —
    those take ids only. `count` > 1 means the name was ambiguous: the tool
    picks no winner and returns every candidate, so the match is unresolved
    until one id is chosen.

    If `count` is 0, the search produced no matches under your current
    admin scope. A zero count is NOT evidence the customer doesn't exist. The most
    likely explanations: casing/spelling difference (try a shorter query
    or just the distinctive part of the name), the customer is outside
    your admin scope, or you typed the name in a language CDESK doesn't
    store it under. A broader query (e.g. just the first 3–4 distinctive
    letters) distinguishes those cases from genuine absence.

    Params: name (required), max_results (default 10, max 100).
    Returns: {matches: [{id, name, alias, city, country, ico}], count}.
    """
)

_CREATE_CUSTOMER_DESC = inspect.cleandoc(
    """
    Create a new customer (company / partner). Only `name` is required;
    everything else is optional but commonly used.

    Address: street, city, postal_code, country (ISO code preferred).

    Slovak tax IDs (ico, dic, icdph) are validated server-side against
    Finstat when the tenant configures that — invalid values surface as
    a validation error.

    Tag handling: pass `tag_ids` as a list of tenant tag ids. Some tenants
    require at least one customer-type classification tag at create time;
    the server will surface that as a clear validation error if so. We
    don't cache tag ids — discover them via the CDESK UI or sb_raw
    queries.

    cdesk_allowed: pass True to enable CDESK-portal access for this
    customer (sends 'Y'). Omit to leave it at the tenant default.

    custom_fields: flat dict of `cfield_<propertyId>_<basepropertyId>`
    keys → values (discover keys via get_customer_custom_fields;
    select/relation fields take the option id). Unrecognized keys are
    ignored with a 200 and reported in the result's `warnings` key.
    NB: the v1 company handler persists custom fields only for EXISTING
    records — if values don't stick on create, re-send them via
    update_customer.

    Returns: {data: <new id>, savedData: {...full record...}}; check the
    optional `warnings` list for silently-dropped/ignored fields
    (e.g. `code` without the company code-field write right).
    """
)

_UPDATE_CUSTOMER_DESC = inspect.cleandoc(
    """
    Partial update — pass only the fields you want to change. Internal
    GET fetches the current record's timestamp_check for the optimistic
    lock; the LLM doesn't see it.

    custom_fields: flat dict of `cfield_<propertyId>_<basepropertyId>`
    keys → values (discover keys via get_customer_custom_fields; read
    stored values via get_customer(id, fieldset="all")). Unrecognized
    keys are ignored with a 200 and reported in the result's `warnings`
    key (not a 400).

    On 409 conflict (record changed elsewhere since fetch), you'll get
    an Optimistic-lock error — re-fetch with get_customer and retry.

    Returns: {data: <id>, savedData: {...full record...}}; check the
    optional `warnings` list for silently-dropped/ignored fields.
    """
)

_DELETE_CUSTOMER_DESC = inspect.cleandoc(
    """
    Delete a customer by id. Fails with 409 if the customer has dependent
    records (deals, tasks, requests, etc.). The 409 names the blocking
    links, which have to be broken before the delete can succeed.
    """
)
