"""Customer module MCP tools.

Tools that expose CDESK's Customer v3 CRUD + an intent helper + the
custom-field catalog. Mirrors the Task tools pattern (CdeskClient + filter
builder + error translator), minus the enum cache (no /v3/company/enums
endpoint).

Optimistic locking is hidden: update_customer internally does GET → grab
timestamp_check → PUT. The LLM never sees the version token. CDESK reads
`timestamp_check` from the PUT body despite the OpenAPI spec claiming
`updated_at`.
"""

from __future__ import annotations

import asyncio

import inspect
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp import base_config
from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.field_policy import (
    CUSTOMER_FIELDS,
    CUSTOMER_UPDATE_FIELDS,
    customer_settings,
    customer_type_tags,
    refuse_if_violated,
    violations,
)
from cdesk_mcp.filters import adapt_sb_for_cdesk, build_customer_filter, encode_sb, ensure_fill_start_date_window
from cdesk_mcp.sort_columns import resolve_sort, sort_note
from cdesk_mcp.tools._helpers import (
    drop_empty_fields,
    list_meta,
    TEXT_MATCH_NOTE,
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
from cdesk_mcp.tools._local_filter import (
    NOT_EQUAL_CMP,
    RANGE_CMPS,
    ColumnRule,
    filtered_page,
    numeric_picker,
    split_conditions,
)
from cdesk_mcp.tz import localize_naive_datetime

# LLM-friendly name → CDESK field name. Centralised so the LLM vocabulary
# stays clean while CDESK quirks live in one place.
_CUSTOMER_CREATE_FIELDS = {
    "tag_ids": "tag_id",  # CDESK uses singular `tag_id` for an array of int ids
    "postal_code": "zip",  # avoid shadowing the Python `zip()` builtin
}

_MAX_PER_PAGE = 100


# What a customer write answers with: CDESK can send the whole customer model
# back — tens of kB, with a serialized settings blob and the mail-server
# configuration — plus many empty fields. None of it tells the model anything
# about the write (passwords are already masked by redact_secrets).
# get_customer still returns a full record when asked.
_CUSTOMER_INTERNAL_PREFIXES = ("pop3_", "smtp_", "imap_", "broad_read_", "error_evaluation_")
_CUSTOMER_INTERNAL_KEYS = frozenset({"auto_mail_server", "settings", "easyclickSettings"})


def _compact_customer_write(result: dict[str, Any]) -> dict[str, Any]:
    for key in ("savedData", "data"):
        record = result.get(key)
        if isinstance(record, dict):
            result[key] = {
                k: v for k, v in drop_empty_fields(record).items()
                if k not in _CUSTOMER_INTERNAL_KEYS
                and not k.startswith(_CUSTOMER_INTERNAL_PREFIXES)
            }
    return result


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
        description=_LIST_CUSTOMERS_DESC + "\n\n" + sort_note("customers") + "\n\n" + TEXT_MATCH_NOTE,
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
                params["sort"] = resolve_sort("customers", sort)
            sb, local = split_conditions(adapt_sb_for_cdesk(sb, "customers"), _customer_condition)
            if sb is not None:
                params["sb"] = encode_sb(sb)
            local_info: dict[str, Any] | None = None
            if not local:
                response, paging = await client.get_page("v3/company", params=params)
                records, meta = unwrap_list(response)
            else:
                records, meta, paging, local_info = await filtered_page(
                    client, "v3/company", params, local, page=page, per_page=per_page,
                    fields=fields, cap=_LOCAL_FILTER_CAP,
                    narrow_hint="narrow the filter first (text_search, status, or another sb_raw condition such as company.city)",
                    note="CDESK cannot apply these conditions; the connector "
                         "fetched every customer matching the rest of the filter and "
                         "applied the condition itself.",
                )
        except ValueError as e:
            raise RuntimeError(f"list_customers input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_customers") from e

        result = {"items": records, "meta": list_meta(meta, paging, dropped=dropped_clauses), "page": page, "per_page": per_page}
        if local_info is not None:
            result["filtered_by_connector"] = local_info
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
            # The tenant's customer FORM rules, read live from core/base-config
            # (no v3 source; v1 customer/settings is admin-only): each field the
            # form requires must be given, and with customer_type at 2 the tags
            # must include a customer-type tag (CDESK checks that on create only).
            customer_cfg = customer_settings(await base_config.load(client))
            refuse_if_violated(violations(CUSTOMER_FIELDS, customer_cfg, {
                "tag_ids": customer_type_tags(tag_ids, customer_cfg),
                "email_general": email_general, "phone_general": phone_general,
                    "mobile_general": mobile_general, "web_page": web_page,
                    "street": street, "city": city, "postal_code": postal_code,
                    "country": country, "ico": ico, "dic": dic, "icdph": icdph,
            }))
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
        return _compact_customer_write(redact_secrets(annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
        )))

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
            # The tenant's customer FORM rules, read live from core/base-config
            # (no v3 source; v1 customer/settings is admin-only): a required
            # field cannot be cleared (left out = unchanged).
            refuse_if_violated(violations(
                CUSTOMER_UPDATE_FIELDS, customer_settings(await base_config.load(client)), {
                    "email_general": email_general, "phone_general": phone_general,
                    "mobile_general": mobile_general, "web_page": web_page,
                    "street": street, "city": city, "postal_code": postal_code,
                    "country": country, "ico": ico, "dic": dic, "icdph": icdph,
                },
                update=True,
            ))
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
        return _compact_customer_write(redact_secrets(annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
        )))

    @mcp.tool(
        description=_DELETE_CUSTOMER_DESC,
        annotations=ToolAnnotations(title="Delete customer", destructiveHint=True),
    )
    async def delete_customer(id: int) -> dict[str, Any]:
        try:
            linked = await _linked_records(client, id)
        except Exception as e:
            raise to_llm_error(e, operation="delete_customer", record_id=id) from e
        if linked:
            raise RuntimeError(
                f"delete_customer refused: customer {id} still has "
                f"{', '.join(linked)}. CDESK would delete the customer and leave "
                f"those records pointing at a customer that no longer exists; "
                f"move them to another customer or delete them first."
            )
        try:
            await client.delete(f"v3/company/{id}")
        except Exception as e:
            raise to_llm_error(e, operation="delete_customer", record_id=id) from e
        # Match delete_task's `{deleted: <id>}` shape for consistency.
        # CDESK's delete response shape varies (None for 204, envelope for 200);
        # the LLM doesn't need that — it just needs confirmation the id is gone.
        return {"deleted": id}


# ---------- helpers (module-private) -------------------------------------

# Records that point at a customer. CDESK refuses a customer delete only for
# an open task: one with closed tasks or fills is deleted and those records
# keep pointing at it. The list filter key per module, and how each is named.
_LINKED_LISTS = (
    ("v3/request", "id_company", "requests"),
    ("v3/task", "company_id", "tasks"),
    ("v3/fill", "company_id", "fills"),
    ("v3/contract", "id_customer", "deals"),
    ("v3/project", "company_id", "projects"),
)


async def _linked_records(client: CdeskClient, customer_id: int) -> list[str]:
    """The kinds of record still linked to the customer, e.g. ["tasks"]."""

    async def has_any(path: str, col: str) -> bool:
        sb: dict[str, Any] = {"group": True, "o": "AND", "query": [
            {"col": col, "c": "=", "q": customer_id, "o": "AND"},
        ]}
        if path == "v3/fill":
            # Without a date window the fill list covers the current month only.
            widened = ensure_fill_start_date_window(sb)
            sb = widened if isinstance(widened, dict) else sb
        rows, _ = unwrap_list(await client.get(path, params={"pg": 1, "pp": 1, "sb": encode_sb(sb)}))
        return bool(rows)

    found = await asyncio.gather(*(has_any(path, col) for path, col, _ in _LINKED_LISTS))
    return [name for (_, _, name), hit in zip(_LINKED_LISTS, found) if hit]


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

    CDESK reads `timestamp_check` from the PUT body (not `updated_at`), so it
    must be present in the GET response.

    fieldset=all is required: the v3 fieldset default (`extended`) does not
    include `timestamp_check` — only `all` carries it."""
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

# Conditions CDESK cannot apply on the customer list, applied by the connector
# (tools/_local_filter.py): a range on `hour_rate` answers HTTP 500 (BETWEEN
# matches nothing), and a range or `!=` on `id` is read as equality.
_LOCAL_FILTER_CAP = 2000
_customer_condition = numeric_picker({
    "hour_rate": ColumnRule(RANGE_CMPS),
    "id": ColumnRule(RANGE_CMPS | {NOT_EQUAL_CMP}),
})



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
      Advanced; mutually exclusive with the typed filters. Only the columns
      known to work are applied server-side — id, name,
      status, hour_rate, ico, dic, icdph, email, phone, created_at,
      updated_at (strict W3C datetimes), the dotted address columns
      company.city / company.street / company.zip / company.country, plus the
      no-col text leaf. So an exact address or tax-id filter IS available
      server-side — prefer it over paging the full set. Clauses on any other
      column (e.g. `code`, `company_num`, `type`, `sla_id`) are STRIPPED, and
      the response carries an `unsupported_filters` block naming them.
      A range on hour_rate (>, >=, <, <=, BETWEEN), and a range or a
      single-value '!=' on id, are applied by this tool, because CDESK
      cannot compare them that way: it fetches every customer
      matching the rest of the filter (up to 2000) and keeps the matching
      ones; the result carries a `filtered_by_connector` block. Such a
      condition must be joined to the rest by AND. `name` '=' with an
      integer filters the customer id. Date presets (TODAY, THIS_WEEK, …)
      on created_at / updated_at are sent as the BETWEEN window they name,
      in tenant time; updated_at falls back to the creation date for a
      customer never updated, so it is never empty. A list of values in
      '=' or '!=' matches any / none of them. '!=' leaves out customers
      with an empty value in that column.

    Pagination: page (1-based), per_page (default 20, max 100).

    `meta.total` is the number of records matching the filter across all pages (CDESK's own count) and `meta.last_page` tells whether this is the last page.

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
    shipping/invoicing address blocks. It is a
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

    KNOWN LIMITATION: the tenant currently ignores customFields writes
    silently — values do not persist. After any write, a
    get_customer(id, fieldset="custom") read shows whether the value actually
    persisted; a value that did not stick was not stored despite the 200.
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
    queries. When the tenant requires a customer type, tag_ids must
    include a customer-type tag (CDESK's customer/supplier tags) — read
    live with the rest of the tenant's customer form: e-mail, phones, web,
    address, IČO, DIČ and IČ DPH are each refused when missing if the form
    requires them (and cannot be cleared by update_customer).
    The customer-type tag also decides whether the company can
    receive requests: when the tenant flags a tag as "requests may be
    added" (CDESK's default customer-type tag does), a company created
    without such a tag is refused by create_request with "Zadajte
    aktívneho zákazníka". Some tenants also require price-list templates
    on a new company, which this tool cannot send.

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

    Side effects (by CDESK): on a tenant with CDESK's mail-server integration
    (the customer's auto_mail_server on the default POP3 server), creating a
    customer also creates a POP3 mailbox for it on that mail server — it stays
    when the customer is deleted.
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

    Side effects (by CDESK): an update can re-run the customer's mailbox setup and
    answer "Mailbox creating error 2 (user already exists!)" — the mailbox exists
    already; the update itself is stored.
    """
)

_DELETE_CUSTOMER_DESC = inspect.cleandoc(
    """
    Delete a customer by id. Refused while any request, task, fill, deal or
    project still belongs to the customer (open or closed); the error names
    which kinds, and those records have to be moved to another customer or
    deleted first. Before deleting, the tool checks each of these lists for
    the customer.
    """
)
