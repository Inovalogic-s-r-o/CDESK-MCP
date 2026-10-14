"""CMDB module MCP tools (read-only).

Nine tools exposing CDESK's CMDB v3 surface: configuration items
(CIs) plus the catalog hierarchy that scopes them. The hierarchy is

    Category → Maingroup → Type → CI

A *type* is what makes a CI a "server", "printer", "place", etc., and
each type carries its own dynamic property schema (fetched via
get_cmdb_type_properties). The v3 API is read-only for CMDB — there
are no create/update/delete endpoints, so no field maps, optimistic
locking, or enum cache here.

CI status enums are tiny and not used for name→id resolution by any
tool, so get_ci_enums fetches them directly instead of wiring a 4th
EnumCache into the startup probe.
"""

from __future__ import annotations

from datetime import datetime, timezone

import asyncio
import inspect
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.filters import adapt_sb_for_cdesk, build_ci_filter, encode_sb
from cdesk_mcp.sort_columns import resolve_sort, sort_note
from cdesk_mcp.tools._helpers import (
    list_meta,
    TEXT_MATCH_NOTE,
    EnumLabel,
    apply_enum_names,
    enum_rows_name_map,
    to_llm_error,
    unsupported_filter_directive,
    unwrap_list,
    unwrap_record,
    wrap_collection,
    validate_fields,
    page_of,
)
from cdesk_mcp.tools._name_lookup import CUSTOMER_NAMES, USER_NAMES, resolve_name_searches
from cdesk_mcp.tools._local_filter import (
    NOT_EQUAL_CMP,
    RANGE_CMPS,
    ColumnRule,
    LocalCondition,
    fetch_rows_with_columns,
    filtered_page,
    numeric_picker,
    split_conditions,
)

_MAX_PER_PAGE = 100

# A CI's `status_id` holds the status enum ID (not its action code). The name
# comes from the live
# v3/cmdb/ci/enums rows, read WITH inactive ones so a CI sitting in a status
# since switched off still gets its name. Type/maingroup/company names are
# already joined in by CDESK (ci_type_name, maingroup_name, company_name).
_CI_ENUM_LABELS = (EnumLabel("status_id", "status", "status_name"),)

# Type/maingroup write keys that carry an icon only when `thumb` is true — the
# v1 handler clears icon/icon_color when it is false, so passing an icon without
# thumb silently does nothing.
_ICON_KEYS = ("icon", "icon_color")


def _unresolvable_maingroup_hint(exc: Exception, maingroup_id: str | None) -> str | None:
    """Actionable text for the one 400 this endpoint answers with an empty body.

    GET /v3/cmdb/ci resolves a non-numeric `maingroupId` as a slug and, finding
    nothing, returns `400 {"data": false}` — a body the generic translator can only echo, leaving the model with "Bad
    request … {"data": false}" and nothing to act on. An unknown *numeric* id, by
    contrast, is a clean empty list, so this failure only ever means "that
    non-numeric value resolved to no maingroup".

    Returns None for anything else, so every other error keeps the normal
    translation.
    """
    if maingroup_id is None or maingroup_id.strip().isdigit():
        return None
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status != 400:
        return None
    return (
        f"list_configuration_items: CDESK could not resolve maingroup_id="
        f"{maingroup_id!r} and rejected the request. Only a NUMERIC maingroup id "
        "reliably works here — get one from list_cmdb_maingroups (its `id`), or "
        "skip the hierarchy and filter by type_id from list_cmdb_types. A "
        "non-numeric value is matched against CMDB *type* slugs (not maingroup "
        "slugs) and most tenants define none, and a comma-separated list is not "
        "supported — one value only."
    )


def register_cmdb_tools(
    mcp: FastMCP,
    client: CdeskClient,
) -> None:
    """Register all CMDB-module tools on the given FastMCP instance."""

    @mcp.tool(
        description=_LIST_CIS_DESC + "\n\n" + sort_note("configuration_items") + "\n\n" + TEXT_MATCH_NOTE,
        annotations=ToolAnnotations(title="List configuration items", readOnlyHint=True),
    )
    async def list_configuration_items(
        text_search: str | None = None,
        category_id: int | None = None,
        maingroup_id: str | None = None,
        type_id: int | None = None,
        company_id: int | None = None,
        status_id: int | None = None,
        owner_id: int | None = None,
        include_deleted: bool = False,
        sb_raw: str | dict[str, Any] | None = None,
        page: int = 1,
        per_page: int = 20,
        sort: str | None = None,
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
            sb = build_ci_filter(
                text_search=text_search,
                status_id=status_id,
                owner_id=owner_id,
                sb_raw=sb_raw,
                dropped_cols_out=dropped_clauses,
            )
            validate_fields(fields)
            params: dict[str, Any] = {"pg": page, "pp": per_page}
            if fields:
                params["returnFields[]"] = fields
            # Hierarchy/company/deleted filters are query params on the
            # endpoint — NOT sb columns (per the CmdbCiSbFilterFields note).
            if category_id is not None:
                params["categoryId"] = category_id
            if maingroup_id is not None:
                params["maingroupId"] = maingroup_id
            if type_id is not None:
                params["typeId"] = type_id
            if company_id is not None:
                params["company"] = company_id
            if include_deleted:
                # withTrashed takes true / 1 / yes; "true" is sent.
                params["withTrashed"] = "true"
            if sort:
                # column / column.desc, checked against the CI sort allow-list
                # (sort_columns.py).
                params["sort"] = resolve_sort("configuration_items", sort)
            sb = await resolve_name_searches(client, sb, {
                "company_id": CUSTOMER_NAMES, "created_by": USER_NAMES, "updated_by": USER_NAMES,
            }, never_match=_NEVER_MATCH)
            sb, local = split_conditions(adapt_sb_for_cdesk(sb, "configuration_items"), _ci_condition)
            sb, owner_conditions = _split_owner_conditions(sb)
            if sb is not None:
                params["sb"] = encode_sb(sb)
            owner_info: dict[str, Any] | None = None
            if owner_conditions:
                records, meta, paging, owner_info = await _owner_filtered_page(
                    client, params, owner_conditions, local,
                    page=page, per_page=per_page, fields=fields,
                )
            elif local:
                records, meta, paging, owner_info = await filtered_page(
                    client, "v3/cmdb/ci", params, local, page=page, per_page=per_page,
                    fields=fields, cap=_LOCAL_FILTER_CAP, narrow_hint=_CI_NARROW_HINT,
                    note="CDESK reads these conditions as equality; the connector "
                         "fetched every CI matching the rest of the filter and "
                         "applied them itself.",
                )
            else:
                response, paging = await client.get_page("v3/cmdb/ci", params=params)
                records, meta = unwrap_list(response)
        except ValueError as e:
            raise RuntimeError(f"list_configuration_items input error: {e}") from e
        except Exception as e:
            hint = _unresolvable_maingroup_hint(e, maingroup_id)
            if hint:
                raise RuntimeError(hint) from e
            raise to_llm_error(e, operation="list_configuration_items") from e

        await _label_cis(client, records)
        result = {"items": records, "meta": list_meta(meta, paging, dropped=dropped_clauses), "page": page, "per_page": per_page}
        if owner_info is not None:
            result["filtered_by_connector"] = owner_info
        if dropped_clauses:
            result["unsupported_filters"] = unsupported_filter_directive(dropped_clauses)
        return result

    @mcp.tool(
        description=_GET_CI_DESC,
        annotations=ToolAnnotations(title="Get configuration item", readOnlyHint=True),
    )
    async def get_configuration_item(
        id: int,
        fields: list[str] | None = None,
    ) -> Any:
        try:
            validate_fields(fields)
            params = {"returnFields[]": fields} if fields else None
            response = await client.get(f"v3/cmdb/ci/{id}", params=params)
        except ValueError as e:
            raise RuntimeError(f"get_configuration_item input error: {e}") from e
        except Exception as e:
            raise to_llm_error(
                e, operation="get_configuration_item", record_id=id,
            ) from e
        record = unwrap_record(response)
        _blank_read_time_purchase_date([record])
        await _label_cis(client, [record])
        return record

    @mcp.tool(
        description=_GET_CI_ENUMS_DESC,
        annotations=ToolAnnotations(title="Get CI enums", readOnlyHint=True),
    )
    async def get_ci_enums(include_inactive: bool = False) -> Any:
        try:
            # CDESK quirk: ANY non-empty `all` value (even "0"/"false")
            # includes inactive statuses — so the param must be omitted
            # entirely unless the caller really wants them.
            params = {"all": "1"} if include_inactive else None
            response = await client.get("v3/cmdb/ci/enums", params=params)
        except Exception as e:
            raise to_llm_error(e, operation="get_ci_enums") from e
        # Returns a bare list of status rows, so it needs the same wrapping as
        # the other collection endpoints: one content block per item otherwise,
        # and ZERO blocks on a tenant with no CI statuses defined.
        return wrap_collection(
            unwrap_record(response),
            kind="configuration-item status enums",
        )

    @mcp.tool(
        description=_LIST_CATEGORIES_DESC,
        annotations=ToolAnnotations(title="List CMDB categories", readOnlyHint=True),
    )
    async def list_cmdb_categories() -> dict[str, Any]:
        try:
            response = await client.get("v3/cmdb/categories")
        except Exception as e:
            raise to_llm_error(e, operation="list_cmdb_categories") from e
        records, meta = unwrap_list(response)
        return {"items": records, "meta": meta}

    @mcp.tool(
        description=_GET_MAINGROUP_DESC,
        annotations=ToolAnnotations(title="Get CMDB maingroup", readOnlyHint=True),
    )
    async def get_cmdb_maingroup(
        id: int,
        fields: list[str] | None = None,
    ) -> Any:
        try:
            validate_fields(fields)
            params = {"returnFields[]": fields} if fields else None
            response = await client.get(f"v3/cmdb/maingroups/{id}", params=params)
        except ValueError as e:
            raise RuntimeError(f"get_cmdb_maingroup input error: {e}") from e
        except Exception as e:
            raise to_llm_error(
                e, operation="get_cmdb_maingroup", record_id=id,
            ) from e
        return unwrap_record(response)

    @mcp.tool(
        description=_GET_TYPE_DESC,
        annotations=ToolAnnotations(title="Get CMDB type", readOnlyHint=True),
    )
    async def get_cmdb_type(
        id: int,
        fields: list[str] | None = None,
    ) -> Any:
        try:
            validate_fields(fields)
            params = {"returnFields[]": fields} if fields else None
            response = await client.get(f"v3/cmdb/types/{id}", params=params)
        except ValueError as e:
            raise RuntimeError(f"get_cmdb_type input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="get_cmdb_type", record_id=id) from e
        return unwrap_record(response)

    @mcp.tool(
        description=_LIST_MAINGROUPS_DESC,
        annotations=ToolAnnotations(title="List CMDB maingroups", readOnlyHint=True),
    )
    async def list_cmdb_maingroups(
        page: int = 1,
        per_page: int = 20,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        # NO category_id parameter, deliberately. GET /v3/cmdb/maingroups
        # documents `categoryId` but ignores it (the v1 filter it delegates to
        # honours only q / active_only / id / changedFrom / exceptId), so the
        # query runs UNFILTERED behind a 200 under every spelling of the
        # parameter.
        #
        # Accepting the parameter and passing it on would report an
        # unfiltered list as filtered, and there is no client-side rescue:
        # maingroups carry category_id=null while their types carry the real
        # category, so post-filtering on the field yields a FALSE EMPTY.
        # list_cmdb_types(category_id=...) is the working path (see the
        # description).
        try:
            if per_page < 1 or per_page > _MAX_PER_PAGE:
                raise ValueError(
                    f"per_page must be between 1 and {_MAX_PER_PAGE} (got {per_page})"
                )
            if page < 1:
                raise ValueError(f"page must be 1 or greater (got {page})")
            validate_fields(fields)
            params: dict[str, Any] = {"pg": page, "pp": per_page}
            if fields:
                params["returnFields[]"] = fields
            response, paging = await client.get_page("v3/cmdb/maingroups", params=params)
        except ValueError as e:
            raise RuntimeError(f"list_cmdb_maingroups input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_cmdb_maingroups") from e
        records, meta = unwrap_list(response)
        return {"items": records, "meta": list_meta(meta, paging, dropped=None), "page": page, "per_page": per_page}

    @mcp.tool(
        description=_LIST_TYPES_DESC,
        annotations=ToolAnnotations(title="List CMDB types", readOnlyHint=True),
    )
    async def list_cmdb_types(
        category_id: int | None = None,
        maingroup_id: str | None = None,
        page: int = 1,
        per_page: int = 20,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            if per_page < 1 or per_page > _MAX_PER_PAGE:
                raise ValueError(
                    f"per_page must be between 1 and {_MAX_PER_PAGE} (got {per_page})"
                )
            if page < 1:
                raise ValueError(f"page must be 1 or greater (got {page})")
            validate_fields(fields)
            params: dict[str, Any] = {"pg": page, "pp": per_page}
            if fields:
                params["returnFields[]"] = fields
            if category_id is not None:
                params["categoryId"] = category_id
            if maingroup_id is not None:
                params["maingroupId"] = maingroup_id
            response, paging = await client.get_page("v3/cmdb/types", params=params)
        except ValueError as e:
            raise RuntimeError(f"list_cmdb_types input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_cmdb_types") from e
        records, meta = unwrap_list(response)
        return {"items": records, "meta": list_meta(meta, paging, dropped=None), "page": page, "per_page": per_page}

    @mcp.tool(
        description=_GET_TYPE_PROPERTIES_DESC,
        annotations=ToolAnnotations(title="Get CMDB type properties", readOnlyHint=True),
    )
    async def get_cmdb_type_properties(
        type_id: int,
        fields: list[str] | None = None,
    ) -> Any:
        try:
            validate_fields(fields)
            params = {"returnFields[]": fields} if fields else None
            response = await client.get(
                f"v3/cmdb/types/{type_id}/properties", params=params
            )
        except ValueError as e:
            raise RuntimeError(f"get_cmdb_type_properties input error: {e}") from e
        except Exception as e:
            raise to_llm_error(
                e, operation="get_cmdb_type_properties", record_id=type_id,
            ) from e
        return wrap_collection(
            unwrap_record(response),
            kind=f"dynamic property definitions on CMDB type {type_id}",
        )

# --- owner filter, applied by the connector -------------------------------

# CDESK's CI list leaves `owner_id` empty on every row and its owner filter
# reads that same empty column, so it never matches. Only the CI detail
# carries the owner: an owner condition is applied here, by reading each
# candidate CI's detail. One read per candidate is why the cap is low.
_OWNER_FILTER_CAP = 300
_LOCAL_FILTER_CAP = 2000
# A condition no record satisfies (`id = 0` is ignored by some endpoints).
_NEVER_MATCH = {"col": "id", "c": "=", "q": -1}
_CI_NARROW_HINT = "narrow the list first (type_id, company_id, status_id or text_search)"

# A range or `!=` on `id`, BETWEEN on `status_id` and a range on `company_id`
# or `type_id` are read as equality by CDESK, and BETWEEN or ISNOTNULL on
# `created_by` / `updated_by` match nothing, so the connector applies them
# (tools/_local_filter.py).
_ci_condition = numeric_picker({
    "id": ColumnRule(RANGE_CMPS | {NOT_EQUAL_CMP}),
    "status_id": ColumnRule(frozenset({"BETWEEN"})),
    "company_id": ColumnRule(RANGE_CMPS),
    "type_id": ColumnRule(RANGE_CMPS),
    "created_by": ColumnRule(frozenset({"BETWEEN", "ISNOTNULL"})),
    "updated_by": ColumnRule(frozenset({"BETWEEN", "ISNOTNULL"})),
})
_OWNER_READ_CONCURRENCY = 8
_OWNER_EQ_CMPS = frozenset({"=", "IS"})
_OWNER_NE_CMPS = frozenset({"!=", "ISNOT"})
_OWNER_NULL_CMPS = frozenset({"ISNULL", "ISNOTNULL"})


def _owner_condition(leaf: dict[str, Any]) -> dict[str, Any]:
    comparator = str(leaf.get("c", "")).upper()
    if comparator in _OWNER_NULL_CMPS:
        return {"c": comparator, "ids": frozenset()}
    if comparator not in _OWNER_EQ_CMPS | _OWNER_NE_CMPS:
        raise ValueError(
            f"owner_id supports '=', '!=', ISNULL and ISNOTNULL (got {leaf.get('c')!r})"
        )
    value = leaf.get("q")
    values = value if isinstance(value, list) else [value]
    ids = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, str)) or not str(v).strip().isdigit():
            raise ValueError(f"owner_id must be a user id or a list of user ids (got {value!r})")
        ids.append(int(v))
    return {"c": comparator, "ids": frozenset(ids)}


def _split_owner_conditions(
    sb: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Take the `owner_id` conditions out of `sb`. They can only be applied
    after the rest of the filter when joined to it by AND, so one inside an
    OR group is refused."""
    conditions: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], and_context: bool) -> dict[str, Any] | None:
        query = node.get("query")
        if not isinstance(query, list):
            return node
        elements = [e for e in query if isinstance(e, dict)]
        child_context = and_context and all(
            str(e.get("o", "AND")).upper() == "AND" for e in elements[1:]
        )
        kept: list[Any] = []
        for element in query:
            if not isinstance(element, dict):
                kept.append(element)
                continue
            if element.get("group") is True or isinstance(element.get("query"), list):
                sub = walk(element, child_context)
                if sub is not None:
                    kept.append(sub)
                continue
            if element.get("col") == "owner_id":
                if not child_context:
                    raise ValueError(
                        "an owner_id condition inside an OR group is not supported: "
                        "the owner is checked after the rest of the filter, which "
                        "only works when it is joined by AND"
                    )
                conditions.append(_owner_condition(element))
                continue
            kept.append(element)
        return {**node, "query": kept} if kept else None

    if sb is None:
        return None, []
    return walk(sb, True), conditions


def _owner_matches(owner: Any, conditions: list[dict[str, Any]]) -> bool:
    owner_id = owner if isinstance(owner, int) and not isinstance(owner, bool) else None
    for cond in conditions:
        c, ids = cond["c"], cond["ids"]
        if c == "ISNULL" and owner_id is not None:
            return False
        if c == "ISNOTNULL" and owner_id is None:
            return False
        if c in _OWNER_EQ_CMPS and owner_id not in ids:
            return False
        if c in _OWNER_NE_CMPS and owner_id in ids:
            return False
    return True


async def _ci_owners(client: CdeskClient, ci_ids: list[int]) -> dict[int, Any]:
    """The owner of each CI, read from its detail (the list leaves it empty)."""
    semaphore = asyncio.Semaphore(_OWNER_READ_CONCURRENCY)

    async def read(ci_id: int) -> tuple[int, Any]:
        async with semaphore:
            record = unwrap_record(await client.get(
                f"v3/cmdb/ci/{ci_id}", params={"returnFields[]": ["id", "owner_id"]}
            ))
        return ci_id, record.get("owner_id") if isinstance(record, dict) else None

    return dict(await asyncio.gather(*(read(i) for i in ci_ids)))


async def _owner_filtered_page(
    client: CdeskClient,
    params: dict[str, Any],
    conditions: list[dict[str, Any]],
    local: list[LocalCondition],
    *,
    page: int,
    per_page: int,
    fields: list[str] | None,
) -> tuple[list[Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The owner conditions, plus any other connector-applied conditions,
    which are checked first so fewer CI details are read."""
    rows, meta, added = await fetch_rows_with_columns(
        client, "v3/cmdb/ci", params, ["id", *(c.col for c in local)], fields=fields,
        id_col="id", cap=_OWNER_FILTER_CAP, narrow_hint=_CI_NARROW_HINT,
    )
    candidates = [
        r for r in rows
        if isinstance(r, dict) and all(c.test(r.get(c.col)) for c in local)
    ]
    ids = [r["id"] for r in candidates if isinstance(r.get("id"), int)]
    owners = await _ci_owners(client, ids)
    matched = []
    for row in candidates:
        ci_id = row.get("id")
        owner = owners.get(ci_id) if isinstance(ci_id, int) else None
        if not _owner_matches(owner, conditions):
            continue
        row = {k: v for k, v in row.items() if k not in added}
        if not fields or "owner_id" in fields:
            row["owner_id"] = owner
        matched.append(row)
    window, paging = page_of(matched, page, per_page)
    info = {
        "conditions": [
            *(c.label for c in local),
            *(f"owner_id {c['c']}" + (f" {sorted(c['ids'])}" if c["ids"] else "")
              for c in conditions),
        ],
        "matched": len(matched),
        "checked": len(rows),
        "note": "CDESK cannot filter CIs by owner; the connector read the owner "
                "of every CI matching the rest of the filter and applied the "
                "condition itself.",
    }
    return window, meta, paging, info




# A CI with no stored buy date is read back with `purchase_date` set to the
# server's current time (the empty value is cast to "now"). A purchase_date
# within this many seconds of the read is that artefact and is shown empty,
# like `guarantee_date`.
_READ_TIME_TOLERANCE_S = 300


def _blank_read_time_purchase_date(records: list[Any], now: datetime | None = None) -> None:
    current = now or datetime.now(timezone.utc)
    for record in records:
        if not isinstance(record, dict):
            continue
        raw = record.get("purchase_date")
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            value = datetime.fromisoformat(raw.strip().replace("Z", "+00:00").replace(" ", "T", 1))
        except ValueError:
            continue
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        if abs((value - current).total_seconds()) <= _READ_TIME_TOLERANCE_S:
            record["purchase_date"] = ""


async def _label_cis(client: CdeskClient, records: list[Any]) -> None:
    """Add status_name next to status_id from the live CI status rows
    (inactive included); best-effort."""
    targets = [r for r in records if isinstance(r, dict) and "status_id" in r]
    if not targets:
        return
    try:
        rows = unwrap_record(await client.get("v3/cmdb/ci/enums", params={"all": "1"}))
    except Exception:  # noqa: BLE001 — labels never fail the tool
        return
    apply_enum_names(targets, _CI_ENUM_LABELS, {
        ("status", "id"): enum_rows_name_map(rows, "id"),
    })


# --- Tool descriptions --------------------------------------------------

_LIST_CIS_DESC = inspect.cleandoc(
    """
    List CMDB configuration items (CIs) — the tenant's tracked objects
    (servers, printers, places, vehicles, ...). The practical way to
    list one kind of object is type_id (from list_cmdb_types); the
    broader hierarchy filters scope by category/maingroup.

    Filters:
      - text_search: substring match on CI name/description.
      - category_id / maingroup_id / type_id: scope to a level of the
        Category → Maingroup → Type hierarchy. Pass maingroup_id as a
        NUMERIC id (from list_cmdb_maingroups). The endpoint nominally
        also takes a slug, but a non-numeric value is matched against CMDB
        *type* slugs — not maingroup slugs — and when nothing matches the
        request FAILS with a 400 rather than returning an empty list
        (an unknown numeric id returns empty, as expected). One value
        only: a comma-separated list is not supported here.
      - company_id: CIs owned by one company (customer).
      - status_id: CI status enum id (resolve names via get_ci_enums).
      - owner_id: owner user id. CDESK cannot filter CIs by owner, so this
        tool reads the owner of every CI matching the other filters (up to
        300 — narrow by type_id, company_id or status_id beyond that) and
        keeps the matching ones; the result carries a `filtered_by_connector`
        block and the real owner_id on each row. In sb_raw, owner_id takes
        '=' / '!=' (a user id or a list) and ISNULL / ISNOTNULL, joined by AND.
      - include_deleted: also return soft-deleted CIs (default False).
      - sb_raw: an sb filter object for the CDESK API v3 CMDB CI list (JSON
        string or object, structured tree form; see docs/cdesk-api-v3.json);
        cannot be combined with the typed filters above
        (hierarchy/company/include_deleted excepted — those are query
        params, not sb). Only the working columns are
        applied — id, name, description, status_id, owner_id (applied by
        this tool, see above), company_id, type_id, created_at (strict W3C datetime
        values only), plus the no-col text leaf. Clauses on any other
        column are STRIPPED and reported in `unsupported_filters` for
        you to apply client-side after paging the full set.
        A range (>, >=, <, <=, BETWEEN) or a single-value '!=' on id,
        BETWEEN on status_id, a range on company_id or type_id, and
        BETWEEN or ISNOTNULL on created_by / updated_by are applied by this
        tool, because CDESK reads them as equality or matches nothing: it
        fetches every CI matching the rest of the filter (up to 2000) and
        keeps the matching ones, reported in `filtered_by_connector`; such
        a condition must be joined to the rest by AND. CONTAINS with a name
        on company_id (customer name) or created_by / updated_by (user name)
        is looked up by the tool and filtered by the matching ids.
        deleted_at is a date column like created_at (values need
        include_deleted to match). Date presets (TODAY, THIS_WEEK, …) are
        sent as the BETWEEN window they name, in tenant time. A list of
        values in '=' or '!=' matches any / none of them.

    NOT offered (the endpoint silently ignores it): parent_id
    (children-of). Fetch the candidate CIs and
    inspect their parent field client-side instead.

      - fields: exact whitelist of top-level field names to keep per CI
        record (returnFields; unknown names silently ignored).
      - sort: see the sort paragraph below.

    Returns paginated items + meta.
    """
)

_GET_CI_DESC = inspect.cleandoc(
    """
    Fetch a single configuration item by id. The detail includes the
    CI's resolved type-driven properties (the dynamic fields defined by
    its CMDB type).

    fields: an exact whitelist of top-level field names to return
    (CDESK's returnFields). Narrowing to the fields actually wanted keeps the
    response small; unknown names are silently ignored.
    """
)

_GET_CI_ENUMS_DESC = inspect.cleandoc(
    """
    Return the configuration-item status enums (id ↔ name), needed to
    resolve a status name into the status_id filter of
    list_configuration_items. Pass include_inactive=True to also list
    inactive statuses.

    Returns {items, count} — plus a `note` when count is 0, meaning no CI
    statuses are defined (an empty result, not a failure).

    Other CI pick-lists are type-scoped: categories/maingroups/types
    come from the list_cmdb_* tools, and per-type property option lists
    from get_cmdb_type_properties.
    """
)

_LIST_CATEGORIES_DESC = inspect.cleandoc(
    """
    List CMDB categories — the top level of the Category → Maingroup →
    Type → CI hierarchy. A small fixed catalog; not paginated.
    """
)

_GET_MAINGROUP_DESC = inspect.cleandoc(
    """
    Fetch a single CMDB maingroup by id (404 if it doesn't exist).
    fields: an exact whitelist of field names to return (returnFields).
    """
)

_GET_TYPE_DESC = inspect.cleandoc(
    """
    Fetch a single CMDB type by id — id, name, description, maingroup,
    category, slug (404 if it doesn't exist). For the dynamic property
    schema a CI of this type carries, call get_cmdb_type_properties.
    fields: an exact whitelist of field names to return (returnFields).
    """
)

_LIST_MAINGROUPS_DESC = inspect.cleandoc(
    """
    List CMDB maingroups — the second level of the hierarchy. Returns full
    maingroup records (no separate detail tool needed). fields: exact
    whitelist of field names to keep per record (returnFields).

    NOT offered: filtering by category. The live endpoint accepts a
    categoryId and ignores it, returning EVERY maingroup (an id, a bogus
    id, and every spelling of the parameter all return the same full set). Do not attempt it by another name, and do
    not filter the results on their `category_id` field either — that
    field is null on maingroups whose types do belong to a category, so
    filtering on it drops real matches.

    To get the maingroups of one category, call
    list_cmdb_types(category_id=...) — that filter IS honored — and take
    the distinct maingroup_id / maingroup_name_with_path values off the
    returned types.
    """
)

_LIST_TYPES_DESC = inspect.cleandoc(
    """
    List CMDB types — the schema-defining level of the hierarchy. A
    type is what makes a CI a 'server', 'printer', 'place', etc. Filter
    with category_id and/or maingroup_id (comma-separated ids allowed).
    For the dynamic property schema a type's CIs carry, call
    get_cmdb_type_properties. fields: exact whitelist of field names to
    keep per record (returnFields).
    """
)

_GET_TYPE_PROPERTIES_DESC = inspect.cleandoc(
    """
    Return the dynamic property definitions a CI of the given type
    carries — the field schema (names, kinds, option lists) you need
    before interpreting the properties on a CI detail.

    Returns {items, count}, plus a `note` when count is 0. An empty result
    is NOT ambiguous: a nonexistent type_id comes back as a clean not-found
    error, so count=0 means the type exists and
    carries no dynamic properties.
    """
)
