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

import inspect
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.filters import build_ci_filter, encode_sb
from cdesk_mcp.tools._helpers import (
    to_llm_error,
    unsupported_filter_directive,
    unwrap_list,
    unwrap_record,
    wrap_collection,
    validate_fields,
)

_MAX_PER_PAGE = 100


def _unresolvable_maingroup_hint(exc: Exception, maingroup_id: str | None) -> str | None:
    """Actionable text for the one 400 this endpoint answers with an empty body.

    GET /v3/cmdb/ci resolves a non-numeric `maingroupId` as a slug and, finding
    nothing, returns `400 {"data": false}` (apiportal CIController.php:890-902) —
    a body the generic translator can only echo, leaving the model with "Bad
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
        description=_LIST_CIS_DESC,
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
                params["withTrashed"] = "1"
            if sort:
                # A BARE column name is the only shape that works here, and it
                # sorts ascending. The v1 handler actually reads sort as an array
                # of [column, direction] pairs (apiportal CIController.php:1095),
                # but sending that shape (sort[0][0]=name) returns HTTP 500 — so
                # a plain string it is, and direction is not selectable. The
                # column is checked against CI::getOrderColsAllowed
                # (Model/CMDB/CI.php:4923); anything outside it emits NO order by
                # at all (BaseModel.php:791-806), i.e. an arbitrary order rather
                # than an error. Live-verified 2026-07-30 — do not restore the
                # old "sort has no observable effect" claim without a retest.
                params["sort"] = sort
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response = await client.get("v3/cmdb/ci", params=params)
        except ValueError as e:
            raise RuntimeError(f"list_configuration_items input error: {e}") from e
        except Exception as e:
            hint = _unresolvable_maingroup_hint(e, maingroup_id)
            if hint:
                raise RuntimeError(hint) from e
            raise to_llm_error(e, operation="list_configuration_items") from e

        records, meta = unwrap_list(response)
        result = {"items": records, "meta": meta, "page": page, "per_page": per_page}
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
        return unwrap_record(response)

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
        # documents `categoryId` but never implemented it: the v3 wrapper
        # forwards the query to the v1 MaingroupController, and
        # Maingroup::whereFilter (apiportal Model/CMDB/Maingroup.php:478-512)
        # has no categoryId branch at all — it honors only q / active_only /
        # id / changedFrom / exceptId. The value is read into
        # maingroupFilterParams (:785) and dropped, so the query runs
        # UNFILTERED behind a 200. Live-confirmed 2026-07-30: categoryId=3,
        # categoryId=99999, category_id, category and categoryIds every one
        # returned the identical full set.
        #
        # Accepting the parameter and passing it on would report an
        # unfiltered list as filtered, and there is no client-side rescue:
        # maingroups carry category_id=null while their types carry the real
        # category, so post-filtering on the field yields a FALSE EMPTY.
        # list_cmdb_types(category_id=...) is the working path (see the
        # description). Re-add this only against a live discrimination test.
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
            response = await client.get("v3/cmdb/maingroups", params=params)
        except ValueError as e:
            raise RuntimeError(f"list_cmdb_maingroups input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_cmdb_maingroups") from e
        records, meta = unwrap_list(response)
        return {"items": records, "meta": meta, "page": page, "per_page": per_page}

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
            response = await client.get("v3/cmdb/types", params=params)
        except ValueError as e:
            raise RuntimeError(f"list_cmdb_types input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_cmdb_types") from e
        records, meta = unwrap_list(response)
        return {"items": records, "meta": meta, "page": page, "per_page": per_page}

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
      - owner_id: owner user id.
      - include_deleted: also return soft-deleted CIs (default False).
      - sb_raw: an sb filter object for the CDESK API v3 CMDB CI list (JSON
        string or object, structured tree form; see docs/cdesk-api-v3.json);
        cannot be combined with the typed filters above
        (hierarchy/company/include_deleted excepted — those are query
        params, not sb). Only the LIVE-VERIFIED working columns are
        applied server-side — id, name, description, status_id,
        owner_id, company_id, type_id, created_at (strict W3C datetime
        values only), plus the no-col text leaf. Clauses on any other
        column are STRIPPED and reported in `unsupported_filters` for
        you to apply client-side after paging the full set.

    NOT offered (the live endpoint silently ignores it — verified
    2026-06-05): parent_id (children-of). Fetch the candidate CIs and
    inspect their parent field client-side instead.

      - fields: exact whitelist of top-level field names to keep per CI
        record (returnFields; unknown names silently ignored).
      - sort: ONE column name, ascending, e.g. sort="name". Default order
        without it is newest-updated first.
        Accepted columns, per the backend's own sort allowlist (name and
        status_id live-verified 2026-07-30, the rest read from it): id,
        name, description, status_id, created_at, updated_at,
        creator_name, updated_by_name, and the joined-name columns
        company_name, ci_type_name, maingroup_name, owner_name,
        place_name, branch_name (plus the asset-management columns
        purchase_date, assignment_date, inventory_number, cost, supplier).
        Three limits, all live-verified 2026-07-30:
          * ASCENDING ONLY. There is no descending form — "-name" is not
            "name desc", it is an unrecognized column (see below). To get
            a descending order, page the set and reverse it yourself.
          * An unrecognized column is SILENTLY IGNORED and the rows come
            back in NO defined order — not the default one. So the raw FK
            columns company_id, type_id and owner_id do nothing here
            (sort by company_name / ci_type_name / owner_name instead),
            and a typo yields an arbitrary order that looks sorted. If the
            order matters for correctness, verify it in the items rather
            than trusting the parameter.
          * Pass ONE bare column name. Do not send an indexed/array form
            (sort[0][0]=...): the endpoint answers HTTP 500 to it.

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
    categoryId and ignores it, returning EVERY maingroup (verified
    2026-07-30: an id, a bogus id, and every spelling of the parameter all
    return the same full set). Do not attempt it by another name, and do
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
    error (verified live 2026-07-29), so count=0 means the type exists and
    carries no dynamic properties.
    """
)
