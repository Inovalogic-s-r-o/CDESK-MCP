"""Knowledge Base module MCP tools — CDESK KB articles.

Eight tools: article CRUD (list / get / get_enums / create / update / delete)
over the v3 endpoints, plus two thin v1-backed discovery tools
(list_kb_spaces / list_kb_categories) — spaces and categories are NOT exposed
via v3, but `space_id` is mandatory on create and `category_id` is one of only
two working list filters.

Cache-less: the enums envelope is `{data: {types, availableTags}}` (no `enums`
key, so EnumCache can't parse it) and `types` is the only resolvable bucket —
type_name is resolved inline against a fresh enums GET.

Notable backend behaviour:
  - the article list filters ONLY on `title` + `category_id`; even the
    universal no-col `q` text leaf is silently ignored;
  - articles are optimistically locked: a PUT must echo `timestamp_check`
    from a fresh GET, and a missing or stale token answers 409;
  - a no-change update returns HTTP 200 with a "no changes" warning body;
  - delete is a soft delete that cascades to the article's comments;
  - the list is silently narrowed to spaces the caller has record rights on.
"""

from __future__ import annotations

import inspect
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.filters import adapt_sb_for_cdesk, build_kb_article_filter, encode_sb
from cdesk_mcp.tools._local_filter import (
    ColumnRule,
    LocalCondition,
    filtered_page,
    null_condition,
    numeric_picker,
    split_conditions,
)
from cdesk_mcp.tools._helpers import (
    list_meta,
    TEXT_MATCH_NOTE,
    EnumLabel,
    apply_enum_names,
    enum_rows_name_map,
    annotate_write_warnings,
    apply_field_scope,
    canon_name,
    to_llm_error,
    unsupported_filter_directive,
    unwrap_list,
    unwrap_record,
    validate_fields,
    validate_fieldset,
    warned_fields,
    write_values_match,
)

# Pagination upper bound — mirrors the other modules' 100 cap.
_MAX_PER_PAGE = 100

# `type_id` holds the type's ACTION CODE (e.g. 1/2/3, with type_id 2 =
# "Poznatok"), not the enum id — see _resolve_type_name. The label
# comes from the tenant's live `types` rows.
_KB_ENUM_LABELS = (EnumLabel("type_id", "types", "type_name", by="action_code"),)

# Compact projection for the type enum rows (the raw rows carry ~30
# presentation fields).
_ENUM_ENTRY_KEYS = ("id", "name", "action_code", "parent_id", "lang_en", "lang_sk", "lang_cs")


def register_knowledgebase_tools(
    mcp: FastMCP,
    client: CdeskClient,
) -> None:
    """Register all Knowledge-Base-module tools on the given FastMCP
    instance. Takes no EnumCache — see the module docstring."""

    @mcp.tool(
        description=_LIST_KB_ARTICLES_DESC + "\n\n" + _SORT_NOTE + "\n\n" + TEXT_MATCH_NOTE,
        annotations=ToolAnnotations(title="List KB articles", readOnlyHint=True),
    )
    async def list_kb_articles(
        title: str | None = None,
        category_id: int | None = None,
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
            dropped_clauses: list[dict[str, Any]] = []
            sb = build_kb_article_filter(
                title=title,
                category_id=category_id,
                sb_raw=sb_raw,
                dropped_cols_out=dropped_clauses,
            )
            order = _parse_sort(sort) if sort else None
            sb, local = split_conditions(adapt_sb_for_cdesk(sb, "kb_articles"), _kb_condition)
            params: dict[str, Any] = {"pg": page, "pp": per_page}
            apply_field_scope(params, fieldset, fields)
            if sb is not None:
                params["sb"] = encode_sb(sb)
            sort_info: dict[str, Any] | None = None
            local_info: dict[str, Any] | None = None
            if order is not None:
                records, meta, paging, sort_info, local_info = await _sorted_page(
                    client, params, order, page=page, per_page=per_page,
                    fieldset=fieldset, fields=fields, conditions=local,
                )
            elif local:
                records, meta, paging, local_info = await filtered_page(
                    client, "v3/knowledgebase/article", params, local,
                    page=page, per_page=per_page, fields=fields, cap=_SORT_CAP,
                    narrow_hint="narrow the list with title or space first",
                    note=_LOCAL_FILTER_NOTE,
                )
            else:
                response, paging = await client.get_page("v3/knowledgebase/article", params=params)
                records, meta = unwrap_list(response)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"list_kb_articles input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_kb_articles") from e

        await _label_articles(client, records)
        result = {"items": records, "meta": list_meta(meta, paging, dropped=dropped_clauses), "page": page, "per_page": per_page}
        if sort_info is not None:
            result["sort"] = sort_info
        if local_info is not None:
            result["filtered_by_connector"] = local_info
        if dropped_clauses:
            result["unsupported_filters"] = unsupported_filter_directive(dropped_clauses)
        return result

    @mcp.tool(
        description=_GET_KB_ARTICLE_DESC,
        annotations=ToolAnnotations(title="Get KB article", readOnlyHint=True),
    )
    async def get_kb_article(
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
            response = await client.get(
                f"v3/knowledgebase/article/{id}", params=params or None
            )
        except ValueError as e:
            raise RuntimeError(f"get_kb_article input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="get_kb_article", record_id=id) from e
        record = unwrap_record(response)
        await _label_articles(client, [record])
        return record

    @mcp.tool(
        description=_GET_KB_ARTICLE_ENUMS_DESC,
        annotations=ToolAnnotations(title="Get KB article enums", readOnlyHint=True),
    )
    async def get_kb_article_enums() -> dict[str, Any]:
        try:
            payload = await _fetch_enums(client)
        except Exception as e:
            raise to_llm_error(e, operation="get_kb_article_enums") from e
        types = payload.get("types")
        slim_types = [
            {k: t[k] for k in _ENUM_ENTRY_KEYS if t.get(k) is not None}
            for t in types
            if isinstance(t, dict)
        ] if isinstance(types, list) else []
        return {
            "types": slim_types,
            "availableTags": payload.get("availableTags", []),
        }

    @mcp.tool(
        description=_CREATE_KB_ARTICLE_DESC,
        annotations=ToolAnnotations(title="Create KB article", destructiveHint=True),
    )
    async def create_kb_article(
        # Optional in the SIGNATURE only, so a missing space is answered with
        # _require_space's actionable message instead of a bare pydantic
        # "Field required ... errors.pydantic.dev/..." validation error, which
        # says nothing about how to find a space. Still mandatory in effect.
        space_id: int | None = None,
        title: str | None = None,
        type_id: int | None = None,
        type_name: str | None = None,
        description: str | None = None,
        public: bool | None = None,
        comments_enabled: bool | None = None,
        category_id: int = 0,
        tag_names: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            _require_space(space_id)
            if not title:
                raise ValueError("title is required")
            resolved_type_id = await _require_type(client, type_name, type_id)
            body: dict[str, Any] = {
                "space_id": space_id,
                "title": title,
                "type_id": resolved_type_id,
                # ALWAYS sent on create: the column is NOT NULL with no
                # default — 0 = "no category" and survives strict-SQL tenants.
                "category_id": category_id,
            }
            if description is not None:
                body["description"] = description
            # Always sent, so the visibility never depends on CDESK's default.
            body["visibility"] = 1 if public else 0
            if comments_enabled is not None:
                body["comments_enabled"] = comments_enabled
            if tag_names:
                body["tags"] = [{"name": n} for n in tag_names]
            response = await client.post("v3/knowledgebase/article", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"create_kb_article input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="create_kb_article") from e
        # The write answers {data: <id>, savedData}; the field check is the
        # read-back in _verify_write, and this call only passes CDESK's own
        # messages through.
        annotated = annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body={}, check_fields=(),
        )
        data = annotated.get("data")
        new_id = data.get("id") if isinstance(data, dict) else data
        result = await _verify_write(
            client, new_id, body, annotated, operation="create_kb_article",
            reported=warned_fields(response),
        )
        await _label_articles(client, [result.get("data")])
        return result

    @mcp.tool(
        description=_UPDATE_KB_ARTICLE_DESC,
        annotations=ToolAnnotations(title="Update KB article", destructiveHint=True),
    )
    async def update_kb_article(
        id: int,
        title: str | None = None,
        type_id: int | None = None,
        type_name: str | None = None,
        description: str | None = None,
        public: bool | None = None,
        comments_enabled: bool | None = None,
        space_id: int | None = None,
        category_id: int | None = None,
        tag_names: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            if type_name is not None and type_id is not None:
                raise ValueError("pass either type_name or type_id, not both")
            resolved_type_id = type_id
            if type_name is not None:
                resolved_type_id = await _resolve_type_name(client, type_name)
            body: dict[str, Any] = {}
            if title is not None:
                body["title"] = title
            if resolved_type_id is not None:
                body["type_id"] = resolved_type_id
            if description is not None:
                body["description"] = description
            if public is not None:
                body["visibility"] = 1 if public else 0
            if comments_enabled is not None:
                body["comments_enabled"] = comments_enabled
            if space_id is not None:
                body["space_id"] = space_id
            if category_id is not None:
                body["category_id"] = category_id
            if tag_names is not None:
                body["tags"] = [{"name": n} for n in tag_names]
            if not body:
                raise ValueError("nothing to update — pass at least one field")
            # Articles are optimistically locked: a PUT without the current
            # `timestamp_check` answers 409, so the token comes from a fresh read.
            current = unwrap_record(await client.get(
                f"v3/knowledgebase/article/{id}", params={"fieldset": "all"}
            ))
            token = current.get("timestamp_check") if isinstance(current, dict) else None
            if token:
                body["timestamp_check"] = token
            response = await client.put(f"v3/knowledgebase/article/{id}", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"update_kb_article input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="update_kb_article", record_id=id) from e
        annotated = annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body={}, check_fields=(),
        )
        result = await _verify_write(
            client, id, body, annotated, operation="update_kb_article",
            reported=warned_fields(response),
        )
        await _label_articles(client, [result.get("data")])
        return result

    @mcp.tool(
        description=_DELETE_KB_ARTICLE_DESC,
        annotations=ToolAnnotations(title="Delete KB article", destructiveHint=True),
    )
    async def delete_kb_article(id: int) -> dict[str, Any]:
        try:
            await client.delete(f"v3/knowledgebase/article/{id}")
        except Exception as e:
            raise to_llm_error(e, operation="delete_kb_article", record_id=id) from e
        return {"deleted": id}

    @mcp.tool(
        description=_LIST_KB_SPACES_DESC,
        annotations=ToolAnnotations(title="List KB spaces", readOnlyHint=True),
    )
    async def list_kb_spaces(per_page: int = 100) -> dict[str, Any]:
        try:
            if per_page < 1:
                raise ValueError(f"per_page must be positive (got {per_page})")
            await _require_kb_enabled(client)
            response = await client.get(
                "knowledgebase/space", params={"perpage": per_page}
            )
        except ValueError as e:
            raise RuntimeError(f"list_kb_spaces input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_kb_spaces") from e
        records, meta = unwrap_list(response)
        return {"items": records, "meta": meta}

    @mcp.tool(
        description=_LIST_KB_CATEGORIES_DESC,
        annotations=ToolAnnotations(title="List KB categories", readOnlyHint=True),
    )
    async def list_kb_categories(
        space_id: int | None = None,
        per_page: int = 100,
    ) -> dict[str, Any]:
        try:
            if per_page < 1:
                raise ValueError(f"per_page must be positive (got {per_page})")
            params: dict[str, Any] = {"perpage": per_page}
            if space_id is not None:
                params["space_id"] = space_id
            await _require_kb_enabled(client)
            response = await client.get("knowledgebase/category", params=params)
        except ValueError as e:
            raise RuntimeError(f"list_kb_categories input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_kb_categories") from e
        records, meta = unwrap_list(response)
        return {"items": records, "meta": meta}


# --- internals ----------------------------------------------------------

async def _require_kb_enabled(client: CdeskClient) -> None:
    """Raise CDESK's own refusal when the knowledge-base module is off. The
    spaces and categories are read from routes outside v3 that answer even
    with the module off; every v3 knowledge-base route is refused then (403,
    code 9), so the small article-enums call decides first."""
    await client.get("v3/knowledgebase/article/enums")

# CDESK ignores `sort` on the article list (always id ascending), so a sorted
# list is built here: every matching article is fetched, sorted, and the
# requested page is cut from the result. The cap bounds how many CDESK calls
# one sorted list may cost.
_SORT_COLUMNS = frozenset({
    "id", "number", "title", "type_id", "space_id", "category_id",
    "created_at", "updated_at",
})
_SORT_CAP = 2000
_SORT_FETCH_PAGE = 100
# What fieldset="base" returns; a sort column outside it is fetched with the
# default fieldset and the rows are trimmed back to these keys.
_BASE_KEYS = frozenset({
    "id", "number", "space_id", "timestamp_check", "title", "type_id", "visibility",
})


def _parse_sort(sort: str) -> tuple[str, bool]:
    """(column, descending) from "col", "col.asc", "col.desc", "-col" or
    "col desc"."""
    text = sort.strip()
    descending = False
    if text.startswith("-"):
        text, descending = text[1:].strip(), True
    else:
        for sep in (".", " "):
            head, _, tail = text.rpartition(sep)
            if head and tail.lower() in ("asc", "desc"):
                text, descending = head.strip(), tail.lower() == "desc"
                break
    if text not in _SORT_COLUMNS:
        raise ValueError(
            f"sort column {text!r} is not supported; valid: "
            f"{', '.join(sorted(_SORT_COLUMNS))}. Add .desc for descending, "
            f"e.g. title.desc"
        )
    return text, descending


def _sort_value(column: str, value: Any) -> tuple[int, Any]:
    """Comparable key: numbers before strings so mixed values never raise;
    titles compare case- and accent-insensitively."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, value)
    text = str(value)
    return (1, canon_name(text) if column == "title" else text)


# Conditions on `category_id` CDESK cannot apply, applied by the connector
# (tools/_local_filter.py): `= 0` (uncategorised), ISNULL and ISNOTNULL match
# every article, and BETWEEN matches only the uncategorised ones. A category
# of 0 means "no category".
_LOCAL_FILTER_NOTE = (
    "CDESK cannot apply these conditions; the connector fetched every article "
    "matching the rest of the filter and applied them itself."
)
_pick_kb_condition = numeric_picker({
    "category_id": ColumnRule(frozenset({"BETWEEN", "ISNULL", "ISNOTNULL"}), zero_is_empty=True),
})


def _kb_condition(leaf: dict[str, Any]) -> LocalCondition | None:
    value = leaf.get("q")
    if (leaf.get("col") == "category_id"
            and str(leaf.get("c", "")).upper() in ("=", "IS")
            and not isinstance(value, (list, bool)) and str(value).strip() == "0"):
        return null_condition({**leaf, "c": "ISNULL"}, zero_is_empty=True)
    return _pick_kb_condition(leaf)


async def _sorted_page(
    client: CdeskClient,
    params: dict[str, Any],
    order: tuple[str, bool],
    *,
    page: int,
    per_page: int,
    fieldset: str | None,
    fields: list[str] | None,
    conditions: list[LocalCondition],
) -> tuple[list[Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Fetch every article matching `params`, keep those passing the
    connector-applied `conditions`, sort by `order`, return the requested page
    as (records, meta, paging, sort_info, filter_info)."""
    column, descending = order
    fetch = {k: v for k, v in params.items() if k not in ("pg", "pp")}
    needed = list(dict.fromkeys([column, *(c.col for c in conditions)]))
    added: list[str] = []
    trim_to_base = False
    if fields:
        added = [col for col in needed if col not in fields]
        if added:
            fetch["returnFields[]"] = [*fields, *added]
    elif fieldset == "base" and any(col not in _BASE_KEYS for col in needed):
        fetch.pop("fieldset", None)
        trim_to_base = True

    rows: list[Any] = []
    meta: dict[str, Any] = {}
    pg = 1
    while True:
        response, paging = await client.get_page(
            "v3/knowledgebase/article", params={**fetch, "pg": pg, "pp": _SORT_FETCH_PAGE}
        )
        batch, batch_meta = unwrap_list(response)
        if pg == 1:
            meta = batch_meta
        total = paging.get("total")
        if isinstance(total, int) and total > _SORT_CAP:
            raise ValueError(
                f"sorting needs all {total} matching articles, more than the "
                f"{_SORT_CAP} this tool sorts; narrow the list with title or "
                f"category_id first, or leave sort out"
            )
        rows.extend(batch)
        if len(rows) > _SORT_CAP:
            raise ValueError(
                f"sorting needs more than {_SORT_CAP} matching articles; narrow the "
                f"list with title or category_id first, or leave sort out"
            )
        if len(batch) < _SORT_FETCH_PAGE or (isinstance(total, int) and len(rows) >= total):
            break
        pg += 1

    checked = len(rows)
    if conditions:
        rows = [r for r in rows if isinstance(r, dict) and all(c.test(r.get(c.col)) for c in conditions)]

    def missing(row: Any) -> bool:
        return not isinstance(row, dict) or row.get(column) in (None, "")

    present = [r for r in rows if not missing(r)]
    present.sort(key=lambda r: _sort_value(column, r[column]), reverse=descending)
    ordered = present + [r for r in rows if missing(r)]

    start = (page - 1) * per_page
    window = ordered[start:start + per_page]
    if added:
        drop = set(added)
        window = [{k: v for k, v in r.items() if k not in drop} if isinstance(r, dict) else r for r in window]
    elif trim_to_base:
        window = [{k: v for k, v in r.items() if k in _BASE_KEYS} if isinstance(r, dict) else r for r in window]
    count = len(ordered)
    page_info = {"total": count, "last_page": start + per_page >= count}
    sort_info = {
        "column": column,
        "descending": descending,
        "sorted_records": count,
        "note": "CDESK cannot sort this list; the connector fetched every "
                "matching article and sorted them (empty values last).",
    }
    filter_info = None
    if conditions:
        filter_info = {
            "conditions": [c.label for c in conditions],
            "matched": count,
            "checked": checked,
            "note": _LOCAL_FILTER_NOTE,
        }
    return window, meta, page_info, sort_info, filter_info

async def _label_articles(client: CdeskClient, records: list[Any]) -> None:
    """Add type_name next to type_id from a live enums read; best-effort."""
    targets = [r for r in records if isinstance(r, dict) and "type_id" in r]
    if not targets:
        return
    try:
        payload = await _fetch_enums(client)
    except Exception:  # noqa: BLE001 — labels never fail the tool
        return
    apply_enum_names(targets, _KB_ENUM_LABELS, {
        ("types", "action_code"): enum_rows_name_map(payload.get("types"), "action_code"),
    })

async def _fetch_enums(client: CdeskClient) -> dict[str, Any]:
    """GET the article enums. Envelope: {data: {types: [...],
    availableTags: [...]}} — the payload is nested under `data` (no `enums`
    key like other modules)."""
    response = await client.get("v3/knowledgebase/article/enums")
    payload = unwrap_record(response)
    return payload if isinstance(payload, dict) else {}


async def _resolve_type_name(client: CdeskClient, type_name: str) -> int:
    """Resolve an article type NAME → its ACTION CODE against a fresh enums
    GET, diacritic- and case-insensitively (incl. lang variants). Misses raise
    with the valid names so the LLM self-corrects in one turn.

    NOT the enum `id`: the article's `type_id` is a small int holding the
    action code (1 Známa chyba / 2 Poznatok / 3 Návod); CDESK refuses an enum
    id.
    """
    payload = await _fetch_enums(client)
    types = payload.get("types")
    entries = [t for t in types if isinstance(t, dict)] if isinstance(types, list) else []
    wanted = canon_name(type_name)
    names: list[str] = []
    for entry in entries:
        aliases = {entry.get(k) for k in ("name", "lang_name", "lang_en", "lang_sk", "lang_cs")}
        primary = entry.get("name")
        if isinstance(primary, str):
            names.append(primary)
        for alias in aliases:
            if isinstance(alias, str) and canon_name(alias) == wanted:
                action_code = entry.get("action_code")
                if isinstance(action_code, int) and not isinstance(action_code, bool):
                    return action_code
                raise RuntimeError(
                    f"Article type {type_name!r} has no action_code in this "
                    f"tenant's enums, so it cannot be stored correctly — pass "
                    f"type_id explicitly (see get_kb_article_enums)."
                )
    raise RuntimeError(
        f"Unknown article type {type_name!r}. Valid types on this tenant: "
        f"{', '.join(names) or '(none — check get_kb_article_enums)'}"
    )


def _require_space(space_id: int | None) -> None:
    """A space is mandatory on create. Checked here so a missing one points
    the model at list_kb_spaces rather than at a bare "Field required"."""
    if space_id is None:
        raise ValueError(
            "space_id is required — every knowledge-base article must live in "
            "a space. Call list_kb_spaces and pass one of its `id` values."
        )
    if space_id <= 0:
        raise ValueError(
            f"space_id must be a positive id from list_kb_spaces (got {space_id})"
        )


# Per-field reasons for a "sent but not stored" warning, so the message says
# what to do rather than just what happened.
_KB_FIELD_NOTES = {
    "type_id": (
        "the article stores the type's ACTION CODE (see get_kb_article_enums "
        "`action_code`)"
    ),
    "space_id": (
        "the space may not exist, or this account may lack record rights on it"
    ),
    "category_id": "the category may not exist, or may belong to another space",
}


# Body field -> the column to re-read it from. `description` and `tags` are
# absent on purpose: the server rewrites embedded file links in the HTML, and
# tags go out as [{name}] but come back as [{id, name}] — neither compares
# literally, so including them would raise FALSE "not stored" alarms.
_READ_BACK_FIELDS: tuple[str, ...] = (
    "title", "type_id", "visibility", "space_id", "category_id",
    "comments_enabled",
)

_PARAM_NAMES: dict[str, str] = {"visibility": "public", "type_id": "type"}


async def _verify_write(
    client: CdeskClient,
    article_id: Any,
    sent_body: dict[str, Any],
    response: dict[str, Any],
    *,
    operation: str,
    reported: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Re-READ the article and warn about anything sent but not stored.

    The write response is not a reliable record of what was stored, so the
    check reads the article back; it costs one extra GET per write. Fields in
    `reported` are already named in CDESK's own warnings and are not repeated.
    """
    if not isinstance(article_id, int):
        return response
    try:
        envelope = await client.get(f"v3/knowledgebase/article/{article_id}")
        stored = unwrap_record(envelope)
    except Exception:  # noqa: BLE001 - verification must never fail the write
        return {
            **response,
            "write_verification": (
                f"The write succeeded, but {operation} could not re-read article "
                f"{article_id} afterwards, so field persistence was NOT "
                f"verified. Do not state that a specific value was stored — "
                f"confirm with get_kb_article."
            ),
        }
    if not isinstance(stored, dict):
        return response
    dropped: list[str] = []
    for column in _READ_BACK_FIELDS:
        if column not in sent_body or column in reported:
            continue
        sent = sent_body[column]
        if sent is None or sent == "":
            continue
        if not write_values_match(sent, stored.get(column)):
            label = _PARAM_NAMES.get(column, column)
            note = _KB_FIELD_NOTES.get(column)
            dropped.append(
                f"{label} (sent {sent!r}, stored {stored.get(column)!r}"
                + (f" — {note}" if note else "") + ")"
            )
    if not dropped:
        return response
    warnings = list(response.get("warnings") or [])
    warnings.append(
        "CDESK accepted the write but did NOT store: " + "; ".join(dropped)
        + ". Verified by re-reading the article. Do not tell the user these "
        "values were saved."
    )
    return {**response, "warnings": warnings}


async def _require_type(
    client: CdeskClient, type_name: str | None, type_id: int | None
) -> int:
    """Create requires an article type — enforce client-side with a clear
    message (CDESK barely validates; DB constraints reject late)."""
    if type_name is not None and type_id is not None:
        raise ValueError("pass either type_name or type_id, not both")
    if type_id is not None:
        return type_id
    if type_name is None:
        raise ValueError(
            "an article type is required — pass type_name (see "
            "get_kb_article_enums 'types') or type_id"
        )
    return await _resolve_type_name(client, type_name)


# --- tool descriptions --------------------------------------------------

_LIST_KB_ARTICLES_DESC = inspect.cleandoc(
    """
    List CDESK knowledge-base articles. Returns a paginated `items` array.
    The server silently narrows the list to articles in KB SPACES the calling
    account has record rights on — a short list may mean limited rights.

    IMPORTANT — this endpoint supports ONLY TWO server-side filters:
      - title: article title WORD search — the value is split into words and
        each word is substring-matched independently ("manual
        A" also matches "manual B" because "A" ⊂ "mAnual"). For an exact
        phrase, filter the returned items client-side.
      - category_id: exact category (positive id; see list_kb_categories)
    There is NO full-text/body search and NO other working column (not even
    the universal no-col text leaf) — to filter by anything else (body text,
    type, space, visibility, dates), page through the full set and filter
    client-side. sb_raw takes a CDESK sb filter object for the CDESK API v3
    knowledge-base article list (JSON string or object, structured tree form;
    see docs/cdesk-api-v3.json); clauses on other columns are stripped and
    reported in `unsupported_filters`. In sb_raw, category_id takes '=', a
    list (any of), '!=' and the ranges; `= 0` (uncategorised), ISNULL,
    ISNOTNULL and BETWEEN are applied by this tool, because CDESK ignores
    them: it fetches every article matching the rest of the filter (up to
    2000) and keeps the matching ones, reported in `filtered_by_connector`.
    CONTAINS on category_id is refused.

    NOTE: fieldset="base" omits the article body; the body (`description`,
    HTML) is in the default "extended" set.
    """
)

_SORT_NOTE = inspect.cleandoc(
    f"""
    sort: ONE column, ascending ("title"), or descending with a .desc suffix
    ("title.desc"; "-title" is read the same). Columns: id, number, title,
    type_id, space_id, category_id, created_at, updated_at. CDESK cannot sort
    this list, so with sort the tool fetches every matching article and sorts
    them itself (titles case- and accent-insensitively, empty values last); the
    result carries a `sort` block saying so. Up to {_SORT_CAP} matching articles
    — narrow with title or category_id beyond that. Without sort the list is in
    id order.
    """
)

_GET_KB_ARTICLE_DESC = inspect.cleandoc(
    """
    Fetch a single knowledge-base article by id. `description` is the article
    body as rich HTML — embedded file links carry server-injected auth hashes
    (so the returned HTML differs from what was stored). fieldset "base"
    omits the body; "extended" (default) includes it.
    """
)

_GET_KB_ARTICLE_ENUMS_DESC = inspect.cleandoc(
    """
    Return the knowledge-base article enums: `types` (tenant article-type
    enum — use an entry's **action_code** as type_id, not its `id` (CDESK
    refuses the id). Passing the name via
    create/update_kb_article's type_name resolves this for you) and `availableTags` (existing
    keyword tags, {id, name}).
    """
)

_CREATE_KB_ARTICLE_DESC = inspect.cleandoc(
    """
    Create a knowledge-base article. Required: space_id (pick via
    list_kb_spaces — the article MUST belong to a space the account has
    record rights on), title, and an article type (type_name or type_id).

    `description` is the article body — rich HTML is allowed. public=true
    makes the article public (visibility 1); public=false or omitted creates
    an INTERNAL article. Ask the user which they want rather than relying on
    the default. category_id is optional (0 = no category; see
    list_kb_categories for the space's categories). tag_names attaches keyword
    tags by name; each name is stored as a tag (a name not yet in
    availableTags creates a new tag). The article number is
    auto-assigned; the creator auto-follows the article and notifications fire
    server-side.

    ARTICLE TYPE: type_name is the simplest route — it resolves to the type's
    action code, which is what the article stores. A type_id must also be the
    action code (1/2/3 — see get_kb_article_enums), not the enum id; CDESK
    refuses anything else. space_id must be a space from list_kb_spaces;
    CDESK refuses any other.
    """
)

_UPDATE_KB_ARTICLE_DESC = inspect.cleandoc(
    """
    Update a knowledge-base article (partial — send only the fields to
    change). Optimistic locking is enforced and handled internally (GET → PUT
    with the version token), so an edit made by someone else in between
    conflicts with HTTP 409 instead of being overwritten. Passing space_id
    MOVES the article to another space (requires
    record rights there). tag_names REPLACES the tag set (unknown names are
    created; tags no longer used are removed). Leaving tag_names out keeps
    the article's current tags.

    NOTE: an update that changes nothing still returns success with a "no
    changes" warning from the server ("Žiadne zmeny na uloženie").
    """
)

_DELETE_KB_ARTICLE_DESC = inspect.cleandoc(
    """
    Delete (soft-delete) a knowledge-base article by id. The article's
    comments are soft-deleted with it; there are no dependency blockers.
    Returns {"deleted": <id>}.
    """
)

_LIST_KB_SPACES_DESC = inspect.cleandoc(
    """
    List knowledge-base SPACES (v1-backed helper — spaces are not exposed via
    the v3 API). A space is the mandatory parent container of every article:
    use an item's id as create_kb_article's space_id. Only spaces the account
    has record rights on are returned. Refused, like every knowledge-base
    tool, while the knowledge-base module is switched off.
    """
)

_LIST_KB_CATEGORIES_DESC = inspect.cleandoc(
    """
    List knowledge-base CATEGORIES (v1-backed helper). Categories are flat
    per-space lists (no hierarchy); optionally narrow by space_id. Use an
    item's id as category_id in create/update_kb_article or as the
    list_kb_articles category filter. Refused, like every knowledge-base
    tool, while the knowledge-base module is switched off.
    """
)
