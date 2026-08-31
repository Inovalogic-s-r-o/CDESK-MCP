"""Task module MCP tools (M7).

Nine tools that expose CDESK's Task v3 CRUD + intent helpers to the LLM.
Wires together CdeskClient (M3), EnumCache (M5), filter builder (M6), and
error translator (M4).

Optimistic locking is hidden: update_task, set_task_status, and the watcher
intent helpers internally do GET → mutate → PUT (re-using the `timestamp_check`
token from the GET response). The LLM never sees the version token.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.enums import EnumCache
from cdesk_mcp.filters import build_task_filter, encode_sb, reject_sb_raw_with_typed
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
)
from cdesk_mcp.tz import localize_naive_datetime, tenant_timezone

# LLM-friendly param name → CDESK field name. Centralised so the LLM
# vocabulary stays clean while the CDESK quirks live in one place.
_TASK_CREATE_FIELDS = {
    "watcher_ids": "task_watchers_ids",
    "tag_ids": "selectedTags",
    "other_solver_ids": "other_solvers_id",  # CDESK uses singular `id` for an array
    "parent_task_id": "task_id",
    "customer_id": "company_id",
    # The deal link: the tool param is `deal_id` (CDESK calls the module
    # Zákazky) but the CDESK body field is still `contract_id`.
    "deal_id": "contract_id",
    "percent_done": "percent_done_manual",
    "duration_gap": "durationGap",  # CDESK camelCase (update-only, not stored)
}

# Pagination upper bound for list_tasks. Architecture.md set 100 as the cap
# to keep response sizes within an LLM's context budget. CDESK itself
# doesn't refuse larger values, so this is purely a defensive guard.
_MAX_PER_PAGE = 100


def register_task_tools(
    mcp: FastMCP,
    client: CdeskClient,
    cache: EnumCache,
) -> None:
    """Register all Task-module tools on the given FastMCP instance. Called
    by build_server when client + cache are both available."""

    @mcp.tool(
        description=_LIST_TASKS_DESC,
        annotations=ToolAnnotations(title="List tasks", readOnlyHint=True),
    )
    async def list_tasks(
        text_search: str | None = None,
        status_name: str | None = None,
        type_name: str | None = None,
        solver_id: int | list[int] | None = None,
        customer_id: int | list[int] | None = None,
        valid_from_after: str | None = None,
        valid_from_before: str | None = None,
        deadline_after: str | None = None,
        deadline_before: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
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
            # Before resolving enums: a name that fails to resolve would report
            # the typo first and hide the fact that sb_raw can't be combined
            # with typed filters at all, costing the agent an extra turn.
            reject_sb_raw_with_typed(sb_raw, {
                "text_search": text_search,
                "status_name": status_name,
                "type_name": type_name,
                "solver_id": solver_id,
                "customer_id": customer_id,
                "valid_from_after": valid_from_after,
                "valid_from_before": valid_from_before,
                "deadline_after": deadline_after,
                "deadline_before": deadline_before,
                "created_after": created_after,
                "created_before": created_before,
            })
            status_id = await resolve_enum_or_raise(cache, "status", status_name, kind="status")
            type_id = await resolve_enum_or_raise(cache, "type", type_name, kind="type")
            # sb_raw clauses on columns the live endpoint doesn't honor are
            # stripped (not forwarded — the backend would silently ignore them
            # and return the unfiltered set anyway) and reported back via
            # `unsupported_filters` so the agent filters the items itself.
            dropped_clauses: list[dict[str, Any]] = []
            sb = build_task_filter(
                text_search=text_search,
                status_id=status_id,
                type_id=type_id,
                solver_id=solver_id,
                customer_id=customer_id,
                valid_from_after=valid_from_after,
                valid_from_before=valid_from_before,
                deadline_after=deadline_after,
                deadline_before=deadline_before,
                created_after=created_after,
                created_before=created_before,
                sb_raw=sb_raw,
                dropped_cols_out=dropped_clauses,
            )
            params: dict[str, Any] = {"pg": page, "pp": per_page}
            apply_field_scope(params, fieldset, fields)
            if sort:
                params["sort"] = sort
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response = await client.get("v3/task", params=params)
        except RuntimeError:
            raise  # already a friendly message (from resolve_enum_or_raise)
        except ValueError as e:
            raise RuntimeError(f"list_tasks input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_tasks") from e

        records, meta = unwrap_list(response)
        result = {"items": records, "meta": meta, "page": page, "per_page": per_page}
        if dropped_clauses:
            result["unsupported_filters"] = unsupported_filter_directive(dropped_clauses)
        return result

    @mcp.tool(
        description=_GET_TASK_DESC,
        annotations=ToolAnnotations(title="Get task", readOnlyHint=True),
    )
    async def get_task(
        id: int,
        fieldset: str | None = None,
        fields: list[str] | None = None,
    ) -> Any:
        try:
            # A non-positive id can't exist, and the backend 500s on it rather
            # than 404ing (live: id=-1 → HTTP 500, while id=0 and a huge id both
            # give a clean "record not found"). Without this guard the client
            # spends its whole 5xx backoff budget on provably bad input and then
            # tells the agent the service is down for 30s. id=0 is included: the
            # backend handles it, but no record can have it either.
            if id < 1:
                raise ValueError(f"id must be a positive record id (got {id})")
            validate_fieldset(fieldset)
            validate_fields(fields)
            params: dict[str, Any] = {}
            if fieldset:
                params["fieldset"] = fieldset
            if fields:
                params["returnFields[]"] = fields
            response = await client.get(f"v3/task/{id}", params=params or None)
        except ValueError as e:
            raise RuntimeError(f"get_task input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="get_task", record_id=id) from e
        return unwrap_record(response)

    @mcp.tool(
        description=_GET_TASK_ENUMS_DESC,
        annotations=ToolAnnotations(title="Get task enums", readOnlyHint=True),
    )
    async def get_task_enums(refresh: bool = False) -> dict[str, list[dict[str, Any]]]:
        try:
            if refresh:
                await cache.refresh()
            else:
                await cache.load()  # populate a cold cache (snapshot is empty otherwise)
        except Exception as e:
            raise to_llm_error(e, operation="get_task_enums") from e
        return cache.snapshot()

    @mcp.tool(
        description=_CREATE_TASK_DESC,
        annotations=ToolAnnotations(title="Create task", destructiveHint=True),
    )
    async def create_task(
        name: str,
        solver_id: int,
        valid_from: str,
        valid_to: str,
        description: str | None = None,
        status_name: str | None = None,
        type_name: str | None = None,
        type_2nd_name: str | None = None,
        place_name: str | None = None,
        customer_id: int | None = None,
        request_id: int | None = None,
        deal_id: int | None = None,
        project_contract_id: int | None = None,
        parent_task_id: int | None = None,
        fullday: bool = False,
        private: bool = False,
        percent_done: int = 0,
        scheduling_mode: int | None = None,
        other_solver_ids: list[int] | None = None,
        watcher_ids: list[int] | None = None,
        tag_ids: list[int] | None = None,
        offer_item_id: int | None = None,
        notify: dict[str, Any] | None = None,
        attachment: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            # Naive datetimes are interpreted as TENANT-LOCAL wall-clock time
            # and sent with an explicit offset: CDESK reads a naive value as
            # UTC (verified live 2026-06-05 — naive 8:00 displayed as 10:00 in
            # the CEST tenant UI), while an explicit offset round-trips
            # correctly. localize_naive_datetime also validates the ISO shape.
            # Via locals: localize_naive_datetime is str | None -> str | None (it
            # maps "" to None), so it can't be assigned straight back onto these
            # required str params. The guard below narrows them for the reassign.
            localized_from = localize_naive_datetime("valid_from", valid_from)
            localized_to = localize_naive_datetime("valid_to", valid_to)
            if not localized_from or not localized_to:
                raise ValueError("valid_from and valid_to are required (ISO-8601)")
            valid_from, valid_to = localized_from, localized_to
            _ensure_chronological(valid_from, valid_to)

            status_id = await resolve_enum_or_raise(cache, "status", status_name, kind="status")
            type_id = await resolve_enum_or_raise(cache, "type", type_name, kind="type")
            # type_2nd is hierarchical: it needs its parent type to resolve. Without
            # type_name, type_id is None and the 2nd-level would resolve unscoped
            # (wrong parent / AmbiguousEnumNameError), so require type_name with it.
            if type_2nd_name is not None and type_id is None:
                raise ValueError(
                    "provide type_name when setting type_2nd_name so the 2nd-level "
                    "type resolves under the right parent"
                )
            type_2nd_id = await resolve_enum_or_raise(
                cache, "type2nd", type_2nd_name, kind="type_2nd", parent_id=type_id
            )
            place_id = await resolve_enum_or_raise(cache, "place", place_name, kind="place")

            body = _build_task_body(
                {
                    "name": name,
                    "solver_id": solver_id,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "description": description,
                    "status": status_id,
                    "type_id": type_id,
                    "type_2nd_id": type_2nd_id,
                    "place_id": place_id,
                    "customer_id": customer_id,
                    "request_id": request_id,
                    "deal_id": deal_id,
                    "project_contract_id": project_contract_id,
                    "parent_task_id": parent_task_id,
                    "fullday": fullday,
                    "private": private,
                    # M7.3: pass 0 through (0 is a valid percent_done value).
                    # _build_task_body drops None but keeps 0.
                    "percent_done": percent_done,
                    "scheduling_mode": scheduling_mode,
                    "other_solver_ids": other_solver_ids,
                    "watcher_ids": watcher_ids,
                    "tag_ids": tag_ids,
                    "offer_item_id": offer_item_id,
                    "notify": notify,
                    "attachment": attachment,
                }
            )
            response = await client.post("v3/task", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"create_task input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="create_task") from e
        return annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=(),
        )

    @mcp.tool(
        description=_UPDATE_TASK_DESC,
        annotations=ToolAnnotations(title="Update task", destructiveHint=True),
    )
    async def update_task(
        id: int,
        name: str | None = None,
        description: str | None = None,
        solver_id: int | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        status_name: str | None = None,
        type_name: str | None = None,
        type_2nd_name: str | None = None,
        place_name: str | None = None,
        customer_id: int | None = None,
        request_id: int | None = None,
        deal_id: int | None = None,
        project_contract_id: int | None = None,
        parent_task_id: int | None = None,
        fullday: bool | None = None,
        private: bool | None = None,
        percent_done: int | None = None,
        scheduling_mode: int | None = None,
        other_solver_ids: list[int] | None = None,
        watcher_ids: list[int] | None = None,
        tag_ids: list[int] | None = None,
        offer_item_id: int | None = None,
        notify: dict[str, Any] | None = None,
        attachment: list[dict[str, Any]] | None = None,
        subordinate_task_id: int | None = None,
        duration_gap: int | None = None,
    ) -> dict[str, Any]:
        try:
            if name is not None and not name.strip():
                raise ValueError("name must be a non-empty string")
            # Naive datetimes = tenant-local wall-clock time; CDESK reads a
            # naive value as UTC (see create_task), so attach the offset here.
            valid_from = localize_naive_datetime("valid_from", valid_from)
            valid_to = localize_naive_datetime("valid_to", valid_to)
            # M7.2: catch valid_to < valid_from before we round-trip. Also
            # cross-check against the current value if only one side is changing
            # — that's the common LLM pattern ("change the deadline to ...").
            if valid_from is not None and valid_to is not None:
                _ensure_chronological(valid_from, valid_to)

            status_id = await resolve_enum_or_raise(cache, "status", status_name, kind="status")
            type_id = await resolve_enum_or_raise(cache, "type", type_name, kind="type")
            # type_2nd is hierarchical: resolving it needs its parent type. On a
            # partial update that sets only type_2nd_name, type_id is None and the
            # 2nd-level would resolve unscoped (wrong parent / AmbiguousEnumNameError),
            # so require type_name alongside it.
            if type_2nd_name is not None and type_id is None:
                raise ValueError(
                    "provide type_name when setting type_2nd_name so the 2nd-level "
                    "type resolves under the right parent"
                )
            type_2nd_id = await resolve_enum_or_raise(
                cache, "type2nd", type_2nd_name, kind="type_2nd", parent_id=type_id
            )
            place_id = await resolve_enum_or_raise(cache, "place", place_name, kind="place")

            current = await _fetch_for_update(client, id)

            # If only one side of the date range is being updated, compare
            # against the *current* stored value to catch inverted ranges.
            if valid_from is not None and valid_to is None:
                current_to = current.get("valid_to")
                if isinstance(current_to, str):
                    _ensure_chronological(valid_from, current_to)
            elif valid_to is not None and valid_from is None:
                current_from = current.get("valid_from")
                if isinstance(current_from, str):
                    _ensure_chronological(current_from, valid_to)
            body = _build_task_body(
                {
                    "name": name,
                    "description": description,
                    "solver_id": solver_id,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "status": status_id,
                    "type_id": type_id,
                    "type_2nd_id": type_2nd_id,
                    "place_id": place_id,
                    "customer_id": customer_id,
                    "request_id": request_id,
                    "deal_id": deal_id,
                    "project_contract_id": project_contract_id,
                    "parent_task_id": parent_task_id,
                    "fullday": fullday,
                    "private": private,
                    "percent_done": percent_done,
                    "scheduling_mode": scheduling_mode,
                    "other_solver_ids": other_solver_ids,
                    "watcher_ids": watcher_ids,
                    "tag_ids": tag_ids,
                    "offer_item_id": offer_item_id,
                    "notify": notify,
                    "attachment": attachment,
                    "subordinate_task_id": subordinate_task_id,
                    "duration_gap": duration_gap,
                }
            )
            _apply_lock_token(body, current)
            response = await client.put(f"v3/task/{id}", json=body)
        except RuntimeError:
            raise
        except ValueError as e:
            raise RuntimeError(f"update_task input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="update_task", record_id=id) from e
        return annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=(),
        )

    @mcp.tool(
        description=_DELETE_TASK_DESC,
        annotations=ToolAnnotations(title="Delete task", destructiveHint=True),
    )
    async def delete_task(id: int) -> dict[str, Any]:
        try:
            await client.delete(f"v3/task/{id}")
        except Exception as e:
            raise to_llm_error(e, operation="delete_task", record_id=id) from e
        return {"deleted": id}

    @mcp.tool(
        description=_SET_TASK_STATUS_DESC,
        annotations=ToolAnnotations(title="Set task status", destructiveHint=True),
    )
    async def set_task_status(id: int, status_name: str) -> dict[str, Any]:
        try:
            status_id = await resolve_enum_or_raise(cache, "status", status_name, kind="status")
            if status_id is None:
                raise RuntimeError("set_task_status requires a non-empty status_name")
            current = await _fetch_for_update(client, id)
            body = {"status": status_id}
            _apply_lock_token(body, current)
            response = await client.put(f"v3/task/{id}", json=body)
        except RuntimeError:
            raise
        except Exception as e:
            raise to_llm_error(e, operation="set_task_status", record_id=id) from e
        return response if isinstance(response, dict) else {"data": response}

    @mcp.tool(
        description=_ADD_WATCHERS_DESC,
        annotations=ToolAnnotations(title="Add task watchers", destructiveHint=True),
    )
    async def add_task_watchers(id: int, watcher_ids: list[int]) -> dict[str, Any]:
        try:
            if not watcher_ids:
                raise RuntimeError("add_task_watchers requires a non-empty watcher_ids list")
            current = await _fetch_for_update(client, id)
            current_watchers = _normalize_int_list(current.get("task_watchers_ids"))
            merged = sorted(set(current_watchers) | set(watcher_ids))
            body = {"task_watchers_ids": merged}
            _apply_lock_token(body, current)
            response = await client.put(f"v3/task/{id}", json=body)
        except RuntimeError:
            raise
        except Exception as e:
            raise to_llm_error(e, operation="add_task_watchers", record_id=id) from e
        return response if isinstance(response, dict) else {"data": response}

    @mcp.tool(
        description=_REMOVE_WATCHERS_DESC,
        annotations=ToolAnnotations(title="Remove task watchers", destructiveHint=True),
    )
    async def remove_task_watchers(id: int, watcher_ids: list[int]) -> dict[str, Any]:
        try:
            if not watcher_ids:
                raise RuntimeError("remove_task_watchers requires a non-empty watcher_ids list")
            current = await _fetch_for_update(client, id)
            current_watchers = _normalize_int_list(current.get("task_watchers_ids"))
            remaining = sorted(set(current_watchers) - set(watcher_ids))
            body = {"task_watchers_ids": remaining}
            _apply_lock_token(body, current)
            response = await client.put(f"v3/task/{id}", json=body)
        except RuntimeError:
            raise
        except Exception as e:
            raise to_llm_error(e, operation="remove_task_watchers", record_id=id) from e
        return response if isinstance(response, dict) else {"data": response}


# --- internals ----------------------------------------------------------

async def _fetch_for_update(client: CdeskClient, id: int) -> dict[str, Any]:
    """Fetch a task and unwrap to the record dict, ready to feed the optimistic
    lock into a follow-up PUT.

    CDESK's optimistic lock reads `timestamp_check` from the PUT body — not
    `updated_at`, despite the OpenAPI documentation claiming otherwise. Verified
    against `apiportal/inovalogic/Wrapper/BaseModel.php::hasBeenUpdated`. We
    require `timestamp_check` to be present; without it we can't safely PUT.

    fieldset=all is required: the v3 fieldset default (`extended`) no longer
    includes `timestamp_check` — only `all` carries it (verified live). It also
    guarantees `task_watchers_ids` is present for the watcher merge logic."""
    envelope = await client.get(f"v3/task/{id}", params={"fieldset": "all"})
    record = unwrap_record(envelope)
    if not isinstance(record, dict):
        raise RuntimeError(
            f"unexpected response shape fetching task {id} for update: "
            f"{type(record).__name__}"
        )
    if "timestamp_check" not in record:
        raise RuntimeError(
            f"task {id} record has no timestamp_check field; cannot apply optimistic lock"
        )
    return record


def _apply_lock_token(body: dict[str, Any], current: dict[str, Any]) -> None:
    """Set body['timestamp_check'] from the fetched record's lock token ONLY when
    truthy. A task that has never been modified has `updated_at` null, so its GET
    returns `timestamp_check: null` — sending null FAILS the PUT, while OMITTING
    the key makes CDESK skip the optimistic-lock check (documented v3 behavior).
    `_fetch_for_update` guarantees the KEY is present (raises if absent), so a
    falsy value here means 'never modified', not a malformed record."""
    token = current.get("timestamp_check")
    if token:
        body["timestamp_check"] = token


def _build_task_body(public_fields: dict[str, Any]) -> dict[str, Any]:
    """Map LLM-facing field names → CDESK field names, dropping Nones.

    Server-computed fields (assign_id, created_by, updated_by, num, close_date)
    are never sent — we don't expose them as tool params, so this layer just
    enforces the rename + None-filter deal."""
    body: dict[str, Any] = {}
    for public_name, value in public_fields.items():
        if value is None:
            continue
        cdesk_name = _TASK_CREATE_FIELDS.get(public_name, public_name)
        body[cdesk_name] = value
    return body


def _ensure_chronological(valid_from: str, valid_to: str) -> None:
    """Reject valid_to < valid_from. Parse both as datetimes to be safe across
    timezone offsets (lexicographic compare would be wrong for mixed offsets)."""
    try:
        vf = datetime.fromisoformat(valid_from)
        vt = datetime.fromisoformat(valid_to)
    except ValueError:
        return  # validate_iso_date already covers this
    # One side may be tz-aware (a localized naive datetime, or CDESK's stored
    # +00:00 value) while the other is naive — after localize_naive_datetime
    # only bare DATES stay naive. Comparing aware vs naive raises TypeError;
    # interpret a naive value in the TENANT timezone, matching the semantics
    # localize gives time-bearing values. Treating it as UTC mis-rejected
    # valid ranges like valid_from="2026-06-05" + valid_to="...T01:00:00"
    # (01:00 CEST = 23:00Z the day BEFORE bare-date midnight read as UTC).
    def _as_aware(dt: datetime) -> datetime:
        return dt.replace(tzinfo=tenant_timezone()) if dt.tzinfo is None else dt

    if _as_aware(vt) < _as_aware(vf):
        raise ValueError(
            f"valid_to ({valid_to!r}) must be on or after valid_from ({valid_from!r})"
        )


def _normalize_int_list(value: Any) -> list[int]:
    """Coerce a possibly-None / possibly-mixed list to list[int]. CDESK has
    been seen to return None or [] for unset multi-value fields."""
    if not value:
        return []
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue  # bool is int subclass; reject
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str):
            try:
                out.append(int(item))
            except ValueError:
                continue
    return out


# --- Tool descriptions (kept at module level so they're easy to find) ---

_LIST_TASKS_DESC = inspect.cleandoc(
    """
    List tasks visible to your CDESK account. Returns a paginated `items`
    array plus `meta`, which on this endpoint does carry `totalItems` and a
    `lastPage` boolean — use them rather than inferring the end from a short
    page.

    Ordering (re-verified live 2026-07-30 — the backend behaviour CHANGED):
    omitting `sort` gives newest-first (id descending), usually what you
    want. A BARE COLUMN NAME now genuinely sorts ASCENDING by that column
    ("name" and "valid_from" both confirmed). But any direction modifier and
    any unrecognised column silently fall back to id ASCENDING with no error
    — "-name", "name desc", "-valid_from", "-id" and a bogus column all did.
    Note "id" and "created_at" are indistinguishable from that fallback on
    ordered data, so don't read them as honored. Practical rule: for
    ascending order by one column pass the bare name; DESCENDING by anything
    other than the default is NOT available server-side — page the set and
    sort client-side, and never assume a "-" prefix was applied.

    Common filters:
      - text_search: free-text across task name/num, solver, creator, customer.
      - status_name: a status name from get_task_enums (resolved to the
        tenant-specific id; matching is diacritic-insensitive and supports
        English/Slovak/Czech variants).
      - solver_id / customer_id: int or list[int] for "task assigned to one
        of these solvers" / "tasks for one of these customers".
      - valid_from_after / valid_from_before: schedule start window.
      - deadline_after / deadline_before: deadline (valid_to) window.
      - created_after / created_before: creation-date window (ISO-8601;
        a bare date is inclusive — start-of-day for after, end-of-day
        for before; a naive datetime is treated as tenant-local time).

    NOT offered (the live CDESK task list silently ignores or breaks on
    them): open/late toggles — to filter by status use status_name; to
    find late tasks use deadline_before=<now>.

    For filter combinations not expressible above, pass sb_raw as an sb filter
    object for the CDESK API v3 task list (JSON string or object, structured
    tree form; see docs/cdesk-api-v3.json).
    Only the LIVE-VERIFIED working columns are applied
    server-side (id, code, status, type_id, solver_id, place_id,
    company_id, request_id, deal_id, project_contract_id,
    created_at, updated_at, close_date (W3C datetime values),
    dateFrom/dateTo, deadlineDateFrom/deadlineDateTo, plus
    the no-col text leaf). Clauses on any other column are STRIPPED, and the
    response carries an `unsupported_filters` block naming them.

    Response shaping (same semantics as get_task): fieldset selects the
    field group per record ("base"/"extended"/"all"/"custom"); fields is
    an exact whitelist of field names (returnFields; union with fieldset).
    Use fieldset="base" or fields=[...] to keep large lists compact.
    """
)

_GET_TASK_DESC = inspect.cleandoc(
    """
    Fetch a single task by id. Returns the task record. Returns an
    error if the task doesn't exist or your account doesn't have admin
    scope over it.

    fieldset selects the field group returned: "base", "extended"
    (CDESK's default when omitted), "all" (base+extended+custom), or
    "custom". Despite the name, "custom" is NOT empty and is not
    user-defined custom fields — on this module it returns the joined and
    computed extras (nested `solver`/`company`/`statusObj` objects,
    `task_watchers_ids`, `is_completed`, `rights`, …). Live field counts on
    this tenant: base 7, extended 24, custom 31, all 55.

    fields: an exact whitelist of field names to return (CDESK's
    returnFields). Composes with fieldset as a union; unknown names are
    silently ignored by CDESK.
    """
)

_GET_TASK_ENUMS_DESC = inspect.cleandoc(
    """
    Return the cached lookup tables for the Task module. Each bucket name
    maps to a list of records with id, display name, and any localized
    name variants. Use this to discover which status / type / tag names
    are valid for create_task / update_task / set_task_status in the
    current CDESK tenant.

    Pass refresh=True to force a fresh fetch (after an admin adds new
    enum values, for example). Cache auto-refreshes on resolve-miss and
    after a 1-hour TTL.
    """
)

_CREATE_TASK_DESC = inspect.cleandoc(
    """
    Create a new task in CDESK.

    Required: name, solver_id, valid_from, valid_to.
    valid_from / valid_to are ISO-8601 datetimes. A datetime WITHOUT a
    timezone offset is interpreted as TENANT-LOCAL wall-clock time
    (CDESK_TIMEZONE, default Europe/Bratislava) and sent with the proper
    offset — so "8:00" means 8:00 on the tenant's clock, not UTC. Pass an
    explicit offset to override. valid_to must be on or
    after valid_from (validated client-side).

    Optional name-based enums (resolved via get_task_enums; pass the
    display name OR any localized variant):
      - status_name
      - type_name / type_2nd_name
      - place_name

    Parent linking (mutually exclusive — CDESK derives assign_id from
    whichever you pass, with precedence deal > project_deal >
    request > parent_task > customer):
      - customer_id, request_id, deal_id, project_contract_id,
        parent_task_id.

    Other useful fields:
      - description (HTML allowed)
      - fullday, private (booleans)
      - percent_done (0-100; locked once status is completed)
      - other_solver_ids, watcher_ids, tag_ids (arrays of ints)

    Advanced / integration fields (safe to omit):
      - offer_item_id: link to an offer item (one-shot — cannot be
        reassigned once set).
      - notify: notification-override object passed verbatim; non-null
        forces notifications even when nothing else changed.
      - attachment: file-upload payload array passed verbatim (CDESK's
        processAttachment shape — requires pre-uploaded file hashes).

    Returns the created task's id + full record (CDESK's savedData).
    """
)

_UPDATE_TASK_DESC = inspect.cleandoc(
    """
    Update an existing task. Pass only the fields you want to change —
    everything else stays as-is.

    Optimistic locking is handled internally: this tool fetches the task
    first to grab its current `timestamp_check` token, then PUTs the change.
    If someone else updated the task between GET and PUT, you'll get a
    clear conflict error explaining how to retry.

    Same name-based enum and parent-linking semantics as create_task,
    plus the same advanced fields (offer_item_id, notify, attachment).
    Setting status to a completed status auto-sets close_date server-side
    and locks percent_done.

    Update-only scheduling helpers:
      - subordinate_task_id + duration_gap: shift dependent subordinate
        tasks by N days when moving dates under auto-scheduling
        (duration_gap is applied, not stored).
    """
)

_DELETE_TASK_DESC = inspect.cleandoc(
    """
    Delete a task by id. Returns {deleted: <id>} on success. If the task
    has dependencies that block deletion (subordinate tasks, etc.), you'll
    get a clear error explaining why.
    """
)

_SET_TASK_STATUS_DESC = inspect.cleandoc(
    """
    Intent helper: change just the status of a task. Equivalent to
    update_task(id, status_name=...) but exists as its own tool because
    "change the status" is a common operation worth surfacing prominently.

    status_name is resolved via get_task_enums (matches across the
    tenant's localized name variants). Internal GET → PUT to handle
    optimistic locking.
    """
)

_ADD_WATCHERS_DESC = inspect.cleandoc(
    """
    Add one or more users to a task's watcher list. Computes the union of
    existing watchers and the new ids — duplicates are removed. Internal
    GET → PUT to preserve existing watchers and handle optimistic locking.

    Requires the `task/fields/watchers` ACL on your account; without it
    the watcher list silently doesn't change.
    """
)

_REMOVE_WATCHERS_DESC = inspect.cleandoc(
    """
    Remove one or more users from a task's watcher list. Computes the
    difference of existing watchers minus the given ids — ids not on the
    list are ignored. Internal GET → PUT.

    Requires the `task/fields/watchers` ACL on your account.
    """
)
