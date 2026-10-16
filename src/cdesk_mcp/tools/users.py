"""User module MCP tools.

Tools that expose CDESK's User v3 CRUD, user groups, whoami, an intent helper
and the custom-field catalog, following the same pattern as the Task and
Customer modules: CdeskClient + filter builder + error translator, no enum
cache.

Special handling:
  - Password: `password` (plain text) OR `send_password_email=True`. Exactly
    one must be provided at create time (checked client-side to fail fast
    rather than round-trip a 4xx).
    CDESK rejects passwords that match the previous value on update with
    HTTP 412; errors.translate_error maps that to a clear "password must
    differ" message.
  - help account flag: a bool mapped to 'Y' or omitted via tools._helpers.yn.

Not exposed:
  - `encrypted_password` (SSO/migration niche)
  - fields the v3 API does not store (it answers with a `not_writable`
    warning): `cm_access` (the CDESK / CMonitor access flags), `test_account`,
    `email_account`, `guest_account`, `id_company`, `type` and `role` (both
    follow from group membership),
    `cm_language`, `timezone`, `ad_username`
  - notification fields requiring the `acl/user/fields/email_settings` ACL
    (CDESK silently ignores them without it)

Optimistic locking is handled internally via timestamp_check (see
_fetch_for_update).
"""

from __future__ import annotations

import inspect
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp import base_config
from cdesk_mcp.cdesk_client import CdeskApiError, CdeskAuthError, CdeskClient
from cdesk_mcp.filters import build_user_filter, encode_sb, user_type_label
from cdesk_mcp.sort_columns import resolve_sort, sort_note
from cdesk_mcp.tools._helpers import (
    forbid_unknown_arguments,
    list_meta,
    TEXT_MATCH_NOTE,
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
    with_warning_texts,
    yn as _yn,
)

# LLM-friendly name → CDESK field name. Centralised in one place so quirks
# don't leak into the tool signatures.
_USER_CREATE_FIELDS = {
    # The POST/PUT read the username from `user`, NOT `login` (the OpenAPI
    # doc says `login`, but a body with `login` 400s with the username-charset
    # error while the same value under `user` succeeds). The LLM-facing param
    # stays `login`.
    "login": "user",
    # The v3 spec documents `sendPassword`, but CDESK reads only
    # `password_action` (which sends the e-mail) and nothing maps one to the
    # other: with `sendPassword` the user is created with no password and no
    # e-mail.
    "send_password_email": "password_action",
    # `group_id` replaces the whole membership; the API also accepts `groups`
    # and `group_ids` as aliases (all three are stored).
    "group_ids": "group_id",
    "mobile_phone": "mobil",                 # CDESK uses cz spelling
}

_MAX_PER_PAGE = 100

# Keys CDESK returns on the user detail that are far too large to hand an LLM
# and are not user data. `grouped_notifications` is the per-user notification
# CONFIGURATION matrix: 22 module groups, each with nested notification rows
# (receiver lists, per-channel hidden flags). It runs to ~350 KB — nearly all
# of a `fieldset="all"` response — which exceeds the MCP tool-result token cap,
# so without this `fieldset="all"` and `"custom"` would fail with "result
# exceeds maximum allowed tokens" and return NOTHING, including `customFields`,
# which is what the custom-field docs read `fieldset="all"` for.
#
# `returnFields[]` does NOT exclude it (fieldset="all" +
# returnFields[id,name,customFields] still returns it), so dropping it here is
# the only way to make the documented path executable.
# Notification settings are managed in the CDESK UI; nothing in this server
# reads them.
_OVERSIZED_USER_KEYS = ("grouped_notifications",)


def _drop_oversized_user_keys(record: dict[str, Any]) -> dict[str, Any]:
    """Strip `_OVERSIZED_USER_KEYS` from a user detail, recording what went.

    Reports the omission in `omitted_fields` rather than dropping silently —
    a caller that genuinely wanted notification config needs to know it was
    removed, not conclude the user has none."""
    dropped = [key for key in _OVERSIZED_USER_KEYS if key in record]
    if not dropped:
        return record
    for key in dropped:
        record.pop(key, None)
    record["omitted_fields"] = {
        key: (
            "Omitted by cdesk-mcp: per-user notification configuration, "
            "~350 KB, which exceeds the tool-result size limit. It is not "
            "user data; notification settings live in the CDESK UI."
        )
        for key in dropped
    }
    return record


# User groups. The v3 spec: user groups "are also solver groups and
# approver groups", and `solver_group_id` is "FK to a user group id". The list
# is trimmed to the identifying fields by default — a group record also carries
# its rights and user settings. CDESK sends three flags on every group:
# `root_group` (one of the tenant's built-in root groups — Administrators,
# Operators, Customer accounts, … — which CDESK never offers as a request's
# solver group), `main_group` (a direct child of a root group) and, on the list,
# `toplevel_group` (no parent group).
_MAX_GROUPS_PER_PAGE = 100
_GROUP_LIST_FIELDS = (
    "id", "name", "description", "enabled", "root_group", "main_group",
    "toplevel_group", "parent_id", "assigned_company_count",
)
# `id_parent` on a group detail is NOT its parent group (it holds the tenant's
# admin account id); the parents are `group_id`.
_GROUP_DETAIL_FIELDS = (
    "id", "name", "nick", "description", "enabled", "root_group", "main_group",
)


# whoami. GET /v3/user/me gives the caller's identity and main roles; the
# company, language, timezone and the request-solver / customer roles come from
# core/base-config — the payload the CDESK web app loads for every logged-in
# user, already used for the form rules — whose `user` block holds the caller's
# own record. That block ALSO holds the session's apitoken (`token`) and `gsh`,
# so only the fields named here ever leave it. Either source alone is enough;
# where both have a field, /v3/user/me wins.
_ME_FIELDS = (
    "id", "user", "name", "email", "type", "job_title_id",
    "is_admin", "is_operator", "is_solver", "is_authorizing_officer",
)
_WHOAMI_FIELDS = (
    ("id", "id"),
    ("login", "user"),
    ("name", "name"),
    ("email", "email"),
    ("company_id", "admin_main_company"),
    ("type", "type"),
    ("job_title_id", "job_title_id"),
    ("language", "lang"),
    ("timezone", "admin_timezone"),
)
_WHOAMI_ROLES = (
    ("admin", "is_admin"),
    ("operator", "is_operator"),
    ("solver", "is_solver"),
    ("request_solver", "is_request_solver"),
    ("customer", "is_customer"),
    ("authorizing_officer", "is_authorizing_officer"),
)


def whoami_record(base: Any, me: Any = None) -> dict[str, Any] | None:
    """The calling user's allow-listed fields from the /v3/user/me record `me`
    and core/base-config's `user` block, or None when neither names a user.
    A base-config block for a different user than `me` is ignored."""
    user = base_config.section(base, "user")
    if not isinstance(user, dict) or not isinstance(user.get("id"), int):
        user = None
    if not isinstance(me, dict) or not isinstance(me.get("id"), int):
        me = None
    if me is None and user is None:
        return None
    if me is not None and user is not None and user["id"] != me["id"]:
        user = None
    source = dict(user or {})
    if me is not None:
        source.update({k: me[k] for k in _ME_FIELDS if k in me})
    out: dict[str, Any] = {key: source.get(src) for key, src in _WHOAMI_FIELDS}
    out["roles"] = {key: bool(source.get(src)) for key, src in _WHOAMI_ROLES}
    return out


def register_user_tools(mcp: FastMCP, client: CdeskClient) -> None:
    """Register all User-module tools on the given FastMCP instance.

    Note: no EnumCache parameter. CDESK has no /v3/user/enums endpoint;
    user-type filter values (filters.USER_TYPE_LABELS) are documented in the
    list_users description."""

    @mcp.tool(
        description=_WHOAMI_DESC,
        annotations=ToolAnnotations(title="Who am I", readOnlyHint=True),
    )
    async def whoami() -> dict[str, Any]:
        try:
            own = unwrap_record(
                await client.get("v3/user/me", params={"returnFields[]": list(_ME_FIELDS)})
            )
        except Exception:  # noqa: BLE001 — core/base-config still names the caller
            own = None
        data = await base_config.load(client)
        me = whoami_record(data, own)
        if me is None:
            raise RuntimeError(
                "whoami: CDESK did not return the calling user's record "
                "(neither /v3/user/me nor core/base-config was readable); try "
                "again, or check the connection with server_info."
            )
        return me

    @mcp.tool(
        description=_LIST_USER_GROUPS_DESC,
        annotations=ToolAnnotations(title="List user groups", readOnlyHint=True),
    )
    async def list_user_groups(
        text_search: str | None = None,
        page: int = 1,
        per_page: int = 50,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            if per_page < 1 or per_page > _MAX_GROUPS_PER_PAGE:
                raise ValueError(
                    f"per_page must be between 1 and {_MAX_GROUPS_PER_PAGE} (got {per_page})"
                )
            if page < 1:
                raise ValueError(f"page must be 1 or greater (got {page})")
            validate_fields(fields)
            params: dict[str, Any] = {
                "pg": page, "pp": per_page,
                "returnFields[]": fields or list(_GROUP_LIST_FIELDS),
            }
            if text_search:
                params["q"] = text_search
            response, paging = await client.get_page("v3/user/groups", params=params)
        except ValueError as e:
            raise RuntimeError(f"list_user_groups input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_user_groups") from e
        records, meta = unwrap_list(response)
        return {"items": records, "meta": list_meta(meta, paging),
                "page": page, "per_page": per_page}

    @mcp.tool(
        description=_GET_USER_GROUP_DESC,
        annotations=ToolAnnotations(title="Get user group", readOnlyHint=True),
    )
    async def get_user_group(id: int) -> dict[str, Any]:
        try:
            response = await client.get(f"v3/user/groups/{id}")
        except Exception as e:
            raise to_llm_error(e, operation="get_user_group", record_id=id) from e
        record = unwrap_record(response)
        if not isinstance(record, dict):
            raise RuntimeError(f"get_user_group: unexpected response for group {id}")
        out = {key: record.get(key) for key in _GROUP_DETAIL_FIELDS}
        members = record.get("assigned_user_id")
        out["member_ids"] = members if isinstance(members, list) else []
        parents = record.get("group_id")
        out["parent_group_ids"] = parents if isinstance(parents, list) else []
        children = record.get("child_group_id")
        out["child_group_ids"] = children if isinstance(children, list) else []
        return out

    forbid_unknown_arguments(mcp, ("list_user_groups",))

    @mcp.tool(
        description=_LIST_USERS_DESC + "\n\n" + sort_note("users") + "\n\n" + TEXT_MATCH_NOTE,
        annotations=ToolAnnotations(title="List users", readOnlyHint=True),
    )
    async def list_users(
        text_search: str | None = None,
        company_id: int | None = None,
        user_type: str | None = None,
        status: str | None = None,
        only_deleted: bool = False,
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
            if user_type is not None:
                user_type = user_type_label(user_type)
            # Unsupported sb_raw columns are stripped and reported via
            # `unsupported_filters` — the agent must filter the items itself.
            dropped_clauses: list[dict[str, Any]] = []
            sb = build_user_filter(
                text_search=text_search,
                company_id=company_id,
                user_type=user_type,
                status=status,
                sb_raw=sb_raw,
                dropped_cols_out=dropped_clauses,
            )
            params: dict[str, Any] = {"pg": page, "pp": per_page}
            apply_field_scope(params, fieldset, fields)
            # onlyDeleted only works as a FLAT query param — the endpoint
            # silently ignores it as an sb clause.
            if only_deleted:
                params["onlyDeleted"] = "1"
            if sort:
                params["sort"] = resolve_sort("users", sort)
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response, paging = await client.get_page("v3/user", params=params)
        except ValueError as e:
            raise RuntimeError(f"list_users input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_users") from e

        records, meta = unwrap_list(response)
        result = {"items": records, "meta": list_meta(meta, paging, dropped=dropped_clauses), "page": page, "per_page": per_page}
        if dropped_clauses:
            result["unsupported_filters"] = unsupported_filter_directive(dropped_clauses)
        return result

    @mcp.tool(
        description=_GET_USER_DESC,
        annotations=ToolAnnotations(title="Get user", readOnlyHint=True),
    )
    async def get_user(
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
            response = await client.get(f"v3/user/{id}", params=params or None)
        except ValueError as e:
            raise RuntimeError(f"get_user input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="get_user", record_id=id) from e
        record = unwrap_record(response)
        if isinstance(record, dict):
            return redact_secrets(_drop_oversized_user_keys(record))
        # Non-dict shape (unexpected, but CDESK envelopes vary): still run the
        # redactor. It recurses into lists, so a list-of-records here is covered
        # too — skipping it would leak credentials through the fallback path.
        return redact_secrets({"data": record})

    @mcp.tool(
        description=_GET_USER_CUSTOM_FIELDS_DESC,
        annotations=ToolAnnotations(title="Get user custom fields", readOnlyHint=True),
    )
    async def get_user_custom_fields() -> Any:
        try:
            response = await client.get("v3/user/custom-fields")
        except Exception as e:
            raise to_llm_error(e, operation="get_user_custom_fields") from e
        return wrap_collection(
            unwrap_record(response),
            kind="custom-field definitions for the User module",
        )

    @mcp.tool(
        description=_FIND_USER_DESC + "\n\n" + TEXT_MATCH_NOTE,
        annotations=ToolAnnotations(title="Find user", readOnlyHint=True),
    )
    async def find_user(text: str, max_results: int = 10) -> dict[str, Any]:
        """Intent helper — full-text search across CDESK's documented user
        q-search columns: id, name, nick, user (login), email, mobil, phone,
        description, personal_number."""
        if max_results < 1 or max_results > _MAX_PER_PAGE:
            raise RuntimeError(
                f"find_user input error: max_results must be 1..{_MAX_PER_PAGE}"
            )
        try:
            sb = build_user_filter(text_search=text)
            params: dict[str, Any] = {"pg": 1, "pp": max_results}
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response = await client.get("v3/user", params=params)
        except Exception as e:
            raise to_llm_error(e, operation="find_user") from e

        records, _meta = unwrap_list(response)
        compact: list[dict[str, Any]] = []
        for r in records:
            if not isinstance(r, dict):
                continue
            compact.append(
                {
                    "id": r.get("id"),
                    "login": r.get("login") or r.get("user"),
                    "email": r.get("email"),
                    "name": r.get("name"),
                    "nick": r.get("nick"),
                    "type": r.get("type"),
                }
            )
        return {"matches": compact, "count": len(compact)}

    @mcp.tool(
        description=_CREATE_USER_DESC,
        annotations=ToolAnnotations(title="Create user", destructiveHint=True),
    )
    async def create_user(
        login: str,
        email: str,
        auto_logout: int,
        ips: str,
        *,
        password: str | None = None,
        send_password_email: bool = False,
        first_name: str | None = None,
        last_name: str | None = None,
        name: str | None = None,
        nick: str | None = None,
        position: str | None = None,
        phone: str | None = None,
        mobile_phone: str | None = None,
        group_ids: list[int] | None = None,
        job_title_id: int | None = None,
        personal_number: str | None = None,
        is_help_account: bool | None = None,
        custom_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            try:
                validate_custom_fields(custom_fields)
            except ValueError as e:
                raise RuntimeError(f"create_user input error: {e}") from e
            # The API requires `name` AND `nick` on create (it rejects the body
            # with "Neplatné meno" / "Neplatná skratka pre výpis" otherwise;
            # first_name+last_name do NOT substitute for name). Fail fast with
            # a clear message.
            if not name or not name.strip():
                raise RuntimeError(
                    "create_user input error: `name` (display name) is "
                    "required by CDESK. Pass it explicitly — first_name/"
                    "last_name alone are not accepted as a substitute."
                )
            if not nick or not nick.strip():
                raise RuntimeError(
                    "create_user input error: `nick` (short list label, "
                    "e.g. initials or a short handle) is required by CDESK."
                )
            # Require EXACTLY ONE password mechanism client-side. CDESK
            # would 4xx without one; supplying both is undefined (it may email a
            # generated password the operator didn't expect). Failing fast saves
            # a round-trip and gives a clearer message.
            if (password is not None) == bool(send_password_email):
                raise RuntimeError(
                    "create_user input error: provide exactly one of `password` "
                    "(plain text) or `send_password_email=True` (CDESK auto-"
                    "generates and emails the password), not both or neither."
                )
            # auto_logout and ips are validated server-side on every
            # create even when CMonitor access is off. Reject obviously
            # invalid values up-front so the user gets a clear message
            # instead of a CDESK 400.
            _validate_auto_logout_and_ips(auto_logout, ips)
            body = _build_user_body(
                {
                    "login": login,
                    "email": email,
                    "auto_logout": auto_logout,
                    "ips": ips,
                    "password": password,
                    # Pass send_password_email through unchanged so an
                    # explicit False reaches CDESK as `password_action: false`.
                    "send_password_email": send_password_email,
                    "first_name": first_name,
                    "last_name": last_name,
                    "name": name,
                    "nick": nick,
                    "position": position,
                    "phone": phone,
                    "mobile_phone": mobile_phone,
                    "group_ids": group_ids,
                    "job_title_id": job_title_id,
                    "personal_number": personal_number,
                    "help_account": _yn(is_help_account),
                }
            )
            # customFields keys are already wire-format (cfield_*) — set after
            # the body build so they bypass the LLM-name→CDESK-name rename map.
            if custom_fields:
                body["customFields"] = custom_fields
            response = await client.post("v3/user", json=body)
            result = response if isinstance(response, dict) else {"data": response}
            result = with_warning_texts(result)
        except (CdeskApiError, CdeskAuthError) as e:
            # CDESK-signaled errors (both RuntimeError subclasses) need
            # translation — must be caught before the plain-RuntimeError
            # passthrough below.
            raise to_llm_error(e, operation="create_user") from e
        except RuntimeError:
            raise  # client-side validators already emit a friendly message
        except Exception as e:
            raise to_llm_error(e, operation="create_user") from e
        return redact_secrets(result)

    @mcp.tool(
        description=_UPDATE_USER_DESC,
        annotations=ToolAnnotations(title="Update user", destructiveHint=True),
    )
    async def update_user(
        id: int,
        *,
        login: str | None = None,
        email: str | None = None,
        auto_logout: int | None = None,
        ips: str | None = None,
        password: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        name: str | None = None,
        nick: str | None = None,
        position: str | None = None,
        phone: str | None = None,
        mobile_phone: str | None = None,
        group_ids: list[int] | None = None,
        job_title_id: int | None = None,
        personal_number: str | None = None,
        is_help_account: bool | None = None,
        custom_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            try:
                validate_custom_fields(custom_fields)
            except ValueError as e:
                raise RuntimeError(f"update_user input error: {e}") from e
            # When present, auto_logout and ips have the same constraints
            # on update as on create.
            if auto_logout is not None or ips is not None:
                _validate_auto_logout_and_ips(
                    auto_logout if auto_logout is not None else 1,
                    ips if ips is not None else "*.*.*.*",
                )
            current = await _fetch_for_update(client, id)
            body = _build_user_body(
                {
                    "login": login,
                    "email": email,
                    "auto_logout": auto_logout,
                    "ips": ips,
                    "password": password,
                    "first_name": first_name,
                    "last_name": last_name,
                    "name": name,
                    "nick": nick,
                    "position": position,
                    "phone": phone,
                    "mobile_phone": mobile_phone,
                    "group_ids": group_ids,
                    "job_title_id": job_title_id,
                    "personal_number": personal_number,
                    "help_account": _yn(is_help_account),
                }
            )
            # customFields keys are already wire-format (cfield_*) — set after
            # the body build so they bypass the LLM-name→CDESK-name rename map.
            if custom_fields:
                body["customFields"] = custom_fields
            body["timestamp_check"] = current["timestamp_check"]
            response = await client.put(f"v3/user/{id}", json=body)
            result = response if isinstance(response, dict) else {"data": response}
            result = with_warning_texts(result)
        except (CdeskApiError, CdeskAuthError) as e:
            # CDESK-signaled errors (both RuntimeError subclasses) need
            # translation — must be caught before the plain-RuntimeError
            # passthrough below.
            raise to_llm_error(e, operation="update_user", record_id=id) from e
        except RuntimeError:
            raise  # client-side validators already emit a friendly message
        except Exception as e:
            raise to_llm_error(e, operation="update_user", record_id=id) from e
        return redact_secrets(result)

    @mcp.tool(
        description=_DELETE_USER_DESC,
        annotations=ToolAnnotations(title="Delete user", destructiveHint=True),
    )
    async def delete_user(id: int) -> dict[str, Any]:
        try:
            await client.delete(f"v3/user/{id}")
        except Exception as e:
            raise to_llm_error(e, operation="delete_user", record_id=id) from e
        return {"deleted": id}


# ---------- helpers (module-private) -------------------------------------


# Group membership (`group_id`), `job_title_id` and `position` (a job-title
# name) are stored and validated by the API — an unknown group, job title or
# position name is a 400 — so they need no read-back.


def _validate_auto_logout_and_ips(auto_logout: int, ips: str) -> None:
    """CDESK validates `auto_logout > 0` and `ips` non-empty on every create
    (and on any update that touches them) even when CMonitor access is off.
    Reject obviously bad values up-front."""
    if not isinstance(auto_logout, int) or isinstance(auto_logout, bool) or auto_logout <= 0:
        raise RuntimeError(
            f"create_user/update_user input error: auto_logout={auto_logout!r} "
            f"must be a positive integer (minutes). Pass 120 if you have no "
            f"specific value."
        )
    if not isinstance(ips, str) or not ips.strip():
        raise RuntimeError(
            "create_user/update_user input error: ips must be a non-empty "
            "string (semicolon-separated IP patterns). Pass '*.*.*.*' to "
            "allow any source."
        )


def _build_user_body(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop None values, rename LLM-friendly names to CDESK field names
    per _USER_CREATE_FIELDS."""
    body: dict[str, Any] = {}
    for k, v in raw.items():
        if v is None:
            continue
        cdesk_key = _USER_CREATE_FIELDS.get(k, k)
        body[cdesk_key] = v
    return body


async def _fetch_for_update(client: CdeskClient, id: int) -> dict[str, Any]:
    """Fetch a user and unwrap to the record dict; require timestamp_check.

    As in tools/tasks.py and tools/customers.py: CDESK reads `timestamp_check` from the PUT body for the optimistic-lock
    check, not `updated_at`.

    fieldset=all is required: the v3 fieldset default (`extended`) does not
    include `timestamp_check` — only `all` carries it."""
    envelope = await client.get(f"v3/user/{id}", params={"fieldset": "all"})
    record = unwrap_record(envelope)
    if not isinstance(record, dict):
        raise RuntimeError(
            f"unexpected response shape fetching user {id} for update: "
            f"{type(record).__name__}"
        )
    if "timestamp_check" not in record:
        raise RuntimeError(
            f"user {id} record has no timestamp_check field; "
            f"cannot apply optimistic lock"
        )
    return record


# ---------- tool descriptions --------------------------------------------

_LIST_USERS_DESC = inspect.cleandoc(
    """
    List CDESK users visible to your account. Returns a paginated list.

    Filters:
    - text_search: full-text across the columns CDESK's text search scans —
      id, name, nick, user (login), email, mobil, phone, description,
      personal_number.
    - company_id: filter to users belonging to a specific customer
      (id_company column).
    - user_type: one of "admin", "operator", "customer", "easyclick",
      "solver", "authorizing-officer", "cm", "guest" ("authorizing_officer" is
      accepted too). There is no label for a plain user. Invalid values are
      rejected with the full list.
      CDESK derives the type from the user's ROOT GROUP; this filter applies
      that derivation. Two consequences:
      (a) a user with NO group membership matches NO user_type value at all
      (membership is set with create_user / update_user `group_ids`);
      (b) the record field `type` ("M"/"C"/...) is
      NOT this derived user_type — a user whose `type` is "C" is matched by
      `solver` when its group is a solver-root group, and is NOT matched by
      `customer`. Do not read one as the other.
    - only_deleted: when true, returns only soft-deleted users.
    - sb_raw: an sb filter object for the CDESK API v3 user list (JSON string
      or object, structured tree form; see docs/cdesk-api-v3.json). Advanced;
      mutually exclusive with the typed filters. Only the columns known to work
      are applied server-side — id, name, email, personal_number,
      id_company, status, type, uname (the login), mobil, id_external, plus
      last_login_date and created_at (strict W3C datetime values), plus the
      no-col text leaf. Clauses on any other column (e.g. `work_position`,
      which is stored as a job_title_id FK) are STRIPPED, and the response
      carries an `unsupported_filters` block naming them.
      NB: an sb_raw `type` leaf takes the same labels as `user_type`;
      CDESK answers any other value with 400 listing them.

    Pagination: page (1-based), per_page (default 20, max 100).

    `meta.total` is the number of records matching the filter across all pages (CDESK's own count) and `meta.last_page` tells whether this is the last page.

    Response shaping (same semantics as get_user): fieldset selects the
    field group per record ("base"/"extended"/"all"/"custom"); fields is an
    exact whitelist of field names (returnFields; union with fieldset).

    Returns: {items: [...], meta: {...}, page, per_page}.
    """
)

_GET_USER_DESC = inspect.cleandoc(
    """
    Fetch one user by id. Returns the record. Errors with a
    not-found message if the id doesn't exist or isn't in your admin
    scope.

    fieldset selects the field group returned: "base", "extended" (CDESK's
    default when omitted), "all", or "custom". Note "custom" is NOT
    "custom fields only": on this module `extended` and `custom` are disjoint
    halves whose union is `all`, so "custom" returns `customFields` plus ~65
    admin/config keys while OMITTING `id`, `name`, `email`, `status`, `type`
    and `timestamp_check`. Only "all" returns identity fields and the stored
    cfield_* values together.

    This tool always omits `grouped_notifications` (the per-user
    notification-settings matrix, ~350 KB — large enough on its own to exceed
    the tool-result size limit) and, whenever it drops it, says so under
    `omitted_fields`. In practice only "all" and "custom" carry the key.
    Notification settings are managed in the CDESK UI.

    fields: an exact whitelist of field names to return (CDESK's
    returnFields). Composes with fieldset as a union; unknown names are
    silently ignored by CDESK. Note that `fields` alone (no fieldset) is
    RESTRICTIVE and is the cheapest way to read a few known keys.
    """
)

_GET_USER_CUSTOM_FIELDS_DESC = inspect.cleandoc(
    """
    List the custom-field definitions configured for the User module.
    Returns {items, count} — plus a `note` when count is 0, meaning the module
    has no custom fields defined (an empty result, not a failure). Each
    baseproperty in `items` carries a ready-to-use key in the form
    `cfield_<propertyId>_<basepropertyId>`.

    Workflow: (1) call this tool; (2) find the field by name; (3) pass
    `custom_fields={"<key>": <value>}` to create_user/update_user
    (scalar for single-value fields; for select/relation fields send the
    option id); (4) read stored values via get_user(id, fieldset="all").

    KNOWN LIMITATION: the tenant currently ignores customFields writes
    silently — values do not persist. After any write, a
    get_user(id, fieldset="custom") read shows whether the value actually
    persisted; a value that did not stick was not stored despite the 200.
    """
)

_FIND_USER_DESC = inspect.cleandoc(
    """
    Intent helper for "find a user by name / login / email / mobil /
    description / personal_number". Runs a full-text search across the
    columns CDESK's text search scans (id, name, nick, user (login),
    email, mobil, phone, description, personal_number) and returns a
    compact list of matches with id, login, email, name, nick, and
    type — enough for the LLM to disambiguate without burning context
    on full records.

    Resolves a name or partial identifier to the id that create/update/delete
    require — those take ids only. `count` > 1 means the identifier was
    ambiguous: the tool picks no winner and returns every candidate, so the
    match is unresolved until one id is chosen.

    If `count` is 0, the search produced no matches under your current
    admin scope. A zero count is NOT evidence the user doesn't exist — they
    may be outside your scope, or the name may differ in casing/spelling. A
    broader query distinguishes those cases from genuine absence.

    Params: text (required), max_results (default 10, max 100).
    Returns: {matches: [{id, login, email, name, nick, type}], count}.
    """
)

_CREATE_USER_DESC = inspect.cleandoc(
    """
    Create a new CDESK user. Required: login (unique; letters, digits,
    space and _ only), email, name (display name — first_name/last_name
    do NOT substitute for it), nick (short list label, e.g. initials),
    auto_logout (CMonitor auto-logout minutes, e.g. 120), and ips
    (allowed login IP range, e.g. `*.*.*.*` for unrestricted).

    NOTE on auto_logout / ips: CDESK validates these on every create
    EVEN when CMonitor access is disabled. Defaults known to work for
    a non-CMonitor user are `auto_logout=120` and `ips='*.*.*.*'`.

    Password (exactly ONE required, enforced client-side):
    - `password`: plain text, CDESK hashes it. It is visible in the MCP tool
      call args and may end up in client transcripts; send_password_email
      avoids that exposure entirely.
    - `send_password_email=True`: CDESK e-mails the new user a link to set
      their own password (the "Vytvoriť heslo prostredníctvom emailu" option
      of the CDESK form). Safer; no plaintext in your AI client's log.

    is_help_account: True marks a help account (non-licensed; CDESK's 'Y').

    Not settable through the API: the user type and role (both follow from
    group membership), the CDESK / CMonitor access flags, the test, e-mail
    and guest account flags, language, timezone and the AD user name — a new
    user takes them from the tenant administrator.

    Grouping:
    - group_ids: the user's group membership (`group_id` in CDESK terms).
      CDESK has no primary group. A group id outside the caller's scope is
      rejected with a 400. A user with no group has no derived user type
      (see list_users).
    - job_title_id: the job title (it drives e.g. which request-catalog
      tiles the user sees); when omitted, the tenant default job title is
      set. An id that is not a job title of the tenant is rejected with a 400.
    - position: the job title by NAME instead of id, matched
      case-insensitively; an unknown name is rejected with a 400. Ignored
      when job_title_id is also sent.

    custom_fields: flat dict of `cfield_<propertyId>_<basepropertyId>`
    keys → values (discover keys via get_user_custom_fields;
    select/relation fields take the option id). Unknown keys → 400.

    Returns: {data: <new id>, savedData: {...the stored user...}}; check the
    optional `warnings` list for fields CDESK ignored.
    """
)

_UPDATE_USER_DESC = inspect.cleandoc(
    """
    Partial update — pass only the fields you want to change. Internal
    GET fetches the current timestamp_check for the optimistic lock;
    the LLM doesn't see it.

    auto_logout and ips, when present, are validated the same way as on
    create: auto_logout must be > 0, ips must be non-empty.

    Password: setting `password` to a new value rotates it. CDESK
    rejects passwords equal to the user's previous password with HTTP
    412 — surfaced as a clear "new password must differ" error. As with
    create_user, a plain-text password here is visible in the AI client's
    tool-call transcript.

    custom_fields: flat dict of `cfield_<propertyId>_<basepropertyId>`
    keys → values (discover keys via get_user_custom_fields; read
    stored values via get_user(id, fieldset="all")).

    group_ids REPLACES the whole group membership: groups not listed are
    left, and an empty list removes the user from every group; omit it to
    keep the membership. job_title_id / position set the job title as on
    create. The fields create_user cannot set cannot be changed here either.

    The display name follows first_name + last_name on update: a `name` sent
    on its own is not stored, and CDESK says so in `warnings`. Send
    first_name / last_name to rename a user.

    Returns: {data: <id>, savedData: {...the stored user...}}; check the
    optional `warnings` list for fields CDESK ignored.
    """
)

_DELETE_USER_DESC = inspect.cleandoc(
    """
    Delete a user by id. Fails with 409 if the user has dependent
    records (open requests, task assignments, etc.). The 409 names the blocking
    links, which have to be broken before the delete can succeed.

    Returns: {deleted: <id>}.
    """
)


_WHOAMI_DESC = inspect.cleandoc(
    """
    The CDESK user this connector acts as — the account whose rights every
    tool call runs with (in http mode, whoever signed in): id, login, name,
    email, company_id (the user's own company), type (CDESK's raw user-type
    letter, "M"/"C"/…; not the derived user_type of list_users), job_title_id,
    language, timezone, and roles (admin, operator, solver, request_solver,
    customer, authorizing_officer). Its `id` is the solver_id / user id that
    "my tasks", "my requests" or "what I logged" refer to.
    """
)

_LIST_USER_GROUPS_DESC = inspect.cleandoc(
    """
    List CDESK user groups — the groups users belong to, which are ALSO the
    solver groups (`solver_group_id` on requests, work orders and catalog
    prefills) and approver groups. Returns id, name, description, enabled,
    root_group, main_group, toplevel_group, parent_id and
    assigned_company_count per group (fields: pick other top-level fields
    instead), plus meta.total.

    root_group=true marks one of the tenant's built-in ROOT groups
    (Administrators, Operators, Customer accounts, …): CDESK does not accept a
    root group as a request's solver group. main_group=true marks a direct
    child of a root group; toplevel_group=true a group with no parent group.
    parent_id is one direct parent group (get_user_group lists all of them).
    The groups the request form offers for one request (its customer, service
    area, catalog) are listed by list_request_solver_groups.

    text_search matches the group name. get_user_group gives one group's
    member user ids.
    """
)

_GET_USER_GROUP_DESC = inspect.cleandoc(
    """
    One CDESK user group (also a solver / approver group): id, name, nick,
    description, enabled, root_group (true = a built-in root group), main_group
    (true = a direct child of a root group), the parent group ids
    (parent_group_ids), the member user ids (member_ids — names via
    get_user / list_users) and the ids of its child groups.
    """
)
