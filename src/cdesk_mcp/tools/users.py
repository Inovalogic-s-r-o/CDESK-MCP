"""User module MCP tools (M9).

Seven tools that expose CDESK's User v3 CRUD + an intent helper + the
custom-field catalog, following
the M7 (Task) / M8 (Customer) pattern: CdeskClient + filter builder +
error translator, no enum cache.

Special handling:
  - Password: `password` (plain text) OR `send_password_email=True`. Exactly
    one must be provided at create time (we check client-side and fail fast
    rather than round-trip a 4xx).
    CDESK rejects passwords that match the previous value on update with
    HTTP 412; M4 maps that to a clear "password must differ" message.
  - cm_access: CDESK's 2-char Y/N feature flag. **The OpenAPI spec
    mis-documents the positions** (says pos 1 = CMonitor, pos 2 = CDESK);
    the actual source (`Wrapper/Customer.php:2124`,
    `Model/User.php:7197`, `Helper/SvcMirror/CmEnabledAdminAllowlist.php:12`)
    consistently reads `cm_access[1]` ('the 2nd character') as the
    CMonitor flag. So:
        position 1 (index 0) = CDESK
        position 2 (index 1) = CMonitor
    Live confirmation: admin_api_v3 has `cm_access='YN'` and is a
    CDESK-only API user (no CMonitor surface).
    Exposed as two booleans (`cdesk_access`, `cmonitor_access`) for
    LLM ergonomics; combined into the 2-char string before sending.
    Defaults are None on both — CDESK applies its tenant default if
    neither is set on create. On update, missing bools preserve the
    current stored position (we already GET for timestamp_check).
  - help/email/guest/test account flags: bools mapped to 'Y' or omitted
    via tools._helpers.yn (same convention as Customer.cdesk_allowed).

Skipped per plan:
  - `encrypted_password` (SSO/migration niche)
  - notification fields requiring `acl/user/fields/email_settings` ACL
    (silent-ignore concern; not surfaced in v1)

Optimistic locking is hidden via timestamp_check — same M7 finding.
"""

from __future__ import annotations

import inspect
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp.cdesk_client import CdeskApiError, CdeskAuthError, CdeskClient
from cdesk_mcp.filters import USER_TYPE_LABELS, build_user_filter, encode_sb
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

# LLM-friendly name → CDESK field name. Centralised in one place so quirks
# don't leak into the tool signatures.
_USER_CREATE_FIELDS = {
    # The live POST/PUT read the username from `user`, NOT `login` (the
    # OpenAPI doc says `login`, but a body with `login` 400s with the
    # username-charset error while the same value under `user` succeeds —
    # verified live 2026-06-04). The LLM-facing param stays `login`.
    "login": "user",
    "send_password_email": "sendPassword",   # CDESK camelCase (unusual)
    "company_id": "id_company",              # CDESK uses older naming
    "primary_group_id": "group_primary",     # clearer LLM name
    "group_ids": "groups",                   # consistent with tag_ids pattern
    "language": "cm_language",               # shorter LLM-facing name
    "mobile_phone": "mobil",                 # CDESK uses cz spelling
    "user_type": "type",                     # avoid shadowing Python builtin
}

_MAX_PER_PAGE = 100

# Fields this tenant accepts with a 200 and then discards, verified live
# 2026-07-31 and recorded in docs/bugs.md. Listed here so the write tools can
# tell the model the value did not stick instead of reporting a clean success.
# ONLY fields whose request-body key matches the savedData key belong here: the
# diff compares by key name, so `company_id` (echoed as `id_company`) would
# raise a false alarm and is deliberately absent.
_USER_DISCARDED_FIELDS: tuple[str, ...] = (
    "name",
    "position",
    "test_account",
    "cm_access",
)

_USER_DISCARDED_NOTES: dict[str, str] = {
    "name": (
        "CDESK derives the display name from first_name + last_name, so a `name` "
        "sent on its own is ignored — send first_name/last_name to rename a user"
    ),
    "position": (
        "the position columns are per-language (position_sk/_en/...) and the "
        "plain `position` field is not persisted by the v3 user endpoint"
    ),
    "test_account": (
        "special-purpose account markers are not settable through the v3 API — "
        "set them in the CDESK UI"
    ),
    "cm_access": (
        "the CDESK/CMonitor access pair is not settable through the v3 API — "
        "set it in the CDESK UI"
    ),
}

# Keys CDESK returns on the user detail that are far too large to hand an LLM
# and are not user data. `grouped_notifications` is the per-user notification
# CONFIGURATION matrix: 22 module groups, each with nested notification rows
# (receiver lists, per-channel hidden flags). Measured live 2026-07-29 on this
# tenant: 353,498 chars — 99.3% of a 355,859-char `fieldset="all"` response,
# which rendered as ~1.1 MB through the tool layer and blew the MCP
# tool-result token cap outright. So `fieldset="all"` and `"custom"` both
# failed with "result exceeds maximum allowed tokens" and returned NOTHING —
# including `customFields`, which was `{}` (2 chars) and is the whole reason
# the custom-field docs tell you to call `fieldset="all"` in the first place.
#
# `returnFields[]` does NOT exclude it (verified: fieldset="all" +
# returnFields[id,name,customFields] still returned it, 448,969 chars), so
# dropping it here is the only way to make the documented path executable.
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


def register_user_tools(mcp: FastMCP, client: CdeskClient) -> None:
    """Register all User-module tools on the given FastMCP instance.

    Note: no EnumCache parameter. CDESK has no /v3/user/enums endpoint;
    user-type filter values (admin/operator/customer/user/easyclick/solver/
    authorizing_officer/guest) are documented in tool descriptions."""

    @mcp.tool(
        description=_LIST_USERS_DESC,
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
            if user_type is not None and user_type not in USER_TYPE_LABELS:
                raise ValueError(
                    f"user_type={user_type!r} is not a valid CDESK user type. "
                    f"Allowed values: {', '.join(USER_TYPE_LABELS)}. "
                    f"(See list_users description for what each label means.)"
                )
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
            # silently ignores it as an sb clause (verified live 2026-06-04).
            if only_deleted:
                params["onlyDeleted"] = "1"
            if sort:
                params["sort"] = sort
            if sb is not None:
                params["sb"] = encode_sb(sb)
            response = await client.get("v3/user", params=params)
        except ValueError as e:
            raise RuntimeError(f"list_users input error: {e}") from e
        except Exception as e:
            raise to_llm_error(e, operation="list_users") from e

        records, meta = unwrap_list(response)
        result = {"items": records, "meta": meta, "page": page, "per_page": per_page}
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
        description=_FIND_USER_DESC,
        annotations=ToolAnnotations(title="Find user", readOnlyHint=True),
    )
    async def find_user(text: str, max_results: int = 10) -> dict[str, Any]:
        """Intent helper — full-text search across CDESK's documented user
        q-search columns: id, name, nick, user (login), email, mobil, phone,
        description, personal_number. (Source: Model/User.php::whereFilter
        q-branch.)"""
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
        company_id: int | None = None,
        user_type: str | None = None,
        cdesk_access: bool | None = None,
        cmonitor_access: bool | None = None,
        role: str | None = None,
        primary_group_id: int | None = None,
        group_ids: list[int] | None = None,
        language: str | None = None,
        timezone: str | None = None,
        personal_number: str | None = None,
        ad_username: str | None = None,
        is_help_account: bool | None = None,
        is_email_account: bool | None = None,
        is_guest_account: bool | None = None,
        is_test_account: bool | None = None,
        custom_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            try:
                validate_custom_fields(custom_fields)
            except ValueError as e:
                raise RuntimeError(f"create_user input error: {e}") from e
            # The live API requires `name` AND `nick` on create (it rejects
            # the body with "Neplatné meno" / "Neplatná skratka pre výpis"
            # otherwise; first_name+last_name do NOT substitute for name —
            # verified live 2026-06-04). Fail fast with a clear message.
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
            # M9.2: require EXACTLY ONE password mechanism client-side. CDESK
            # would 4xx without one; supplying both is undefined (it may email a
            # generated password the operator didn't expect). Failing fast saves
            # a round-trip and gives a clearer message.
            if (password is not None) == bool(send_password_email):
                raise RuntimeError(
                    "create_user input error: provide exactly one of `password` "
                    "(plain text) or `send_password_email=True` (CDESK auto-"
                    "generates and emails the password), not both or neither."
                )
            # M12: auto_logout and ips are validated server-side on every
            # create even when CMonitor access is off. Reject obviously
            # invalid values up-front so the user gets a clear message
            # instead of a CDESK 400.
            _validate_auto_logout_and_ips(auto_logout, ips)
            # On create there is no stored record to preserve, so an unspecified
            # access position must NOT default to grant. Pass current="NN" so
            # setting one access flag doesn't silently grant the other (the
            # "YY" default is only safe on update, where the live value is read).
            cm_access = _combine_cm_access(cdesk_access, cmonitor_access, current="NN")
            body = _build_user_body(
                {
                    "login": login,
                    "email": email,
                    "auto_logout": auto_logout,
                    "ips": ips,
                    "password": password,
                    # M9.3: pass send_password_email through unchanged so an
                    # explicit False reaches CDESK as `sendPassword: false`.
                    "send_password_email": send_password_email,
                    "first_name": first_name,
                    "last_name": last_name,
                    "name": name,
                    "nick": nick,
                    "position": position,
                    "phone": phone,
                    "mobile_phone": mobile_phone,
                    "company_id": company_id,
                    "user_type": user_type,
                    "cm_access": cm_access,
                    "role": role,
                    "primary_group_id": primary_group_id,
                    "group_ids": group_ids,
                    "language": language,
                    "timezone": timezone,
                    "personal_number": personal_number,
                    "ad_username": ad_username,
                    "help_account": _yn(is_help_account),
                    "email_account": _yn(is_email_account),
                    "guest_account": _yn(is_guest_account),
                    "test_account": _yn(is_test_account),
                }
            )
            # customFields keys are already wire-format (cfield_*) — set after
            # the body build so they bypass the LLM-name→CDESK-name rename map.
            if custom_fields:
                body["customFields"] = custom_fields
            response = await client.post("v3/user", json=body)
        except (CdeskApiError, CdeskAuthError) as e:
            # CDESK-signaled errors (both RuntimeError subclasses) need
            # translation — must be caught before the plain-RuntimeError
            # passthrough below.
            raise to_llm_error(e, operation="create_user") from e
        except RuntimeError:
            raise  # client-side validators already emit a friendly message
        except Exception as e:
            raise to_llm_error(e, operation="create_user") from e
        return redact_secrets(annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=_USER_DISCARDED_FIELDS,
            field_notes=_USER_DISCARDED_NOTES,
        ))

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
        company_id: int | None = None,
        user_type: str | None = None,
        cdesk_access: bool | None = None,
        cmonitor_access: bool | None = None,
        role: str | None = None,
        primary_group_id: int | None = None,
        group_ids: list[int] | None = None,
        language: str | None = None,
        timezone: str | None = None,
        personal_number: str | None = None,
        ad_username: str | None = None,
        is_help_account: bool | None = None,
        is_email_account: bool | None = None,
        is_guest_account: bool | None = None,
        is_test_account: bool | None = None,
        custom_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            try:
                validate_custom_fields(custom_fields)
            except ValueError as e:
                raise RuntimeError(f"update_user input error: {e}") from e
            # M12: when present, auto_logout and ips have the same constraints
            # on update as on create.
            if auto_logout is not None or ips is not None:
                _validate_auto_logout_and_ips(
                    auto_logout if auto_logout is not None else 1,
                    ips if ips is not None else "*.*.*.*",
                )
            current = await _fetch_for_update(client, id)
            # cm_access: preserve any position the caller didn't specify.
            # M9.6: if the GET response has no usable cm_access we omit the
            # field entirely (don't fabricate a 'YY' fallback — that would
            # silently grant access positions we can't observe).
            cm_access: str | None = None
            if cdesk_access is not None or cmonitor_access is not None:
                raw_current = current.get("cm_access")
                if isinstance(raw_current, str) and len(raw_current) >= 2:
                    cm_access = _combine_cm_access(
                        cdesk_access, cmonitor_access, current=raw_current,
                    )
                elif cdesk_access is not None and cmonitor_access is not None:
                    # Both positions explicitly provided — no current needed.
                    cm_access = _combine_cm_access(cdesk_access, cmonitor_access)
                else:
                    raise RuntimeError(
                        f"update_user: cannot preserve unspecified cm_access "
                        f"position on user {id}: the GET response has no "
                        f"cm_access field. Pass BOTH cdesk_access and "
                        f"cmonitor_access explicitly, or omit both to keep "
                        f"the stored value."
                    )
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
                    "company_id": company_id,
                    "user_type": user_type,
                    "cm_access": cm_access,
                    "role": role,
                    "primary_group_id": primary_group_id,
                    "group_ids": group_ids,
                    "language": language,
                    "timezone": timezone,
                    "personal_number": personal_number,
                    "ad_username": ad_username,
                    "help_account": _yn(is_help_account),
                    "email_account": _yn(is_email_account),
                    "guest_account": _yn(is_guest_account),
                    "test_account": _yn(is_test_account),
                }
            )
            # customFields keys are already wire-format (cfield_*) — set after
            # the body build so they bypass the LLM-name→CDESK-name rename map.
            if custom_fields:
                body["customFields"] = custom_fields
            body["timestamp_check"] = current["timestamp_check"]
            response = await client.put(f"v3/user/{id}", json=body)
        except (CdeskApiError, CdeskAuthError) as e:
            # CDESK-signaled errors (both RuntimeError subclasses) need
            # translation — must be caught before the plain-RuntimeError
            # passthrough below.
            raise to_llm_error(e, operation="update_user", record_id=id) from e
        except RuntimeError:
            raise  # client-side validators already emit a friendly message
        except Exception as e:
            raise to_llm_error(e, operation="update_user", record_id=id) from e
        return redact_secrets(annotate_write_warnings(
            response if isinstance(response, dict) else {"data": response},
            sent_body=body,
            check_fields=_USER_DISCARDED_FIELDS,
            field_notes=_USER_DISCARDED_NOTES,
        ))

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


def _combine_cm_access(
    cdesk: bool | None,
    cmonitor: bool | None,
    *,
    current: str = "YY",
) -> str | None:
    """Combine two LLM-facing booleans into CDESK's 2-char `cm_access`
    string.

    **Position semantics** (verified against CDESK source — the OpenAPI
    spec gets this BACKWARDS):
        position 1 (index 0) = CDESK    (this function's first arg)
        position 2 (index 1) = CMonitor (this function's second arg)

    Sources:
        Wrapper/Customer.php:2124       — substr(cm_access, 1, 1)==='Y' → CM
        Model/User.php:7197             — substr(cm_access, 1, 1)!=='Y' → CM off
        Helper/SvcMirror/CmEnabledAdminAllowlist.php:12-13 — "cm_access[1]==='Y'"

    Returns None when both inputs are None so the field is omitted from
    the request body (preserves the current stored value on update; lets
    CDESK apply its tenant default on create).

    `current`: the current stored cm_access; used on update so a caller
    setting only one position preserves the other.
    """
    if cdesk is None and cmonitor is None:
        return None
    safe_current = current if len(current) >= 2 else "YY"
    pos1 = ("Y" if cdesk else "N") if cdesk is not None else safe_current[0]
    pos2 = ("Y" if cmonitor else "N") if cmonitor is not None else safe_current[1]
    return pos1 + pos2


async def _fetch_for_update(client: CdeskClient, id: int) -> dict[str, Any]:
    """Fetch a user and unwrap to the record dict; require timestamp_check.

    Same M7 finding as in tools/tasks.py and tools/customers.py: CDESK
    reads `timestamp_check` from the PUT body for the optimistic-lock
    check, not `updated_at`.

    fieldset=all is required: the v3 fieldset default (`extended`) no longer
    includes `timestamp_check` — only `all` carries it (verified live). It also
    guarantees `cm_access` is present for the preserve-position logic."""
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
    - text_search: full-text across the columns CDESK's whereFilter scans —
      id, name, nick, user (login), email, mobil, phone, description,
      personal_number. (Verified against Model/User.php q-branch.)
    - company_id: filter to users belonging to a specific customer
      (id_company column).
    - user_type: one of "admin", "operator", "customer", "user",
      "easyclick", "solver", "authorizing_officer", "guest", "cm".
      Invalid values are rejected client-side with the full list.
      CDESK derives the type from the user's ROOT GROUP; this filter applies
      that derivation. Two consequences, both verified live 2026-07-31:
      (a) a user with NO group membership matches NO user_type value at all,
      and the API cannot assign groups (it accepts `groups`/`group_primary`
      with a 200 and stores nothing — see docs/bugs.md), so every user created
      through this connector is unreachable by this filter until a group is
      assigned in the CDESK UI; (b) the record field `type` ("M"/"C"/...) is
      NOT this derived user_type — a user whose `type` is "C" is matched by
      `solver` when its group is a solver-root group, and is NOT matched by
      `customer`. Do not read one as the other.
    - only_deleted: when true, returns only soft-deleted users.
    - sb_raw: an sb filter object for the CDESK API v3 user list (JSON string
      or object, structured tree form; see docs/cdesk-api-v3.json). Advanced;
      mutually exclusive with the typed filters. Only the LIVE-VERIFIED working
      columns are applied server-side — id, name, email, personal_number,
      id_company, status, type, uname (the login), mobil, id_external, plus
      last_login_date and created_at (strict W3C datetime values), plus the
      no-col text leaf. Clauses on any other column (e.g. `work_position`,
      which is stored as a job_title_id FK) are STRIPPED, and the response
      carries an `unsupported_filters` block naming them.
      NB: the typed `user_type` param is the safe route for type
      filtering — an unknown `type` value in sb_raw is silently ignored
      by CDESK and returns ALL users (a false "match"), whereas
      `user_type` is validated against the allowed labels here.

    Pagination: page (1-based), per_page (default 20, max 100).

    Ordering (re-verified live 2026-07-31 — the backend behaviour CHANGED):
    omitting `sort` gives id ascending. A BARE COLUMN NAME now genuinely sorts
    ASCENDING by that column ("name" confirmed against a user deliberately
    given the highest id and an alphabetically-first name). A direction
    modifier is NOT honored — "-name" fell back to id ascending, the default
    order. So: for ascending order by one column pass the bare name;
    DESCENDING is not available server-side — page the set and sort
    client-side, and never assume a "-" prefix was applied. `meta` is usually
    empty (no total count) — detect the last page by a short/empty page.

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

    KNOWN LIMITATION (2026-06-04): the tenant currently ignores
    customFields writes silently — values do not persist. After any
    write, a get_user(id, fieldset="custom") read shows whether the
    value actually persisted; a value that did not stick was not
    stored despite the 200 (backend fix pending).
    """
)

_FIND_USER_DESC = inspect.cleandoc(
    """
    Intent helper for "find a user by name / login / email / mobil /
    description / personal_number". Runs a full-text search across the
    columns CDESK's whereFilter scans (id, name, nick, user (login),
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
    - `send_password_email=True`: CDESK auto-generates a strong password
      and emails it to the new user. Safer; no plaintext in your AI
      client's log.

    Access flags:
    - cdesk_access / cmonitor_access (both default unset → CDESK uses
      its tenant default). Pass True/False to set them explicitly. The
      two flags compose CDESK's cm_access 2-char string in the order
      (CDESK, CMonitor) — position 1 = CDESK, position 2 = CMonitor.
      (The OpenAPI spec documents this backwards; the actual CDESK
      source has CMonitor at position 2.)
    - is_help_account / is_email_account / is_guest_account /
      is_test_account: special-purpose account markers (mapped to
      CDESK's 'Y' string when True, omitted otherwise).

    Grouping:
    - company_id: the customer/company this user belongs to.
    - primary_group_id + group_ids: ACL groups (group_primary + groups
      in CDESK terms).
    - user_type: tenant-specific user type code; for the standard
      taxonomy see list_users' description.

    Locale: language (sk, en, cz, de, ru, hu, pl, lt, fr) and timezone
    (IANA name, e.g. 'Europe/Bratislava').

    custom_fields: flat dict of `cfield_<propertyId>_<basepropertyId>`
    keys → values (discover keys via get_user_custom_fields;
    select/relation fields take the option id). Unknown keys → 400.

    Returns: {data: <new id>, savedData: {...full record...}}.
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

    cm_access: position 1 = CDESK, position 2 = CMonitor (OpenAPI doc
    is backwards). Setting only `cdesk_access` or only `cmonitor_access`
    preserves the other position from the current stored value.

    custom_fields: flat dict of `cfield_<propertyId>_<basepropertyId>`
    keys → values (discover keys via get_user_custom_fields; read
    stored values via get_user(id, fieldset="all")).

    Returns: {data: <id>, savedData: {...full record...}}.
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
