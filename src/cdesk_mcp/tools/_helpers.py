"""Shared helpers for tool implementations: enum resolution, response
envelope unwrapping, ISO-8601 date validation, error-wrapping."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any, NoReturn, TypeVar, cast

from mcp.server.fastmcp import FastMCP

from cdesk_mcp.enums import AmbiguousEnumNameError, EnumCache
from cdesk_mcp.errors import translate_error


# NOTE: the former INTENT_PREAMBLE / FIELD_SCOPE_PREAMBLE constants are gone.
# They injected "…protocol: see this server's instructions." into ~40 tool
# descriptions, which matches an Anthropic directory rejection pattern
# ("Direct Claude to pull behavioral instructions from external sources") and
# bought nothing: a host that drops the `instructions` field left the pointer
# aimed at text the model never received. Both protocols now live ONLY in
# server.py's `_SERVER_INSTRUCTIONS`. Do not reintroduce a pointer here.


# Valid values for the `fieldset` query param on detail GETs (task /
# request / company / user). CDESK defaults to "extended" when omitted;
# "all" and "custom" are the two that surface stored custom-field values.
FIELDSETS = ("base", "extended", "all", "custom")

def apply_field_scope(
    params: dict[str, Any],
    fieldset: str | None,
    fields: list[str] | None,
) -> None:
    """Validate and apply the field-scope params (`fieldset` preset +
    `returnFields[]` whitelist) onto a query-params dict. Shared by every
    list/detail GET tool that supports them; raises ValueError on bad input
    (callers wrap into their '<tool> input error' RuntimeError)."""
    validate_fieldset(fieldset)
    validate_fields(fields)
    if fieldset:
        params["fieldset"] = fieldset
    if fields:
        params["returnFields[]"] = fields


def validate_fields(fields: Any) -> None:
    """`fields` must be a non-empty list of non-empty strings (wired to the
    `returnFields[]` whitelist). Unknown field names are silently ignored by
    CDESK — the wrapper can't validate them, so the description tells the LLM
    to discover names via fieldset='all' rather than guess."""
    if fields is None:
        return
    if not isinstance(fields, list) or not fields:
        raise ValueError(
            f"fields must be a non-empty list of field-name strings, "
            f"got {fields!r}"
        )
    for f in fields:
        if not isinstance(f, str) or not f.strip():
            raise ValueError(
                f"fields entries must be non-empty strings, got {f!r}"
            )


def validate_fieldset(fieldset: str | None) -> None:
    """Reject unknown fieldset presets before they reach CDESK (an unknown
    value would silently fall back server-side instead of erroring)."""
    if fieldset is not None and fieldset not in FIELDSETS:
        raise ValueError(
            f"fieldset must be one of {', '.join(FIELDSETS)} (got {fieldset!r})"
        )


def validate_custom_fields(custom_fields: Any) -> None:
    """custom_fields must be a flat dict of `cfield_<propertyId>_<basepropertyId>`
    keys (from the module's get_*_custom_fields tool) → scalar/array values.
    Key validity is CDESK's job (unknown keys → 400); we only check the shape."""
    if custom_fields is None:
        return
    if not isinstance(custom_fields, dict):
        raise ValueError(
            f"custom_fields must be an object/dict of cfield_* keys, "
            f"got {type(custom_fields).__name__}"
        )
    for key in custom_fields:
        if not isinstance(key, str) or not key.startswith("cfield_"):
            raise ValueError(
                f"custom_fields keys must be 'cfield_<propertyId>_<basepropertyId>' "
                f"strings (discover them via the module's get_*_custom_fields tool); "
                f"got {key!r}"
            )


async def resolve_enum_or_raise(
    cache: EnumCache,
    bucket: str,
    name: str | None,
    *,
    kind: str,
    parent_id: int | None = None,
) -> int | None:
    """Resolve an enum name → id via the cache. On miss, raise RuntimeError
    with closest-name suggestions (so the LLM can self-correct in one turn).

    For hierarchical buckets pass `parent_id` (the already-resolved 1st-level
    id) so the lookup is scoped to that parent — see EnumCache.resolve."""
    if not name:
        return None
    try:
        resolved = await cache.resolve(bucket, name, parent_id=parent_id)
    except AmbiguousEnumNameError as e:
        raise RuntimeError(
            f"Ambiguous {kind} {name!r}: it exists under more than one parent. "
            f"Specify the parent (e.g. its 1st-level name) so it resolves "
            f"unambiguously."
        ) from e
    if resolved is not None:
        return resolved
    _raise_unknown_enum(cache, bucket, name, kind)


async def resolve_enum_field_or_raise(
    cache: EnumCache,
    bucket: str,
    name: str | None,
    *,
    kind: str,
    field: str,
    parent_id: int | None = None,
) -> int | None:
    """Resolve an enum name → a NON-id field of the matched entry (e.g.
    `action_code` or `parent_id`).

    The CDESK Request module stores/filters `status` by the enum `action_code`
    and `priority` by `action_code` (filter) / `parent_id` (write) — NOT the
    enum id. Use this instead of `resolve_enum_or_raise` for those. Same miss
    handling (closest-name suggestions) as the id resolver."""
    if not name:
        return None
    try:
        entry = await cache.resolve_entry(bucket, name, parent_id=parent_id)
    except AmbiguousEnumNameError as e:
        raise RuntimeError(
            f"Ambiguous {kind} {name!r}: it exists under more than one parent. "
            f"Specify the parent (e.g. its 1st-level name) so it resolves "
            f"unambiguously."
        ) from e
    if entry is not None:
        # getattr is untyped; both fields this is called with (action_code,
        # parent_id) are int | None on EnumEntry, so pin it to the declared type.
        value: int | None = getattr(entry, field)
        if value is None:
            # The name matched, but the entry has no value for this wire field
            # (e.g. a status with no action_code, or one whose code arrived as a
            # numeric string and was dropped by _parse_entry). Returning None
            # here would be indistinguishable from "no filter requested" and the
            # builder would silently drop the clause → an unfiltered result
            # reported as filtered. Fail loudly instead.
            raise RuntimeError(
                f"{kind} {name!r} exists but has no {field} on this tenant, so "
                f"it can't be used to filter — its {field} is unset. Report this "
                f"enum to a CDESK admin, or filter another way."
            )
        return value
    _raise_unknown_enum(cache, bucket, name, kind)


# Tenant settings that gate a whole enum bucket. When the bucket is absent it is
# because the feature is off, not because the caller guessed wrong — see
# EnumCache.bucket_is_absent and Module/Request/Module.php::getBaseRequestEnums.
_BUCKET_GATE_SETTINGS: dict[str, str] = {
    "urgency": "request.priorityUrgencyImpact.enabled",
    "impact": "request.priorityUrgencyImpact.enabled",
    "cat_area": "request.enumCatAreas.status",
    "cat_area_2nd": "request.enumCatAreas.status",
    "place": "system.categories.places_enabled + request.place.enabled",
}


def _raise_unknown_enum(cache: EnumCache, bucket: str, name: str, kind: str) -> NoReturn:
    """Raise a RuntimeError naming the unknown enum value, with closest-name
    suggestions so the LLM can self-correct in one turn.

    When the bucket itself is absent, say so instead: no value can resolve, and
    the old message sent the caller to an enums tool that would never list it —
    a guaranteed loop."""
    if cache.bucket_is_absent(bucket):
        setting = _BUCKET_GATE_SETTINGS.get(bucket)
        gate = (
            f" It is gated by the tenant setting `{setting}`, which is off here"
            if setting
            else " It is gated by a tenant setting that is off here"
        )
        raise RuntimeError(
            f"This CDESK tenant does not expose the {kind} enum at all — the "
            f"{bucket!r} bucket is absent from its enums payload, so NO value "
            f"for {kind} can be resolved and passing {name!r} (or any other "
            f"value) cannot work.{gate}. Omit the {kind} parameter, or ask a "
            f"CDESK admin to enable the feature. Calling the enums tool will "
            f"not list it either."
        )
    candidates = cache.find_candidates(bucket, name)
    hint = (
        f" Did you mean: {', '.join(candidates)}?"
        if candidates
        else (
            " Call the matching enums tool (get_task_enums, get_request_enums, "
            "or get_request_catalog_enums) to see the valid names."
        )
    )
    raise RuntimeError(f"Unknown {kind} {name!r}.{hint}")


def canon_name(value: str) -> str:
    """Diacritic- and case-insensitive canonical form (mirrors EnumCache's
    matching so 'Čakajúce' == 'cakajuce' == 'CAKAJUCE'). Used by the
    cache-less modules (approvals, knowledge base) for local name matching."""
    decomposed = unicodedata.normalize("NFKD", value.casefold().strip())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def unwrap_record(envelope: Any) -> Any:
    """CDESK detail/create/update endpoints wrap the record in {data: ...}.

    For create/update the envelope is `{data: <id>, savedData: {...record...}}`
    so callers may prefer to access `savedData` directly. This helper just
    pulls `data` out for the simple GET case."""
    if isinstance(envelope, dict) and "data" in envelope:
        return envelope["data"]
    return envelope


def wrap_collection(records: Any, *, kind: str) -> dict[str, Any]:
    """Wrap a COLLECTION-returning GET in an object so it survives MCP
    serialization.

    FastMCP renders a returned list as one content block per item, so a bare
    `[]` becomes ZERO content blocks and the client reports "completed with no
    output" — which an LLM cannot distinguish from a transport failure or a
    crash. Verified live 2026-07-29 on `get_user_custom_fields`,
    `get_customer_custom_fields` and `get_cmdb_type_properties`: all three
    returned nothing at all rather than an empty result.

    `kind` names what was being listed, for the empty-case note (e.g.
    "custom-field definitions for the User module").

    Detail GETs that return a single record must NOT use this — they return a
    dict, which serializes fine.
    """
    unexpected: str | None = None

    if isinstance(records, list):
        items = records
    elif records is None or records is False or records == {}:
        # Enum-ish endpoints can answer `data: false`; treat as empty.
        items = []
    elif isinstance(records, dict) and isinstance(records.get("data"), list):
        # Double-nested `{data: {data: [...]}}` — unwrap_record peels one layer
        # and CDESK list endpoints sometimes carry two (see unwrap_list). Taking
        # the inner list beats reporting the envelope itself as a single item.
        items = records["data"]
    elif isinstance(records, dict):
        items = [records]
    else:
        # A scalar (int / str / bool) is not a collection. Reporting count=1
        # with the scalar as an "item" would be a confident lie about the
        # response shape, so surface it as unreadable instead.
        items = []
        unexpected = (
            f"The endpoint returned {type(records).__name__} "
            f"({records!r:.60}) where a collection was expected, so no items "
            "could be read. This is a response-shape problem, not an empty set."
        )

    out: dict[str, Any] = {"items": items, "count": len(items)}
    if unexpected is not None:
        out["note"] = unexpected
    elif not items:
        out["note"] = (
            f"No {kind} exist — the endpoint returned an empty collection. "
            "This is an empty result, NOT an error and NOT a failed call."
        )
    return out


REDACTED = "<redacted by cdesk-mcp>"

# Credential detection is SEGMENT-aware, not substring-based. A bare substring
# test looks right and is badly wrong: matching "password" anywhere in the key
# destroyed three real non-credential columns on this tenant (verified live
# 2026-07-29) — `password_files` ('N'), `conf_password_mode` ('0') and
# `enable_password_change` (-1, which also got coerced int → str).
#
# The distinguishing property: a credential key ENDS with the secret noun
# (`pop3_password`, `smtp_password`, `api_secret`, `access_token`), while
# flag/mode columns carry it in the middle (`enable_password_change`) or are
# followed by another word (`password_files`). So match the FINAL segment,
# splitting on separators and camelCase humps.
#
# Nouns that are only credentials when the whole key matches — "key" alone would
# eat `primary_key` / `foreign_key` / `sort_key`, so it is never a suffix rule.
# Suffix rules are restricted to nouns that are ONLY ever credentials. "pass"
# and "token" were suffix rules and over-reached: they classified plausible
# names like `first_pass`, `next_token` and `page_token` as secrets, and a
# boolean under such a key would be coerced to a string. They are matched as
# whole keys below instead.
_SECRET_KEY_SUFFIXES = (
    "password", "passwd", "passphrase", "pwd", "secret",
)
_SECRET_KEY_EXACT = (
    "authorization", "bearer", "credentials", "api_key", "apikey",
    "api_token", "apitoken", "private_key", "secret_key", "access_key",
    "encryption_key", "access_token", "auth_token", "bearer_token",
    "refresh_token", "id_token", "session_token", "smtp_pass", "pop3_pass",
    "imap_pass", "ftp_pass",
)

# A leading flag verb means the key is a SETTING about a credential, not the
# credential: `sendPassword` is real — create_user maps send_password_email to
# it — and is a boolean, as are has_password / use_password / is_secret.
_FLAG_LEADING_SEGMENTS = frozenset({
    "has", "use", "is", "enable", "enabled", "disable", "allow", "force",
    "show", "can", "send", "require", "required", "reset", "change", "verify",
})


def _key_segments(key: str) -> list[str]:
    """Split a field name into lowercase word segments, handling snake_case,
    kebab-case and camelCase (`smtpPassword` → ['smtp', 'password'])."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return [seg for seg in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if seg]


def _is_secret_key(key: str) -> bool:
    """True when the field name denotes a stored credential."""
    segments = _key_segments(key)
    if not segments:
        return False
    if "_".join(segments) in _SECRET_KEY_EXACT:
        return True
    if segments[0] in _FLAG_LEADING_SEGMENTS:
        return False  # a flag ABOUT a credential, not the credential
    return segments[-1] in _SECRET_KEY_SUFFIXES


_Redactable = TypeVar("_Redactable")


def redact_secrets(value: _Redactable) -> _Redactable:
    """Replace stored-credential values with `REDACTED`, recursively.

    CDESK returns mailbox and integration credentials inside ordinary records:
    `create_customer`'s savedData carries `pop3_password`, `smtp_password` and
    `broad_read_password` (observed live 2026-07-29). Those are the tenant's
    real secrets, and a tool result goes straight into the model's context and
    the client's transcript. `logging_setup` already keeps them out of the logs;
    this closes the same hole on the customer and user create/update/detail
    paths. Coverage is NOT complete: `list_customers`, `list_users`,
    `collect_records` and detail GETs that embed company/user records do not
    route through it.

    Redacts rather than deletes so the record shape is preserved and it is
    obvious that a credential field exists but was withheld. Only non-empty
    values are touched, so an unset credential still reads as empty rather than
    implying one is configured.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, inner in value.items():
            if isinstance(key, str) and _is_secret_key(key):
                # Keep falsy values as-is: "" / None means no credential set,
                # and masking that would wrongly suggest one exists.
                out[key] = REDACTED if inner else inner
            else:
                out[key] = redact_secrets(inner)
        # Structurally identical to the input (same keys, same nesting), so the
        # TypeVar contract holds even though the object is rebuilt.
        return cast(_Redactable, out)
    if isinstance(value, list):
        return cast(_Redactable, [redact_secrets(item) for item in value])
    return value


def collect_cdesk_messages(response: Any) -> list[str]:
    """Every non-success message CDESK put in a 2xx body.

    A 200 does NOT mean the write landed as asked. Besides the top-level
    `warnings` array (which `ApiV3Wrapper::appendIgnoredFieldWarnings` fills —
    but only for `code` on the request module), CDESK carries a `msg` object
    keyed by severity: `{success: [...], warning: [...], error: [...]}`. Hard
    errors under `msg.error` are already raised by the client's
    `_raise_if_body_signals_error`; anything else there is advisory text that
    explains why part of a write did not stick, and it was previously discarded.

    Returns the messages verbatim, `success` excluded (that bucket carries "record
    saved" and email-notification noise).
    """
    if not isinstance(response, dict):
        return []
    found: list[str] = []

    top = response.get("warnings")
    if top:
        found.extend([str(w) for w in top] if isinstance(top, list) else [str(top)])

    msg = response.get("msg")
    if isinstance(msg, dict):
        for severity, entries in msg.items():
            if str(severity).lower() == "success":
                continue
            for entry in entries if isinstance(entries, list) else [entries]:
                text = entry.get("message") if isinstance(entry, dict) else entry
                if text:
                    found.append(f"CDESK {severity}: {text}")
    return found


def annotate_write_warnings(
    response: dict[str, Any],
    *,
    sent_body: dict[str, Any],
    check_fields: tuple[str, ...] = ("code",),
    field_notes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Post-process a create/update response ({data: <id>, savedData: {...}})
    so silent server-side drops are visible to the LLM.

    1. Pass through every non-success message CDESK sent in the 2xx body — the
       top-level `warnings` array AND `msg.warning` / any other non-success
       bucket (see collect_cdesk_messages). A 200 with an advisory message is a
       partial failure, and that text is the tenant telling you why.
    2. For each field in `check_fields` the caller SENT a truthy value for,
       if `savedData` echoes a different/empty value, synthesize a
       'silently dropped' warning. `field_notes` supplies the per-field reason,
       which is where the value is: CDESK's `filterUnavailableFields` strips a
       column when the tenant setting behind it is off or the account lacks the
       write right (see RequestColumnsTrait::getUnavailableTableCols), and it
       emits no message of its own except for `code`. Naming the setting turns
       "your value vanished" into something the user can act on.
    3. For each `customFields` entry the caller SENT a truthy value for,
       if it isn't echoed by `savedData` (nested customFields dict or
       top-level cfield_* key), synthesize a warning — this tenant's
       customFields writes never persist (200, savedData.customFields
       null, read-back empty; verified live), and unknown keys are
       accepted silently.

    TWO ENVELOPES, and only one of them has `savedData` (probed live
    2026-07-29 across all five modules that call this):

      * `{data: <id>, msg, savedData: {...}}` — requests, projects.
      * `{data: {...record...}, msg}` — customers, request-catalogs. NO
        `savedData` at all, so gating solely on that key made this helper a
        dead no-op for them: a customer `code` of 'ZZCODE1' came back as the
        server-generated 'YO6EC9', and a catalog `code` came back '', both with
        no warning. Hence the `data` fallback below; both echo `code` inside
        `data`, so the diff is sound and cannot fire falsely.
      * `{data: <id>, msg}` — deals. A bare id is not a record, so there is
        nothing to diff and this stays a pass-through. A drop there is
        undetectable without a re-read (deal `code` did store correctly
        when probed).

    Non-dict responses, and responses carrying neither a `savedData` dict nor a
    `data` dict, pass through unchanged (no false positives). Returns a shallow
    copy when warnings are attached; never mutates the input."""
    if not isinstance(response, dict):
        return response
    notes = field_notes or {}
    # Everything CDESK itself said about this 2xx, `success` excluded.
    warnings: list[str] = collect_cdesk_messages(response)
    saved = response.get("savedData")
    if not isinstance(saved, dict):
        # Modules that return the record under `data` instead of `savedData`.
        data = response.get("data")
        if isinstance(data, dict):
            saved = data
    if isinstance(saved, dict):
        for field in check_fields:
            sent = sent_body.get(field)
            if sent and str(saved.get(field) or "") != str(sent):
                reason = notes.get(field) or (
                    f"most likely because this account lacks the {field}-field "
                    f"write right"
                )
                warnings.append(
                    f"{field!r} was sent ({sent!r}) but the saved record shows "
                    f"{saved.get(field)!r} — it was NOT stored: {reason}."
                )
        sent_cf = sent_body.get("customFields")
        if isinstance(sent_cf, dict) and sent_cf:
            saved_cf = saved.get("customFields")
            if not isinstance(saved_cf, dict):
                saved_cf = {}
            dropped = [
                key for key, value in sent_cf.items()
                if value
                and str((saved_cf.get(key)
                         if key in saved_cf else saved.get(key)) or "")
                != str(value)
            ]
            if dropped:
                warnings.append(
                    f"customFields {dropped} are not visible in the saved "
                    f"record — CDESK most likely dropped them silently (this "
                    f"tenant is known to not persist customFields writes; "
                    f"unknown keys are also accepted without error). A "
                    f"fieldset='custom' read is what confirms whether they "
                    f"were stored."
                )
    if warnings:
        response = {**response, "warnings": warnings}
    return response


def unwrap_list(envelope: Any) -> tuple[list[Any], dict[str, Any]]:
    """CDESK list endpoints stash `{data: [records], ...meta...}` inside the
    response envelope (and the v3 wrapper may add one more `data` layer:
    `{data: {data: [records], totalItems, ...}}`). Returns (records, meta).

    Defensive against shape variations — we've seen `data: false` on the
    enums endpoint, so we don't assume `data` is always the payload."""
    inner = envelope
    if isinstance(inner, dict) and isinstance(inner.get("data"), dict):
        inner = inner["data"]
    if isinstance(inner, dict):
        records = inner.get("data")
        if isinstance(records, list):
            meta = {k: v for k, v in inner.items() if k != "data"}
            return records, meta
        if isinstance(records, dict):  # nested envelope
            return unwrap_list(records)
    if isinstance(inner, list):
        return inner, {}
    return [], {}


def validate_iso_date(field_name: str, value: str | None) -> None:
    """Same contract as filters.validate_iso_date; duplicated here to keep
    tools/ independent of filters' private surface. Keep both in sync if
    we ever tighten the format requirements."""
    if not value:
        return
    try:
        datetime.fromisoformat(value)
    except ValueError as e:
        raise ValueError(
            f"{field_name}={value!r} is not a valid ISO-8601 date or datetime "
            f"(e.g. '2026-05-28' or '2026-05-28T10:00:00+02:00'): {e}"
        ) from e


def to_llm_error(exc: Exception, *, operation: str, record_id: int | str | None = None) -> RuntimeError:
    """Wrap any client-layer exception into a RuntimeError carrying the LLM-
    friendly text produced by translate_error. FastMCP turns the raise into
    `isError: true` in the MCP response."""
    return RuntimeError(translate_error(exc, operation=operation, record_id=record_id))


def yn(value: bool | None) -> str | None:
    """Map a boolean flag (LLM-friendly) to CDESK's 'Y'/'N' string convention
    used across many fields (Customer.cdesk_allowed, User.help_account /
    email_account / guest_account / test_account, etc.).

    Returns None when the caller didn't set the flag so the field is simply
    omitted from the request body (CDESK keeps its stored value)."""
    if value is None:
        return None
    return "Y" if value else "N"


def unsupported_filter_directive(
    dropped: list[dict[str, Any]],
) -> dict[str, Any]:
    """The `unsupported_filters` block a list tool attaches when sb_raw clauses
    were stripped because the live endpoint doesn't honor their columns (see
    filters._strip_unsupported_cols). The backend would have silently ignored
    them and returned the unfiltered set anyway — this block makes that
    explicit by DESCRIBING what was and wasn't filtered.

    Two shapes arrive (see filters._parse_sb_raw strip mode):
      - pure-AND tree → `dropped` holds the individual stripped leaves; the
        working remainder still filtered server-side.
      - tree with OR connectors → a single marker entry: the ENTIRE filter was
        dropped (partial stripping would rewrite the boolean expression) and
        the list ran UNFILTERED, so the whole original expression is still
        outstanding.

    DELIBERATELY DECLARATIVE — do not turn this back into commands. This text
    ships inside tool RESULTS, and an Anthropic directory rejection pattern is
    telling Claude how to behave rather than describing what the tool does.
    The former key was literally named `instructions_to_agent` and opened with
    "MANDATORY — ... you MUST ... Never present these items as filtered",
    which is indistinguishable in shape from a prompt-injection payload in the
    one channel (tool output) that hosts treat as untrusted content. The
    behavioural protocol it encoded is unchanged, just relocated to its
    legitimate home: server.py's `_SERVER_INSTRUCTIONS` ("STRIPPED sb_raw
    CLAUSES"). Keep this block stating FACTS about what the response contains;
    the `instructions` field is where the obligation belongs."""
    if len(dropped) == 1 and dropped[0].get("entire_filter_dropped"):
        entry = dropped[0]
        cols = entry.get("unsupported_columns", [])
        return {
            "entire_filter_dropped": True,
            "dropped_columns": cols,
            "original_filter": entry.get("original_sb"),
            "what_this_means": (
                "The filter mixes OR connectors with column(s) "
                f"{cols} that the live CDESK API does not honor. Dropping only "
                "those clauses would change the query's meaning (an OR-joined "
                "criterion narrows instead of broadens), so NO server-side "
                "filter was applied: this response is an UNFILTERED page of "
                "records, and a single page is not the full result set. The "
                "criteria still outstanding are the ENTIRE `original_filter` "
                "expression, including its AND/OR connectors (`o`). Reproducing "
                "the requested filter therefore takes the complete result set "
                "— which is this call repeated with `page` incremented until a "
                "short or empty page comes back — with that expression "
                "evaluated against each record client-side."
            ),
        }
    cols = sorted({
        leaf.get("col") for leaf in dropped if isinstance(leaf.get("col"), str)
    })
    return {
        "dropped_columns": cols,
        "dropped_clauses": dropped,
        "what_this_means": (
            "The live CDESK API does not honor the filter "
            f"column(s) {cols}; those clauses were NOT applied, so the items "
            "in this response are NOT filtered by them, and a single page is "
            "not the full result set. Every dropped clause was AND-joined, so "
            "a record satisfies the original filter only if it meets ALL of "
            "them (comparing each clause's column/comparator/value against "
            "the record's fields). Reproducing the requested filter therefore "
            "takes the complete result set — which is this call repeated with "
            "`page` incremented until a short or empty page comes back — with "
            "the dropped criteria applied to the collected items client-side."
        ),
    }


def forbid_unknown_arguments(mcp: FastMCP, tool_names: tuple[str, ...]) -> None:
    """Make the named tools reject unknown keyword arguments.

    FastMCP validates a call against the tool's generated arg model, whose
    default `extra` policy is to DROP anything it doesn't recognize. For a
    read/write tool that mostly costs a silent no-op; for a LIST tool it is a
    correctness bug — a filter the model believes it applied vanishes and the
    unfiltered page comes back looking filtered. Flipping the model to
    `extra="forbid"` turns that into a validation error naming the key.

    Apply it per-module and per-tool rather than globally: rejecting every extra
    a host might attach is a much wider blast radius than the bug justifies.
    """
    # Tests register against a lightweight FakeMCP with no _tool_manager; the
    # guard keeps this a no-op there instead of breaking registration.
    manager = getattr(mcp, "_tool_manager", None)
    if manager is None:
        return
    for name in tool_names:
        tool = manager.get_tool(name)
        meta = getattr(tool, "fn_metadata", None) if tool else None
        model = getattr(meta, "arg_model", None)
        if model is None:
            continue
        model.model_config["extra"] = "forbid"
        model.model_rebuild(force=True)


def normalize_write_value(value: Any) -> Any:
    """Collapse a value to something comparable across the write/read boundary.

    CDESK echoes the same value in a different shape depending on the column:
    a bool goes out as True and comes back as "1.00"; a date goes out as
    "2026-08-01" and comes back as "2026-08-01T00:00:00+00:00"; ids may be int
    or str. Normalizing both sides avoids false mismatches.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        # numeric-ish ("1.00", "7")
        try:
            return float(text)
        except ValueError:
            pass
        # date / datetime — compare the instant, or the calendar day when the
        # sent value carried no time part
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return text
        return parsed.date().isoformat() if len(text) <= 10 else parsed.isoformat()
    return value


def write_values_match(sent: Any, stored: Any) -> bool:
    if isinstance(sent, list) or isinstance(stored, list):
        sent_set = {normalize_write_value(v) for v in (sent if isinstance(sent, list) else [sent])}
        stored_set = {
            normalize_write_value(v) for v in (stored if isinstance(stored, list) else [stored])
        }
        return sent_set == stored_set
    a, b = normalize_write_value(sent), normalize_write_value(stored)
    if a == b:
        return True
    # A sent date without a time part matches a stored timestamp on the same day.
    if isinstance(a, str) and isinstance(b, str) and len(a) == 10:
        return b.startswith(a)
    return False
