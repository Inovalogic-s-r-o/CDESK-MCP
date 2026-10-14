"""Shared helpers for tool implementations: enum resolution, response
envelope unwrapping, ISO-8601 date validation, error-wrapping."""

from __future__ import annotations

import contextvars
import functools
import inspect
import re
import unicodedata
from datetime import datetime, timezone
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, TypeVar, cast

from mcp.server.fastmcp import FastMCP

from difflib import SequenceMatcher

from cdesk_mcp.enums import _FUZZY_MIN_RATIO, AmbiguousEnumNameError, EnumCache
from cdesk_mcp.errors import translate_error
from cdesk_mcp.tz import tenant_timezone


# Valid values for the `fieldset` query param on detail GETs (task /
# request / company / user). CDESK defaults to "extended" when omitted;
# "all" and "custom" are the two that surface stored custom-field values.
FIELDSETS = ("base", "extended", "all", "custom")

# Appended to every description that exposes a free-text filter. CDESK's
# text search splits the value on whitespace and substring-matches each word
# INDEPENDENTLY, so a multi-word value is not a phrase and word order is
# ignored: 'mandatory mcp' matches a record named "mcp test mandatory", and
# 'and' matches "mandatory". This matters beyond ergonomics —
# verify_claims checks that a quote is a literal substring of the record, NOT
# that the record is the right one, so a false text match can survive
# verification with an accurate quote from the wrong record.
TEXT_MATCH_NOTE = (
    "Text matching is WORD-SPLIT, not phrase: the value is split on whitespace "
    "and each word is substring-matched independently (case-insensitive, "
    "AND-joined, word order ignored), so \"manual A\" also matches a record "
    "titled \"manual B\". Short words are near-useless as filters — to target "
    "one specific record, filter on its id or code, not its name."
)

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


# Set while a tool that is not read-only runs (mark_write_tools). The enum
# resolvers then refuse a value the tenant has switched off (in_use = 0), as
# the CDESK form does; read-only tools — list filters — still accept it, since
# old records can sit in a value that is no longer in use.
_WRITE_CALL: contextvars.ContextVar[bool] = contextvars.ContextVar("cdesk_write_call", default=False)


def mark_write_tools(mcp: FastMCP) -> None:
    """Wrap every registered tool whose annotations are not read-only so the
    enum resolvers know a write is running. Applied once in build_server; a
    no-op on the test FakeMCP (tests that need it set _WRITE_CALL directly)."""
    manager = getattr(mcp, "_tool_manager", None)
    if manager is None:
        return
    for tool in manager.list_tools():
        annotations = getattr(tool, "annotations", None)
        if annotations is not None and annotations.readOnlyHint is True:
            continue
        fn = getattr(tool, "fn", None)
        if fn is None:
            continue

        def wrap(inner: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(inner)
            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                token = _WRITE_CALL.set(True)
                try:
                    result = inner(*args, **kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                    return result
                finally:
                    _WRITE_CALL.reset(token)
            return wrapped

        tool.fn = wrap(fn)
        tool.is_async = True


def _refuse_if_not_in_use(cache: EnumCache, bucket: str, entry: Any, name: str, kind: str) -> None:
    if not _WRITE_CALL.get() or getattr(entry, "in_use", None) is not False:
        return
    rows = cache.snapshot().get(bucket, [])
    active = [str(r["name"]) for r in rows
              if isinstance(r, dict) and r.get("in_use") is not False and r.get("name")]
    raise RuntimeError(
        f"{kind} {name!r} is not in use on this tenant (switched off in CDESK, so "
        f"its form does not offer it); in use: {', '.join(active) or '(none)'}."
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
    if _WRITE_CALL.get():
        # A write needs the entry, not just the id, to see its in_use flag.
        try:
            entry = await cache.resolve_entry(bucket, name, parent_id=parent_id)
        except AmbiguousEnumNameError as e:
            raise RuntimeError(
                f"Ambiguous {kind} {name!r}: it exists under more than one parent. "
                f"Specify the parent (e.g. its 1st-level name) so it resolves "
                f"unambiguously."
            ) from e
        if entry is not None:
            _refuse_if_not_in_use(cache, bucket, entry, name, kind)
            return int(entry.id)
        _raise_unknown_enum(cache, bucket, name, kind)
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
        _refuse_if_not_in_use(cache, bucket, entry, name, kind)
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
# EnumCache.bucket_is_absent.
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
    pointing the caller at an enums tool that never lists it would only loop."""
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


@dataclass(frozen=True)
class EnumLabel:
    """One enum-valued record field and how to read its label.

    `by` says which part of the enum entry the record stores: the enum `id`,
    its `action_code` (Request status/priority — NOT the id), or the `key` of a
    key/value bucket (the request type "druh", e.g. "H"). `name_key` is the key
    the label is written to, matching the tool parameter that takes the name."""

    field: str
    bucket: str
    name_key: str
    by: Literal["id", "action_code", "key"] = "id"


async def add_enum_names(
    cache: EnumCache, records: Iterable[Any], labels: tuple[EnumLabel, ...]
) -> None:
    """Write each enum field's label next to its raw value, in place.

    CDESK returns enum fields as bare numbers, and for requests `priority: 2`
    is an ACTION CODE whose label may be "dlhodobá" while another entry is
    literally named "2 - vysoká" — read as a name, the number is the wrong
    answer. The raw value is kept untouched (write-warning
    diffs and update paths read it); `<field>_name` is inserted right after it.

    Additive and best-effort: an existing key is never overwritten (CDESK's own
    `fieldset=all` already sends e.g. `cat_type_name`), a value with no label
    (empty, inactive, bucket switched off) gets no key, and an enum fetch
    failure leaves the records exactly as CDESK sent them."""
    targets = [r for r in records if isinstance(r, dict)]
    if not targets or not any(lbl.field in r for r in targets for lbl in labels):
        return
    try:
        await cache.load()
    except Exception:  # noqa: BLE001 — labels are a convenience, never a failure
        return
    maps: dict[tuple[str, str], dict[Any, str]] = {}
    for lbl in labels:
        if (lbl.bucket, lbl.by) in maps:
            continue
        if lbl.by == "action_code":
            maps[lbl.bucket, lbl.by] = cache.action_code_name_map(lbl.bucket)
        elif lbl.by == "key":
            maps[lbl.bucket, lbl.by] = cache.key_name_map(lbl.bucket)
        else:
            maps[lbl.bucket, lbl.by] = cache.id_name_map(lbl.bucket)
    apply_enum_names(targets, labels, maps)


def enum_rows_name_map(rows: Any, by: str) -> dict[Any, str]:
    """value → display name over raw enum rows, for modules that fetch their
    enums themselves instead of through an EnumCache. Rows are the tenant's
    live payload ({id, action_code, name, …} or {key, value}); nothing here is
    a fixed label. First match wins, like EnumCache.action_code_name_map.

    `lang_name` wins over `name` when present: the v1 work-order settings rows
    keep the base English text in `name` ("New") and the tenant's display name
    in `lang_name` ("Nový", what the CDESK UI and the v3 enums show); in v3
    rows the two are equal."""
    out: dict[Any, str] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get(by)
        if by == "key":
            name = row.get("value")
        else:
            lang_name = row.get("lang_name")
            name = lang_name if isinstance(lang_name, str) and lang_name else row.get("name")
        if isinstance(value, str) and value.isdigit() and by != "key":
            value = int(value)
        if value is not None and not isinstance(value, bool) and isinstance(name, str) and name:
            out.setdefault(value, name)
    return out


def apply_enum_names(
    records: Iterable[Any],
    labels: tuple[EnumLabel, ...],
    maps: Mapping[tuple[str, str], Mapping[Any, str]],
) -> None:
    """Insert `<field>_name` after each enum field, from ready-made name maps
    keyed by (bucket, by). Same rules as add_enum_names: additive, never
    overwrites, no key when there is no label. A label whose map is missing
    is skipped."""
    for rec in records:
        if not isinstance(rec, dict):
            continue
        additions: dict[str, tuple[str, str]] = {}  # field -> (name_key, label)
        for lbl in labels:
            if lbl.field not in rec or lbl.name_key in rec:
                continue
            if any(key == lbl.name_key for key, _ in additions.values()):
                continue  # an alias field (req_status vs status) already labelled it
            value = rec[lbl.field]
            if isinstance(value, str) and value.isdigit() and lbl.by != "key":
                value = int(value)
            if value is None or isinstance(value, (bool, dict, list)):
                continue
            label = maps.get((lbl.bucket, lbl.by), {}).get(value)
            if label:
                additions[lbl.field] = (lbl.name_key, label)
        if not additions:
            continue
        # Rebuild so each label sits right after its raw field.
        rebuilt: dict[str, Any] = {}
        for key, val in rec.items():
            rebuilt[key] = val
            if key in additions:
                name_key, label = additions[key]
                rebuilt[name_key] = label
        rec.clear()
        rec.update(rebuilt)


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
    crash (e.g. `get_user_custom_fields`, `get_customer_custom_fields` and
    `get_cmdb_type_properties` on a tenant with no definitions).

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
# would mask real non-credential columns — `password_files` ('N'),
# `conf_password_mode` ('0') and `enable_password_change` (-1, which would also
# be coerced int → str).
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
# and "token" are not suffix rules: they would classify plausible names like
# `first_pass`, `next_token` and `page_token` as secrets, and a boolean under
# such a key would be coerced to a string. They are matched as whole keys
# below instead.
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
# credential: `sendPassword` (the v3 spec's name for the e-mail-a-password
# flag) is a boolean, as are has_password / use_password / is_secret.
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
    `broad_read_password`. Those are the tenant's
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
    `warnings` array (which CDESK fills only for `code` on the request
    module), CDESK carries a `msg` object
    keyed by severity: `{success: [...], warning: [...], error: [...]}`. Hard
    errors under `msg.error` are already raised by the client's
    `_raise_if_body_signals_error`; anything else there is advisory text that
    explains why part of a write did not stick.

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
       which is where the value is: CDESK strips a column when the tenant
       setting behind it is off or the account lacks the write right, and it
       emits no message of its own except for `code`. Naming the setting turns
       "your value vanished" into something the user can act on.
    3. For each `customFields` entry the caller SENT a truthy value for,
       if it isn't echoed by `savedData` (nested customFields dict or
       top-level cfield_* key), synthesize a warning — customFields writes
       can fail to persist with a 200 (savedData.customFields null, read-back
       empty), and unknown keys are accepted silently.

    TWO ENVELOPES, and only one of them has `savedData`:

      * `{data: <id>, msg, savedData: {...}}` — requests, projects.
      * `{data: {...record...}, msg}` — customers, request-catalogs. NO
        `savedData` at all, so the diff falls back to `data` (a customer
        `code` can come back server-generated, a catalog `code` empty); both
        echo `code` inside `data`, so the diff is sound and cannot fire falsely.
      * `{data: <id>, msg}` — deals. A bare id is not a record, so there is
        nothing to diff and this stays a pass-through. A drop there is
        undetectable without a re-read.

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
            # write_values_match, not a string compare: CDESK echoes a datetime
            # in UTC (09:30+01:00 → 08:30+00:00), a bool as "1.00", an id as a
            # string — the same value, not a dropped one.
            if sent and not write_values_match(sent, saved.get(field)):
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


def notify_block_value(given: int | None) -> int | None:
    """The `notify_block` a write sends: only what the call passed.
    CDESK notifies by default; a call passes notify_block=1 when the user does
    not want a notification for that write."""
    return given


_NOTIFY_FLAGS = frozenset({"notify_block", "notify_customer_block", "notify_sms_block"})


def without_notify_flags(body: dict[str, Any]) -> dict[str, Any]:
    """The write body minus the notification flags — what an update actually
    changes. An update that carries only notify_block changes nothing and is
    refused as empty."""
    return {k: v for k, v in body.items() if k not in _NOTIFY_FLAGS}


def request_status_group(action_code: Any) -> str | None:
    """CDESK's group of a request status action code (its solved and closed
    status groups; open = below 80) — fixed system codes, the same on every
    tenant. None for a code in no group."""
    if isinstance(action_code, bool) or not isinstance(action_code, int):
        return None
    if action_code < 80:
        return "open"
    if action_code in (80, 81, 85, 86):
        return "completed"
    if action_code in (90, 91, 95):
        return "closed"
    return None


def task_status_group(action_code: Any) -> str | None:
    """CDESK's group of a task status action code (open below 80, closed from
    80)."""
    if isinstance(action_code, bool) or not isinstance(action_code, int):
        return None
    return "open" if action_code < 80 else "closed"


def with_status_groups(rows: Any, group_of: Callable[[Any], str | None]) -> Any:
    """Copies of the status enum rows with `status_group` beside the action
    code (the cached rows are not touched)."""
    if not isinstance(rows, list):
        return rows
    out: list[Any] = []
    for row in rows:
        if isinstance(row, dict):
            row = {**row, "status_group": group_of(row.get("action_code"))}
        out.append(row)
    return out


def list_meta(
    meta: dict[str, Any], paging: dict[str, Any], *, dropped: Any = None,
) -> dict[str, Any]:
    """A list result's `meta`: what the body carried, plus `total` (records
    matching the filter, across all pages) and `last_page` from the v3 paging
    headers (CdeskClient.get_page). When the tool stripped filter clauses CDESK
    cannot apply (`dropped`), CDESK counted WITHOUT them, so the total would
    describe a different set — it is left out and `total_omitted` says why."""
    out = dict(meta)
    if paging.get("last_page") is not None:
        out["last_page"] = paging["last_page"]
    total = paging.get("total")
    if total is not None:
        if dropped:
            out["total_omitted"] = (
                "CDESK counted the records without the criteria listed under "
                "unsupported_filters, so its total does not describe this result"
            )
        else:
            out["total"] = total
    return out


def unwrap_list(envelope: Any) -> tuple[list[Any], dict[str, Any]]:
    """CDESK list endpoints stash `{data: [records], ...meta...}` inside the
    response envelope (and the v3 wrapper may add one more `data` layer:
    `{data: {data: [records], totalItems, ...}}`). Returns (records, meta).

    Defensive against shape variations — the enums endpoint can answer
    `data: false`, so `data` is not assumed to always be the payload."""
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
    were stripped because the endpoint doesn't honor their columns (see
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

    DELIBERATELY DECLARATIVE, not commands. This text ships inside tool
    RESULTS — the channel hosts treat as untrusted content — and an imperative
    there ("you MUST ... never present these items as filtered") is
    indistinguishable in shape from a prompt-injection payload. A tool result
    describes what it contains; it does not tell the model how to behave. The
    behavioural rule lives in server.py's `_SERVER_INSTRUCTIONS` ("STRIPPED
    sb_raw CLAUSES"). Keep this block stating FACTS about what the response
    contains; the `instructions` field is where the obligation belongs."""
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
    correctness failure — a filter the model believes it applied vanishes and the
    unfiltered page comes back looking filtered. Flipping the model to
    `extra="forbid"` turns that into a validation error naming the key.

    Apply it per-module and per-tool rather than globally: rejecting every extra
    a host might attach is a much wider blast radius than the problem justifies.
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


FIELDS_NOT_RETURNED_NOTE = (
    "CDESK returned none of these requested fields on any record here — CDESK "
    "drops a field name it does not know without a word (a field it knows comes "
    "back even when empty). Usually a misspelled or wrong name: fieldset='all' "
    "on one record shows the real names."
)


def fields_not_returned(fields: Any, records: Any) -> list[str]:
    """The requested `fields` that appear on none of the returned records — the
    names CDESK silently dropped. A known field comes back even when null
    (`due_date: None`), an unknown one (`bogus_field`, `req_due_date`, users'
    `id_company`) is simply absent. Empty when there is nothing to compare (no fields, no records)."""
    if not isinstance(fields, list) or not fields:
        return []
    rows = [r for r in (records if isinstance(records, list) else [records]) if isinstance(r, dict)]
    if not rows:
        return []
    seen: set[str] = set()
    for row in rows:
        seen.update(row)
    return [f for f in fields if isinstance(f, str) and f not in seen]


def _with_fields_check(result: Any, fields: Any) -> Any:
    if not isinstance(result, dict):
        return result
    items = result.get("items")
    records = items if isinstance(items, list) else result
    missing = fields_not_returned(fields, records)
    if missing:
        result["fields_not_returned"] = {"fields": missing, "note": FIELDS_NOT_RETURNED_NOTE}
    return result


# --- what the LLM receives: compact enum rows and list items ------------------
# CDESK's enum rows carry icon and colour styles, admin/sort/timestamp columns,
# internal settings flags and the name in every language, so an enums payload
# runs to tens of thousands of characters; list items carry many empty values
# (often half or more of their fields). Only the model's context pays for this.

_ENUM_ROW_NOISE = frozenset({
    "custom_icon_class", "predefined_icon_color_class", "custom_icon_color_style",
    "custom_icon_background_style", "edit", "delete", "sort", "order",
    "enum_name_id", "admin_id", "updated_at", "created_at", "deleted_at",
    "lang_id", "lang_iso_code", "lang_name", "settings",
})
_ENUM_ROW_DROP_IF_EMPTY = frozenset({"short_name", "description", "char_code"})
_LANG_COLUMN = re.compile(r"^lang_[a-z]{2,3}$")


def _is_enum_row(value: dict[str, Any]) -> bool:
    return "id" in value and "name" in value and any(k in _ENUM_ROW_NOISE for k in value)


def compact_enum_rows(value: Any) -> Any:
    """Strip CDESK's per-row noise from every enum row in an enums payload,
    recursively: keep id, name, action_code, parent_id, in_use (as a bool) and
    anything the connector added; drop styles, bookkeeping columns, the
    internal `settings` flags and the per-language names (the tools resolve a
    name in any language themselves). Other dicts and lists pass through."""
    if isinstance(value, list):
        return [compact_enum_rows(v) for v in value]
    if not isinstance(value, dict):
        return value
    if not _is_enum_row(value):
        return {k: compact_enum_rows(v) for k, v in value.items()}
    out: dict[str, Any] = {}
    for key, inner in value.items():
        if key in _ENUM_ROW_NOISE or _LANG_COLUMN.match(key):
            continue
        if key in _ENUM_ROW_DROP_IF_EMPTY and inner in (None, ""):
            continue
        if key == "in_use" and isinstance(inner, (int, str)) and not isinstance(inner, bool):
            try:
                inner = int(inner) > 0
            except ValueError:
                pass
        out[key] = inner
    return out


def drop_empty_fields(item: Any) -> Any:
    """A list item without its empty fields (None, "", [], {}); 0 and False
    are values and stay."""
    if not isinstance(item, dict):
        return item
    return {k: v for k, v in item.items() if not (v is None or v == "" or v == [] or v == {})}


def _wrap_result(mcp: FastMCP, select: Callable[[Any], bool], transform: Callable[[Any], Any]) -> None:
    manager = getattr(mcp, "_tool_manager", None)
    if manager is None:
        return
    for tool in manager.list_tools():
        if not select(tool) or getattr(tool, "fn", None) is None:
            continue

        def wrap(inner: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(inner)
            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                result = inner(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return transform(result)
            return wrapped

        tool.fn = wrap(tool.fn)
        tool.is_async = True


def _compact_list_result(result: Any) -> Any:
    if isinstance(result, dict) and isinstance(result.get("items"), list):
        result["items"] = [drop_empty_fields(i) for i in result["items"]]
    return result


def _compact_write_result(result: Any) -> Any:
    # A write answers with the saved record; its empty fields say nothing about
    # the write.
    if isinstance(result, dict) and isinstance(result.get("savedData"), dict):
        result["savedData"] = drop_empty_fields(result["savedData"])
    return result


def _is_write_tool(tool: Any) -> bool:
    annotations = getattr(tool, "annotations", None)
    return not (annotations is not None and annotations.readOnlyHint is True)


def compact_llm_payloads(mcp: FastMCP) -> None:
    """Trim what the LLM receives (applied once in build_server, AFTER
    annotate_unreturned_fields, so that check still sees CDESK's own keys):
    get_*_enums results lose the enum-row noise, list_* items and the
    savedData of write answers their empty fields."""
    _wrap_result(mcp, lambda t: t.name.startswith("get_") and t.name.endswith("_enums"),
                 compact_enum_rows)
    _wrap_result(mcp, lambda t: t.name.startswith("list_"), _compact_list_result)
    _wrap_result(mcp, _is_write_tool, _compact_write_result)


def annotate_unreturned_fields(mcp: FastMCP) -> None:
    """Wrap every registered tool that takes a `fields` whitelist so its result
    names the requested fields CDESK silently dropped (`fields_not_returned`).
    Applied once, after all modules registered (server.build_server); a no-op on
    the test FakeMCP, which has no tool manager."""
    manager = getattr(mcp, "_tool_manager", None)
    if manager is None:
        return
    for tool in manager.list_tools():
        fn = getattr(tool, "fn", None)
        if fn is None or "fields" not in inspect.signature(fn).parameters:
            continue

        def wrap(inner: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(inner)
            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                result = inner(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return _with_fields_check(result, kwargs.get("fields"))
            return wrapped

        tool.fn = wrap(fn)
        tool.is_async = True


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
        if len(text) <= 10:
            return parsed.date().isoformat()
        # The same instant in another offset is the same value: CDESK echoes a
        # datetime sent as 09:30+01:00 back as 08:30+00:00.
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.isoformat()
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
    # A sent date without a time part matches a stored timestamp on the same day
    # in tenant time: CDESK stores the bare date as tenant midnight and some
    # modules echo it in UTC (2026-08-01 → 2026-07-31T22:00:00+00:00).
    if isinstance(a, str) and isinstance(b, str) and len(a) == 10 and isinstance(stored, str):
        return _tenant_day(stored) == a
    return False


def _tenant_day(value: str) -> str:
    """The YYYY-MM-DD of a stored timestamp in tenant time; the literal date
    part when it carries no offset or does not parse."""
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return text[:10]
    if parsed.tzinfo is None:
        return text[:10]
    return parsed.astimezone(tenant_timezone()).strftime("%Y-%m-%d")


def apply_clear_fields(
    body: dict[str, Any],
    clear_fields: Iterable[str] | None,
    allowed: Mapping[str, str],
) -> None:
    """Send an explicit `null` for each field named in `clear_fields`, which is
    how a v3 PUT clears a nullable field (an omitted field keeps its value).

    `allowed` maps the tool's parameter name to the wire name, and only covers
    fields the spec documents as clearable. Runs AFTER the body builders,
    which drop None values. Raises ValueError for a name that cannot be
    cleared, and for one the same call also sets. Mutates `body` in place."""
    for name in clear_fields or ():
        wire = allowed.get(name)
        if wire is None:
            raise ValueError(
                f"clear_fields: {name!r} cannot be cleared; clearable fields: "
                f"{', '.join(sorted(allowed))}"
            )
        if body.get(wire) is not None:
            raise ValueError(
                f"clear_fields: {name!r} is also being set in this call — "
                f"set it or clear it, not both"
            )
        body[wire] = None


_LANG_NAME_KEY = re.compile(r"^lang_(name|[a-z]{2,3})$")


def match_status_name(rows: Any, canon: str) -> int | None:
    """The `action_code` of the enum row whose `name` or any `lang_*` variant
    equals `canon` (already passed through canon_name). For tenant status enums
    read from the API (project and work-order statuses)."""
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        code = row.get("action_code")
        if isinstance(code, bool) or not isinstance(code, int):
            continue
        for key, value in row.items():
            if (key == "name" or _LANG_NAME_KEY.match(key)) and isinstance(value, str) \
                    and canon_name(value) == canon:
                return code
    return None


def closest_row_names(rows: Any, name: str, max_count: int = 5) -> list[str]:
    """The display names of the enum rows closest to `name`, scored against
    `name` and every `lang_*` variant (diacritic- and case-insensitively) —
    the "Did you mean" list for enum rows read without the cache. Rows below
    the same similarity floor as EnumCache.find_candidates are left out."""
    wanted = canon_name(name)
    scored: list[tuple[float, str]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            continue
        best = 0.0
        for key, value in row.items():
            if (key == "name" or _LANG_NAME_KEY.match(key)) and isinstance(value, str):
                best = max(best, SequenceMatcher(None, canon_name(value), wanted).ratio())
        if best >= _FUZZY_MIN_RATIO:
            scored.append((best, row["name"]))
    scored.sort(key=lambda t: t[0], reverse=True)
    return list(dict.fromkeys(n for _, n in scored))[:max_count]


def status_choices(rows: Any) -> str:
    """"code=name, ..." for an error message, from the tenant's status rows."""
    return ", ".join(
        f"{row.get('action_code')}={row.get('name')}"
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and row.get("name")
    )


async def fetch_all_rows(
    client: Any,
    path: str,
    params: dict[str, Any],
    *,
    cap: int,
    narrow_hint: str,
    page_size: int = 100,
) -> tuple[list[Any], dict[str, Any]]:
    """Every row of a list endpoint matching `params` (paging params are
    replaced), in CDESK's order, for filtering the connector applies itself
    when CDESK cannot. Refuses with `narrow_hint` once more than `cap` rows
    would be needed, so one call cannot pull an unbounded set. Returns
    (rows, meta of the first page)."""
    fetch = {k: v for k, v in params.items() if k not in ("pg", "pp")}
    rows: list[Any] = []
    meta: dict[str, Any] = {}
    page = 1
    while True:
        response, paging = await client.get_page(path, params={**fetch, "pg": page, "pp": page_size})
        batch, batch_meta = unwrap_list(response)
        if page == 1:
            meta = batch_meta
        total = paging.get("total")
        if (isinstance(total, int) and total > cap) or len(rows) + len(batch) > cap:
            count = total if isinstance(total, int) and total > cap else f"more than {cap}"
            raise ValueError(
                f"this filter has to be applied by the connector to every matching "
                f"record ({count}), more than the {cap} it checks; {narrow_hint}"
            )
        rows.extend(batch)
        if len(batch) < page_size or (isinstance(total, int) and len(rows) >= total):
            return rows, meta
        page += 1


def page_of(rows: list[Any], page: int, per_page: int) -> tuple[list[Any], dict[str, Any]]:
    """The requested page of rows filtered by the connector, with the paging
    info list_meta expects."""
    start = (page - 1) * per_page
    return rows[start:start + per_page], {"total": len(rows), "last_page": start + per_page >= len(rows)}
