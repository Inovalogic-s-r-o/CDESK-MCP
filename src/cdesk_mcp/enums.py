"""Tenant-specific enum lookups for the Task module.

Hides per-tenant enum ids behind human-readable names: the LLM (or end user)
can say 'Otvorené' or 'received' or 'prijate'; we resolve to the tenant's id.

Matching is diacritic-insensitive (NFD-decompose + drop combining marks) and
case-insensitive (casefold). Both display name and every `lang_<locale>`
variant participate, so the same status is reachable across languages.

Lifetime — nothing is cached across tool calls. Enums change on the tenant at
any time (a status renamed, a category added, a module switched on), so every
tool call reads them from the API afresh:
- The first lookup in a call fetches the endpoint; later lookups in the SAME
  call reuse that response (a create resolving status, priority and type makes
  one request, not three).
- The parsed response lives in a ContextVar scope. The MCP server handles each
  request in its own asyncio task, which starts from a copy of the context
  without that scope, so the next call fetches again.
- `refresh()` re-fetches within the current call.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol

log = logging.getLogger(__name__)

_DEFAULT_ENUMS_PATH = "v3/task/enums"
_FUZZY_MIN_RATIO = 0.3
# lang_<2-3 letter locale code> — accepts lang_en, lang_sk, lang_cs etc.
# Rejects lang_id, lang_iso_code, lang_name, lang_updated_at, lang_description.
_LANG_KEY_PATTERN = re.compile(r"^lang_[a-z]{2,3}$")


class _ClientProto(Protocol):
    """Minimal CdeskClient interface — lets tests pass stubs without circular imports."""

    async def get(
        self, path: str, *, params: dict[str, Any] | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class EnumEntry:
    id: int
    name: str
    parent_id: int | None = None
    action_code: int | None = None
    lang_names: tuple[str, ...] = field(default_factory=tuple)

    def all_names(self) -> tuple[str, ...]:
        # Deduplicate while preserving order.
        seen: set[str] = set()
        result: list[str] = []
        for n in (self.name, *self.lang_names):
            if n and n not in seen:
                seen.add(n)
                result.append(n)
        return tuple(result)


class AmbiguousEnumNameError(ValueError):
    """A name matches entries under more than one parent in a hierarchical
    bucket (e.g. the same `type2nd` label exists under two different `type`s)
    and no parent was supplied to disambiguate. Raising beats silently
    resolving to whichever entry happened to be stored first — which could
    attach a 2nd-level value under the wrong parent."""

    def __init__(self, bucket: str, name: str, match_count: int) -> None:
        self.bucket = bucket
        self.name = name
        self.match_count = match_count
        super().__init__(
            f"{name!r} is ambiguous in {bucket!r}: it matches {match_count} "
            f"entries under different parents. Specify the parent to disambiguate."
        )


@dataclass
class _Loaded:
    """One parsed enums response."""

    buckets: dict[str, list[EnumEntry]]
    # Key/value-shaped buckets (e.g. the request `type` "druh": {key,value})
    # that don't fit the int-id EnumEntry model but are still needed — e.g.
    # the type code for create_request. Surfaced via snapshot().
    raw_buckets: dict[str, list[dict[str, Any]]]
    # The endpoint's `settings` object, when it sends one. Carries the tenant
    # gates a module needs before a write — notably `enabled`, which is how a
    # switched-off module is detectable on the endpoints CDESK does NOT gate
    # (deal create/delete return 200 with the module off).
    settings: dict[str, Any]


_EMPTY = _Loaded(buckets={}, raw_buckets={}, settings={})

# The enums fetched during the current tool call, keyed by EnumCache instance.
# Unset (None) until the first lookup of a call; see the module docstring.
_CALL_SCOPE: contextvars.ContextVar[dict[int, _Loaded] | None] = contextvars.ContextVar(
    "cdesk_enum_call_scope", default=None,
)


def _call_scope() -> dict[int, _Loaded]:
    scope = _CALL_SCOPE.get()
    if scope is None:
        scope = {}
        _CALL_SCOPE.set(scope)
    return scope


class EnumCache:
    """Name ↔ id lookups over one enums endpoint, read fresh on every tool call
    (the name is historical — nothing is kept between calls)."""

    def __init__(
        self,
        client: _ClientProto,
        endpoint: str = _DEFAULT_ENUMS_PATH,
    ) -> None:
        self._client = client
        self._endpoint = endpoint
        self._lock = asyncio.Lock()

    def _state(self) -> _Loaded:
        """This call's parsed response, or an empty one before load()."""
        scope = _CALL_SCOPE.get()
        return (scope or {}).get(id(self), _EMPTY)

    @property
    def _buckets(self) -> dict[str, list[EnumEntry]]:
        return self._state().buckets

    @property
    def _raw_buckets(self) -> dict[str, list[dict[str, Any]]]:
        return self._state().raw_buckets

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def bucket_names(self) -> list[str]:
        return list(self._buckets.keys())

    @property
    def settings(self) -> dict[str, Any]:
        """The endpoint's `settings` object ({} when it sends none)."""
        return self._state().settings

    @property
    def loaded(self) -> bool:
        """Whether the enums were already fetched in the current tool call."""
        return id(self) in (_CALL_SCOPE.get() or {})

    @property
    def is_stale(self) -> bool:
        """Kept for callers of the old TTL cache: true until this call fetched."""
        return not self.loaded

    async def load(self) -> None:
        """Fetch the enums unless the current tool call already did."""
        scope = _call_scope()
        async with self._lock:
            if id(self) in scope:
                return
            scope[id(self)] = await self._fetch()

    async def refresh(self) -> None:
        """Fetch the enums again, within the current tool call."""
        scope = _call_scope()
        async with self._lock:
            scope[id(self)] = await self._fetch()

    async def _fetch(self) -> _Loaded:
        response = await self._client.get(self._endpoint)
        buckets = _parse_response(response, self._endpoint)
        settings = response.get("settings") if isinstance(response, dict) else None
        loaded = _Loaded(
            buckets=buckets,
            raw_buckets=_extract_keyvalue_buckets(response),
            settings=settings if isinstance(settings, dict) else {},
        )
        log.info(
            "EnumCache[%s] fetched: %s",
            self._endpoint,
            {b: len(e) for b, e in buckets.items()},
        )
        return loaded

    async def resolve(
        self,
        bucket: str,
        name: str,
        *,
        parent_id: int | None = None,
        allow_refresh: bool = True,
    ) -> int | None:
        """Resolve an enum name → its enum `id`. See `resolve_entry` for the
        matching semantics; this just returns the matched entry's id."""
        entry = await self.resolve_entry(
            bucket, name, parent_id=parent_id, allow_refresh=allow_refresh
        )
        return entry.id if entry is not None else None

    async def resolve_entry(
        self,
        bucket: str,
        name: str,
        *,
        parent_id: int | None = None,
        allow_refresh: bool = True,
    ) -> EnumEntry | None:
        """Diacritic + case-insensitive lookup across display name and lang_* aliases,
        returning the full matched `EnumEntry`.

        Most callers want `resolve` (the id). Use this when the wire value is a
        DIFFERENT field of the entry — e.g. the Request module stores/filters
        `status` and `priority` by `action_code` (status) / `parent_id` (priority
        on write), NOT the enum id.

        For hierarchical buckets (e.g. `type2nd`, `cat_area_2nd`), pass the
        already-resolved `parent_id` so only children of that parent are
        considered — the spec requires a 2nd-level value to be a child of the
        chosen 1st-level one. When `parent_id` is None and the name matches
        entries under multiple parents, raises AmbiguousEnumNameError rather
        than silently picking the first.

        The enums are fetched on the first lookup of the tool call (unless
        `allow_refresh` is False, which then answers None); a fetch failure
        degrades to a None result. A miss is final — the data is this call's.
        """
        if not self.loaded:
            if not allow_refresh:
                return None
            try:
                await self.load()
            except Exception as e:
                log.warning(
                    "Enum load failed: %s: %s", type(e).__name__, e,
                )
                return None
        return self._match_cached(bucket, name, parent_id)

    def _match_cached(
        self, bucket: str, name: str, parent_id: int | None = None,
    ) -> EnumEntry | None:
        canon_input = _canon(name)
        matches: list[EnumEntry] = []
        for entry in self._buckets.get(bucket, []):
            if parent_id is not None and entry.parent_id != parent_id:
                continue  # scope hierarchical buckets to the chosen parent
            if any(_canon(candidate) == canon_input for candidate in entry.all_names()):
                matches.append(entry)
        if not matches:
            return None
        # An unscoped lookup that lands on entries under different parents is
        # genuinely ambiguous — the same name lives under multiple parents, so
        # picking the first would risk the wrong parent. (Names are unique
        # within a parent, so this never fires once parent_id is supplied, nor
        # for flat buckets whose entries share parent_id=None.)
        if parent_id is None and len({e.parent_id for e in matches}) > 1:
            raise AmbiguousEnumNameError(bucket, name, len(matches))
        return matches[0]

    def list_names(self, bucket: str) -> list[str]:
        """Display names only (canonical). For all aliases per entry, use snapshot()."""
        return [e.name for e in self._buckets.get(bucket, [])]

    def id_name_map(self, bucket: str) -> dict[Any, str]:
        """id → display name for ONE bucket. Cheaper than snapshot() (which
        re-materializes every bucket) when only one bucket's labels are needed —
        e.g. resolving a record's status id to a readable name."""
        out: dict[Any, str] = {}
        for entry in self._buckets.get(bucket, []):
            if entry.id is not None and isinstance(entry.name, str):
                out[entry.id] = entry.name
        return out

    def action_code_name_map(self, bucket: str) -> dict[Any, str]:
        """action_code → display name for ONE bucket. Needed for the Request
        module, whose records store status/priority by enum `action_code`
        (e.g. 30) rather than the enum id (e.g. 183756) — so id_name_map()
        always misses on those records. On collision (two statuses sharing a
        code) first-match wins, matching how the action_code filter groups
        them."""
        out: dict[Any, str] = {}
        for entry in self._buckets.get(bucket, []):
            if entry.action_code is not None and isinstance(entry.name, str):
                out.setdefault(entry.action_code, entry.name)
        return out

    def find_candidates(
        self,
        bucket: str,
        name: str,
        max_count: int = 5,
        min_ratio: float = _FUZZY_MIN_RATIO,
    ) -> list[str]:
        """Closest known names by SequenceMatcher ratio. Returns the *alias*
        that produced the best match, not just the canonical name — so a
        diacritic-typo on a Slovak status returns the Slovak form, not the
        English one. Entries below `min_ratio` are suppressed so garbage
        input doesn't surface random-looking suggestions."""
        canon_input = _canon(name)
        scored: list[tuple[float, str]] = []
        for entry in self._buckets.get(bucket, []):
            best_ratio = 0.0
            best_alias = entry.name
            for candidate in entry.all_names():
                ratio = SequenceMatcher(None, _canon(candidate), canon_input).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_alias = candidate
            if best_ratio >= min_ratio:
                # Score against every alias (so a diacritic-stripped or
                # English typo still finds the entry) but SUGGEST the tenant's
                # own display name. Returning the matched alias surfaced
                # Lithuanian/Hungarian/Czech names as suggestions on a Slovak
                # tenant — `no-such-status` came back "Did you mean:
                # sustabdyta, in progress AS, činnosti nasazení?", which is
                # noise the caller cannot act on. `best_alias` is kept for the
                # tie-break only.
                scored.append((best_ratio, entry.name or best_alias))
        scored.sort(key=lambda t: t[0], reverse=True)
        seen: set[str] = set()
        unique: list[str] = []
        for _ratio, name in scored:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        return unique[:max_count]

    def bucket_is_absent(self, bucket: str) -> bool:
        """True when the tenant's enums payload carries no usable entry for
        `bucket` — either the key is missing entirely or every row lacks a name.

        CDESK gates whole buckets on tenant settings: `urgency`/`impact` need
        `request.priorityUrgencyImpact.enabled`, `cat_area`/`cat_area_2nd` need
        `request.enumCatAreas.status > 0`, `place` needs
        `system.categories.places_enabled` + `request.place.enabled`
        (Module/Request/Module.php::getBaseRequestEnums). When a bucket is
        absent, NO value can ever resolve, so "unknown value" is the wrong
        error — the caller needs to be told the feature is off rather than sent
        to an enums tool that will never list it.
        """
        entries = self._buckets.get(bucket)
        if not entries:
            return True
        return not any(entry.name for entry in entries)

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """JSON-serializable view of this call's enums. Used by the get_*_enums
        tools (load() first)."""
        out: dict[str, list[dict[str, Any]]] = {}
        for bucket, entries in self._buckets.items():
            out[bucket] = []
            for entry in entries:
                record: dict[str, Any] = {"id": entry.id, "name": entry.name}
                if entry.parent_id is not None:
                    record["parent_id"] = entry.parent_id
                if entry.action_code is not None:
                    record["action_code"] = entry.action_code
                if entry.lang_names:
                    record["lang_names"] = list(entry.lang_names)
                out[bucket].append(record)
        # Surface key/value buckets (e.g. request `type`) that the int-id model
        # can't hold — the LLM needs these codes (e.g. the create_request type
        # code "H"). Don't overwrite a typed bucket of the same name.
        # Distinct loop variables: `entries` above is list[EnumEntry], while these
        # are raw key/value dicts — reusing the name would conflate the two types.
        for raw_bucket, raw_entries in self._raw_buckets.items():
            # Use the raw key/value entries when the typed path produced nothing
            # for this bucket (it parsed 0 int-id entries, e.g. `type`).
            if not out.get(raw_bucket):
                out[raw_bucket] = [dict(e) for e in raw_entries]
        return out


def _parse_response(
    response: object, endpoint: str = _DEFAULT_ENUMS_PATH,
) -> dict[str, list[EnumEntry]]:
    if not isinstance(response, dict):
        raise ValueError(f"expected dict response from /{endpoint}")
    enums = response.get("enums")
    if not isinstance(enums, dict):
        raise ValueError("response missing 'enums' object")

    buckets: dict[str, list[EnumEntry]] = {}
    for bucket_name, records in enums.items():
        if not isinstance(records, list):
            continue
        parsed: list[EnumEntry] = []
        for r in records:
            if not isinstance(r, dict):
                continue
            # Key/value-shaped entries (no int id) are surfaced separately by
            # _extract_keyvalue_buckets — skip quietly rather than warn each load.
            if "key" in r and not isinstance(r.get("id"), int):
                continue
            try:
                parsed.append(_parse_entry(r))
            except ValueError as e:
                log.warning("Skipping malformed enum entry in bucket %r: %s", bucket_name, e)
        buckets[bucket_name] = parsed
    return buckets


def _extract_keyvalue_buckets(response: object) -> dict[str, list[dict[str, Any]]]:
    """Collect key/value-shaped enum buckets (entries with `key`/`value` and no
    usable int `id`) that the int-id EnumEntry model can't hold — e.g. the
    request `type` ("druh") bucket `[{"key":"H","value":"Helpdesk"}]`. These
    codes are needed (e.g. create_request `type_code`), so they are surfaced via
    snapshot(). Returns {bucket: [{"key":..., "value":...}, ...]}."""
    out: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(response, dict):
        return out
    enums = response.get("enums")
    if not isinstance(enums, dict):
        return out
    for bucket_name, records in enums.items():
        if not isinstance(records, list):
            continue
        kv: list[dict[str, Any]] = []
        for r in records:
            if not isinstance(r, dict) or "key" not in r:
                continue
            if isinstance(r.get("id"), int) and not isinstance(r.get("id"), bool):
                continue  # real int-id entry — handled by the typed path
            kv.append({"key": r.get("key"), "value": r.get("value")})
        if kv:
            out[bucket_name] = kv
    return out


def _parse_entry(record: dict[str, Any]) -> EnumEntry:
    raw_id = record.get("id")
    # bool is an int subclass in Python; reject it explicitly so True/False
    # never accidentally become an enum id.
    if isinstance(raw_id, bool):
        raise ValueError(f"enum entry has boolean 'id' (rejected): {record!r}")
    id_int: int
    if isinstance(raw_id, int):
        id_int = raw_id
    elif isinstance(raw_id, str):
        try:
            id_int = int(raw_id)
        except ValueError as e:
            raise ValueError(
                f"enum entry has non-numeric string 'id': {record!r}"
            ) from e
    else:
        raise ValueError(f"enum entry missing 'id' or has unsupported type: {record!r}")

    name = record.get("name")
    name_str = name if isinstance(name, str) else ""

    # Allowlist match: lang_<2-3 letter locale code>. Rejects lang_id (int),
    # lang_iso_code (locale code itself), lang_name (duplicate of name), and
    # any future fields like lang_updated_at, lang_description that might
    # accidentally widen the resolver surface.
    lang_names: list[str] = []
    for k, v in record.items():
        if not _LANG_KEY_PATTERN.match(k):
            continue
        if isinstance(v, str) and v:
            lang_names.append(v)

    parent_id = record.get("parent_id")
    parent_id_int = parent_id if isinstance(parent_id, int) and not isinstance(parent_id, bool) else None

    # Accept a numeric-STRING action_code too (symmetry with `id` above): some
    # tenants/endpoints return it as "30" rather than 30, and silently dropping
    # it to None would make status/priority filters no-op (see
    # resolve_enum_field_or_raise). A bool is an int subclass — reject it.
    action_code = record.get("action_code")
    action_code_int: int | None
    if isinstance(action_code, bool):
        action_code_int = None
    elif isinstance(action_code, int):
        action_code_int = action_code
    elif isinstance(action_code, str):
        try:
            action_code_int = int(action_code)
        except ValueError:
            action_code_int = None
    else:
        action_code_int = None

    return EnumEntry(
        id=id_int,
        name=name_str,
        parent_id=parent_id_int,
        action_code=action_code_int,
        lang_names=tuple(lang_names),
    )


def _canon(text: str) -> str:
    """Diacritic-stripped, case-folded form for matching."""
    nfd = unicodedata.normalize("NFD", text)
    no_diacritics = "".join(c for c in nfd if not unicodedata.combining(c))
    return no_diacritics.casefold()
