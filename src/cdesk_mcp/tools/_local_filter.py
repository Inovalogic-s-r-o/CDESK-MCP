"""Filter conditions the connector applies itself, for list endpoints where
CDESK cannot apply them correctly (it answers HTTP 500, ignores the condition,
or reads it as something else).

Such a condition is taken out of the `sb` filter sent to CDESK; every record
matching the rest of the filter is fetched (up to a cap) and the condition is
checked on each row. Only conditions joined to the rest by AND can be taken out
without changing the filter's meaning, so one inside an OR group is refused.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from cdesk_mcp.tools._helpers import fetch_all_rows, page_of

RANGE_CMPS = frozenset({">", ">=", "<", "<=", "BETWEEN"})
NOT_EQUAL_CMP = "!="
NULL_CMPS = frozenset({"ISNULL", "ISNOTNULL"})


@dataclass(frozen=True)
class LocalCondition:
    """A condition applied to each fetched row: `test` gets the row's value of
    `col`; `label` is how the result reports it."""

    col: str
    label: str
    test: Callable[[Any], bool]


def as_number(col: str, value: Any) -> float:
    """A comparison value or a row value as a number; decimal strings
    ("12.50", "12,50") included."""
    if isinstance(value, bool):
        raise ValueError(f"{col}: the comparison value must be a number (got {value!r})")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", "."))
        except ValueError:
            pass
    raise ValueError(f"{col}: the comparison value must be a number (got {value!r})")


def range_condition(leaf: dict[str, Any], row_col: str | None = None) -> LocalCondition:
    """`>`, `>=`, `<`, `<=` or `BETWEEN` on a numeric column. A row whose value
    is empty or not a number does not match. `row_col` is the row key holding
    the value when it differs from the filter column."""
    col = str(leaf["col"])
    comparator = str(leaf["c"]).upper()
    low = as_number(col, leaf.get("q"))
    high: float | None = None
    if comparator == "BETWEEN":
        if leaf.get("q2") is None:
            raise ValueError(f"{col}: BETWEEN needs both q and q2")
        high = as_number(col, leaf.get("q2"))

    def test(raw: Any) -> bool:
        if raw is None or raw == "":
            return False
        try:
            value = as_number(col, raw)
        except ValueError:
            return False
        if comparator == "BETWEEN":
            return high is not None and low <= value <= high
        return {">": value > low, ">=": value >= low, "<": value < low, "<=": value <= low}[comparator]

    label = (f"{col} BETWEEN {low:g} AND {high:g}" if comparator == "BETWEEN"
             else f"{col} {comparator} {low:g}")
    return LocalCondition(row_col or col, label, test)


def not_equal_condition(leaf: dict[str, Any], row_col: str | None = None) -> LocalCondition:
    """`!=` one number on a numeric (id) column. A row with no value matches:
    it is not equal to the number."""
    col = str(leaf["col"])
    value = as_number(col, leaf.get("q"))

    def test(raw: Any) -> bool:
        if raw is None or raw == "":
            return True
        try:
            return as_number(col, raw) != value
        except ValueError:
            return True

    return LocalCondition(row_col or col, f"{col} != {value:g}", test)


def null_condition(
    leaf: dict[str, Any], row_col: str | None = None, *, zero_is_empty: bool = False
) -> LocalCondition:
    """ISNULL / ISNOTNULL (and `= 0` read as ISNULL where 0 means "none").
    Empty is None or ""; with `zero_is_empty`, also 0."""
    col = str(leaf["col"])
    want_empty = str(leaf["c"]).upper() != "ISNOTNULL"

    def test(raw: Any) -> bool:
        empty = raw is None or raw == "" or (zero_is_empty and str(raw).strip() in ("0", "0.0"))
        return empty is want_empty

    return LocalCondition(row_col or col, f"{col} {'ISNULL' if want_empty else 'ISNOTNULL'}", test)


@dataclass(frozen=True)
class ColumnRule:
    """Which comparators on a filter column the connector applies itself, and
    the row key that holds the column's value. With `zero_is_empty`, 0 counts
    as no value (a category 0 is "uncategorised")."""

    comparators: frozenset[str]
    row_col: str | None = None
    zero_is_empty: bool = False


def numeric_picker(
    rules: Mapping[str, ColumnRule],
) -> Callable[[dict[str, Any]], LocalCondition | None]:
    """A `split_conditions` picker taking the ranges, the single-value `!=` and
    the ISNULL / ISNOTNULL that `rules` names. A `!=` with a list of values
    stays in the filter."""

    def pick(leaf: dict[str, Any]) -> LocalCondition | None:
        col, comparator = leaf.get("col"), leaf.get("c")
        if not isinstance(col, str) or not isinstance(comparator, str):
            return None
        rule = rules.get(col)
        c = comparator.upper()
        if rule is None or c not in rule.comparators:
            return None
        if c in RANGE_CMPS:
            return range_condition(leaf, rule.row_col)
        if c == NOT_EQUAL_CMP and not isinstance(leaf.get("q"), list):
            return not_equal_condition(leaf, rule.row_col)
        if c in NULL_CMPS:
            return null_condition(leaf, rule.row_col, zero_is_empty=rule.zero_is_empty)
        return None

    return pick


def split_conditions(
    sb: dict[str, Any] | None,
    pick: Callable[[dict[str, Any]], LocalCondition | None],
) -> tuple[dict[str, Any] | None, list[LocalCondition]]:
    """Take out of `sb` every leaf for which `pick` returns a condition.
    Returns the remaining filter (None when nothing is left) and the
    conditions."""
    conditions: list[LocalCondition] = []

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
            condition = pick(element)
            if condition is None:
                kept.append(element)
                continue
            if not child_context:
                raise ValueError(
                    f"this condition on {element.get('col')!r} "
                    f"({element.get('c')!r}) inside an OR group is not supported: "
                    f"CDESK cannot apply it, and the connector can only apply it "
                    f"after the rest of the filter when it is joined by AND. "
                    f"Rephrase the filter with AND, or run one query per OR branch."
                )
            conditions.append(condition)
        return {**node, "query": kept} if kept else None

    if sb is None:
        return None, []
    return walk(sb, True), conditions


async def fetch_rows_with_columns(
    client: Any,
    path: str,
    params: dict[str, Any],
    needed: list[str],
    *,
    fields: list[str] | None,
    id_col: str,
    cap: int,
    narrow_hint: str,
) -> tuple[list[Any], dict[str, Any], list[str]]:
    """Every row matching `params`, each carrying the `needed` columns.
    Returns (rows, meta, added): `added` are the columns the caller did not ask
    for, to be removed from the rows it returns.

    With an explicit `fields` whitelist the needed columns are added to it.
    Otherwise the rows come as requested, and a needed column they lack is read
    in a second pass (`returnFields[]` of the id and that column) and merged in
    by id: how CDESK combines `fieldset` with `returnFields[]` differs per
    endpoint, so the two are not sent together."""
    fetch = dict(params)
    added: list[str] = []
    if fields:
        added = [col for col in [*needed, id_col] if col not in fields]
        if added:
            fetch["returnFields[]"] = [*fields, *dict.fromkeys(added)]
    rows, meta = await fetch_all_rows(client, path, fetch, cap=cap, narrow_hint=narrow_hint)
    present = set().union(*(r.keys() for r in rows if isinstance(r, dict))) if rows else set()
    missing = [col for col in needed if col not in present]
    if rows and missing:
        extra_params = {k: v for k, v in params.items() if k not in ("fieldset", "returnFields[]")}
        extra_params["returnFields[]"] = [id_col, *missing]
        extra, _ = await fetch_all_rows(client, path, extra_params, cap=cap, narrow_hint=narrow_hint)
        by_id = {r.get(id_col): r for r in extra if isinstance(r, dict)}
        rows = [
            {**r, **{col: by_id.get(r.get(id_col), {}).get(col) for col in missing}}
            if isinstance(r, dict) else r
            for r in rows
        ]
        added = list(dict.fromkeys([*added, *missing]))
    return rows, meta, added


async def filtered_page(
    client: Any,
    path: str,
    params: dict[str, Any],
    conditions: list[LocalCondition],
    *,
    page: int,
    per_page: int,
    fields: list[str] | None,
    cap: int,
    narrow_hint: str,
    note: str,
    id_col: str = "id",
) -> tuple[list[Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fetch every row matching `params`, keep those passing all `conditions`,
    and return the requested page as (rows, meta, paging, info). Columns the
    conditions need are fetched even when `fields` / `fieldset` leave them
    out, and removed from the returned rows again."""
    needed = list(dict.fromkeys(c.col for c in conditions))
    rows, meta, added = await fetch_rows_with_columns(
        client, path, params, needed, fields=fields, id_col=id_col,
        cap=cap, narrow_hint=narrow_hint,
    )
    matched = [
        r for r in rows
        if isinstance(r, dict) and all(c.test(r.get(c.col)) for c in conditions)
    ]
    if added:
        drop = set(added)
        matched = [{k: v for k, v in r.items() if k not in drop} for r in matched]
    window, paging = page_of(matched, page, per_page)
    info = {
        "conditions": [c.label for c in conditions],
        "matched": len(matched),
        "checked": len(rows),
        "note": note,
    }
    return window, meta, paging, info
