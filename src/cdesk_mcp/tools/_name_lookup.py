"""Name searches on id columns, resolved by the connector.

CDESK documents `CONTAINS <name>` on some id columns (a CI's or project's
`company_id`, an approval's `created_by`) as a search by the customer's or
user's name, but answers it with no rows. The connector looks the name up in
the customer or user list instead and sends the matching ids as
`(col = id1 OR col = id2 …)`, which CDESK applies correctly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cdesk_mcp.filters import encode_sb
from cdesk_mcp.tools._helpers import fetch_all_rows

NAME_SEARCH_CMPS = frozenset({"CONTAINS", "API_CONTAINS"})
_LOOKUP_CAP = 200


@dataclass(frozen=True)
class NameLookup:
    """Where a name is looked up: the list endpoint and the column searched."""

    path: str
    search_col: str
    what: str


CUSTOMER_NAMES = NameLookup("v3/company", "name", "customer")
USER_NAMES = NameLookup("v3/user", "name", "user")


def _is_name(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not value.strip().isdigit()


async def _matching_ids(client: Any, lookup: NameLookup, name: str) -> list[int]:
    sb = {"group": True, "o": "AND", "query": [
        {"col": lookup.search_col, "c": "CONTAINS", "q": name, "o": "AND"},
    ]}
    rows, _ = await fetch_all_rows(
        client, lookup.path, {"sb": encode_sb(sb), "fieldset": "base"},
        cap=_LOOKUP_CAP,
        narrow_hint=f"use a more specific {lookup.what} name, or filter by id",
    )
    return [r["id"] for r in rows if isinstance(r, dict) and isinstance(r.get("id"), int)]


async def resolve_name_searches(
    client: Any,
    sb: dict[str, Any] | None,
    lookups: Mapping[str, NameLookup],
    *,
    never_match: dict[str, Any],
) -> dict[str, Any] | None:
    """Replace each `CONTAINS <name>` leaf on a column in `lookups` with an OR
    group of `col = id` for the matching records. When no record matches, the
    leaf becomes `never_match`, a condition no record satisfies."""
    if sb is None:
        return None
    wanted: dict[tuple[str, str], NameLookup] = {}

    def collect(element: Any) -> None:
        if not isinstance(element, dict):
            return
        query = element.get("query")
        if isinstance(query, list):
            for e in query:
                collect(e)
            return
        col = element.get("col")
        if (isinstance(col, str) and col in lookups
                and str(element.get("c", "")).upper() in NAME_SEARCH_CMPS
                and _is_name(element.get("q"))):
            wanted[(col, str(element["q"]).strip())] = lookups[col]

    collect(sb)
    if not wanted:
        return sb
    keys = list(wanted)
    found = await asyncio.gather(*(_matching_ids(client, wanted[k], k[1]) for k in keys))
    ids = dict(zip(keys, found))

    def rewrite(element: Any) -> Any:
        if not isinstance(element, dict):
            return element
        query = element.get("query")
        if isinstance(query, list):
            return {**element, "query": [rewrite(e) for e in query]}
        col = element.get("col")
        value = element.get("q")
        key = (col, str(value).strip()) if isinstance(col, str) else None
        if key not in ids or str(element.get("c", "")).upper() not in NAME_SEARCH_CMPS:
            return element
        matches = ids[key]
        joiner = element.get("o", "AND")
        if not matches:
            return {**never_match, "o": joiner}
        leaves = [
            {"col": col, "c": "=", "q": i, "o": "AND" if n == 0 else "OR"}
            for n, i in enumerate(matches)
        ]
        return {"group": True, "o": joiner, "query": leaves}

    out = rewrite(sb)
    return out if isinstance(out, dict) else sb
