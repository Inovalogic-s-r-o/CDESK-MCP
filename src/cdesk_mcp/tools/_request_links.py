"""How a request is identified to people: the number CDESK shows them and the
link that opens the request in CDESK.

`system_number` ("CDR-1234") is the tenant's prefix + the request's sequence
number; `request_url` opens the request in the CDESK web interface (the
reader still signs in and needs the right to see it). The request list
returns both when asked for them by name. The request detail GET returns
`system_number` but has no `request_url` at all, so a single request's link is
read through the list endpoint, filtered to its id.
"""

from __future__ import annotations

from typing import Any

from cdesk_mcp.cdesk_client import CdeskClient
from cdesk_mcp.filters import encode_sb
from cdesk_mcp.tools._helpers import unwrap_list

SYSTEM_NUMBER = "system_number"
REQUEST_URL = "request_url"
LINK_FIELDS = (SYSTEM_NUMBER, REQUEST_URL)


def id_filter(request_id: int) -> str:
    """The encoded sb that narrows the request list to one id."""
    return encode_sb({
        "group": True, "o": "AND",
        "query": [{"col": "id_req", "c": "=", "q": request_id, "o": "AND"}],
    })


async def read_request_links(
    client: CdeskClient, request_id: Any, *, include_deleted: bool = False,
) -> dict[str, Any]:
    """{system_number, request_url} of one request, each key only when CDESK
    sent a value. Best-effort: empty when the read fails or the id is not an
    int, because it only adds to an answer that is already complete."""
    if not isinstance(request_id, int) or isinstance(request_id, bool):
        return {}
    params: dict[str, Any] = {
        "pg": 1, "pp": 1, "fieldset": "base",
        "returnFields[]": list(LINK_FIELDS), "sb": id_filter(request_id),
    }
    if include_deleted:
        params["includeDeleted"] = 1
    try:
        response, _ = await client.get_page("v3/request", params=params)
        records, _ = unwrap_list(response)
    except Exception:  # noqa: BLE001
        return {}
    row = next((r for r in records if isinstance(r, dict)), None)
    if row is None or row.get("id", request_id) != request_id:
        return {}
    return {key: row[key] for key in LINK_FIELDS if row.get(key)}


def add_links(record: Any, links: dict[str, Any]) -> Any:
    """Put the link fields onto a record that does not carry them yet."""
    if isinstance(record, dict):
        for key, value in links.items():
            if not record.get(key):
                record[key] = value
    return record
