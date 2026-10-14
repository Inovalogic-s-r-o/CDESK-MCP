"""MCP tools for the CNB-specific bulk CMDB export.

CDESK servers configured for CNB expose one endpoint beyond the curated v3
surface: ``GET /cnb/cmdb/export`` — a dump of EVERY CMDB CI record with a fixed
column set and no field selection. It is paged: `pg` (1-based) and `pp`
(max 5000) are REQUIRED, and a call without them is a 400 (code 6). On a server
whose CMDB lacks the export's custom fields it answers code 14; the spec
documents 404 for a server without the endpoint.

Because the dump is the whole dataset, returning it inline would overwhelm the
LLM's context. So ``export_cmdb_ci`` reads every page, parks the rows in a per-tenant
TTL snapshot (``ExportSnapshotStore``) and returns a small manifest;
``get_cmdb_export_page`` serves windows of rows (with optional column
projection) from that snapshot. Everything flows through tool results, so this
works identically for a locally-run server and for the HTTP-hosted server that
web Claude connects to — a server-side file would be unreadable by a remote LLM.

If a user only wants a filtered slice or a few columns (not the whole tenant),
``list_configuration_items`` (the /v3/cmdb/ci listing) is the better tool — it
supports filters, pagination, and returnFields[]. This export is specifically
the whole-dataset, fixed-columns bulk dump.
"""

from __future__ import annotations

import inspect
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cdesk_mcp import base_config
from cdesk_mcp.cdesk_client import CdeskApiError, CdeskClient
from cdesk_mcp.tools._export_cache import ExportSnapshotStore
from cdesk_mcp.tools._helpers import to_llm_error, validate_fields

_EXPORT_PATH = "cnb/cmdb/export"
_MAX_PAGE = 500
# Rows per export request — the server's maximum (`pp` above it is a 400).
_EXPORT_PAGE_SIZE = 5000
# Hard stop for the page loop (2.5M rows), so a server that ignored `pg` and
# kept answering full pages could not loop forever.
_MAX_EXPORT_PAGES = 500
_NON_CNB_MSG = (
    "This CDESK server is not a CNB server, so the bulk CI export endpoint "
    "(cnb/cmdb/export) isn't available here — it exists only on the CNB customer "
    "server."
)
# CDESK signals "unknown/inaccessible property fields" as code 14 — as HTTP 400
# or, on some servers, as HTTP 200 with a body-level error. The export's fixed
# column set references custom fields not present on every server.
# Code 14 is ALSO the "pp above the maximum" answer, so the message decides
# whenever there is one; the bare code only when CDESK sent no text.
_MISSING_FIELDS_CODE = 14


def _cdesk_errors(body: object) -> tuple[list[int], str]:
    """Pull (codes, joined-message) from a CDESK {msg:{error:[...]}} body."""
    codes: list[int] = []
    msgs: list[str] = []
    msg = body.get("msg") if isinstance(body, dict) else None
    if isinstance(msg, dict) and isinstance(msg.get("error"), list):
        for entry in msg["error"]:
            if isinstance(entry, dict):
                code = entry.get("code")
                if isinstance(code, int) and not isinstance(code, bool):
                    codes.append(code)
                text = entry.get("message")
                if isinstance(text, str) and text:
                    msgs.append(text)
    return codes, "; ".join(msgs)


def _is_missing_fields_error(codes: list[int], joined: str) -> bool:
    low = joined.lower()
    if low:
        # CDESK's Slovak phrasing names the property fields ("pole_vlastností")
        # it couldn't resolve; the pp-limit error shares code 14 but not this.
        return "pole_vlastnos" in low or ("polia" in low and "prístupn" in low)
    return _MISSING_FIELDS_CODE in codes


def _missing_fields_msg(detail: str) -> str:
    base = (
        "The CNB CMDB export can't run on this CDESK server: its configured column "
        "set references CMDB custom fields that aren't defined (or aren't accessible) "
        "here. The export's columns are defined for the CNB tenant's CMDB schema, so "
        "it fails on any server that lacks those custom fields. To run it, use the "
        "CNB tenant, or have the export's column configuration repointed to fields "
        "that exist on this server."
    )
    return f"{base} CDESK reported: {detail}" if detail else base


def _safe_json(response: httpx.Response | None) -> object:
    if response is None or not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def _export_rows(response: object) -> list[Any]:
    """One page of the export: a raw JSON array, NOT a {data, ...} envelope
    (an envelope is tolerated defensively)."""
    if isinstance(response, list):
        return response
    if isinstance(response, dict) and isinstance(response.get("data"), list):
        return list(response["data"])
    return []


def _columns_of(rows: list[Any]) -> list[str]:
    """The fixed column set, taken from the first record (the export guarantees
    a uniform shape). Empty when there are no records."""
    if rows and isinstance(rows[0], dict):
        return sorted(rows[0].keys())
    return []


def register_cnb_tools(
    mcp: FastMCP,
    client: CdeskClient,
    snapshot_store: ExportSnapshotStore,
) -> None:
    """Register the CNB-only tools (bulk export + paged retrieval)."""

    @mcp.tool(
        description=_EXPORT_DESC,
        annotations=ToolAnnotations(title="Export CMDB CI", readOnlyHint=True),
    )
    async def export_cmdb_ci() -> dict[str, Any]:
        rows: list[Any] = []
        truncated = False
        data = await base_config.load(client)
        if isinstance(data, dict) and data.get("cnb_server") is False:
            raise RuntimeError(
                "export_cmdb_ci works only on the CNB tenant (it exports CNB's own "
                "CI fields); this CDESK server is not one. list_configuration_items "
                "pages through CIs on any tenant."
            )
        try:
            for page in range(1, _MAX_EXPORT_PAGES + 1):
                batch = _export_rows(await client.get(
                    _EXPORT_PATH, params={"pg": page, "pp": _EXPORT_PAGE_SIZE},
                ))
                rows.extend(batch)
                # The total and last-page flag come only as response headers,
                # which the client does not surface, so a short page is the
                # end. A page larger than `pp` means the server ignored paging
                # and sent everything at once.
                if len(batch) != _EXPORT_PAGE_SIZE:
                    break
            else:
                truncated = True
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            # The generic 404 translation says "Record not found", which is
            # wrong here — the spec's 404 means the endpoint does not exist on
            # this server.
            if status == 404:
                raise RuntimeError(_NON_CNB_MSG) from e
            if status == 400:
                codes, joined = _cdesk_errors(_safe_json(e.response))
                if _is_missing_fields_error(codes, joined):
                    raise RuntimeError(_missing_fields_msg(joined)) from e
            raise to_llm_error(e, operation="export_cmdb_ci") from e
        except CdeskApiError as e:
            # Some servers deliver the invalid-fields case as HTTP 200 + a
            # body-level error (code 14).
            codes, joined = _cdesk_errors(e.body)
            if _is_missing_fields_error(codes, joined):
                raise RuntimeError(_missing_fields_msg(joined)) from e
            raise to_llm_error(e, operation="export_cmdb_ci") from e
        except Exception as e:
            raise to_llm_error(e, operation="export_cmdb_ci") from e

        export_id = snapshot_store.put(rows)
        manifest: dict[str, Any] = {
            "export_id": export_id,
            "total_count": len(rows),
            "columns": _columns_of(rows),
            "sample": rows[:5],
            "note": (
                "The full export is cached server-side, not returned here. The rows "
                "are readable in windows via get_cmdb_export_page(export_id, "
                "offset, limit, fields). The snapshot expires after a few "
                "minutes; once it has, a page call reports it is gone and only "
                "a fresh export_cmdb_ci run yields a usable export_id."
            ),
        }
        if truncated:
            manifest["warnings"] = [
                f"The export stopped after {_MAX_EXPORT_PAGES} pages of "
                f"{_EXPORT_PAGE_SIZE} rows each, so it holds the first "
                f"{len(rows)} rows and may not be the whole dataset."
            ]
        return manifest

    @mcp.tool(
        description=_PAGE_DESC,
        annotations=ToolAnnotations(title="Get CMDB export page", readOnlyHint=True),
    )
    async def get_cmdb_export_page(
        export_id: str,
        offset: int = 0,
        limit: int = 50,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            if offset < 0:
                raise ValueError(f"offset must be 0 or greater (got {offset})")
            if limit < 1 or limit > _MAX_PAGE:
                raise ValueError(
                    f"limit must be between 1 and {_MAX_PAGE} (got {limit})"
                )
            validate_fields(fields)
        except ValueError as e:
            raise RuntimeError(f"get_cmdb_export_page input error: {e}") from e

        rows = snapshot_store.get(export_id)
        if rows is None:
            raise RuntimeError(
                f"No cached export for id {export_id!r} on this CDESK server — it "
                f"expired, was replaced by a newer export, or was created against a "
                f"different server. Re-run export_cmdb_ci to get a fresh export_id."
            )

        window = rows[offset : offset + limit]
        if fields:
            keep = set(fields)
            window = [
                {k: v for k, v in row.items() if k in keep}
                if isinstance(row, dict)
                else row
                for row in window
            ]
        total = len(rows)
        return {
            "items": window,
            "total_count": total,
            "offset": offset,
            "limit": limit,
            "returned": len(window),
            "has_more": offset + len(window) < total,
            "columns": _columns_of(rows),
        }


_EXPORT_DESC = inspect.cleandoc(
    """
    CNB ONLY: dump EVERY CMDB configuration item of the tenant with a fixed
    column set (the /cnb/cmdb/export endpoint). Available only on the CNB
    customer server — on any other server this reports that it isn't available.

    The full dataset is cached server-side rather than returned inline (it can
    be very large). This tool returns a manifest only: an `export_id`, the
    `total_count`, the `columns`, and a 5-row `sample`. The rows themselves
    are read in windows by get_cmdb_export_page, which takes the `export_id`.

    This endpoint has a fixed column set and no filtering. For a filtered
    subset or a few columns of CIs, list_configuration_items is the tool that
    supports filters, pagination, and field selection.
    """
)

_PAGE_DESC = inspect.cleandoc(
    """
    Return a window of rows from a cached CMDB export created by export_cmdb_ci.

    Pass the `export_id` from the manifest, plus `offset` (0-based) and `limit`
    (1–500). Optionally pass `fields` to keep only those columns in each row
    (the column names are in the manifest's `columns`). Returns `items` plus
    `total_count`, `offset`, `limit`, `returned`, and `has_more` so you can page
    through the whole set. An expired snapshot or unknown id is reported as such;
    a fresh export_cmdb_ci run is what produces a usable `export_id` again.
    """
)
