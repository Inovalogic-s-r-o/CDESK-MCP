"""Translate raw client exceptions into LLM-friendly text.

Pure functions, no I/O. Tools wrap their CDESK calls in try/except and pass
the exception here to get a string suitable for an MCP tool error result.

CDESK error body shape (verified live against cmpp.seal.sk):
    {
      "data": null,
      "msg": {
        "error": [{"code": <int>, "message": "<localized text>"}]
      }
    }

The translator extracts CDESK's own code + message and surfaces them so the
LLM can interpret the actual cause for the user (e.g. "Modul vypnutý" / code 9
= MESSAGE_DISABLED_FEATURE → the Task module is disabled in tenant settings,
not an ACL denial). This is the difference between "I couldn't do that" and
"I couldn't do that because <specific reason the user can act on>".

Every returned string is run through token-scrub regexes as a safety net.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from cdesk_mcp.cdesk_client import CdeskApiError, CdeskAuthError

# Scrub patterns, applied in order on every returned string:
#   1. `apitoken <hex>` — the literal header form.
#   2. Bare 40-hex string — the documented CDESK apitoken shape. May incidentally
#      match SHA-1 hashes in stack traces, which is acceptable.
#   3. JSON `"password": "<value>"` / `"encrypted_password": "<value>"` — defense
#      in depth: if a CDESK 4xx ever echoes the request body in its diagnostic
#      response, we don't surface the password to the LLM or client logs. The
#      regex matches both pretty-printed and compact JSON, with single or double
#      quotes, and any value characters up to the closing quote.
_TOKEN_PATTERN = re.compile(r"apitoken\s+[0-9a-fA-F]{8,}", re.IGNORECASE)
_BARE_HEX_PATTERN = re.compile(r"\b[0-9a-fA-F]{40}\b")
_PASSWORD_KEY_PATTERN = re.compile(
    r'(["\'](?:password|encrypted_password|api_key|apikey|secret)["\']\s*:\s*)'
    r'(["\'])([^"\']*?)\2',
    re.IGNORECASE,
)
_BODY_PREVIEW_MAX = 200


def translate_error(
    exc: Exception,
    *,
    operation: str,
    record_id: int | str | None = None,
) -> str:
    """Convert a raw exception from CdeskClient into LLM-friendly text.

    - `operation`: short name of the failing tool ("get_task", "update_customer").
    - `record_id`: id of the record being operated on, if applicable.
    """
    return _scrub(_translate(exc, operation, record_id))


def _translate(exc: Exception, operation: str, record_id: int | str | None) -> str:
    if isinstance(exc, CdeskAuthError):
        return f"While performing '{operation}': {exc}"
    if isinstance(exc, CdeskApiError):
        return _translate_cdesk_api_error(exc, operation, record_id)
    if isinstance(exc, httpx.HTTPStatusError):
        return _translate_status(exc, operation, record_id)
    if isinstance(exc, httpx.RequestError):
        path_hint = _request_path_hint(exc)
        return (
            f"Network error talking to CDESK during '{operation}'{path_hint}: "
            f"{type(exc).__name__}."
        )
    return f"Unexpected error during '{operation}': {type(exc).__name__}: {exc}"


def _translate_cdesk_api_error(
    exc: CdeskApiError, operation: str, record_id: int | str | None,
) -> str:
    """CDESK signaled an error inside a 2xx response body. Use the same body
    extraction as HTTPStatusError so the LLM gets the actual CDESK message."""
    body = exc.body
    rid_suffix = f" (id={record_id})" if record_id is not None else ""
    cdesk = _format_cdesk_reason(body)

    # Body-level 409 is the optimistic-lock conflict — try to recognise.
    raw_text = _body_to_text(body)
    lower = raw_text.lower()
    codes, _messages = _extract_cdesk_messages(body)
    # Match the optimistic-lock conflict TIGHTLY: its real signatures are the
    # 409 code, CDESK's Slovak phrase "...medzičasom aktualizovaný", or the
    # framework's `recordUpdated` key. The old loose "record"+"updat" pair also
    # caught validation/ACL messages ("record cannot be updated: ...", "you may
    # not update this record") and wrongly told the LLM to re-fetch and retry an
    # operation that fails identically every time.
    is_lock_conflict = (
        409 in codes
        or "konflikt" in lower
        or "medzičasom aktualizovaný" in lower
        or "recordupdated" in lower.replace(" ", "").replace("_", "")
    )

    if is_lock_conflict:
        base = (
            f"Optimistic-lock conflict during '{operation}'{rid_suffix}: the "
            f"record was modified by someone else (or the timestamp_check we "
            f"sent didn't match). Re-fetch with the appropriate get_* tool "
            f"and retry"
        )
        return _append_cdesk(base, cdesk) + "."

    base = f"CDESK rejected '{operation}'{rid_suffix} (body-level error)"
    if cdesk:
        return _append_cdesk(base, cdesk) + "."
    return f"{base}."


def _translate_status(
    exc: httpx.HTTPStatusError,
    operation: str,
    record_id: int | str | None,
) -> str:
    status = exc.response.status_code
    body = _safe_body(exc.response)
    rid_suffix = f" (id={record_id})" if record_id is not None else ""
    cdesk = _format_cdesk_reason(body)

    if status == 400:
        return _compose("Bad request", operation, cdesk, body, prefix_only=True)
    if status == 401:
        base = (
            "CDESK rejected the request as unauthorized even after refreshing the "
            "session"
        )
        suffix = (
            ". The credentials are likely wrong or revoked; check "
            "CDESK_LOGIN / CDESK_PASSWORD."
        )
        return _append_cdesk(base, cdesk) + suffix
    if status == 403:
        base = f"CDESK rejected '{operation}'{rid_suffix} (HTTP 403)"
        if cdesk:
            return _append_cdesk(base, cdesk) + "."
        return (
            f"{base}. No additional detail in the response — likely an ACL denial "
            f"or a disabled module. Check the tenant's ACL settings and the "
            f"module-enabled global settings."
        )
    if status == 404:
        base = f"Record not found{rid_suffix} during '{operation}'"
        if cdesk:
            return _append_cdesk(base, cdesk) + "."
        return (
            f"{base}, or your account doesn't have admin scope over it."
        )
    if status == 409:
        return _translate_409(body, operation, rid_suffix, cdesk)
    if status == 412:
        base = f"Operation '{operation}' rejected (HTTP 412)"
        if cdesk:
            return _append_cdesk(base, cdesk) + "."
        return f"{base}: the new password matches the previous one."
    if status == 422:
        return _compose("Validation failed", operation, cdesk, body, prefix_only=True)
    if status == 429:
        base = (
            f"CDESK rate limit hit during '{operation}'. The client retried but "
            f"the limit persisted; try again shortly"
        )
        return _append_cdesk(base, cdesk) + "."
    if 500 <= status < 600:
        base = (
            f"CDESK returned HTTP {status} after the client retried with "
            f"exponential backoff. The upstream service appears unavailable — "
            f"wait at least 30 seconds before retrying"
        )
        return _append_cdesk(base, cdesk) + "."
    return (
        f"CDESK returned HTTP {status} during '{operation}': "
        f"{cdesk or _body_preview(body) or '<no body>'}"
    )


def _translate_409(
    body: Any, operation: str, rid_suffix: str, cdesk_reason: str,
) -> str:
    # Keyword search runs on raw body text (not the truncated preview) so a
    # discriminator past char 200 still classifies correctly.
    # TODO(open-questions Q3): validate the recordUpdated / linked keywords
    # against captured 409 bodies once the materials/ docs are accessible.
    raw_text = _body_to_text(body)
    lower = raw_text.lower()
    if "recordupdated" in lower or "updated_at" in lower:
        base = (
            f"Optimistic-lock conflict{rid_suffix}: the record was modified by "
            f"someone else since it was last fetched. Re-fetch the record with "
            f"the appropriate get_* tool and retry '{operation}'"
        )
        return _append_cdesk(base, cdesk_reason) + "."
    if any(word in lower for word in ("depend", "linked", "reference")):
        base = (
            f"Cannot complete '{operation}'{rid_suffix}: the record has "
            f"dependent records (linked tasks, deals, etc.) blocking it"
        )
        return _append_cdesk(base, cdesk_reason) + "."
    base = f"Conflict (HTTP 409) during '{operation}'{rid_suffix}"
    if cdesk_reason:
        return _append_cdesk(base, cdesk_reason) + "."
    return f"{base}: {_body_preview(body) or '<no body>'}"


def _compose(
    prefix: str, operation: str, cdesk_reason: str, body: Any, *, prefix_only: bool,
) -> str:
    """Used for 400 / 422 where the CDESK reason is the most informative part."""
    if cdesk_reason:
        return f"{prefix} for '{operation}'. {cdesk_reason}."
    return f"{prefix} for '{operation}': {_body_preview(body) or '<no body>'}"


def _append_cdesk(base: str, cdesk_reason: str) -> str:
    if cdesk_reason:
        return f"{base}. {cdesk_reason}"
    return base


def _extract_cdesk_messages(body: Any) -> tuple[list[int], list[str]]:
    """Pull (codes, messages) lists from CDESK's response body.

    Primary shape (v3 endpoints, verified live):
        {"data": ..., "msg": {"error": [{"code": <int>, "message": <str>}, ...]}}

    Fallback shapes:
        - /auth/login uses {"errno": <int>, "data": {...}}
        - Some endpoints may put a top-level `message` or `error` string.
        - The API-v3 gatekeeper returns a top-level `error` OBJECT, not a
          string: {"error": {"message": "Prístup k API v3 vyžaduje licenciu
          GOLD …", "code": 1}}. This is the shape every 403 from the v3
          licence/ACL guard takes (verified live 2026-07-31 on two accounts,
          across v3/user and v3/contract). Missing it made translate_error
          fall through to "No additional detail in the response — likely an
          ACL denial or a disabled module", which is actively WRONG here: the
          cause is the account's licence tier, so the admin was sent to the
          ACL and module-enabled screens instead of the licence assignment.
    """
    codes: list[int] = []
    messages: list[str] = []

    if not isinstance(body, dict):
        return codes, messages

    msg = body.get("msg")
    if isinstance(msg, dict):
        entries = msg.get("error")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                code = entry.get("code")
                if isinstance(code, int) and not isinstance(code, bool):
                    codes.append(code)
                text = entry.get("message")
                if isinstance(text, str) and text:
                    messages.append(text)

    # Fallback: some endpoints (notably /auth/login per the OpenAPI spec) use
    # a different shape. Don't double-count if msg.error already captured.
    if not codes:
        errno = body.get("errno")
        if isinstance(errno, int) and not isinstance(errno, bool):
            codes.append(errno)
    if not messages:
        for key in ("message", "error"):
            value = body.get(key)
            if isinstance(value, str) and value:
                messages.append(value)
                break
            # Nested object form: {"error": {"message": ..., "code": ...}}.
            if isinstance(value, dict):
                text = value.get("message")
                if isinstance(text, str) and text:
                    messages.append(text)
                    code = value.get("code")
                    # Only when nothing else supplied a code, so the rendered
                    # reason never shows two unrelated codes.
                    if (
                        not codes
                        and isinstance(code, int)
                        and not isinstance(code, bool)
                    ):
                        codes.append(code)
                    break

    return codes, messages


def _format_cdesk_reason(body: Any) -> str:
    """Render the CDESK-side reason as a single string fragment, or empty when
    the body had no extractable detail. Designed to be concatenated onto the
    end of our LLM-facing template:
        "<our base sentence>. CDESK said: 'Modul vypnutý' (code 9)."
    """
    codes, messages = _extract_cdesk_messages(body)
    if messages:
        joined = "; ".join(messages)
        if codes:
            return f"CDESK said: {joined!r} (code {codes[0]})"
        return f"CDESK said: {joined!r}"
    if codes:
        return f"CDESK error code {codes[0]}"
    return ""


def _request_path_hint(exc: httpx.RequestError) -> str:
    """Path-only diagnostic for network errors. Host omitted so logs/messages
    don't carry tenant URLs around. httpx's `RequestError.request` is a
    property that raises RuntimeError when the exception wasn't constructed
    with a Request — so we have to catch rather than null-check."""
    try:
        request = exc.request
    except RuntimeError:
        return ""
    try:
        path = request.url.path
    except (AttributeError, ValueError):
        return ""
    return f" at {path}" if path else ""


def _safe_body(response: httpx.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        # json.JSONDecodeError is a ValueError subclass.
        return response.text


def _body_to_text(body: Any) -> str:
    """Serialize body to text for substring searches. Not truncated."""
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(body)


def _body_preview(body: Any) -> str:
    """Truncated serialization for inclusion in error messages."""
    text = _body_to_text(body)
    if len(text) > _BODY_PREVIEW_MAX:
        return text[:_BODY_PREVIEW_MAX] + "..."
    return text


def _scrub(text: str) -> str:
    text = _TOKEN_PATTERN.sub("apitoken <REDACTED>", text)
    text = _BARE_HEX_PATTERN.sub("<REDACTED>", text)
    # Redact secret-key VALUES while keeping the key visible so the LLM
    # can still tell *which* field was problematic.
    text = _PASSWORD_KEY_PATTERN.sub(r"\1\2<REDACTED>\2", text)
    return text
