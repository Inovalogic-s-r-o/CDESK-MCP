"""Text normalization for the grounded-answer tools.

CDESK record bodies are often rich HTML ("description (HTML allowed)"). The
grounded-answer feature (collect_records / verify_claims) has to (a) show
readable text to the LLM and (b) check that a quoted span really occurs in a
record. Both need the HTML and whitespace noise removed first, or a verbatim
quote copied from the rendered text would never match the raw stored value.

Two flavors:
- normalize_for_display: strip tags + unescape entities + collapse whitespace,
  KEEP case — what the LLM reads.
- normalize_for_match: same, plus casefold — used on BOTH the stored text and
  the LLM's quote before the substring check, so trivial case/whitespace/HTML
  differences don't cause a real quote to be rejected.

MIN_QUOTE_CHARS guards against a too-short quote (e.g. "phone") slipping the
substring check as meaningless "evidence".
"""

from __future__ import annotations

import html
import re

# Minimum length (raw, stripped) for a quote to count as evidence. Short
# fragments match too easily and don't actually pin a claim to a request.
MIN_QUOTE_CHARS = 15

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip(text: str) -> str:
    """Remove HTML tags, decode entities, collapse runs of whitespace."""
    no_tags = _TAG_RE.sub(" ", text)
    unescaped = html.unescape(no_tags)
    return _WS_RE.sub(" ", unescaped).strip()


def normalize_for_display(text: object) -> str:
    """Reader-facing text: tags/entities/whitespace cleaned, case preserved."""
    if not isinstance(text, str):
        return ""
    return _strip(text)


def normalize_for_match(text: object) -> str:
    """Comparison form: normalize_for_display + casefold. Apply to BOTH the
    stored request text and the candidate quote before the substring test."""
    if not isinstance(text, str):
        return ""
    return _strip(text).casefold()
