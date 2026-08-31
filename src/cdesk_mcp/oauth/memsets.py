"""Tiny process-local hardening sets for the stateless OAuth provider.

Self-encoded tokens carry all session state, so there is no datastore. Two
behaviours can't be expressed by a self-contained token alone, and a small
in-memory set fills each gap:

* **used auth codes** — OAuth requires an authorization code to be single-use.
  The code's hash is recorded on redemption; a replay within its short lifetime
  is rejected.
* **revoked grants** — ``revoke_token`` records the grant id so the session
  stops working immediately on this process.

Both are process-local and lost on restart, which is safe: auth codes expire on
their own within minutes, and revocation also best-effort logs the apitoken out
at CDESK. Entries store only a SHA-256 of the secret (never the raw value) and
are pruned lazily by an embedded expiry, so the sets stay bounded.
"""

from __future__ import annotations

import hashlib
import time


class ExpiringKeySet:
    """A set of hashed keys, each with an expiry. Membership tests and inserts
    prune expired entries, so memory stays bounded without a background task.

    Single-threaded asyncio use: no awaits between read and mutate."""

    def __init__(self) -> None:
        self._exp: dict[str, float] = {}

    @staticmethod
    def _h(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _prune(self, now: float) -> None:
        if not self._exp:
            return
        dead = [k for k, exp in self._exp.items() if exp <= now]
        for k in dead:
            self._exp.pop(k, None)

    def add(self, value: str, *, ttl_seconds: float) -> None:
        now = time.time()
        self._prune(now)
        self._exp[self._h(value)] = now + ttl_seconds

    def __contains__(self, value: str) -> bool:
        now = time.time()
        exp = self._exp.get(self._h(value))
        if exp is None:
            return False
        if exp <= now:
            self._exp.pop(self._h(value), None)
            return False
        return True
