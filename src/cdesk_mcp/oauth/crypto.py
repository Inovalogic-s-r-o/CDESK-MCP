"""Symmetric encryption for the stateless OAuth tokens (which carry the CDESK credential).

Wraps ``cryptography.fernet`` (AES-128-CBC + HMAC). The key is a urlsafe-base64
32-byte Fernet key supplied via ``CDESK_ENCRYPTION_KEY``; generate one with::

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

from cryptography.fernet import Fernet


class TokenCipher:
    """Encrypt/decrypt short secret strings with Fernet."""

    def __init__(self, key: str) -> None:
        # Fernet validates the key (urlsafe-base64, 32 bytes) and raises
        # ValueError on a bad key — surface that early, at startup.
        self._fernet = Fernet(key.encode("ascii"))

    @classmethod
    def generate(cls) -> TokenCipher:
        """An ephemeral cipher with a fresh random key — for the in-memory dev
        fallback (state is lost on restart anyway, so a per-process key is fine)."""
        return cls(Fernet.generate_key().decode("ascii"))

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str, *, ttl_seconds: int | None = None) -> str:
        """Decrypt a token, optionally rejecting it if older than ``ttl_seconds``.

        Fernet stamps each token with its creation time, so ``ttl`` gives a hard,
        storage-free expiry: a token past its lifetime raises ``InvalidToken``
        exactly like a tampered/forged one. This is how the stateless provider
        expires self-encoded auth codes and tokens without a datastore."""
        return self._fernet.decrypt(
            token.encode("ascii"), ttl=ttl_seconds
        ).decode("utf-8")
