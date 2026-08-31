"""Token/authorization-code models for the remote (http) OAuth server.

These extend the SDK's ``AccessToken`` / ``AuthorizationCode`` with the CDESK
login the grant maps to. ``cdesk_login`` is kept for diagnostics only — the live
``CdeskClient`` is held out-of-band in the provider, since a live object can't
live on a pydantic model.

In the **stateless** design each artifact (auth code, access token, refresh
token) is Fernet-encrypted into the opaque string the client holds — there is no
server-side store. The per-session ``CdeskCredential`` therefore rides *inside*
the model as ``cred``: it is serialized into the ciphertext at issue and read
back on decrypt, so the server can rebuild the user's ``CdeskClient`` from the
token alone. ``cred`` only ever exists encrypted-in-transit (inside the token)
or in process memory after decrypt — never at rest in a datastore.
"""

from __future__ import annotations

from mcp.server.auth.provider import AccessToken, AuthorizationCode, RefreshToken
from pydantic import BaseModel


class CdeskCredential(BaseModel):
    """Per-session CDESK credential material, carried (encrypted) inside the
    OAuth tokens: the login it maps to, the current apitoken, and the CDESK
    refresh token used to renew it.

    ``cdesk_refresh_token`` is the long-lived user secret and the durable one —
    CDESK does not rotate it on renew, so a token that embeds it can always
    reconstruct a working client. The name is explicit so it is never confused
    with the OAuth refresh token we issue to Claude.

    ``base_url`` is the CDESK server this session targets — chosen by the user on
    the login page (one of the hosted servers, or a custom URL). It rides in the
    credential so every per-request reconstruction, renewal, and logout hits the
    right server, and so the choice is isolated per user. Empty on legacy tokens;
    the provider falls back to its configured default in that case."""

    login: str
    apitoken: str
    cdesk_refresh_token: str | None = None
    base_url: str = ""


class CdeskAuthCode(AuthorizationCode):
    """Authorization code that also remembers which CDESK user it belongs to.

    ``cdesk_login`` is kept for diagnostics; ``cred`` carries the validated CDESK
    credential so the token exchange can mint session tokens without any store.
    The whole model is Fernet-encrypted into the opaque ``code`` string."""

    cdesk_login: str
    cred: CdeskCredential | None = None


class CdeskAccessToken(AccessToken):
    """Access token carrying the CDESK login it maps to (diagnostics), the stable
    ``grant_id`` (revocation key), and the ``cred`` the per-request resolver
    rebuilds the user's ``CdeskClient`` from. ``grant_id`` defaults to "" for
    older/test constructions and is set at issue time. The whole model is
    Fernet-encrypted into the opaque bearer ``token`` string."""

    cdesk_login: str
    grant_id: str = ""
    cred: CdeskCredential | None = None


class CdeskRefreshToken(RefreshToken):
    """OAuth refresh token we issue to Claude. Carries the ``grant_id`` of the
    CDESK session it belongs to (revocation key), the RFC 8707 ``resource``
    audience to re-bind refreshed access tokens, and the ``cred`` used to mint
    new access tokens. The whole model is Fernet-encrypted into the opaque
    refresh-token string."""

    grant_id: str
    resource: str | None = None
    cred: CdeskCredential | None = None
