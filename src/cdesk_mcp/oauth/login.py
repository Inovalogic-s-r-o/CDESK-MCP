"""CDESK login/consent page + route for the remote (http) OAuth flow.

Mounted unauthenticated by design: this is where the end user proves their CDESK
identity (login + password) before the provider mints a one-time authorization
code. The opaque OAuth session id is carried in a hidden form field.

Hardening: every HTML response carries clickjacking / sniffing headers, and the
POST is rate-limited per client IP (defense-in-depth — CDESK throttles logins
server-side and a fronting WAF should rate-limit too).
"""

from __future__ import annotations

import html
import logging

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from cdesk_mcp.cdesk_client import CdeskAuthError
from cdesk_mcp.oauth._connector import (
    ConnectorProbe,
    ProbeFn,
    Resolver,
    host_is_private,
    probe_connector,
    redact_userinfo,
)
from cdesk_mcp.oauth._web import _client_ip, _RateLimiter, _secure_html, _secure_json
from cdesk_mcp.oauth.provider import CdeskOAuthProvider

log = logging.getLogger(__name__)

_LOGIN_MAX_ATTEMPTS = 10
_LOGIN_WINDOW_SECONDS = 60.0
# The probe gets its own budget (see register_login_route) so a run of mistyped
# addresses can't consume the sign-in allowance.
_PROBE_MAX_ATTEMPTS = 10
# Someone is watching a spinner: far shorter than the 30s service timeout.
_PROBE_TIMEOUT_SECONDS = 5.0


# CDESK-branded consent page. The CSS lives in a plain (non-f) string so its
# literal `{ }` don't collide with the f-string interpolation in the page
# builder; the logo is an inline SVG approximation of the CDESK orange badge.
_CDESK_LOGO_SVG = (
    '<svg class="brand-logo" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" '
    'role="img" aria-label="CDESK logo">'
    '<circle cx="20" cy="20" r="19" fill="#F47920"/>'
    '<path d="M11.5 21 l5.2 5.2 L28.5 13" stroke="#fff" stroke-width="3.6" '
    'stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
    "</svg>"
)

_LOGIN_STYLE = """<style>
  :root{--orange:#F47920;--orange-dark:#d8660f;--ink:#2f3640;--muted:#6b7280;--line:#dfe3e8;--bg:#f4f6f8;
        --ms-blue:#2b7cd3;--ms-blue-dark:#2367b3;}
  *{box-sizing:border-box;}
  body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);
       color:var(--ink);margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:1rem;}
  .card{background:#fff;width:100%;max-width:25rem;border:1px solid var(--line);border-radius:10px;
        box-shadow:0 6px 24px rgba(0,0,0,.08);overflow:hidden;}
  .brand{display:flex;align-items:center;gap:.65rem;padding:1.1rem 1.5rem;border-bottom:1px solid var(--line);}
  .brand-logo{width:38px;height:38px;flex:0 0 auto;}
  .brand-name{font-weight:800;font-size:1.3rem;letter-spacing:.5px;line-height:1;}
  .brand-name sup{color:var(--orange);font-size:.55em;}
  .brand-sub{font-size:.6rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-top:3px;}
  .body{padding:1.5rem;}
  h1{font-size:1.05rem;margin:0 0 .3rem;}
  p{color:var(--muted);font-size:.9rem;margin:.3rem 0;}
  .client{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem;
          margin:.6rem 0 1rem;color:var(--ink);font-size:.85rem;}
  .err{color:#c0392b;font-size:.85rem;margin:.5rem 0;}
  label{display:block;margin:.85rem 0 .3rem;font-size:.78rem;font-weight:600;color:var(--ink);}
  input:not([type=hidden]),select{width:100%;padding:.6rem .7rem;border:1px solid var(--line);border-radius:7px;
                           font-size:.95rem;color:var(--ink);background:#fff;}
  input:not([type=hidden]):focus,select:focus{outline:none;border-color:var(--orange);box-shadow:0 0 0 3px rgba(244,121,32,.18);}
  .warn{background:#fff6ef;border:1px solid #f3c79a;border-radius:8px;padding:.6rem .8rem;margin:.5rem 0 0;
        font-size:.78rem;color:#8a4b16;line-height:1.4;}
  .hint{font-size:.75rem;color:var(--muted);margin:.35rem 0 0;line-height:1.4;}
  .hint code{background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:0 .2rem;
             font-size:.95em;}
  button{width:100%;margin-top:1.3rem;padding:.7rem 1rem;background:var(--orange);color:#fff;font-weight:700;
         font-size:.95rem;border:none;border-radius:7px;cursor:pointer;}
  button:hover{background:var(--orange-dark);}
  .ms-btn{display:flex;align-items:center;justify-content:center;gap:.55rem;margin-top:0;
          background:var(--ms-blue);color:#fff;font-weight:600;}
  .ms-btn:hover{background:var(--ms-blue-dark);}
  .ms-btn svg{width:17px;height:17px;flex:0 0 auto;}
  .alt-sep{margin:1.3rem 0 .6rem;text-align:center;font-size:.68rem;font-weight:700;
           letter-spacing:.09em;text-transform:uppercase;color:var(--muted);}
  .footer{padding:.8rem 1.5rem;border-top:1px solid var(--line);font-size:.72rem;color:var(--muted);text-align:center;}
  [hidden]{display:none !important;}
  .server-head{display:flex;align-items:baseline;justify-content:space-between;gap:.5rem;}
  .server-head label{margin-bottom:.3rem;}
  .change{font-size:.75rem;font-weight:600;color:var(--orange-dark);background:none;border:none;
          padding:0;margin:0;width:auto;cursor:pointer;text-decoration:underline;}
  .change:hover{background:none;color:var(--orange);}
  input[readonly]{background:var(--bg);color:var(--muted);}
  .note{background:#fff6ef;border:1px solid #f3c79a;border-radius:8px;padding:.6rem .8rem;
        margin:.7rem 0 0;font-size:.78rem;color:#8a4b16;line-height:1.4;}
  button[disabled]{opacity:.65;cursor:progress;}
</style>"""


# Shown under the server-URL field. We do NOT validate the address (no SSRF
# guard, by design) — responsibility for a correct/trusted URL is the user's.
_CUSTOM_WARNING = (
    "⚠️ Make sure this is your real CDESK server URL and is spelled correctly. "
    "Your CDESK login and password will be sent to whatever address you enter "
    "here. You accept full responsibility for the address being correct and for "
    "any loss or theft of your credentials or data caused by an incorrect or "
    "untrusted address."
)


# The Microsoft 4-square glyph in white — the form CDESK's own login page uses on
# its blue Microsoft button (the 4-colour logo needs a light background, and this
# button is blue).
_MS_LOGO_SVG = (
    '<svg viewBox="0 0 21 21" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="1" y="1" width="9" height="9" fill="#fff"/>'
    '<rect x="11" y="1" width="9" height="9" fill="#fff"/>'
    '<rect x="1" y="11" width="9" height="9" fill="#fff"/>'
    '<rect x="11" y="11" width="9" height="9" fill="#fff"/></svg>'
)

# Turns the one-page form into the two-step flow, and drives the Microsoft button.
#
# PROGRESSIVE ENHANCEMENT, deliberately. The HTML ships as the old single-step form
# with everything visible and enabled; this script is what hides step 2 and shows
# the Verify button. So a browser with JS off gets exactly the page that worked
# before, and no <noscript> stylesheet is needed. Two consequences that are easy to
# get wrong if this is ever inverted:
#   * step 2's inputs are `required`; a hidden required field makes Chrome refuse to
#     submit with "An invalid form control is not focusable" and NO visible error.
#     They are therefore `disabled` while hidden (which removes them from validation
#     AND submission) and re-enabled on reveal.
#   * #server is never hidden and never `disabled` — a disabled field isn't
#     submitted, so the POST would arrive with an empty server immediately after a
#     successful verify. It goes `readonly` instead, which keeps it in the payload.
#     It is also the same element in both steps: the address shown is by
#     construction the address submitted.
#
# The Microsoft button starts `hidden` and is revealed only by a probe that reports
# an azure connector. It needs this script to work at all, so gating it on the
# probe costs nothing and stops us offering SSO on servers that don't have it.
#
# Inline script is allowed by our CSP. Plain (non-f) string so the braces don't
# collide with the page f-string.
_TWO_STEP_JS = """<script>
  (function(){
    var form=document.getElementById('loginForm');
    var server=document.getElementById('server');
    var verifyBtn=document.getElementById('verifyBtn');
    var step1=document.getElementById('step1extra');
    var step2=document.getElementById('step2');
    var changeBtn=document.getElementById('changeServer');
    var probeErr=document.getElementById('probeErr');
    var probeNote=document.getElementById('probeNote');
    var msBtn=document.getElementById('msLoginBtn');
    var altSep=document.getElementById('altSep');
    if(!form||!server||!verifyBtn||!step2){return;}
    var probeUrl=verifyBtn.getAttribute('data-probe-url');
    var session=verifyBtn.getAttribute('data-session');
    var step2Inputs=step2.querySelectorAll('input,select');
    var inFlight=null;

    function showMs(on){
      if(msBtn){msBtn.hidden=!on;}
      if(altSep){altSep.hidden=!on;}
    }

    function setStep2(on){
      step2.hidden=!on;
      for(var i=0;i<step2Inputs.length;i++){step2Inputs[i].disabled=!on;}
      if(step1){step1.hidden=on;}
      verifyBtn.hidden=on;
      if(changeBtn){changeBtn.hidden=!on;}
      server.readOnly=on;
    }
    function back(){
      form.setAttribute('data-verified-for','');
      if(probeNote){probeNote.hidden=true;}
      showMs(false);
      setStep2(false);
      server.focus();
    }
    function reveal(value,note,azure,focus){
      form.setAttribute('data-verified-for',value);
      setStep2(true);
      if(probeErr){probeErr.hidden=true;probeErr.textContent='';}
      if(probeNote){
        probeNote.hidden=!note;
        probeNote.textContent=note||'';
      }
      showMs(azure);
      if(focus){
        var l=document.getElementById('login');
        if(l){l.focus();}
      }
    }
    function fail(message){
      if(probeErr){
        probeErr.textContent=message;
        probeErr.hidden=false;
      }
      server.focus();
    }

    function verify(){
      var value=server.value.trim();
      if(!value){
        if(server.reportValidity){server.reportValidity();}else{server.focus();}
        return;
      }
      if(inFlight){inFlight.abort();}
      var ctrl=new AbortController();
      inFlight=ctrl;
      // Our own server, not the customer's — but the tunnel can drop, so bound it.
      var timer=setTimeout(function(){ctrl.abort();},8000);
      verifyBtn.disabled=true;
      verifyBtn.textContent='Checking\\u2026';
      if(probeErr){probeErr.hidden=true;}
      fetch(probeUrl,{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        credentials:'omit',
        body:JSON.stringify({session:session,server:value})
      }).then(function(r){return r.json();}).then(function(d){
        // A slow probe of server A must not decide anything about server B.
        if(d&&d.server!==server.value.trim()){return;}
        if(d&&d.blocked){fail(d.message||'That address could not be used.');return;}
        reveal(value,(d&&d.message)||'',!!(d&&d.azure),true);
      }).catch(function(){
        // Reaching OUR server failed (offline, tunnel dropped, aborted). Never let
        // that block sign-in — see the "don't add a pre-flight probe back" comment
        // on the authenticate() call below. Proceed without the Microsoft button.
        reveal(value,'We could not check that server just now. You can still sign in.',false,true);
      }).then(function(){
        clearTimeout(timer);
        if(inFlight===ctrl){inFlight=null;}
        verifyBtn.disabled=false;
        verifyBtn.textContent='Verify server';
      });
    }

    verifyBtn.addEventListener('click',verify);
    verifyBtn.hidden=false;
    server.addEventListener('keydown',function(e){
      // Enter in the server field means "verify", not "submit a half-empty form".
      if(e.key==='Enter'){e.preventDefault();verify();}
    });
    server.addEventListener('input',function(){
      if(server.value.trim()!==form.getAttribute('data-verified-for')){back();}
    });
    if(changeBtn){
      changeBtn.addEventListener('click',function(e){e.preventDefault();back();});
    }
    // Single source of truth for "which address is this page verified for": the
    // form attribute. Re-checked on load AND on bfcache restore (Safari/Firefox
    // hand back the old DOM), so a restored page can never show step 2 for an
    // address that is no longer in the field.
    function sync(){
      var v=form.getAttribute('data-verified-for');
      if(v&&v===server.value.trim()){
        reveal(v,'',false,false);
        // Server-rendered step 2 (an error re-render after the password was
        // rejected). The page has no idea whether this server offers Entra, so
        // ask quietly — the only thing it changes is the button's visibility.
        fetch(probeUrl,{
          method:'POST',
          headers:{'Content-Type':'application/json'},
          credentials:'omit',
          body:JSON.stringify({session:session,server:v})
        }).then(function(r){return r.json();}).then(function(d){
          if(d&&d.azure&&d.server===server.value.trim()){showMs(true);}
        }).catch(function(){});
      }else{
        setStep2(false);
        showMs(false);
      }
    }
    sync();
    window.addEventListener('pageshow',sync);

    if(msBtn){
      msBtn.addEventListener('click',function(){
        var value=server.value.trim();
        if(!value){
          if(server.reportValidity){server.reportValidity();}else{server.focus();}
          return;
        }
        var url=msBtn.getAttribute('data-start-url')+'?session='+encodeURIComponent(session)
                +'&server='+encodeURIComponent(value);
        window.location.href=url;
      });
    }
  })();
</script>"""


def _signin_error_text(e: CdeskAuthError) -> str:
    """User-facing text for a rejected CDESK sign-in.

    CdeskAuthError messages are written for operators — the 401 one names
    CDESK_LOGIN / CDESK_PASSWORD, which means nothing to someone typing their
    credentials into this page — so the two classified cases get their own
    wording and only unclassified failures fall back to the raw text."""
    if e.status == 401:
        return "CDESK sign-in failed. The CDESK login or password was wrong."
    if e.status == 302:
        return (
            "CDESK sign-in failed. This account requires two-factor "
            "authentication, which the connector cannot complete."
        )
    return f"CDESK sign-in failed: {e}"


# What a probe outcome means to the person typing the address, and — the load-
# bearing half — whether it may stop them.
#
# BLOCKING IS ONLY EVER FOR A CONFIDENT NEGATIVE. See the "don't add a pre-flight
# probe back" comment on the authenticate() call below: a /auth/me probe once made
# every login depend on one endpoint's health, and when it 500'd on an otherwise
# healthy tenant nobody could sign in. /api/auth/connector is public and
# documented, which makes it a better probe, not a safe gate. So an answer that
# merely fails to confirm — a timeout, a 401 from a Basic-auth proxy, a 404, a 5xx
# — warns and lets the sign-in proceed. Only "we know this is wrong" stops it.
#
# `unreachable` covers DNS failure AND connection refused with one wording on
# purpose: distinguishing them would turn this route into a "does this host exist"
# oracle for anything our server can reach.
_PROBE_BLOCKING: dict[str, str] = {
    "unreachable": (
        "We couldn't reach that address. Check the spelling, and that the server "
        "is reachable from the internet."
    ),
    "tls": (
        "That server's HTTPS certificate couldn't be verified. If it's an internal "
        "server, try http:// instead."
    ),
    "not_json": "That address is a website, but not a CDESK server.",
    "not_cdesk": "That address is a website, but not a CDESK server.",
}

_PROBE_ADVISORY: dict[str, str] = {
    "timeout": (
        "That address didn't respond in time. It may be behind a firewall, or the "
        "port may be wrong. You can still sign in below."
    ),
    "http_error": (
        "That address answered, but not like a CDESK server. You can still sign in "
        "below, or go back and check the address."
    ),
}

# Shown instead of the detailed wording when the address resolves inside a private
# range on a deployment that accepts arbitrary servers — otherwise the differences
# between "refused", "timed out" and "404" make this an unauthenticated port
# scanner for the network our server sits in.
_PROBE_VAGUE = "We couldn't confirm that server. You can still sign in below."


_PROBE_PROTECTED = (
    "That address is protected and wouldn't tell us about itself. You can still "
    "sign in below."
)


def _probe_reply(result: ConnectorProbe, *, vague: bool) -> tuple[bool, str, bool]:
    """``(blocked, message, azure)`` for a probe outcome. ``vague`` suppresses the
    detail (see ``_PROBE_VAGUE``) and, with it, any blocking — we won't refuse an
    address on evidence we've decided not to state."""
    if result.status == "ok":
        return False, "", result.has_azure
    if vague:
        return False, _PROBE_VAGUE, False
    if result.status == "http_error" and result.http_status in (401, 403):
        # A CDESK behind a Basic-auth proxy really does answer this way; the
        # password POST that follows carries the credentials and may well work.
        return False, _PROBE_PROTECTED, False
    blocking = _PROBE_BLOCKING.get(result.status)
    if blocking is not None:
        return True, blocking, False
    return False, _PROBE_ADVISORY.get(result.status, _PROBE_VAGUE), False


# Prompts for the three required fields. Without this the browser supplies its own
# text in ITS OWN language — a Slovak Chrome pops up "Zadajte adresu URL" on a page
# written entirely in English — and that text can be neither worded nor translated
# by us. setCustomValidity replaces it; clearing on input is mandatory, or the field
# stays permanently invalid once it has been flagged. Inline script is allowed by
# our CSP. Plain (non-f) string so the braces don't collide with the page f-string.
_REQUIRED_FIELD_JS = """<script>
  (function(){
    var prompts={
      server:"Enter your CDESK server address.",
      login:"Enter your CDESK login.",
      password:"Enter your CDESK password."
    };
    Object.keys(prompts).forEach(function(id){
      var el=document.getElementById(id);
      if(!el){return;}
      el.addEventListener('invalid',function(){el.setCustomValidity(prompts[id]);});
      el.addEventListener('input',function(){el.setCustomValidity('');});
    });
  })();
</script>"""


def _azure_button_html(start_url: str) -> str:
    """The Microsoft (Entra) sign-in block that sits *below* the submit button.

    Deliberately mirrors CDESK's own login screen: a small uppercase caption
    ("use another service to sign in") followed by the blue Microsoft button, so
    someone who knows the CDESK login page finds the same control in the same
    place. The label matches CDESK's wording verbatim, English on every locale.

    Ships ``hidden``: it is revealed only once the probe has confirmed the chosen
    server actually has an azure connector, so we never offer Microsoft sign-in on
    a server that would answer with an error page. ``start_url`` is absolute (built
    from CDESK_PUBLIC_URL) — the old root-absolute ``/login/azure/start`` broke
    under path-prefix hosting."""
    return (
        '<div class="alt-sep" id="altSep" hidden>Use another service to sign in</div>'
        f'<button type="button" id="msLoginBtn" class="ms-btn" hidden '
        f'data-start-url="{html.escape(start_url, quote=True)}">'
        f"{_MS_LOGO_SVG}<span>Sign in with Microsoft</span></button>"
    )


def _server_url_field_html(value: str) -> str:
    """A single CDESK-server URL field — the user pastes their server address.
    ``value`` is what was submitted on a failed attempt, so it isn't retyped.

    Deliberately ``type="text"`` rather than ``type="url"``: a url input's browser
    validation rejects a scheme-less address outright, with a popup we can't word,
    before the server ever sees it. Since we now accept ``cdesk.example.com`` and
    add the ``https://`` ourselves, that check would block the most natural input.
    ``inputmode=url`` still gets the right mobile keyboard, and the address is
    validated (and canonicalized) server-side by ``normalize_base_url``."""
    prefill = f' value="{html.escape(value, quote=True)}"' if value else ""
    return (
        '<div class="server-head">'
        '<label for="server">CDESK server URL</label>'
        '<button type="button" class="change" id="changeServer" hidden>Change</button>'
        "</div>"
        '<input id="server" name="server" type="text" inputmode="url" required '
        'autocapitalize="none" autocorrect="off" spellcheck="false" autocomplete="url" '
        f'placeholder="cdesk.example.com"{prefill}>'
        '<div class="err" id="probeErr" hidden></div>'
        '<div class="note" id="probeNote" hidden></div>'
        '<div id="step1extra">'
        '<div class="hint">Just the address you use in your browser — '
        '<code>https://</code> is added if you leave it out.</div>'
        f'<div class="warn">{html.escape(_CUSTOM_WARNING)}</div>'
        "</div>"
    )


def _login_page_html(
    session: str,
    error: str | None = None,
    *,
    client_name: str | None = None,
    redirect_host: str | None = None,
    selected: str = "",
    azure_enabled: bool = False,
    verified: bool = False,
    probe_url: str = "/login/probe",
    azure_start_url: str = "/login/azure/start",
) -> str:
    """CDESK-branded login/consent form (no template engine). Posts back to
    /login with the opaque OAuth session id carried in a hidden field. When
    known, identifies the requesting client + redirect target so the user can
    see who they're authorizing (anti-phishing). When ``azure_enabled`` a blue
    "Sign in with Microsoft" button below the submit — same wording and
    placement as CDESK's own login screen — starts the Office365 SSO for the
    selected CDESK server (connector id discovered at /login/azure/start)."""
    safe_session = html.escape(session, quote=True)
    error_block = f'<div class="err">{html.escape(error)}</div>' if error else ""
    if client_name:
        dest = (
            f" It will be redirected to <code>{html.escape(redirect_host)}</code>."
            if redirect_host else ""
        )
        client_block = (
            f'<div class="client"><strong>{html.escape(client_name)}</strong> is '
            f"requesting access to your CDESK account.{dest}</div>"
        )
    else:
        client_block = ""
    # Empty by default — the user pastes their own CDESK server URL. `selected`
    # only re-fills the field after a failed attempt so they needn't retype it.
    server_block = _server_url_field_html(selected)
    azure_block = _azure_button_html(azure_start_url) if azure_enabled else ""
    # `verified` re-opens the page directly at step 2. Set on every error that
    # happens AFTER the address was accepted — a rejected password must not send
    # the user back to re-verify a server that was fine. The JS treats this
    # attribute as the single source of truth and drops back to step 1 the moment
    # the field stops matching it.
    verified_for = html.escape(selected, quote=True) if verified and selected else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in to CDESK</title>
{_LOGIN_STYLE}
</head>
<body>
<div class="card">
  <div class="brand">
    {_CDESK_LOGO_SVG}
    <div>
      <div class="brand-name">CDESK<sup>&reg;</sup></div>
      <div class="brand-sub">Powerful Service Desk</div>
    </div>
  </div>
  <div class="body">
    {client_block}
    <h1>Sign in to CDESK</h1>
    <p>Sign in with your CDESK account to authorize this connector.</p>
    {error_block}
    <form method="post" action="" id="loginForm" data-verified-for="{verified_for}">
      <input type="hidden" name="session" value="{safe_session}">
      {server_block}
      <button type="button" id="verifyBtn" hidden
              data-probe-url="{html.escape(probe_url, quote=True)}"
              data-session="{safe_session}">Verify server</button>
      <div id="step2">
        <label for="login">CDESK login</label>
        <input id="login" name="login" autocomplete="username" required>
        <label for="password">Password</label>
        <input id="password" name="password" type="password" autocomplete="current-password" required>
        <button type="submit">Sign in</button>
        {azure_block}
      </div>
    </form>
  </div>
  <div class="footer">You'll only authorize access your CDESK account already has.</div>
</div>
{_REQUIRED_FIELD_JS}
{_TWO_STEP_JS}
</body></html>"""


def register_login_route(
    mcp: FastMCP,
    provider: CdeskOAuthProvider,
    *,
    trust_forwarded: bool = False,
    azure_enabled: bool = True,
    probe: ProbeFn = probe_connector,
    resolve: Resolver | None = None,
) -> None:
    """Mount the GET/POST /login consent page and POST /login/probe
    (unauthenticated by design).

    The page is two-step: the user enters only their CDESK server address and
    presses "Verify server", which asks /login/probe what that server is and what
    it offers; the credential fields are revealed in place. When ``azure_enabled``
    a "Sign in with Microsoft" button (see oauth/azure_login.py) is
    revealed too, but only for a server whose connector list actually contains an
    azure entry.

    ``probe`` and ``resolve`` are injected rather than imported at the call site so
    tests can drive every branch without network or DNS."""
    limiter = _RateLimiter(_LOGIN_MAX_ATTEMPTS, _LOGIN_WINDOW_SECONDS)
    # A separate budget from the password form's on purpose: five mistyped server
    # addresses must not eat half the allowance for sign-in attempts.
    probe_limiter = _RateLimiter(_PROBE_MAX_ATTEMPTS, _LOGIN_WINDOW_SECONDS)
    public = provider.public_url.rstrip("/")
    probe_url = f"{public}/login/probe"
    azure_start_url = f"{public}/login/azure/start"
    # With an allowlist configured there is no SSRF surface — the probe can only
    # reach servers the operator named — so the detailed diagnostics are safe. It
    # is the open, accept-any-server deployment that must not become a port
    # scanner for whatever network this process sits in.
    guard_private = provider.allow_custom_base_url

    def _page(
        session: str,
        error: str | None = None,
        *,
        client_name: str | None = None,
        redirect_host: str | None = None,
        selected: str = "",
        verified: bool = False,
    ) -> str:
        return _login_page_html(
            session, error, client_name=client_name, redirect_host=redirect_host,
            selected=selected, azure_enabled=azure_enabled, verified=verified,
            probe_url=probe_url, azure_start_url=azure_start_url,
        )

    @mcp.custom_route("/login/probe", methods=["POST"])  # type: ignore[untyped-decorator]
    async def login_probe(request: Request) -> Response:
        """What is at this address, and does it offer Microsoft sign-in?

        POST rather than GET so the customer's hostname stays out of history,
        Referer and proxy logs, and so the JSON content type makes it a non-simple
        CORS request that no cross-origin page can read the answer to.

        Answers ``{server, blocked, message, azure}``. ``server`` echoes back
        exactly what was sent so the page can discard a slow reply about an address
        the user has since changed. The canonical URL is deliberately NOT sent
        back: a base URL may carry ``user:pw@`` (Basic-auth proxies are supported),
        and writing a redacted form into the field would destroy it."""
        if not probe_limiter.allow(_client_ip(request, trust_forwarded)):
            return _secure_json({"blocked": True, "message": "Too many attempts. "
                                 "Please wait a minute and try again."}, status_code=429)
        if request.headers.get("sec-fetch-site", "same-origin") != "same-origin":
            return _secure_json({"blocked": True, "message": "Unexpected request."},
                                status_code=400)
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            payload = None
        if not isinstance(payload, dict):
            return _secure_json({"blocked": True, "message": "Unexpected request."},
                                status_code=400)
        session = str(payload.get("session", ""))
        typed = str(payload.get("server", "")).strip()

        # Bounded to a live login session. Not real authorization — client
        # registration is open — but it keeps the route from being a standalone
        # probing service and gives ops an id to correlate on. Checked BEFORE any
        # outbound request is made.
        if not session or not await provider.peek_session(session):
            return _secure_json(
                {"server": typed, "blocked": True,
                 "message": "This sign-in link has expired. Please restart the "
                            "connection from your client."},
                status_code=400,
            )

        base_url, url_problem = provider.check_base_url(typed)
        if base_url is None:
            return _secure_json(
                {"server": typed, "blocked": True,
                 "message": url_problem or "That CDESK server address can't be used.",
                 "azure": False},
            )

        result = await probe(base_url, timeout_seconds=_PROBE_TIMEOUT_SECONDS)
        vague = False
        if result.status != "ok" and guard_private:
            vague = await host_is_private(base_url, resolve)
        blocked, message, azure = _probe_reply(result, vague=vague)
        log.debug(
            "probe %s -> %s (http %s) blocked=%s",
            redact_userinfo(base_url), result.status, result.http_status, blocked,
        )
        return _secure_json(
            {"server": typed, "blocked": blocked, "message": message,
             "azure": azure and azure_enabled},
        )

    @mcp.custom_route("/login", methods=["GET", "POST"])  # type: ignore[untyped-decorator]
    async def login(request: Request) -> Response:  # pragma: no cover - exercised live
        if request.method == "GET":
            session = request.query_params.get("session", "")
            ctx = await provider.login_context(session) if session else None
            if ctx is None:
                return _secure_html(
                    "<h1>Invalid or expired login link</h1>"
                    "<p>Please restart the connection from your client.</p>",
                    status_code=400,
                )
            return _secure_html(
                _page(
                    session,
                    client_name=ctx["client_name"],
                    redirect_host=ctx["redirect_host"],
                )
            )

        # Brute-force guard (defense-in-depth; CDESK + the edge throttle too).
        if not limiter.allow(_client_ip(request, trust_forwarded)):
            return _secure_html(
                "<h1>Too many attempts</h1>"
                "<p>Please wait a minute and try again.</p>",
                status_code=429,
            )

        form = await request.form()
        session = str(form.get("session", ""))
        login_name = str(form.get("login", "")).strip()
        password = str(form.get("password", ""))
        server = str(form.get("server", "")).strip()

        if not session or not await provider.peek_session(session):
            return _secure_html(
                "<h1>Invalid or expired login link</h1>"
                "<p>Please restart the connection from your client.</p>",
                status_code=400,
            )
        if not login_name or not password:
            return _secure_html(
                _page(session, "Enter both your CDESK login and password.", selected=server,
                      verified=bool(server)),
                status_code=400,
            )

        # The user pastes their CDESK server URL; check_base_url canonicalizes it
        # (adding a missing https://) and enforces the allowlist / custom-URL
        # policy, returning the specific reason when it won't accept the address so
        # the page can say which of the dozen possible mistakes this one was.
        base_url, url_problem = provider.check_base_url(server)
        if base_url is None:
            return _secure_html(
                _page(session, url_problem or "That CDESK server address can't be used.",
                      selected=server),
                status_code=400,
            )

        # Validate the credentials with a temporary password-bearing client against
        # the chosen server. authenticate() performs exactly one call — the
        # documented POST /auth/login — which both proves the credentials and
        # captures the apitoken + refresh token we carry onward.
        #
        # It deliberately does NOT probe a second endpoint afterwards. This used to
        # trigger the same lazy login via a GET /auth/me probe; that made every
        # connector login depend on one undocumented endpoint, and when it started
        # returning 500 on a tenant whose data endpoints were all fine, nobody
        # could sign in. /auth/login already distinguishes 200 / 401 / 302 (2FA),
        # so the extra round trip added a failure mode and no information. The
        # probe helper has been deleted — don't add one back.
        validation_client = provider.build_cdesk_client(login_name, password, base_url)
        try:
            await validation_client.authenticate()  # POST /auth/login
        except CdeskAuthError as e:
            await validation_client.close()
            # The operator-facing detail goes to the log; the page shows the
            # human wording (see _signin_error_text).
            log.info("CDESK login rejected for %r: %s", login_name, e)
            return _secure_html(
                _page(session, _signin_error_text(e), selected=server, verified=True),
                status_code=401,
            )
        except Exception as e:  # network / unexpected
            await validation_client.close()
            log.warning("CDESK login probe failed: %s: %s", type(e).__name__, e)
            return _secure_html(
                _page(session, "Could not reach CDESK. Check the server and try again.",
                      selected=server, verified=True),
                status_code=502,
            )

        # Capture the issued tokens, then discard the password-bearing client —
        # the password must not live past this point (only the apitoken + CDESK
        # refresh token are carried onward, encrypted into the issued OAuth token).
        apitoken = validation_client.token
        refresh = validation_client.refresh_token
        await validation_client.close()

        if not apitoken:  # defensive: a successful authenticate() implies a token
            return _secure_html(
                _page(session, "CDESK did not issue a session token. Please try again.",
                      selected=server, verified=True),
                status_code=502,
            )
        if not refresh:
            # No refresh token means the session can't outlive the apitoken's
            # inactivity window; still usable, but flag it for ops.
            log.warning("CDESK login for %r returned no refresh token", login_name)

        # Mint the one-time code; the chosen server is embedded in the credential.
        try:
            redirect_url = await provider.complete_login(
                session, login=login_name, apitoken=apitoken,
                refresh_token=refresh, base_url=base_url,
            )
        except KeyError:
            return _secure_html(
                "<h1>Login session expired</h1><p>Please restart the connection.</p>",
                status_code=400,
            )
        except ValueError:
            return _secure_html(
                _page(session, "That CDESK server isn't allowed.", selected=server,
                      verified=True),
                status_code=400,
            )
        return RedirectResponse(redirect_url, status_code=302)
