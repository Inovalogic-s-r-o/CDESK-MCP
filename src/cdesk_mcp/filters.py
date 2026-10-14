"""Build CDESK structured `sb` filter objects from typed kwargs.

Pure functions, no I/O. Each `build_*_filter` returns either a `dict` ready
to be JSON-encoded and passed as the `sb` query parameter, or `None` if no
filters were specified (the caller should omit `sb` entirely in that case).

Output shape:
    {
      "group": true,
      "o": "AND",
      "query": [
        {"q": "search text", "o": "AND"},
        {"col": "status", "c": "=", "q": 30, "o": "AND"},
        {"col": "created_at", "c": ">=", "q": "2026-01-01T00:00:00+00:00", "o": "AND"},
        ...
      ]
    }

Comparator inventory (CDESK's sb comparators):
    Equality: `=` / `IS` / `YES`, `!=` / `ISNOT` / `NO`
    Truth:    `TRUE`, `FALSE`
    Null:     `ISNULL`, `ISNOTNULL`
    Range:    `BETWEEN` (uses q + q2), `>`, `>=`, `<`, `<=`
    Text:     `CONTAINS`, `CONTAINSNOT`, `IS_LITERAL`, `MATCHAGAINST`
    Date:     `TODAY`, `YESTERDAY`, `TOMORROW`, `THIS_WEEK`, ...,
              `FOR_THE_LAST` (q + q2), `FOR_THE_NEXT` (q + q2)

Each builder accepts `sb_raw` — a JSON string or an already-parsed dict
(FastMCP pre-parses JSON-looking string args into dicts, so both arrive).
Accepted forms: group tree or primary-text. The documented CDQL shorthand
is rejected (the server silently ignores it — see _validate_sb_shape).
When set, the typed kwargs **must NOT also be set** (a ValueError is
raised) — silent override is a UX trap. The value is validated to have
the basic structural shape, and every `col` leaf is checked against the
per-module allowlist of columns that actually filter (CDESK has no
allowlist of its own — an unknown column is silently ignored and the query
runs unfiltered). Non-working columns are handled in one of two modes:
  - strict (default; grounding): raise ValueError — silently broadening an
    absence_query would corrupt truth verdicts.
  - strip (list tools; pass `dropped_cols_out=[]`): remove the non-working
    clauses, append them to the list, and return the (broader) filter — the
    tool then returns the items together with the criteria that were not
    applied.
Values on date columns are normalized to strict W3C form on the raw path too
(_normalize_sb_date_values / _W3C_DATE_SB_COLS): CDESK silently discards a
value it cannot parse as strict W3C, so a bare 'YYYY-MM-DD' would drop the
clause and return the unfiltered set with a 200. Column allowlisting alone
does not protect a working column from an unusable value.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from cdesk_mcp.tz import tenant_timezone

# Resource-specific column maps. Public param name → CDESK column name.
# Centralized so the LLM-friendly vocabulary and CDESK internal vocabulary
# diverge cleanly.
_TASK_COLS = {
    "customer_id": "company_id",  # CDESK Task links to customer via `company_id`
    "solver_id": "solver_id",
    "type_id": "type_id",
    "status": "status",
    # Range-shaped param columns (the column name itself encodes direction).
    "deadline_after": "deadlineDateFrom",   # → tasks.valid_to >= q
    "deadline_before": "deadlineDateTo",    # → tasks.valid_to <= q
    "valid_from_after": "dateFrom",         # → tasks.valid_from >= q
    "valid_from_before": "dateTo",          # → tasks.valid_from <= q
    # Created-date window. CDESK parses the value STRICTLY as W3C
    # ('Y-m-d\TH:i:sP'); anything else is silently discarded and the clause
    # matches ALL rows. _w3c_datetime normalizes our ISO inputs to that exact
    # shape, which is why these are safe to offer.
    "created_after": "created_at",          # → tasks.created_at >= q (W3C)
    "created_before": "created_at",         # → tasks.created_at <= q (W3C)
    # NB: `open`/`late` sb clauses are unusable: the sb TRUE/FALSE comparator
    # delivers the STRING 'TRUE' where CDESK expects boolean true (so it never
    # fires), and `late` with a real boolean fails with HTTP 500. Also ignored
    # among the documented sb cols: num, name, description, type_2nd_id,
    # assign_id, percent_done_manual, valid_from, valid_to, deleted_at;
    # `task_id` returns HTTP 500.
}

_CUSTOMER_COLS = {
    "status": "status",
}

_USER_COLS = {
    "company_id": "id_company",     # User uses `id_company`, not `company_id`
    "user_type": "type",
    "status": "status",             # single-char flag (e.g. "A" = active)
    # NB: onlyDeleted is NOT an sb column — the endpoint ignores it via `sb`
    # (returns active users) and honors it only as the FLAT query param
    # `onlyDeleted=1`. The tool sends it flat.
}

# CDESK accepts these exact labels for the user `type` filter and derives the
# user-group lookup from each. The actual `type` *column* stores a single
# character (e.g. 'M') — different concept; only the filter accepts these
# labels. Lives here (beside build_user_filter) so both the list tool and the
# grounding resolver validate against ONE source instead of a tool importing it
# from a sibling.
USER_TYPE_LABELS = (
    "admin", "operator", "customer", "easyclick",
    "solver", "authorizing-officer", "cm", "guest",
)


def user_type_label(value: str) -> str:
    """The CDESK user-type label for `value`: case-insensitive, `_` read as
    `-` (so "authorizing_officer" is accepted). Anything else raises with the
    valid labels — CDESK itself answers an unknown label with 400."""
    label = value.strip().lower().replace("_", "-")
    if label not in USER_TYPE_LABELS:
        raise ValueError(
            f"user_type={value!r} is not a CDESK user type. Allowed values: "
            f"{', '.join(USER_TYPE_LABELS)}."
        )
    return label

# Request module — per RequestSbFilterFields schema in the OpenAPI doc.
# Surface high-value filters only; the OpenAPI lists ~30 columns but most
# would just bloat the tool signature without value to the LLM.
_REQUEST_COLS = {
    "customer_id": "id_company",        # FK to cdesk_companies.id
    "solver_id": "id_solver",
    "solver_group_id": "solver_group_id",
    # Request status/priority filter on the sb keys `status`/`priority` by
    # enum ACTION_CODE — NOT the prefixed `req_status` column nor the enum id.
    # A clause on `req_status`/`req_priority` is silently ignored (match-all).
    "status_id": "status",
    "priority_id": "priority",
    # Category type: the sb `type` column filters the request's cat_type_id,
    # so a RESOLVED cat_type_id goes in under col `type`.
    # `baseType` filters the base-type LETTER ('H'/'C'/…), passed through as-is.
    "cat_type_id": "type",
    "base_type": "baseType",
    "catalog_id": "catalog_id",
    "deal_id": "id_rc",
    "project_contract_id": "project_contract_id",
    "branch_id": "branch_id",
    "place_id": "place_id",
    "superior_request_id": "superior_request_id",
    # Periodicity: a generated request carries the parent periodical-request id
    # in periodical_request_id (null on ad-hoc requests). The typed `periodic`
    # bool filters via ISNULL/ISNOTNULL on this column (see build_request_filter).
    "periodic": "periodical_request_id",
    # Created-date window. Same mechanics as the task module: the sb key is
    # `created_at` (NOT the documented `idatetime` column), and it requires
    # STRICT W3C datetime values (a bare date is silently discarded → match-all).
    "created_after": "created_at",          # → cd_req.idatetime >= q (W3C)
    "created_before": "created_at",         # → cd_req.idatetime <= q (W3C)
    # Due-date bounds via the legacy v1 keys, which take the value raw: a bare
    # 'YYYY-MM-DD' and the strict-W3C form behave IDENTICALLY, so no
    # normalization is needed (unlike created_at above). `dateFrom` filters
    # cleanly. `dateTo` is NULL-PERMISSIVE: requests with no due date pass the
    # upper bound, so due_before alone does NOT mean "has a due date before X".
    # Pair it with due_after to exclude the undated rows; the tool description
    # says so too.
    "due_after": "dateFrom",                # → req_due_date >= q (raw value)
    "due_before": "dateTo",                 # → req_due_date <= q (raw value)
    # Of the documented RequestSbFilterFields, these are IGNORED by the
    # endpoint (the clause returns the full set): req_title, req_status,
    # req_priority (use the action_code keys `status`/`priority` instead),
    # sla_id, cat_type_id, cat_area_2nd_id, percent_done_manual, deleted_at,
    # req_due_date, req_close_date, idatetime. The columns that do filter are
    # listed in _REQUEST_SB_COLS below.
}

# Request Catalog — of the plain documented RequestCatalogSbFilterFields
# columns only `id` and `type` (the base-type string code, e.g. "H") filter;
# name, code, type_id, desc, cat_*, change_manager_*, approval_rule_id,
# auto_close_request, created_at, updated_at and deleted_at are silently
# ignored (the dotted `request_catalog.*` forms in _REQUEST_CATALOG_SB_COLS do
# filter). The default-cols `q` text leaf works.
_REQUEST_CATALOG_COLS = {
    "type_code": "type",
}

# --- sb_raw column allowlists -------------------------------------------
#
# CDESK has NO allowlist of its own: an sb clause on a column it doesn't
# recognize is silently never read — the query runs UNFILTERED and returns 200,
# presenting unfiltered data as filtered. Many documented
# <Module>SbFilterFields columns do not actually filter, so sb_raw is gated
# client-side to the set of columns that do, per module; a leaf on any other
# column is rejected with the working list. The no-col `q` text leaf works
# everywhere except on knowledge-base articles (see below) and is otherwise
# always allowed.

# Task: the documented columns that filter + the undocumented v1 date keys.
# `contract_id` (the deal link) is NOT here: on tasks it is ignored (filtering
# by one deal returns the whole unfiltered list), although the same column does
# filter on v3/fill.
_TASK_SB_COLS = frozenset({
    "id", "code", "title", "status", "type_id", "solver_id", "place_id", "company_id",
    "request_id", "project_contract_id",
    # undocumented v1 date keys (the documented valid_from/
    # valid_to columns are silently ignored)
    "dateFrom", "dateTo", "deadlineDateFrom", "deadlineDateTo",
    "created_by", "updated_by",
    # date columns below work ONLY with strict W3C datetime values — the
    # typed created_after/created_before params normalize for you.
    "created_at", "updated_at", "close_date",
})

# Request: the working columns + the undocumented action_code keys
# `status`/`priority` (the documented req_status/req_priority are ignored).
# `type` filters the request's cat_type_id — pass the INTEGER cat-type enum id,
# NOT the letter ('H' matches 0); `baseType` filters the base-type LETTER
# ('H'/'C'/…). `dateFrom`/`dateTo` are the due-date >= / <= bounds (raw value);
# the other date columns need strict W3C datetime values.
_REQUEST_SB_COLS = frozenset({
    "id_req", "req_num", "code", "urgency_id", "impact_id", "id_company",
    "catalog_id", "id_solver", "solver_group_id", "id_rc",
    "project_contract_id", "cat_area_id", "branch_id", "place_id",
    "change_manager_id", "superior_request_id", "request_weight",
    "percent_done", "status", "priority", "dateFrom", "dateTo", "type",
    "baseType",
    # periodical_request_id: ISNULL = ad-hoc only, ISNOTNULL = periodic only,
    # `=`/array = a specific periodical-request parent (per RequestSbFilterFields).
    "periodical_request_id",
    # date columns below: strict W3C datetime values only.
    # reaction_due_date is also honored by CDESK but deliberately left out:
    # its matching behaviour is unconfirmed.
    "created_at", "assign_date", "udatetime", "due_date", "close_date",
})

# Fill (fulfillment / work record) — LLM param → CDESK sb col.
# `invoiced`/`signed` are booleans surfaced as the integer flag cols
# `invoiced_status`/`is_signed`. The date window is a single `start_date`
# BETWEEN leaf (see build_fill_filter) — `=` on start_date matches nothing.
_FILL_COLS = {
    "solver_id": "solver_id",
    "company_id": "company_id",
    "request_id": "request_id",
    "deal_id": "contract_id",
    "project_contract_id": "project_contract_id",
    "assign_id": "assign_id",
    "place_id": "place_id",
    "invoiced": "invoiced_status",
    "signed": "is_signed",
    "rma": "rma",
    "worked_window": "start_date",   # BETWEEN(worked_from, worked_to)
}

# Fill: every documented FillSbFilterFields column filters EXCEPT `request` —
# its ISNULL matches the request-linked fills too (match-all), so it's excluded
# as a trap (link/ISNULL filtering works via the `task`/`contract`/`work_order`
# keys). `start_date` works ONLY via BETWEEN with strict W3C bounds (`=`
# matches nothing); the list defaults to the current calendar month when no
# window is sent.
_FILL_SB_COLS = frozenset({
    "fill_id", "assign_id", "company_id", "solver_id", "request_id",
    "contract_id", "project_contract_id", "place_id", "description",
    "used_material", "task", "contract", "work_order", "duration",
    "invoiced_status", "rma", "is_signed", "start_date",
    # end_date filters on both an exact value and a range.
    "end_date",
})

# The fill DATE TRAPS, deliberately absent from the set above (each is ignored —
# a far-future value returns the unfiltered set):
# `date_from`, `date_to`, `valid_from`, `valid_to`. The last two are the names
# `create_fill` takes for the same fields, so they are the natural thing for a
# caller to reach for; the filterable columns are `start_date` / `end_date`.
# Also note the list projection carries NO date column at all, so a real value
# cannot be discovered from a list response.

# Customer (company): use `email` (NOT `email_general`, which is ignored);
# `company.*` are the dotted address columns. `created_at`/`updated_at` take
# strict W3C datetime values only. The other documented date cols
# (create_date/change_date/membership_validity[_date]) are ignored.
_CUSTOMER_SB_COLS = frozenset({
    "id", "name", "status", "hour_rate",
    "ico", "dic", "icdph", "email", "phone",
    "company.city", "company.street", "company.zip", "company.country",
    "created_at", "updated_at",
})

# User: 10 of the 20 documented columns + the label-form `type` filter
# (admin/operator/... — the documented single-char codes are ignored).
_USER_SB_COLS = frozenset({
    "id", "name", "email", "personal_number", "id_company", "status", "type",
    # work_position is left out: CDESK stores it as the job_title_id FK, not
    # the documented string column.
    "uname", "mobil", "id_external",
    # date columns: strict W3C datetime values only. Other documented date
    # columns are silently ignored.
    "last_login_date", "created_at",
})

# Request catalog: the dotted `request_catalog.*` forms filter; the plain
# `name`/`code`/`type_id`/`desc`/`cat_type_id`/`cat_area_id` forms are ignored.
# `companyId` returns the catalogs available to that company.
_REQUEST_CATALOG_SB_COLS = frozenset({
    "id", "type", "companyId", "request_catalog.name",
    "request_catalog.cat_type_id", "request_catalog.type_id",
    "request_catalog.cat_area_id", "job_title_id",
})

# CMDB CI: `parent_id` is silently IGNORED (a real and a bogus id both return
# the full set) — excluded. `created_at` needs strict W3C values (a bare date
# silently matches all; same trap as tasks/requests).
_CI_SB_COLS = frozenset({
    "id", "name", "description", "status_id", "owner_id", "company_id",
    "type_id",
    "created_by", "updated_by",
    # date columns: strict W3C datetime values only. status_date is ignored.
    "created_at", "updated_at",
    # deleted_at is NOT null-permissive (`>= 1990` does not match the nulls),
    # unlike the request `dateTo` upper bound.
    "deleted_at",
})


# Approval module — per ApprovalSbFilterFields schema in the OpenAPI doc
# (docs/cdesk-api-v3.json). The four request/work-order-linked cols
# (request_text, request_applicant, request_catalog_id, work_order_entered_by)
# force the matching assign kind server-side. CDESK implements more cols
# (rule_name, workorder_text, daysoff_*, state_changed, work_order_catalog_id,
# ...) — those are not offered.
_APPROVAL_SB_COLS = frozenset({
    "id", "name", "assign", "state", "approver_id", "processed_approver",
    "processed_state", "solverOrDemand", "request_text", "request_applicant",
    "request_catalog_id", "work_order_entered_by", "created_by",
    # date filter — strict W3C datetime values only (same date parsing as
    # tasks/requests; a bare date silently matches ALL rows).
    "created_at",
})

# Approval — LLM param → CDESK sb col (filter keys).
_APPROVAL_COLS = {
    "scope": "solverOrDemand",
    "state": "state",
    "assign": "assign",
    "approver_id": "approver_id",
    "request_text": "request_text",
    "created_after": "created_at",
    "created_before": "created_at",
}

# Project module — per ProjectSbFilterFields schema in the OpenAPI doc
# (docs/cdesk-api-v3.json). The economic-stat cols (agreed/real/plan
# revenue/lien/external_costs/gross_margin, planned_hours) and `duration`
# filter by equality only: a range comparator answers HTTP 500, and on `id` it
# is read as equality. list_projects applies range conditions on these columns
# itself (tools/_local_filter.py). Values are DECIMAL STRINGS on the
# money columns ('0.00'), an int on `duration`.
# CDESK also implements undocumented keys (leader_id, manager_id,
# project_team, project_watchers_ids, percent_done, date_start/date_end/
# created_at/updated_at — strict W3C) — those are not offered.
# `status_id` values are the STABLE ACTION CODES (10 open, 20 in progress,
# 30 pending, 80 canceled, 90 completed). `company_id` is int-vs-string
# sensitive (a non-numeric CONTAINS value searches the company name instead).
_PROJECT_SB_COLS = frozenset({
    "id", "title", "code", "company_id", "status_id",
    "project_contract_code", "project_contract_title", "tags",
    # economic stats (see above)
    "duration", "planned_hours",
    "agreed_revenue", "real_revenue",
    "plan_lien", "real_lien",
    "agreed_external_costs", "real_external_costs",
    "agreed_gross_margin", "real_gross_margin", "plan_gross_margin",
})


# Work order — per WorkOrderSbFilterFields schema in the OpenAPI doc
# (docs/cdesk-api-v3.json). The list always inner-joins the parent request, so
# request-derived cols are available. `work_order_catalog_id` is a WRITE/link
# field, NOT filterable — excluded. `company_id` filters the linked request's
# company (there is no company column on work orders). `status_id` accepts the
# stable action codes (0-9) or the literals 'open' / 'all'. Date columns need
# strict W3C datetime values. The no-col `q` text leaf spans
# name/description/number/solver/company.
_WORKORDER_SB_COLS = frozenset({
    "id", "number", "name", "description", "status_id", "solver_id",
    "solver_group_id", "request_id", "company_id", "entered_by", "applicant",
    # date columns — strict W3C datetime values only.
    "assign_due_date", "due_date", "created_at", "updated_at",
})

# Work order — LLM param → CDESK sb col.
_WORKORDER_COLS = {
    "number": "number",
    "name": "name",
    "description": "description",
    "status_id": "status_id",
    "solver_id": "solver_id",
    "solver_group_id": "solver_group_id",
    "request_id": "request_id",
    "customer_id": "company_id",          # → linked request's cd_req.id_company
    "entered_by": "entered_by",
    "assign_due_after": "assign_due_date",
    "assign_due_before": "assign_due_date",
    "due_after": "due_date",
    "due_before": "due_date",
    "created_after": "created_at",
    "created_before": "created_at",
    "updated_after": "updated_at",
    "updated_before": "updated_at",
}


# Knowledge base articles — per KnowledgeBaseSbFilterFields. CDESK implements
# ONLY these two keys; every other col is silently ignored. UNIQUELY, the no-col
# `q` text leaf is ALSO ignored — build_kb_article_filter strips/rejects it
# instead of allowing it like every other module.
_KB_ARTICLE_SB_COLS = frozenset({"title", "category_id"})


# Deal module — per ContractSbFilterFields schema in the OpenAPI doc
# (docs/cdesk-api-v3.json). `subtree` works on deal codes shaped
# "<PREFIX> <number>/<suffix>" and only with a `*` or `@` in its value (see
# adapt_sb_for_cdesk). The boolean cols `mine`/`open`/`late`/`wdl` are
# deliberately EXCLUDED (sb boolean delivery can mis-fire server-side, as on
# tasks; the `status` keywords 'open'/'closed' cover the same need reliably).
# Filter-key naming is NOT the column naming: id filters id_rc, `title` filters
# rc_title, `id_customer` filters id_company.
_DEAL_SB_COLS = frozenset({
    "id", "title", "code", "rc_num", "id_customer", "company.name",
    "status", "rc_status", "type", "phase", "responsible_person_id",
    "tag", "subtree", "for_invoicing",
    # due-date bounds: the legacy dateFrom/dateTo keys take the value raw
    # (no W3C parsing server-side); `due_date` itself is a strict-W3C date
    # filter like created_at/updated_at below.
    "dateFrom", "dateTo", "due_date",
    # strict W3C datetime values only (a bare date silently matches ALL rows,
    # same trap as tasks/requests).
    "created_at", "updated_at", "last_invoice_issue_date",
    "updated_by",
})

# Deal columns that filter by id equality only. `updated_by` takes a user id
# or a list of ids with `=`; a name fragment with CONTAINS matches nothing.
_DEAL_EQUALITY_ONLY_SB_COLS = frozenset({"updated_by"})
_SB_EQUALITY_CMPS = frozenset({"=", "IS"})

# Columns whose sb value CDESK parses strictly as W3C. A value in any other
# shape — notably a bare 'YYYY-MM-DD' — is silently discarded: the clause
# EVAPORATES, the query runs UNFILTERED, and the response is still 200. That is
# indistinguishable from a successful filter at the call site, so every value on
# these columns is normalized by _normalize_sb_date_values before the filter
# leaves this module — the same guarantee _w3c_datetime already gives the typed
# params.
#
# The legacy v1 keys dateFrom/dateTo/deadlineDateFrom/deadlineDateTo are
# deliberately NOT listed: CDESK takes their value raw (a bare 'YYYY-MM-DD' and
# strict W3C behave identically).

# Builder param → the param name the TOOL exposes, where they differ (the tool
# resolves an enum NAME to the id the builder takes). Used only to phrase the
# sb_raw-conflict error in the caller's own vocabulary: naming `status_id` at an
# agent that sent `status_name` tells it to remove a parameter it cannot send.
_TASK_PUBLIC_PARAMS = {"status_id": "status_name", "type_id": "type_name"}

_W3C_DATE_SB_COLS = frozenset({
    "created_at", "updated_at", "close_date", "due_date", "assign_date",
    "assign_due_date", "udatetime", "last_login_date",
    "last_invoice_issue_date", "start_date", "deleted_at",
})

# Comparators that compare a date column against a date VALUE. Everything
# else on a date column carries no value to normalize (ISNULL/ISNOTNULL,
# TRUE/FALSE, TODAY/THIS_WEEK/…) or carries a count+unit pair rather than a
# date (FOR_THE_LAST/FOR_THE_NEXT), and must be left untouched.
_SB_DATE_VALUE_CMPS = frozenset({
    "=", "!=", "IS", "ISNOT", ">=", "<", "<=", "BETWEEN",
})

# '>' is NOT in the set above: on a date column CDESK applies no condition
# for it at all, so the clause is a silent MATCH-ALL regardless of the value.
# Normalizing its value would only make an unusable clause look handled, so it
# is refused outright with the working alternative — see
# _normalize_sb_date_values.
_SB_DATE_REJECTED_CMPS = frozenset({">"})

# Bare-date expansion direction: '<=' (on or before) means the END of the named
# day; '>=', '<', '=' and '!=' mean its start. (For '='/'IS'/'!='/'ISNOT'
# CDESK re-expands to the whole day itself, so the choice there is cosmetic.)
_SB_DATE_END_OF_DAY_CMPS = frozenset({"<="})

# The legacy v1 date keys. These take their value RAW (no strict W3C parse),
# so they must NOT be reshaped — but an unparseable value still silently
# matches ALL rows exactly like the W3C columns (e.g. dateFrom >= 'last
# Tuesday'). So they get validation without normalization.
_V1_RAW_DATE_SB_COLS = frozenset({
    "dateFrom", "dateTo", "deadlineDateFrom", "deadlineDateTo",
})

# Range comparators. On a set-membership column these are never a filter, only
# a lie in one of two directions (see _RANGE_UNSAFE_TASK_SB_COLS).
_SB_RANGE_CMPS = frozenset({">", ">=", "<", "<=", "BETWEEN"})

# Task columns that do NOT honor a range comparator:
#   id, company_id   -> treated as `=`: CDESK reads only the VALUE and filters
#                       IN (value), so `id > 6300` silently means
#                       `id IN (6300)` and `BETWEEN 6320..6339` means
#                       IN (6320, 6339) — the two endpoints only.
#                       UNDER-reports: a confident empty answer.
#   solver_id, status, created_by -> every row (match-all).
#   updated_by       -> '>'/'<' empty, '>='/'<=' match-all. Wrong both ways.
# `code` and `title` are deliberately NOT here: they apply the comparator and
# genuinely filter (lexicographically — '10' < '100' — which is surprising on
# numeric-looking codes but is a real filter, not a lie).
# type_id/place_id/request_id/deal_id/project_contract_id are NOT here either:
# their range behaviour is unconfirmed, and over-rejecting breaks a working
# query.
_RANGE_UNSAFE_TASK_SB_COLS = frozenset({
    "id", "company_id", "solver_id", "status", "created_by", "updated_by",
})

# Deal — LLM param → CDESK sb col (filter keys).
_DEAL_COLS = {
    "title": "title",
    "code": "code",
    "customer_id": "id_customer",       # filters cd_real_contracts.id_company
    "customer_name": "company.name",    # LIKE across name/alias/customer code
    "status": "status",                 # 'open'/'closed' keyword, id, or array
    "cat_type_id": "type",              # int → cat_type_id
    "phase_id": "phase",                # rc_phase_id
    "responsible_person_id": "responsible_person_id",
    "for_invoicing": "for_invoicing",
    "due_after": "dateFrom",            # → due_date >= q (raw value)
    "due_before": "dateTo",             # → due_date <= q (raw value)
    "created_after": "created_at",      # → insert_date (strict W3C)
    "created_before": "created_at",
    "updated_after": "updated_at",      # → update_date (strict W3C)
    "updated_before": "updated_at",
}


# CMDB CI module — per CmdbCiSbFilterFields schema in the OpenAPI doc.
# Only direct cmdb_ci columns are sb-filterable; category/maingroup/type/
# company/deleted are query parameters on GET /v3/cmdb/ci, NOT sb columns
# (the doc calls this out explicitly), so they're handled by the tool.
# `parent_id` is not offered: CDESK silently ignores that sb clause.
_CI_COLS = {
    "status_id": "status_id",
    "owner_id": "owner_id",
}


def build_task_filter(
    *,
    text_search: str | None = None,
    status_id: int | None = None,
    status_group: str | None = None,
    type_id: int | None = None,
    solver_id: int | list[int] | None = None,
    customer_id: int | list[int] | None = None,
    valid_from_after: str | None = None,
    valid_from_before: str | None = None,
    deadline_after: str | None = None,
    deadline_before: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the Task list endpoint.

    LLM-friendly arg names; the builder maps to CDESK internal columns
    (see _TASK_COLS). Resolution of name→id (e.g. status name → tenant id)
    is the caller's responsibility — pass already-resolved ints here.

    Date params accept ISO-8601 strings (e.g. "2026-05-28" or
    "2026-05-28T10:00:00+02:00"). They're validated client-side so
    malformed input fails fast instead of producing an opaque 400.
    created_after/created_before are additionally normalized to the strict
    W3C datetime form the backend's date parser requires — a bare date would
    otherwise be SILENTLY ignored server-side (see _TASK_COLS note). A naive
    datetime is treated as tenant-local time (CDESK_TIMEZONE); a bare date means
    start-of-day for `after` and end-of-day for `before` (both inclusive).

    only_open/only_late are deliberately not offered — the sb TRUE/FALSE
    path can't reach those filters (see the _TASK_COLS note).
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"},
            public_names=_TASK_PUBLIC_PARAMS,
        )
        return _parse_sb_raw(
            sb_raw, allowed_cols=_TASK_SB_COLS, dropped_cols_out=dropped_cols_out,
            range_unsafe_cols=_RANGE_UNSAFE_TASK_SB_COLS,
        )

    # The v1 dateFrom/dateTo/deadlineDate* keys take their value raw, and a
    # bare 'YYYY-MM-DD' on an UPPER bound therefore means MIDNIGHT — it
    # silently excludes the whole named day. Expand the upper bounds to
    # end-of-day so they match the inclusive semantics `created_before`
    # already has, and so the two neighbouring param pairs stop disagreeing.
    #
    # These keys accept a naive, space-separated or offset-bearing datetime
    # alike and IGNORE the offset (+02:00 and +00:00 give identical results),
    # so emitting the tenant-local W3C form is both accepted and semantically
    # right — CDESK compares wall-clock.
    valid_from_after = _w3c_datetime("valid_from_after", valid_from_after)
    valid_from_before = _w3c_datetime(
        "valid_from_before", valid_from_before, end_of_day=True
    )
    deadline_after = _w3c_datetime("deadline_after", deadline_after)
    deadline_before = _w3c_datetime(
        "deadline_before", deadline_before, end_of_day=True
    )
    created_after_w3c = _w3c_datetime("created_after", created_after)
    created_before_w3c = _w3c_datetime(
        "created_before", created_before, end_of_day=True
    )

    clauses: list[dict[str, Any]] = []
    if text_search:
        clauses.append(_text_leaf(text_search))
    if status_id is not None:
        clauses.append(_eq_leaf(_TASK_COLS["status"], status_id))
    if status_group is not None:
        clauses.append(_eq_leaf(
            _TASK_COLS["status"], _status_group_value("tasks", status_group, TASK_STATUS_GROUPS),
        ))
    if type_id is not None:
        clauses.append(_eq_leaf(_TASK_COLS["type_id"], type_id))
    if solver_id is not None:
        clauses.append(_eq_leaf(_TASK_COLS["solver_id"], solver_id))
    if customer_id is not None:
        clauses.append(_eq_leaf(_TASK_COLS["customer_id"], customer_id))
    if valid_from_after:
        clauses.append(_eq_leaf(_TASK_COLS["valid_from_after"], valid_from_after))
    if valid_from_before:
        clauses.append(_eq_leaf(_TASK_COLS["valid_from_before"], valid_from_before))
    if deadline_after:
        clauses.append(_eq_leaf(_TASK_COLS["deadline_after"], deadline_after))
    if deadline_before:
        clauses.append(_eq_leaf(_TASK_COLS["deadline_before"], deadline_before))
    if created_after_w3c:
        clauses.append(_cmp_leaf(_TASK_COLS["created_after"], ">=", created_after_w3c))
    if created_before_w3c:
        clauses.append(_cmp_leaf(_TASK_COLS["created_before"], "<=", created_before_w3c))

    return _wrap(clauses)


def build_customer_filter(
    *,
    text_search: str | None = None,
    status: str | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the Customer list endpoint.

    `text_search` matches across name, customer code, alias, country, city,
    street, zip, IČO, DIČ, IČ DPH (CDESK's default fields for company `q`) —
    which is how country/city are searched; CDESK's `sb` filter has no
    standalone country/city column (see CustomerSbFilterFields). For an exact
    country/city match use sb_raw against a column that exists.
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"}
        )
        return _parse_sb_raw(
            sb_raw, allowed_cols=_CUSTOMER_SB_COLS, dropped_cols_out=dropped_cols_out
        )

    clauses: list[dict[str, Any]] = []
    if text_search:
        clauses.append(_text_leaf(text_search))
    if status is not None:
        clauses.append(_eq_leaf(_CUSTOMER_COLS["status"], status))

    return _wrap(clauses)


def build_user_filter(
    *,
    text_search: str | None = None,
    company_id: int | list[int] | None = None,
    user_type: str | None = None,
    status: str | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the User list endpoint.

    `user_type` takes one of CDESK's type-filter labels (USER_TYPE_LABELS),
    checked by user_type_label.

    A deleted-users toggle is deliberately NOT offered here: the endpoint
    ignores an `onlyDeleted` sb clause (silent no-op returning active users)
    and honors only the flat `onlyDeleted=1` query param, which the caller
    must pass alongside the `sb` (see list_users).
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"}
        )
        return _parse_sb_raw(
            sb_raw, allowed_cols=_USER_SB_COLS, dropped_cols_out=dropped_cols_out
        )

    clauses: list[dict[str, Any]] = []
    if text_search:
        clauses.append(_text_leaf(text_search))
    if company_id is not None:
        clauses.append(_eq_leaf(_USER_COLS["company_id"], company_id))
    if user_type:
        clauses.append(_eq_leaf(_USER_COLS["user_type"], user_type_label(user_type)))
    if status:
        clauses.append(_eq_leaf(_USER_COLS["status"], status))

    return _wrap(clauses)


# CDESK's own status groups. The v3 request and task lists apply them when
# `status` carries the group value — the same quick filters the CDESK UI
# offers. The v3 spec documents `status` only as an action code, so the group
# values live here.
REQUEST_STATUS_GROUPS: dict[str, int] = {
    "open": 100,                  # otvorené — action code < 80
    "completed": 250,             # vyriešené — 80, 81, 85, 86
    "closed": 200,                # uzavreté — 90, 91, 95 only
    "closed_and_completed": 210,  # vyriešené, uzavreté — both of the above
    "waiting_for_response": 260,  # čaká na odozvu — open and waiting for a reaction
}
TASK_STATUS_GROUPS = ("open", "closed")  # < 80 (or no status) / >= 80


def _status_group_value(kind: str, group: str, groups: Any) -> Any:
    if group not in groups:
        raise ValueError(
            f"status_group must be one of {', '.join(groups)} for {kind} (got {group!r})"
        )
    return groups[group] if isinstance(groups, dict) else group


def build_request_filter(
    *,
    text_search: str | None = None,
    status_id: int | None = None,
    status_group: str | None = None,
    priority_id: int | None = None,
    cat_type_id: int | None = None,
    base_type: str | None = None,
    customer_id: int | list[int] | None = None,
    solver_id: int | list[int] | None = None,
    solver_group_id: int | None = None,
    catalog_id: int | None = None,
    deal_id: int | None = None,
    project_contract_id: int | None = None,
    branch_id: int | None = None,
    place_id: int | None = None,
    superior_request_id: int | None = None,
    periodic: bool | None = None,
    due_after: str | None = None,
    due_before: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the Request list endpoint.

    LLM-friendly arg names; the builder maps to CDESK internal columns
    (see _REQUEST_COLS). Name→id resolution (status_name → status_id) is
    the caller's responsibility — pass already-resolved ints here.

    periodic narrows by request periodicity via ISNULL/ISNOTNULL on
    periodical_request_id: True → periodic (generated) requests only,
    False → ad-hoc requests only, None → no constraint.

    created_after/created_before accept ISO-8601 and are normalized to the
    strict W3C datetime form the backend's date parser requires — a bare
    date would otherwise be SILENTLY ignored server-side (see _REQUEST_COLS
    note). A naive datetime is treated as tenant-local time (CDESK_TIMEZONE); a
    bare date means start-of-day for `after` / end-of-day for `before`.

    cat_type_id filters the request CATEGORY type via the sb `type` column
    (resolve a cat_type NAME → id with the request enum cache's "cat_type_id"
    bucket; pass the int here). base_type filters the base-type LETTER
    ('H'/'C'/…) via the sb `baseType` column.

    due_after/due_before ride the legacy raw-value dateFrom/dateTo keys
    (validated ISO, passed through — same as the task/deal deadline
    params). CAVEAT: the dateTo upper bound is NULL-permissive server-side —
    requests with no due date pass it — so due_before alone is not "has a due
    date before X"; combine it with due_after to exclude undated requests.

    A deleted toggle and sla_id are deliberately not offered — the v3 request
    list silently ignores those sb columns (see _REQUEST_COLS).
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"}
        )
        return _parse_sb_raw(
            sb_raw, allowed_cols=_REQUEST_SB_COLS, dropped_cols_out=dropped_cols_out
        )

    _validate_iso_date("due_after", due_after)
    _validate_iso_date("due_before", due_before)
    created_after_w3c = _w3c_datetime("created_after", created_after)
    created_before_w3c = _w3c_datetime(
        "created_before", created_before, end_of_day=True
    )

    clauses: list[dict[str, Any]] = []
    if text_search:
        clauses.append(_text_leaf(text_search))
    if status_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["status_id"], status_id))
    if status_group is not None:
        clauses.append(_eq_leaf(
            _REQUEST_COLS["status_id"],
            _status_group_value("requests", status_group, REQUEST_STATUS_GROUPS),
        ))
    if priority_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["priority_id"], priority_id))
    if cat_type_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["cat_type_id"], cat_type_id))
    if base_type is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["base_type"], base_type))
    if customer_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["customer_id"], customer_id))
    if solver_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["solver_id"], solver_id))
    if solver_group_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["solver_group_id"], solver_group_id))
    if catalog_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["catalog_id"], catalog_id))
    if deal_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["deal_id"], deal_id))
    if project_contract_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["project_contract_id"], project_contract_id))
    if branch_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["branch_id"], branch_id))
    if place_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["place_id"], place_id))
    if superior_request_id is not None:
        clauses.append(_eq_leaf(_REQUEST_COLS["superior_request_id"], superior_request_id))
    if periodic is not None:
        clauses.append(_null_leaf(_REQUEST_COLS["periodic"], is_null=not periodic))
    if due_after:
        clauses.append(_eq_leaf(_REQUEST_COLS["due_after"], due_after))
    if due_before:
        clauses.append(_eq_leaf(_REQUEST_COLS["due_before"], due_before))
    if created_after_w3c:
        clauses.append(_cmp_leaf(_REQUEST_COLS["created_after"], ">=", created_after_w3c))
    if created_before_w3c:
        clauses.append(_cmp_leaf(_REQUEST_COLS["created_before"], "<=", created_before_w3c))

    return _wrap(clauses)


# The list defaults to the current calendar month unless a date window is sent;
# these bounds let a one-sided worked_from/worked_to still ride the
# start_date BETWEEN path (the only comparator the column honors).
_FILL_FAR_PAST = "2000-01-01T00:00:00"
_FILL_FAR_FUTURE = "2099-12-31T23:59:59"


def build_fill_filter(
    *,
    text_search: str | None = None,
    solver_id: int | list[int] | None = None,
    company_id: int | list[int] | None = None,
    request_id: int | list[int] | None = None,
    deal_id: int | list[int] | None = None,
    project_contract_id: int | None = None,
    assign_id: int | None = None,
    place_id: int | None = None,
    invoiced: bool | None = None,
    signed: bool | None = None,
    rma: bool | None = None,
    worked_from: str | None = None,
    worked_to: str | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the Fill (fulfillment) list endpoint.

    LLM-friendly arg names; the builder maps to CDESK internal columns (see
    _FILL_COLS). Pass already-resolved ints — Fill has no enum *name* params.

    worked_from/worked_to bound fills.valid_from via a single `start_date`
    BETWEEN leaf (the only form the column honors — `=` does not filter).
    Each accepts ISO-8601 and is normalized to the
    strict W3C datetime BETWEEN bounds the backend requires (worked_from
    start-of-day, worked_to end-of-day); a one-sided window fills the other
    bound with a far past/future sentinel. IMPORTANT: with NO window the v1 list
    returns only the CURRENT CALENDAR MONTH — always pass a window to widen.

    invoiced/signed/rma are booleans surfaced as the integer flag columns
    invoiced_status/is_signed/rma (True→1, False→0).
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"},
        )
        return _parse_sb_raw(
            sb_raw, allowed_cols=_FILL_SB_COLS, dropped_cols_out=dropped_cols_out
        )

    clauses: list[dict[str, Any]] = []
    if text_search:
        clauses.append(_text_leaf(text_search))
    if solver_id is not None:
        clauses.append(_eq_leaf(_FILL_COLS["solver_id"], solver_id))
    if company_id is not None:
        clauses.append(_eq_leaf(_FILL_COLS["company_id"], company_id))
    if request_id is not None:
        clauses.append(_eq_leaf(_FILL_COLS["request_id"], request_id))
    if deal_id is not None:
        clauses.append(_eq_leaf(_FILL_COLS["deal_id"], deal_id))
    if project_contract_id is not None:
        clauses.append(_eq_leaf(_FILL_COLS["project_contract_id"], project_contract_id))
    if assign_id is not None:
        clauses.append(_eq_leaf(_FILL_COLS["assign_id"], assign_id))
    if place_id is not None:
        clauses.append(_eq_leaf(_FILL_COLS["place_id"], place_id))
    if invoiced is not None:
        clauses.append(_eq_leaf(_FILL_COLS["invoiced"], 1 if invoiced else 0))
    if signed is not None:
        clauses.append(_eq_leaf(_FILL_COLS["signed"], 1 if signed else 0))
    if rma is not None:
        clauses.append(_eq_leaf(_FILL_COLS["rma"], 1 if rma else 0))
    if worked_from or worked_to:
        frm = _w3c_datetime("worked_from", worked_from) or (
            _w3c_datetime("worked_from", _FILL_FAR_PAST)
        )
        to = _w3c_datetime("worked_to", worked_to, end_of_day=True) or (
            _w3c_datetime("worked_to", _FILL_FAR_FUTURE)
        )
        clauses.append(_between_leaf(_FILL_COLS["worked_window"], frm, to))

    return _wrap(clauses)


def ensure_fill_start_date_window(
    sb_raw: str | dict[str, Any],
) -> str | dict[str, Any]:
    """Guarantee a raw fill filter bounds `start_date` so a grounded collect
    scans full history instead of the v3 list's silent current-calendar-month
    default.

    The fill list returns ONLY the current month when no start_date window is
    sent, so an sb_raw that filters by some other
    column (duration, used_material, a link col) would scan just this month and
    could turn a real older fill into a false confirmed_absence — the exact
    failure mode the typed-path window injection prevents. If the parsed tree
    already targets start_date we trust the caller's bound; otherwise we AND-in a
    wide one by wrapping. Wrapping (rather than appending into the existing
    query) is structure-agnostic: it preserves the caller's boolean expression
    even when its top-level connector is OR. A value we can't parse is returned
    unchanged for build_fill_filter to reject with its normal error."""
    if isinstance(sb_raw, dict):
        parsed: dict[str, Any] = sb_raw
    else:
        try:
            loaded = json.loads(sb_raw)
        except json.JSONDecodeError:
            return sb_raw
        if not isinstance(loaded, dict):
            return sb_raw
        parsed = loaded
    if _sb_tree_targets_col(parsed, _FILL_COLS["worked_window"]):
        return parsed
    window = _between_leaf(
        _FILL_COLS["worked_window"],
        _w3c_datetime("worked_from", _FILL_FAR_PAST),
        _w3c_datetime("worked_to", _FILL_FAR_FUTURE, end_of_day=True),
    )
    return {"group": True, "o": "AND", "query": [parsed, window]}


def _sb_tree_targets_col(sb: dict[str, Any], col: str) -> bool:
    """True if any leaf anywhere in the sb tree filters on `col` (recurses into
    nested group nodes)."""
    query = sb.get("query")
    if not isinstance(query, list):
        return False
    for leaf in query:
        if not isinstance(leaf, dict):
            continue
        if leaf.get("group") is True or isinstance(leaf.get("query"), list):
            if _sb_tree_targets_col(leaf, col):
                return True
            continue
        if leaf.get("col") == col:
            return True
    return False


def build_request_catalog_filter(
    *,
    text_search: str | None = None,
    type_code: str | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the Request Catalog list endpoint.

    `text_search` matches across name / code / desc (CDESK's default
    fields). `type_code` filters by the base request type string code
    (e.g. "H") — the only working typed column besides `id`; type_id /
    auto_close_request / deleted_at are deliberately not offered (the
    live endpoint silently ignores them — see _REQUEST_CATALOG_COLS).
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"}
        )
        return _parse_sb_raw(
            sb_raw,
            allowed_cols=_REQUEST_CATALOG_SB_COLS,
            dropped_cols_out=dropped_cols_out,
        )

    clauses: list[dict[str, Any]] = []
    if text_search:
        clauses.append(_text_leaf(text_search))
    if type_code:
        clauses.append(_eq_leaf(_REQUEST_CATALOG_COLS["type_code"], type_code))

    return _wrap(clauses)


def build_ci_filter(
    *,
    text_search: str | None = None,
    status_id: int | None = None,
    owner_id: int | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the CMDB configuration-item list endpoint.

    Only direct cmdb_ci columns go through `sb` (see _CI_COLS). The
    hierarchy filters (category/maingroup/type), company, and the
    deleted toggle are query parameters on GET /v3/cmdb/ci — the tool
    passes those alongside, never through this builder.

    `parent_id` is deliberately not offered — the endpoint silently
    ignores the sb clause (see _CI_COLS note).
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"}
        )
        return _parse_sb_raw(
            sb_raw, allowed_cols=_CI_SB_COLS, dropped_cols_out=dropped_cols_out
        )

    clauses: list[dict[str, Any]] = []
    if text_search:
        clauses.append(_text_leaf(text_search))
    if status_id is not None:
        clauses.append(_eq_leaf(_CI_COLS["status_id"], status_id))
    if owner_id is not None:
        clauses.append(_eq_leaf(_CI_COLS["owner_id"], owner_id))

    return _wrap(clauses)


def build_project_filter(
    *,
    text_search: str | None = None,
    title: str | None = None,
    code: str | None = None,
    customer_id: int | list[int] | None = None,
    status_id: int | list[int] | None = None,
    project_contract_code: str | None = None,
    project_contract_title: str | None = None,
    tag_id: int | list[int] | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the Project list endpoint.

    `status_id` takes the stable ACTION CODES (10 open, 20 in progress,
    30 pending, 80 canceled, 90 completed) — name resolution is the caller's
    job. `customer_id` must be an int (the backend treats a non-numeric value
    as a company-name search instead). `text_search` rides the no-col `q`
    leaf, which for projects spans id, title, company name/alias/code, tags
    and leader/manager names. The documented economic-stat cols and
    `duration` are available through sb_raw (see _PROJECT_SB_COLS).
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"}
        )
        return _parse_sb_raw(
            sb_raw, allowed_cols=_PROJECT_SB_COLS, dropped_cols_out=dropped_cols_out
        )

    clauses: list[dict[str, Any]] = []
    if text_search:
        clauses.append(_text_leaf(text_search))
    if title:
        clauses.append({"col": "title", "c": "CONTAINS", "q": title, "o": "AND"})
    if code:
        clauses.append({"col": "code", "c": "CONTAINS", "q": code, "o": "AND"})
    if customer_id is not None:
        clauses.append(_eq_leaf("company_id", customer_id))
    if status_id is not None:
        clauses.append(_eq_leaf("status_id", status_id))
    if project_contract_code:
        clauses.append(
            {"col": "project_contract_code", "c": "CONTAINS",
             "q": project_contract_code, "o": "AND"}
        )
    if project_contract_title:
        clauses.append(
            {"col": "project_contract_title", "c": "CONTAINS",
             "q": project_contract_title, "o": "AND"}
        )
    if tag_id is not None:
        clauses.append(_eq_leaf("tags", tag_id))

    return _wrap(clauses)


def build_workorder_filter(
    *,
    text_search: str | None = None,
    number: int | str | None = None,
    name: str | None = None,
    description: str | None = None,
    status_id: int | str | None = None,
    solver_id: int | list[int] | None = None,
    solver_group_id: int | None = None,
    request_id: int | list[int] | None = None,
    customer_id: int | None = None,
    entered_by: int | None = None,
    assign_due_after: str | None = None,
    assign_due_before: str | None = None,
    due_after: str | None = None,
    due_before: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the Work Order list endpoint.

    `status_id` takes the stable ACTION CODES (0 new … 5 solving … 8 finished,
    9 canceled) or the literal keywords 'open' / 'all' — name resolution is the
    caller's job. `customer_id` filters the linked request's company. `name`/
    `description` are CONTAINS; `text_search` is the no-col `q` leaf. The four
    date windows are normalized to the strict W3C form the backend's date
    filter requires (a bare date would otherwise silently match all rows).
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"}
        )
        return _parse_sb_raw(
            sb_raw, allowed_cols=_WORKORDER_SB_COLS, dropped_cols_out=dropped_cols_out
        )

    ad_after = _w3c_datetime("assign_due_after", assign_due_after)
    ad_before = _w3c_datetime("assign_due_before", assign_due_before, end_of_day=True)
    d_after = _w3c_datetime("due_after", due_after)
    d_before = _w3c_datetime("due_before", due_before, end_of_day=True)
    c_after = _w3c_datetime("created_after", created_after)
    c_before = _w3c_datetime("created_before", created_before, end_of_day=True)
    u_after = _w3c_datetime("updated_after", updated_after)
    u_before = _w3c_datetime("updated_before", updated_before, end_of_day=True)

    clauses: list[dict[str, Any]] = []
    if text_search:
        clauses.append(_text_leaf(text_search))
    if number is not None:
        clauses.append(_eq_leaf(_WORKORDER_COLS["number"], number))
    if name:
        clauses.append({"col": "name", "c": "CONTAINS", "q": name, "o": "AND"})
    if description:
        clauses.append(
            {"col": "description", "c": "CONTAINS", "q": description, "o": "AND"}
        )
    if status_id is not None:
        clauses.append(_eq_leaf(_WORKORDER_COLS["status_id"], status_id))
    if solver_id is not None:
        clauses.append(_eq_leaf(_WORKORDER_COLS["solver_id"], solver_id))
    if solver_group_id is not None:
        clauses.append(_eq_leaf(_WORKORDER_COLS["solver_group_id"], solver_group_id))
    if request_id is not None:
        clauses.append(_eq_leaf(_WORKORDER_COLS["request_id"], request_id))
    if customer_id is not None:
        clauses.append(_eq_leaf(_WORKORDER_COLS["customer_id"], customer_id))
    if entered_by is not None:
        clauses.append(_eq_leaf(_WORKORDER_COLS["entered_by"], entered_by))
    if ad_after:
        clauses.append(_cmp_leaf("assign_due_date", ">=", ad_after))
    if ad_before:
        clauses.append(_cmp_leaf("assign_due_date", "<=", ad_before))
    if d_after:
        clauses.append(_cmp_leaf("due_date", ">=", d_after))
    if d_before:
        clauses.append(_cmp_leaf("due_date", "<=", d_before))
    if c_after:
        clauses.append(_cmp_leaf("created_at", ">=", c_after))
    if c_before:
        clauses.append(_cmp_leaf("created_at", "<=", c_before))
    if u_after:
        clauses.append(_cmp_leaf("updated_at", ">=", u_after))
    if u_before:
        clauses.append(_cmp_leaf("updated_at", "<=", u_before))

    return _wrap(clauses)


def build_kb_article_filter(
    *,
    title: str | None = None,
    category_id: int | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the Knowledge Base article list endpoint.

    The smallest builder in the module: the article filter implements
    ONLY `title` (contains) and `category_id` (positive int, exact). There is
    deliberately NO text_search param — uniquely, this endpoint has no `q`
    branch, so the universal no-col text leaf is silently ignored too;
    sb_raw text leaves are stripped/rejected (allow_text_leaves=False).
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"}
        )
        return _parse_sb_raw(
            sb_raw,
            allowed_cols=_KB_ARTICLE_SB_COLS,
            dropped_cols_out=dropped_cols_out,
            allow_text_leaves=False,
        )

    clauses: list[dict[str, Any]] = []
    if title:
        clauses.append({"col": "title", "c": "CONTAINS", "q": title, "o": "AND"})
    if category_id is not None:
        if category_id <= 0:
            raise ValueError(
                f"category_id must be a positive id (got {category_id}) — the "
                f"backend ignores non-positive values"
            )
        clauses.append(_eq_leaf("category_id", category_id))

    return _wrap(clauses)


def build_deal_filter(
    *,
    text_search: str | None = None,
    title: str | None = None,
    code: str | None = None,
    customer_id: int | list[int] | None = None,
    customer_name: str | None = None,
    status: int | str | None = None,
    cat_type_id: int | None = None,
    phase_id: int | None = None,
    responsible_person_id: int | None = None,
    for_invoicing: bool | None = None,
    due_after: str | None = None,
    due_before: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the Deal list endpoint.

    Name→id resolution is the caller's responsibility. `status` accepts the
    backend's literal keywords 'open' / 'closed' (action_code < / >= 30) or an
    already-resolved value: a tenant status ENUM id (> 80 → matched on
    rc_status_id) or a standard ACTION CODE (10/20/25/30/80 → matched on
    rc_status). `cat_type_id` filters via the `type` key; `phase_id` via
    `phase` (rc_phase_id).

    due_after/due_before ride the legacy raw-value dateFrom/dateTo keys
    (validated ISO, passed through — same as the task deadline params).
    created_/updated_ windows use the strict-W3C date filters and are
    normalized like the task module's created window (see _w3c_datetime).
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"}
        )
        parsed = _parse_sb_raw(
            sb_raw, allowed_cols=_DEAL_SB_COLS, dropped_cols_out=dropped_cols_out
        )
        if parsed is not None:
            _require_equality_on_cols(parsed, _DEAL_EQUALITY_ONLY_SB_COLS)
        return parsed

    _validate_iso_date("due_after", due_after)
    _validate_iso_date("due_before", due_before)
    created_after_w3c = _w3c_datetime("created_after", created_after)
    created_before_w3c = _w3c_datetime(
        "created_before", created_before, end_of_day=True
    )
    updated_after_w3c = _w3c_datetime("updated_after", updated_after)
    updated_before_w3c = _w3c_datetime(
        "updated_before", updated_before, end_of_day=True
    )

    clauses: list[dict[str, Any]] = []
    if text_search:
        clauses.append(_text_leaf(text_search))
    if title:
        clauses.append(
            {"col": _DEAL_COLS["title"], "c": "CONTAINS", "q": title, "o": "AND"}
        )
    if code:
        clauses.append(
            {"col": _DEAL_COLS["code"], "c": "CONTAINS", "q": code, "o": "AND"}
        )
    if customer_id is not None:
        clauses.append(_eq_leaf(_DEAL_COLS["customer_id"], customer_id))
    if customer_name:
        clauses.append(
            {"col": _DEAL_COLS["customer_name"], "c": "CONTAINS",
             "q": customer_name, "o": "AND"}
        )
    if status is not None:
        clauses.append(_eq_leaf(_DEAL_COLS["status"], status))
    if cat_type_id is not None:
        clauses.append(_eq_leaf(_DEAL_COLS["cat_type_id"], cat_type_id))
    if phase_id is not None:
        clauses.append(_eq_leaf(_DEAL_COLS["phase_id"], phase_id))
    if responsible_person_id is not None:
        clauses.append(
            _eq_leaf(_DEAL_COLS["responsible_person_id"], responsible_person_id)
        )
    if for_invoicing is not None:
        clauses.append(
            _eq_leaf(_DEAL_COLS["for_invoicing"], 1 if for_invoicing else 0)
        )
    if due_after:
        clauses.append(_eq_leaf(_DEAL_COLS["due_after"], due_after))
    if due_before:
        clauses.append(_eq_leaf(_DEAL_COLS["due_before"], due_before))
    if created_after_w3c:
        clauses.append(
            _cmp_leaf(_DEAL_COLS["created_after"], ">=", created_after_w3c)
        )
    if created_before_w3c:
        clauses.append(
            _cmp_leaf(_DEAL_COLS["created_before"], "<=", created_before_w3c)
        )
    if updated_after_w3c:
        clauses.append(
            _cmp_leaf(_DEAL_COLS["updated_after"], ">=", updated_after_w3c)
        )
    if updated_before_w3c:
        clauses.append(
            _cmp_leaf(_DEAL_COLS["updated_before"], "<=", updated_before_w3c)
        )

    return _wrap(clauses)


def build_approval_filter(
    *,
    text_search: str | None = None,
    scope_id: int | None = None,
    state_id: int | None = None,
    assign_id: int | None = None,
    approver_id: int | None = None,
    request_text: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    sb_raw: str | dict[str, Any] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Structured filter for the Approval list endpoint.

    Pass already-resolved ints — the approval enums are PHP class constants
    (not tenant data), so the tool layer resolves names via its static maps:
    scope_id = solverOrDemand preset (10=all … 13=mine-waiting), state_id =
    approvals.state (1=waiting … 7=returned for replenishment), assign_id =
    parent kind (1=request, 2=days-off, 3=work order, 4=simple approval).

    request_text searches the linked request's title/text (and forces
    request-kind approvals server-side). created_after/created_before accept
    ISO-8601 and are normalized to the strict W3C datetime form the backend's
    date parser requires — a bare date would otherwise be SILENTLY ignored
    (same mechanics as tasks/requests; see _w3c_datetime).
    """
    if sb_raw is not None:
        _reject_typed_with_sb_raw(
            sb_raw, locals(), exclude={"sb_raw", "dropped_cols_out"}
        )
        return _parse_sb_raw(
            sb_raw, allowed_cols=_APPROVAL_SB_COLS, dropped_cols_out=dropped_cols_out
        )

    created_after_w3c = _w3c_datetime("created_after", created_after)
    created_before_w3c = _w3c_datetime(
        "created_before", created_before, end_of_day=True
    )

    clauses: list[dict[str, Any]] = []
    if text_search:
        clauses.append(_text_leaf(text_search))
    if scope_id is not None:
        clauses.append(_eq_leaf(_APPROVAL_COLS["scope"], scope_id))
    if state_id is not None:
        clauses.append(_eq_leaf(_APPROVAL_COLS["state"], state_id))
    if assign_id is not None:
        clauses.append(_eq_leaf(_APPROVAL_COLS["assign"], assign_id))
    if approver_id is not None:
        clauses.append(_eq_leaf(_APPROVAL_COLS["approver_id"], approver_id))
    if request_text:
        clauses.append(
            {"col": _APPROVAL_COLS["request_text"], "c": "CONTAINS",
             "q": request_text, "o": "AND"}
        )
    if created_after_w3c:
        clauses.append(
            _cmp_leaf(_APPROVAL_COLS["created_after"], ">=", created_after_w3c)
        )
    if created_before_w3c:
        clauses.append(
            _cmp_leaf(_APPROVAL_COLS["created_before"], "<=", created_before_w3c)
        )

    return _wrap(clauses)


# --- internals ----------------------------------------------------------

def _text_leaf(text: str) -> dict[str, Any]:
    """Default-cols text search leaf (no `col`). CDESK's default field set
    per resource is broad (Task: name/num/solver/creator/company; Customer:
    name/code/alias/country/city/street/zip/IČO/DIČ/IČ DPH; User: id/name/
    nick/login/email/phone/personal_number)."""
    return {"q": text, "o": "AND"}


def _eq_leaf(col: str, value: Any) -> dict[str, Any]:
    """Equality leaf. value may be a single scalar or a list — CDESK treats
    a list as set membership (IN) for many fields; on the columns in
    VALUE_LIST_COLS the list tools send it one leaf per value instead
    (expand_value_lists)."""
    if value is None:
        raise ValueError(f"refusing to emit equality leaf for {col!r} with None value")
    return {"col": col, "c": "=", "q": value, "o": "AND"}


def _cmp_leaf(col: str, comparator: str, value: Any) -> dict[str, Any]:
    """Comparator leaf (>=, <=, >, <, BETWEEN-less single bound). Used for the
    date windows, where CDESK reads the comparator from `c` and the bound from
    `q`."""
    if value is None:
        raise ValueError(
            f"refusing to emit {comparator!r} leaf for {col!r} with None value"
        )
    return {"col": col, "c": comparator, "q": value, "o": "AND"}


def _null_leaf(col: str, *, is_null: bool) -> dict[str, Any]:
    """ISNULL / ISNOTNULL leaf. CDESK honors the UPPERCASE condition codes
    'ISNULL' / 'ISNOTNULL' — the lowercase 'null'/'!null' forms are silently
    ignored."""
    return {"col": col, "c": "ISNULL" if is_null else "ISNOTNULL", "q": "", "o": "AND"}


def _between_leaf(col: str, frm: Any, to: Any) -> dict[str, Any]:
    """BETWEEN leaf (q=from, q2=to). Date BETWEEN bounds must be strict W3C
    datetimes or the clause is silently dropped — callers normalize first."""
    if frm is None or to is None:
        raise ValueError(f"refusing to emit BETWEEN leaf for {col!r} with a None bound")
    return {"col": col, "c": "BETWEEN", "q": frm, "q2": to, "o": "AND"}


def _w3c_datetime(
    field_name: str, value: str | None, *, end_of_day: bool = False
) -> str | None:
    """Validate an ISO-8601 date/datetime and normalize it to the STRICT W3C
    form CDESK's date filter requires, in UTC: 'YYYY-MM-DDTHH:MM:SS+00:00'.

    CDESK parses sb date values strictly as W3C and silently discards any
    other shape (notably a bare 'YYYY-MM-DD'), so the clause matches ALL rows.
    Normalizing here is what makes a date filter safe to expose at all.

    Semantics: a naive datetime is treated as TENANT-LOCAL wall-clock time
    (CDESK_TIMEZONE, default Europe/Bratislava), because users speak local
    time. A bare date expands to start-of-day, or end-of-day when
    end_of_day=True (so 'before 2026-06-01' includes the whole of June 1st),
    in tenant time. The result is always converted to UTC: CDESK stores UTC
    and reads the clock part of a filter value as UTC, ignoring its offset,
    so '20:00+02:00' would be compared as 20:00 UTC.
    """
    if not value:
        return None
    raw = value.strip()
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as e:
        raise ValueError(
            f"{field_name}={value!r} is not a valid ISO-8601 date or datetime "
            f"(e.g. '2026-05-28' or '2026-05-28T10:00:00+02:00'): {e}"
        ) from e
    # Bare date detection is STRUCTURAL (no time part), not length-based:
    # fromisoformat also accepts the ISO-8601 basic format '20260604' (8
    # chars) and week dates, which a len==10 check would misclassify as
    # datetimes and silently break the documented end-of-day inclusivity.
    is_bare_date = "T" not in raw and " " not in raw and ":" not in raw
    if is_bare_date and end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tenant_timezone())
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


# Public name for tools that compare a bound themselves (e.g. list_deals
# due_before), so their bound means exactly what a filter bound means.
w3c_datetime = _w3c_datetime


def _normalize_sb_date_bound(
    col: str, key: str, value: Any, *, end_of_day: bool
) -> Any:
    """Normalize one date bound of an sb_raw leaf, or raise if it can't be."""
    if not isinstance(value, str) or not value.strip():
        return value  # int/None/empty — not a date literal, leave it alone
    try:
        return _w3c_datetime(f"sb_raw {col}.{key}", value, end_of_day=end_of_day)
    except ValueError as e:
        raise ValueError(
            f"sb_raw leaf on date column {col!r} has an unusable {key} value "
            f"{value!r}: {e} — the backend parses date values as strict W3C "
            f"inside a swallowed try, so a value it cannot parse makes the "
            f"clause match ALL rows and returns UNFILTERED data as filtered. "
            f"Pass an ISO-8601 date ('2026-06-01') or datetime "
            f"('2026-06-01T10:00:00+02:00')."
        ) from e


def _reject_range_on_membership_cols(
    sb: dict[str, Any], range_unsafe_cols: frozenset[str]
) -> None:
    """Reject range comparators on columns that only do set membership.

    The column allowlist answers "is this column read at all"; it cannot answer
    "is this comparator read on it". On these columns a range comparator is
    silently either collapsed to equality or ignored entirely, so the caller
    gets a confidently wrong answer — empty where rows exist, or every row
    where few match. Recurses into nested groups."""
    query = sb.get("query")
    if not isinstance(query, list):
        return
    for leaf in query:
        if not isinstance(leaf, dict):
            continue
        if leaf.get("group") is True or isinstance(leaf.get("query"), list):
            _reject_range_on_membership_cols(leaf, range_unsafe_cols)
            continue
        col = leaf.get("col")
        comparator = leaf.get("c")
        if (
            isinstance(col, str)
            and col in range_unsafe_cols
            and isinstance(comparator, str)
            and comparator.upper() in _SB_RANGE_CMPS
        ):
            raise ValueError(
                f"sb_raw comparator {comparator!r} is not honored on column "
                f"{col!r}: the live endpoint filters this column by set "
                f"MEMBERSHIP only, so a range comparator silently returns the "
                f"wrong rows (either just the exact value, or every row) "
                f"rather than failing. Use '=' with a single value or a list "
                f"of values (e.g. {{'col': {col!r}, 'c': '=', 'q': [1, 2, 3]}}), "
                f"or page the set and compare client-side."
            )


def _require_equality_on_cols(sb: dict[str, Any], cols: frozenset[str]) -> None:
    """Reject any comparator but equality on columns that filter by id only.
    Another comparator returns no rows, which reads as a confident "none".
    Recurses into nested groups."""
    query = sb.get("query")
    if not isinstance(query, list):
        return
    for leaf in query:
        if not isinstance(leaf, dict):
            continue
        if leaf.get("group") is True or isinstance(leaf.get("query"), list):
            _require_equality_on_cols(leaf, cols)
            continue
        col = leaf.get("col")
        comparator = leaf.get("c")
        if (
            isinstance(col, str)
            and col in cols
            and isinstance(comparator, str)
            and comparator.upper() not in _SB_EQUALITY_CMPS
        ):
            raise ValueError(
                f"sb_raw comparator {comparator!r} is not usable on column "
                f"{col!r}: it filters by user id only. Use '=' with a user id "
                f"or a list of ids (e.g. {{'col': {col!r}, 'c': '=', 'q': [1, 2]}}); "
                f"find the id with find_user."
            )


def _validate_sb_v1_date(col: str, value: Any) -> None:
    """Reject an unparseable value on a legacy v1 raw-value date key.

    These keys are not reshaped (a bare date is a legitimate bound there), but
    an unparseable value still silently matches every row, so it must not be
    forwarded."""
    if not isinstance(value, str) or not value.strip():
        return
    try:
        datetime.fromisoformat(value.strip())
    except ValueError as e:
        raise ValueError(
            f"sb_raw leaf on date column {col!r} has an unusable value "
            f"{value!r}: {e} — an unparseable date makes the clause match ALL "
            f"rows and returns UNFILTERED data as filtered. Pass an ISO-8601 "
            f"date ('2026-06-01') or datetime ('2026-06-01T23:59:59')."
        ) from e


def _normalize_sb_date_values(
    sb: dict[str, Any], allowed_cols: frozenset[str] | None
) -> dict[str, Any]:
    """Rewrite every date-column value in an sb_raw tree to strict W3C form.

    Without this, `sb_raw` reaches the very same date columns the typed params
    guard (_W3C_DATE_SB_COLS) with a completely unvalidated value — and a bare
    'YYYY-MM-DD' there does not fail, it silently drops the clause and returns
    the whole table. Normalizing here is what makes those columns safe to
    expose on the raw path at all, exactly as _w3c_datetime does for the typed
    params (`created_after` & co.).

    Leaves on columns outside `allowed_cols` are left untouched so the caller's
    own allowlist check (_validate_sb_cols / _strip_unsupported_cols) still
    produces its more specific "column is not honored" message. Returns a new
    tree; the caller's dict is never mutated. Recurses into nested groups.
    """
    out = dict(sb)
    query = out.get("query")
    if not isinstance(query, list):
        return out
    normalized: list[Any] = []
    for leaf in query:
        if not isinstance(leaf, dict):
            normalized.append(leaf)
            continue
        if leaf.get("group") is True or isinstance(leaf.get("query"), list):
            normalized.append(_normalize_sb_date_values(leaf, allowed_cols))
            continue
        col = leaf.get("col")
        comparator = leaf.get("c")
        if not isinstance(col, str) or (
            allowed_cols is not None and col not in allowed_cols
        ):
            normalized.append(leaf)  # the col check owns this leaf
            continue
        c = comparator.upper() if isinstance(comparator, str) else ""
        is_date_col = col in _W3C_DATE_SB_COLS or col in _V1_RAW_DATE_SB_COLS
        if is_date_col and c in _SB_DATE_REJECTED_CMPS:
            raise ValueError(
                f"sb_raw comparator {comparator!r} is not usable on date column "
                f"{col!r}: the live backend appends no condition for it, so the "
                f"clause would match ALL rows and return UNFILTERED data as "
                f"filtered. Use '>=' with the next instant you want included "
                f"(or BETWEEN for a window)."
            )
        if col in _V1_RAW_DATE_SB_COLS:
            # Validate only — these keys take the value raw and reshaping them
            # would change their meaning (a bare date is a legitimate bound).
            _validate_sb_v1_date(col, leaf.get("q"))
            _validate_sb_v1_date(col, leaf.get("q2"))
            normalized.append(leaf)
            continue
        if col not in _W3C_DATE_SB_COLS or c not in _SB_DATE_VALUE_CMPS:
            normalized.append(leaf)
            continue
        fixed = dict(leaf)
        if c == "BETWEEN":
            # Lower bound opens the day, upper bound closes it.
            fixed["q"] = _normalize_sb_date_bound(
                col, "q", leaf.get("q"), end_of_day=False
            )
            fixed["q2"] = _normalize_sb_date_bound(
                col, "q2", leaf.get("q2"), end_of_day=True
            )
        else:
            fixed["q"] = _normalize_sb_date_bound(
                col, "q", leaf.get("q"),
                end_of_day=c in _SB_DATE_END_OF_DAY_CMPS,
            )
        normalized.append(fixed)
    out["query"] = normalized
    return out


def _wrap(clauses: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not clauses:
        return None
    return {"group": True, "o": "AND", "query": clauses}


# A list of values in one sb leaf (`{"col": c, "c": "=", "q": [v1, v2]}`) is
# matched wrongly on most columns — no rows, only the first value, or a
# different set — while a single value matches correctly. On these columns the
# list is sent as one leaf per value instead: `=` becomes `(c = v1 OR c = v2)`
# and `!=` becomes `(c != v1 AND c != v2)`, inside a group so the leaf keeps
# its place in the filter.
VALUE_LIST_COLS: dict[str, frozenset[str]] = {
    "approvals": frozenset({"id", "name", "assign", "state", "solverOrDemand"}),
    "configuration_items": frozenset({"id", "name", "status_id", "company_id", "type_id"}),
    "customers": frozenset({
        "id", "name", "status", "ico", "dic", "icdph", "email", "phone",
        "company.city", "company.street", "company.zip", "company.country",
    }),
    "deals": frozenset({
        "id", "title", "code", "rc_num", "id_customer", "status", "type", "phase",
    }),
    "projects": frozenset({"id", "title", "code", "company_id", "status_id", "duration"}),
    "kb_articles": frozenset({"category_id"}),
}
# On the deal title a single multi-word `!=` value excludes each of its words
# separately, while a `!=` list is matched one phrase per value. So there a
# `!=` list is left as it is and a single `!=` value is sent as a one-value
# list.
_NE_SENT_AS_LIST: dict[str, frozenset[str]] = {"deals": frozenset({"title"})}
_VALUE_LIST_EQ_CMPS = frozenset({"=", "IS"})
_VALUE_LIST_NE_CMPS = frozenset({"!="})


def expand_value_lists(sb: dict[str, Any] | None, module: str) -> dict[str, Any] | None:
    """Rewrite each `=` / `!=` leaf holding a list of values, on the columns
    VALUE_LIST_COLS names for `module`, into one leaf per value (see there),
    and a single `!=` value on the columns in _NE_SENT_AS_LIST into a
    one-value list. Other leaves are returned unchanged."""
    cols = VALUE_LIST_COLS[module]
    ne_as_list = _NE_SENT_AS_LIST.get(module, frozenset())

    def expand(element: Any) -> Any:
        if not isinstance(element, dict):
            return element
        if element.get("group") is True or isinstance(element.get("query"), list):
            query = element.get("query")
            if not isinstance(query, list):
                return element
            return {**element, "query": [expand(e) for e in query]}
        values = element.get("q")
        comparator = str(element.get("c", "")).upper()
        if element.get("col") in ne_as_list and comparator in _VALUE_LIST_NE_CMPS:
            return element if isinstance(values, list) else {**element, "q": [values]}
        if element.get("col") not in cols or not isinstance(values, list) or not values:
            return element
        if comparator in _VALUE_LIST_EQ_CMPS:
            joiner = "OR"
        elif comparator in _VALUE_LIST_NE_CMPS:
            joiner = "AND"
        else:
            return element
        leaves = [
            {**element, "q": value, "o": "AND" if i == 0 else joiner}
            for i, value in enumerate(values)
        ]
        if len(leaves) == 1:
            return {**leaves[0], "o": element.get("o", "AND")}
        return {"group": True, "o": element.get("o", "AND"), "query": leaves}

    if sb is None:
        return None
    expanded = expand(sb)
    return expanded if isinstance(expanded, dict) else sb


# Date presets (`{"col": "created_at", "c": "THIS_WEEK"}`) are ignored by these
# modules' list endpoints, which then return every record. A preset is sent as
# the BETWEEN window it names instead, computed in tenant time and sent in UTC
# like every other date bound (see _w3c_datetime). Weeks start on Monday.
# FOR_THE_LAST / FOR_THE_NEXT are applied correctly and are left as they are.
_DATE_PRESET_MODULES = frozenset({"approvals", "configuration_items", "customers", "deals"})
_DATE_PRESETS = frozenset({
    "TODAY", "YESTERDAY", "TOMORROW", "THIS_WEEK", "LASTWEEK", "THIS_MONTH",
    "LASTMONTH", "THIS_YEAR", "PREVIOUS_YEAR", "LAST24H", "NEXT24H",
})


def _preset_window(preset: str, now: datetime) -> tuple[datetime, datetime]:
    """[start, end) of a date preset around `now` (tenant-local, aware)."""
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month = day.replace(day=1)
    next_month = (month + timedelta(days=32)).replace(day=1)
    year = day.replace(month=1, day=1)
    week = day - timedelta(days=day.weekday())
    windows = {
        "TODAY": (day, day + timedelta(days=1)),
        "YESTERDAY": (day - timedelta(days=1), day),
        "TOMORROW": (day + timedelta(days=1), day + timedelta(days=2)),
        "THIS_WEEK": (week, week + timedelta(days=7)),
        "LASTWEEK": (week - timedelta(days=7), week),
        "THIS_MONTH": (month, next_month),
        "LASTMONTH": ((month - timedelta(days=1)).replace(day=1), month),
        "THIS_YEAR": (year, year.replace(year=year.year + 1)),
        "PREVIOUS_YEAR": (year.replace(year=year.year - 1), year),
        "LAST24H": (now - timedelta(hours=24), now),
        "NEXT24H": (now, now + timedelta(hours=24)),
    }
    return windows[preset]


def _utc_w3c(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _preset_as_between(leaf: dict[str, Any], now: datetime) -> dict[str, Any]:
    start, end = _preset_window(str(leaf["c"]).upper(), now)
    rest = {k: v for k, v in leaf.items() if k not in ("c", "q", "q2")}
    # BETWEEN includes both bounds.
    return {**rest, "c": "BETWEEN", "q": _utc_w3c(start), "q2": _utc_w3c(end - timedelta(seconds=1))}


# The deal `status` keyword `completed` matches nothing in CDESK; it names the
# COMPLETED action code.
_DEAL_STATUS_COMPLETED = 80

# Comparators refused per module and column, with the working form named for
# the caller. CDESK applies none of these as written: it ignores them or
# answers a different question.
_REFUSED_CMPS: dict[str, dict[str, tuple[frozenset[str], str]]] = {
    "approvals": {
        "solverOrDemand": (
            frozenset({"!=", "ISNOT", ">", ">=", "<", "<=", "BETWEEN", "CONTAINS", "ISNULL", "ISNOTNULL"}),
            "it is a scope preset; use '=' with one of 10-16, or a list of them",
        ),
        "approver_id": (
            frozenset({"ISNULL", "ISNOTNULL"}),
            "CDESK answers it with no approvals at all; filter by an approver id with '='",
        ),
        "processed_state": (
            frozenset({"ISNULL", "ISNOTNULL"}),
            "CDESK answers it with no approvals at all; filter by a state with '='",
        ),
        "processed_approver": (
            frozenset({">", ">=", "<", "<=", "BETWEEN"}),
            "it holds the user who decided; use '=' / '!=' with a user id, ISNULL "
            "(not decided yet) or ISNOTNULL (decided)",
        ),
    },
    "deals": {
        "rc_status": (
            frozenset({">", ">=", "<=", "!=", "ISNOT", "BETWEEN"}),
            "CDESK always applies rc_status as 'strictly less than the value'; "
            "use '<', or the `status` column for an exact status",
        ),
        "for_invoicing": (
            frozenset({"<", "<=", "!=", "ISNOT", "BETWEEN", "CONTAINS"}),
            "use '=' 1 (an amount ready for invoicing) or '=' 0 (none)",
        ),
    },
    "kb_articles": {
        "category_id": (
            frozenset({"CONTAINS", "API_CONTAINS"}),
            "it takes a category id; list_kb_categories lists them",
        ),
    },
}
_SOLVER_OR_DEMAND_PRESETS = frozenset(range(10, 17))


def _check_solver_or_demand(leaf: dict[str, Any]) -> None:
    values = leaf.get("q")
    for value in values if isinstance(values, list) else [values]:
        text = str(value).strip()
        if isinstance(value, bool) or not text.isdigit() or int(text) not in _SOLVER_OR_DEMAND_PRESETS:
            raise ValueError(
                f"solverOrDemand {value!r} is not a scope preset: use 10 (all), "
                f"11 (mine), 12 (team), 13 (mine waiting), 14 (team waiting), "
                f"15 (mine created) or 16 (team created). CDESK ignores any other "
                f"value and returns every approval."
            )


def _check_subtree(value: Any) -> None:
    """CDESK applies `subtree` only when the value holds `*` or `@`; any other
    value is ignored and every deal comes back."""
    if not isinstance(value, str) or ("*" not in value and "@" not in value):
        raise ValueError(
            f"subtree {value!r} needs a '*' or '@' form, for deal codes shaped "
            f"'<PREFIX> <number>/<suffix>': 'ZML*' = codes starting 'ZML ', "
            f"'ZML*2026' = 'ZML <n>/2026', 'ZML@' = codes starting 'ZML ' without "
            f"the '<n>/<n>' part, '@' = codes not starting with a word and a space. "
            f"CDESK ignores any other value and returns every deal; for a plain "
            f"code search use `code`."
        )


def _is_completed_keyword(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "completed"


def _for_invoicing_as_equality(leaf: dict[str, Any]) -> dict[str, Any]:
    """`for_invoicing > 0` is documented but ignored (every deal comes back);
    `= 1` is applied. Only a bound meaning "has an amount" is accepted."""
    comparator = str(leaf.get("c", "")).upper()
    value = leaf.get("q")
    bound = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if bound is None or bound < 0 or (comparator == ">" and bound >= 1) or (comparator == ">=" and bound > 1):
        raise ValueError(
            "for_invoicing only tells whether a deal has an amount ready for "
            "invoicing: use '=' 1 (or '> 0') for those deals, '=' 0 for the others"
        )
    return {**leaf, "c": "=", "q": 1}


def adapt_sb_for_cdesk(
    sb: dict[str, Any] | None, module: str, *, now: datetime | None = None
) -> dict[str, Any] | None:
    """Rewrite an sb filter for the list endpoint of `module` so that CDESK
    applies it as meant: lists of values (expand_value_lists), date presets as
    BETWEEN windows, and the value forms CDESK misreads. A condition CDESK
    cannot apply at all raises ValueError naming the working form."""
    if sb is None:
        return None
    when = (now or datetime.now(timezone.utc)).astimezone(tenant_timezone())
    refused = _REFUSED_CMPS.get(module, {})

    def adapt(element: Any) -> Any:
        if not isinstance(element, dict):
            return element
        query = element.get("query")
        if isinstance(query, list):
            return {**element, "query": [adapt(e) for e in query]}
        if element.get("group") is True:
            return element
        col = element.get("col")
        comparator = str(element.get("c", "")).upper()
        value = element.get("q")
        if isinstance(col, str) and col in refused and comparator in refused[col][0]:
            raise ValueError(
                f"sb_raw comparator {element.get('c')!r} is not usable on column "
                f"{col!r}: {refused[col][1]}."
            )
        if module == "approvals" and col == "solverOrDemand":
            _check_solver_or_demand(element)
        if (module in _DATE_PRESET_MODULES and col in _W3C_DATE_SB_COLS
                and comparator in _DATE_PRESETS):
            return _preset_as_between(element, when)
        if (module == "customers" and col == "name" and comparator in ("=", "IS")
                and isinstance(value, int) and not isinstance(value, bool)):
            # Documented as matching the company id, but answered with no rows.
            return {**element, "col": "id"}
        if module == "deals" and col == "status":
            if _is_completed_keyword(value):
                return {**element, "q": _DEAL_STATUS_COMPLETED}
            if isinstance(value, list):
                return {**element, "q": [
                    _DEAL_STATUS_COMPLETED if _is_completed_keyword(v) else v for v in value
                ]}
        if module == "deals" and col == "for_invoicing" and comparator in (">", ">="):
            return _for_invoicing_as_equality(element)
        if module == "deals" and col == "subtree":
            _check_subtree(value)
        return element

    adapted = adapt(sb)
    if not isinstance(adapted, dict):
        return sb
    return expand_value_lists(adapted, module) if module in VALUE_LIST_COLS else adapted


def encode_sb(sb: dict[str, Any]) -> str:
    """Serialize an `sb` filter object for the `sb` query parameter.

    CDESK base64-decodes the `sb` value and THEN JSON-parses it (per the v3
    `sb` parameter spec). A plain `json.dumps` string fails that base64-decode
    and is silently ignored — the endpoint then returns every record, so the
    filter must be base64-encoded."""
    return base64.b64encode(json.dumps(sb).encode("utf-8")).decode("ascii")


def _parse_sb_raw(
    sb_raw: str | dict[str, Any],
    *,
    allowed_cols: frozenset[str] | None = None,
    dropped_cols_out: list[dict[str, Any]] | None = None,
    allow_text_leaves: bool = True,
    range_unsafe_cols: frozenset[str] | None = None,
) -> dict[str, Any] | None:
    # Accept an already-parsed object too: FastMCP pre-parses JSON-looking
    # string arguments into dicts before validation, so a str-only contract
    # would reject every JSON-object string an MCP client sends.
    if isinstance(sb_raw, dict):
        parsed = sb_raw
    else:
        try:
            parsed = json.loads(sb_raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"sb_raw is not valid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise ValueError(
                f"sb_raw must decode to a JSON object, got {type(parsed).__name__}"
            )
    _validate_sb_shape(parsed)
    # Before ANY return below: a bare date on a strict-W3C date column is a
    # silent match-all server-side, so normalize values on the raw path just
    # like the typed params do.
    parsed = _normalize_sb_date_values(parsed, allowed_cols)
    # A working column with a comparator it doesn't honor is the same class of
    # failure as an unusable value — a confident wrong answer behind a 200.
    if range_unsafe_cols:
        _reject_range_on_membership_cols(parsed, range_unsafe_cols)
    if allowed_cols is None:
        return parsed
    if dropped_cols_out is None:
        # Strict mode (grounding): a non-working column is an error — silently
        # broadening an absence_query would corrupt truth verdicts.
        _validate_sb_cols(parsed, allowed_cols, allow_text_leaves=allow_text_leaves)
        return parsed
    # Strip mode (list tools): remove the non-working clauses, report them via
    # dropped_cols_out, and let the caller report the criteria that were not
    # applied alongside the (broader) result. May return None when every clause
    # was dropped — the list then runs unfiltered.
    #
    # Stripping single leaves is only semantics-preserving in a PURE-AND tree:
    # dropping an OR-joined leaf NARROWS the result (records matching only that
    # leaf are never fetched, and client-side filtering of the narrowed set
    # computes an intersection where a union was asked), and dropping a middle
    # leaf rewires the o-chain into a different boolean expression. In those
    # cases drop the ENTIRE filter instead — running unfiltered and handing the
    # agent the WHOLE original expression is always correct, just less
    # efficient.
    local_dropped: list[dict[str, Any]] = []
    cleaned = _strip_unsupported_cols(
        parsed, allowed_cols, local_dropped, allow_text_leaves=allow_text_leaves
    )
    if not local_dropped:
        return cleaned
    if _sb_tree_is_pure_and(parsed):
        dropped_cols_out.extend(local_dropped)
        return cleaned
    dropped_cols_out.append({
        "entire_filter_dropped": True,
        "original_sb": parsed,
        "unsupported_columns": sorted({
            leaf["col"] for leaf in local_dropped
            if isinstance(leaf.get("col"), str)
        }),
    })
    return None


def _sb_tree_is_pure_and(sb: dict[str, Any]) -> bool:
    """True iff every connector in the tree is AND (or absent — leaves default
    to AND). Only then is removing an individual leaf semantics-preserving;
    any OR anywhere makes partial stripping rewrite the query's meaning."""
    o = sb.get("o")
    if o is not None and str(o).upper() != "AND":
        return False
    query = sb.get("query")
    if not isinstance(query, list):
        return True
    for leaf in query:
        if not isinstance(leaf, dict):
            continue
        if leaf.get("group") is True or isinstance(leaf.get("query"), list):
            if not _sb_tree_is_pure_and(leaf):
                return False
            continue
        lo = leaf.get("o")
        if lo is not None and str(lo).upper() != "AND":
            return False
    return True


def _strip_unsupported_cols(
    sb: dict[str, Any],
    allowed_cols: frozenset[str],
    dropped: list[dict[str, Any]],
    *,
    allow_text_leaves: bool = True,
) -> dict[str, Any] | None:
    """Remove sb leaves on columns the live endpoint doesn't honor.

    Forwarding them would be worse than dropping them: CDESK silently skips
    unknown columns and returns the UNFILTERED set anyway (see
    _validate_sb_cols) — stripping client-side just makes that explicit, so
    the caller can report which criteria were not applied.

    allow_text_leaves=False additionally strips no-col `q` text leaves and the
    primary-text form's top-level `q` (recorded under col "q") — for modules
    whose filter has no `q` branch at all (knowledge base), where even
    the universal text search is silently ignored.

    Dropped leaves are appended to `dropped`. Recurses into nested group
    nodes; a group whose leaves were all dropped is removed entirely. Returns
    the cleaned tree, or None when nothing filtering remains (no leaves and no
    non-empty primary-text `q`)."""
    query = sb.get("query")
    kept: list[Any] = []
    if isinstance(query, list):
        for leaf in query:
            if not isinstance(leaf, dict):
                kept.append(leaf)
                continue
            # Nested group node — strip inside; drop it if it empties.
            if leaf.get("group") is True or isinstance(leaf.get("query"), list):
                sub = _strip_unsupported_cols(
                    leaf, allowed_cols, dropped,
                    allow_text_leaves=allow_text_leaves,
                )
                if sub is not None:
                    kept.append(sub)
                continue
            col = leaf.get("col")
            if col is None:
                if allow_text_leaves:
                    kept.append(leaf)  # default-cols text search works
                else:
                    dropped.append({"col": "q", **leaf})
            elif isinstance(col, str) and col in allowed_cols:
                kept.append(leaf)  # working column
            else:
                dropped.append(leaf)
    out = dict(sb)
    # Primary-text form: a top-level q normally still filters by itself, but
    # when text leaves aren't honored it must be stripped and reported too.
    top_q = out.get("q")
    has_top_q = isinstance(top_q, str) and bool(top_q.strip())
    if has_top_q and not allow_text_leaves:
        dropped.append({"col": "q", "q": top_q, "o": "AND"})
        out.pop("q", None)
        has_top_q = False
    if kept:
        out["query"] = kept
        return out
    out.pop("query", None)
    # Group form with no surviving leaves carries no filter at all.
    if out.get("group") is True:
        return None
    if has_top_q:
        return out
    return None


def _validate_sb_cols(
    sb: dict[str, Any],
    allowed_cols: frozenset[str] | None,
    *,
    allow_text_leaves: bool = True,
) -> None:
    """Reject sb_raw leaves on columns the live endpoint doesn't honor.

    CDESK has NO allowlist: an sb clause on an unknown column is silently
    never read — the query runs UNFILTERED and returns 200, so the caller would
    present unfiltered data as filtered. sb_raw is therefore gated to the
    per-module set of columns that filter. `allowed_cols=None` skips the check.
    The no-col `q` text leaf passes unless allow_text_leaves is False. Recurses
    into nested group nodes."""
    if allowed_cols is None:
        return
    top_q = sb.get("q")
    if not allow_text_leaves and isinstance(top_q, str) and top_q.strip():
        raise ValueError(
            "this module's live endpoint ignores the text-search `q` leaf "
            "entirely (its filter has no q branch) — the clause would be "
            "silently ignored and the result would be UNFILTERED data "
            f"presented as filtered. Working columns: {sorted(allowed_cols)}."
        )
    query = sb.get("query")
    if not isinstance(query, list):
        return
    for leaf in query:
        if not isinstance(leaf, dict):
            continue
        # Nested group node — validate its leaves too.
        if leaf.get("group") is True or isinstance(leaf.get("query"), list):
            _validate_sb_cols(leaf, allowed_cols, allow_text_leaves=allow_text_leaves)
            continue
        col = leaf.get("col")
        if col is None:
            if allow_text_leaves:
                continue  # text leaf — default-cols q search works
            raise ValueError(
                "this module's live endpoint ignores the text-search `q` "
                "leaf entirely (its filter has no q branch) — the clause "
                "would be silently ignored and the result would be "
                "UNFILTERED data presented as filtered. Working columns: "
                f"{sorted(allowed_cols)}."
            )
        if not isinstance(col, str) or col not in allowed_cols:
            raise ValueError(
                f"sb_raw column {col!r} is not honored by the live CDESK "
                f"endpoint — the clause would be silently ignored and the "
                f"result would be UNFILTERED data presented as filtered. "
                f"Working columns: {sorted(allowed_cols)} (plus the no-col "
                f"text leaf {{'q': ...}})."
            )


def _validate_sb_shape(sb: dict[str, Any]) -> None:
    """Lightweight structural check. Verifies the shape looks like one of the
    three documented forms (CDQL shorthand, group, or primary-text) — but
    doesn't validate individual `col` names (that's CDESK's job) or
    comparator values."""
    # CDQL shorthand: {"cdql": "col = value AND status != 3"}. The OpenAPI
    # doc says it is compiled to the structured form server-side, but the
    # server IGNORES it (a cdql clause on a nonexistent id returns ALL records
    # while the same clause in structured form returns 0). A silently-ignored
    # filter would present unfiltered data as filtered, so it is rejected.
    if "cdql" in sb:
        raise ValueError(
            "sb_raw 'cdql' shorthand is documented but the live CDESK server "
            "currently ignores it (the filter silently matches ALL records). "
            "Use the structured tree form instead: {'group': true, 'o': 'AND', "
            "'query': [{'col': ..., 'c': '=', 'q': ...}]}"
        )

    has_group = sb.get("group") is True
    has_query = "query" in sb
    has_text = "q" in sb

    if has_group:
        if not has_query:
            raise ValueError("sb_raw with 'group: true' must also include a 'query' list")
        query = sb["query"]
        if not isinstance(query, list):
            raise ValueError(
                f"sb_raw 'query' must be a list, got {type(query).__name__}"
            )
        for i, item in enumerate(query):
            if not isinstance(item, dict):
                raise ValueError(
                    f"sb_raw query[{i}] must be a dict, got {type(item).__name__}"
                )
        if "o" not in sb:
            raise ValueError("sb_raw with 'group: true' must include 'o' ('AND' or 'OR')")
    elif has_query or has_text:
        # primary form: top-level may carry 'q' and optionally a nested 'query'.
        if has_query and not isinstance(sb["query"], list):
            raise ValueError(
                f"sb_raw 'query' must be a list, got {type(sb['query']).__name__}"
            )
    else:
        # None of the supported forms — warn via error so the caller knows
        # CDESK is unlikely to interpret this.
        raise ValueError(
            "sb_raw must be a group form ({'group': true, 'o': 'AND'|'OR', "
            "'query': [...]}) or a primary-search form ({'q': '...', ...})"
        )


def reject_sb_raw_with_typed(
    sb_raw: str | dict[str, Any] | None, typed: dict[str, Any]
) -> None:
    """Public conflict guard, keyed by the caller's OWN param names.

    The builders check this too, but only after the tool has resolved enum
    NAMES to ids — and resolution can fail first, so an agent that passed both
    `sb_raw` and a misspelled `status_name` would be told about the typo, fix
    it, retry, and only then learn the two can't be combined at all. Tools call
    this before resolving so the real problem surfaces on the first turn."""
    if sb_raw is None:
        return
    _reject_typed_with_sb_raw(sb_raw, typed, exclude=set())


def _reject_typed_with_sb_raw(
    sb_raw: str | dict[str, Any],
    locals_dict: dict[str, Any],
    *,
    exclude: set[str],
    public_names: dict[str, str] | None = None,
) -> None:
    """If sb_raw is provided, no typed kwargs may also be set — silent override
    would mislead an LLM that believes its typed args are being applied.

    `public_names` renames builder-internal params to the ones the TOOL
    actually exposes, for the builders where the two diverge (a tool takes
    `status_name` and resolves it to the `status_id` this builder receives).
    Without it the error orders the agent to remove a parameter it never sent
    and cannot send, costing a wasted self-correction turn."""
    conflicting: list[str] = []
    for name, value in locals_dict.items():
        if name in exclude:
            continue
        # A kwarg counts as "set" unless it's an unset default: None, a default
        # False bool flag, or "". Use identity for False so a meaningful int 0
        # (e.g. catalog_id=0) is NOT treated as unset (0 == False would otherwise
        # let it slip past this guard).
        if value is not None and value is not False and value != "":
            conflicting.append((public_names or {}).get(name, name))
    if conflicting:
        raise ValueError(
            "sb_raw cannot be combined with typed filter kwargs; "
            f"either drop sb_raw or remove: {', '.join(sorted(conflicting))}"
        )


def _validate_iso_date(field_name: str, value: str | None) -> None:
    """Accept ISO-8601 date or datetime, with or without timezone. Reject the
    obvious things (free text, MM/DD/YYYY) before we send them to CDESK."""
    if not value:
        return
    try:
        # fromisoformat accepts date ('2026-05-28'), datetime, and timezones in 3.11+.
        datetime.fromisoformat(value)
    except ValueError as e:
        raise ValueError(
            f"{field_name}={value!r} is not a valid ISO-8601 date or datetime "
            f"(e.g. '2026-05-28' or '2026-05-28T10:00:00+02:00'): {e}"
        ) from e
