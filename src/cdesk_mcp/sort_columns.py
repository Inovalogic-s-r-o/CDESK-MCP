"""Which columns each CDESK v3 list can sort by (UAT B9).

CDESK honors `sort` only for the columns its model allow-lists
(`getOrderColsAllowed`, the same on every tenant); any other value is dropped
without a word and the page comes back in default order
(`Wrapper/BaseModel.php` ordering). The direction is a `.asc` / `.desc`
suffix (`Middleware/before_requestJsonObject.php` splits the value on '.',
default ascending) — so a table-qualified name such as `creator.name` cannot
be sent and is left out. Verified live on cmpp 2026-10-12 (tasks valid_from
asc/desc, requests priority asc/desc, an unknown column ignored). The v3
spec documents only "Sort column" — see docs/bugs/missing_endpoints.md.

Generated from the backend source (apiportal/inovalogic) on 2026-10-12.
"""

from __future__ import annotations

import re

SORT_COLUMNS: dict[str, frozenset[str]] = {
    # Model/Approval.php
    "approvals": frozenset({
        "approval_created_by_name", "assign_id", "created_at", "id", "name", "ruleName",
        "state", "state_changed",
    }),
    # Model/CMDB/CI.php
    "configuration_items": frozenset({
        "admin_solver_name", "asset_management_physical_status",
        "asset_management_status", "assetmanagement_assigned_user_name",
        "assetmanagement_responsible_user_name", "assignment_date", "branch_name",
        "ci_type_name", "company_name", "cost", "costcenter_name",
        "costcenter_responsible_person_name", "created_at", "creator_name",
        "description", "disposal_date", "id", "inv_code", "inventory_number",
        "invoice_id", "item_approver_name", "item_manager_name", "liquidation_date",
        "maingroup_name", "name", "owner_name", "place_name", "propertyarea_name",
        "propertyarea_responsible_person_name", "purchase_date",
        "service_it_manager_name", "service_owner_name", "sla", "slo", "status_id",
        "supplier", "supplier_id", "technical_owner_name", "technical_solver_name",
        "type_mark", "updated_at", "updated_by_name",
    }),
    # Model/Customer.php
    "customers": frozenset({
        "accounting_language", "alias", "city", "company_ext_curr", "company_note",
        "company_num", "country", "customer", "dic", "email_general", "email_invoice",
        "fax", "hour_rate", "icdph", "ico", "id", "inv_city", "inv_country", "inv_name",
        "inv_street", "inv_zip", "legalName", "membership", "membership_validity_date",
        "mobile_general", "name", "phone_general", "ship_city", "ship_country",
        "ship_name", "ship_street", "ship_zip", "sla", "street", "web_page", "zip",
    }),
    # Model/Contract.php
    "deals": frozenset({
        "billing_hourly_rate", "cat_type_id", "customer_name", "for_invoicing",
        "for_invoicing_sum", "for_invoicing_sum_w_vat", "id", "insert_date",
        "last_invoice_issue_date", "last_invoiced_sum", "last_invoiced_sum_w_vat",
        "orderer_contact_name", "param_request", "rc_code_sort", "rc_num",
        "rc_status_id", "rc_title", "rc_type", "req_count", "update_date",
    }),
    # Model/Fill.php
    "fills": frozenset({
        "approved", "assign_id", "billing_item_hourly_price", "billing_item_title",
        "company_name", "contract_cat_type_id", "contract_code", "contract_num",
        "contract_orderer", "contract_title", "description", "duration",
        "duration_hours", "duration_out_of_workday", "from_to", "id", "internal",
        "inv_fill_num", "invoiceDate", "invoice_number", "invoiced", "invoiced_at",
        "name", "num", "other_time", "param_request", "parking_fee", "place_name",
        "quantity", "real_time", "reqAcceptedDate", "reqCloseDate", "req_accepted_rate",
        "req_creator", "request_cat_area_id", "request_cat_type_id",
        "request_description", "request_id", "request_num", "request_solver_name",
        "request_status", "request_title", "solver_name", "task_description",
        "task_name", "task_num", "total_duration", "travel_distance", "travel_fixed",
        "travel_period", "type", "used_material", "valid_from", "valid_to",
        "work_order_id", "workorder_description",
    }),
    # (no allow-list: the primary key only)
    "kb_articles": frozenset({
        "id",
    }),
    # Model/ProjectContract.php
    "projects": frozenset({
        "agreed_external_costs", "agreed_gross_margin", "agreed_revenue", "code",
        "company_id", "created_at", "created_by", "date_end", "date_start", "duration",
        "expectation_id", "gba", "gba_ag", "gba_bg", "id", "leader_id", "manager_id",
        "phase_id", "plan_gross_margin", "plan_lien", "plan_revenue",
        "plan_revenue_converted", "planned_hours", "priority_id", "projectDateEnd",
        "projectDateStart", "projectDuration", "project_id", "real_external_costs",
        "real_gross_margin", "real_lien", "real_revenue", "real_revenue_converted",
        "status_id", "title", "type_id", "updated_at", "updated_by",
    }),
    # Model/Request/Catalog.php allow-lists only `id`, yet the v3 catalog list
    # ignores sort altogether (live 2026-10-12: sort=id came back id-descending,
    # the default) — nothing can be sorted.
    "request_catalogs": frozenset(),
    # Model/Request.php
    "requests": frozenset({
        "accepted_date", "assign_date", "assign_due_date", "calendar_date_start",
        "cat_area_id", "cat_type_id", "close_as_date", "close_date", "close_reason_id",
        "company_alias", "creator", "customer_name", "customer_reaction_date",
        "date_from", "date_to", "due_date", "due_date_clock", "entered_by_merged",
        "enteredby_name", "fill_duration_summary", "id", "id_rc", "id_req", "idatetime",
        "invoice_date", "last_reaction_date", "missed_due_date_time",
        "missed_reaction_due_date_time", "msgProcessing_toEmail", "num_alias",
        "param_request", "priority", "project_contract_code", "project_contract_title",
        "rc_code", "rc_num", "rc_title", "reaction_date", "reaction_datetime",
        "reaction_due_date", "reaction_due_date_clock", "req_accepted_date",
        "req_close_as_date", "req_close_date", "req_due_date", "req_in_progress_from",
        "req_num", "req_status", "req_title", "req_year", "req_year_num",
        "request_weight", "short_company_name", "solver_name", "solver_reaction_date",
        "source", "status", "title", "udatetime", "updated_name", "voip_call_count",
        "voip_duration_sum",
    }),
    # Model/Task.php
    "tasks": frozenset({
        "code", "connection", "created_at", "created_by", "customerName", "id",
        "missed_valid_to_time", "name", "num", "placeName", "solver", "status",
        "type_id", "valid_from", "valid_to",
    }),
    # Model/User.php
    "users": frozenset({
        "email", "enable_password_change", "fullname", "id", "id_parent",
        "job_title_id", "last_login_date", "last_login_type", "license_selected",
        "name", "nick", "param_request", "position", "status", "user", "user_int_ext",
    }),
    # Model/WorkOrder.php
    "work_orders": frozenset({
        "assign_due_date", "changer_name", "created_at", "creator_name", "due_date",
        "enteredby_name", "id", "missed_due_date_time", "name", "number", "request",
        "request_id", "solver_group_name", "solver_name", "status_id", "updated_at",
    }),
}


def resolve_sort(module: str, sort: str) -> str:
    """The `sort` value to send for `module`: "column" (ascending) or
    "column.desc". Accepts "column", "column.asc", "column.desc", "-column" and
    "column desc" / "column asc". Raises ValueError for a column the module
    cannot sort by, listing the ones it can."""
    allowed = SORT_COLUMNS[module]
    if not allowed:
        raise ValueError(
            f"CDESK does not sort {module.replace('_', ' ')} (sort is ignored on this "
            f"list); leave sort out and order the items yourself"
        )
    text = sort.strip()
    descending = False
    if text.startswith("-"):
        text, descending = text[1:].strip(), True
    else:
        for sep in (".", " "):
            head, _, tail = text.rpartition(sep)
            if head and tail.lower() in ("asc", "desc"):
                text, descending = head.strip(), tail.lower() == "desc"
                break
    # CDESK appends the tenant's custom-field columns to some lists at runtime
    # (CustomField::appendCustomFieldsToOrderCols) — cfield_<id> passes through.
    if text not in allowed and not re.fullmatch(r"cfield_\d+", text):
        raise ValueError(
            f"sort column {text!r} is not one CDESK can sort {module.replace('_', ' ')} by "
            f"(it would be ignored silently); valid: {', '.join(sorted(allowed))}. "
            f"Add .desc for descending, e.g. {sorted(allowed)[0]}.desc"
        )
    return f"{text}.desc" if descending else text



# A few columns worth naming in a description where the full list is long.
_EXAMPLES: dict[str, tuple[str, ...]] = {
    "requests": ("title", "status", "priority", "due_date", "idatetime", "close_date",
                 "solver_name", "customer_name"),
    "fills": ("valid_from", "duration", "solver_name", "company_name", "request_id"),
    "customers": ("name", "city", "ico", "company_num"),
    "projects": ("title", "date_start", "date_end", "status_id", "created_at"),
    "configuration_items": ("name", "status_id", "company_name", "ci_type_name",
                            "created_at", "updated_at"),
    # the column used in the example when the whole list is shown
    "tasks": ("valid_from",), "users": ("name",), "work_orders": ("due_date",),
    "approvals": ("created_at",), "deals": ("rc_title",), "kb_articles": ("id",),
}


def sort_note(module: str) -> str:
    """The `sort` paragraph of a list tool's description, from SORT_COLUMNS."""
    allowed = sorted(SORT_COLUMNS[module])
    if not allowed:
        return ("sort: CDESK ignores sorting on this list — leave it out and order "
                "the items yourself; a sort value is refused.")
    if len(allowed) <= 20:
        cols = "Columns: " + ", ".join(allowed) + "."
    else:
        cols = ("Columns include " + ", ".join(_EXAMPLES.get(module, tuple(allowed[:6])))
                + "; a refused value comes back with the full list.")
    example = _EXAMPLES.get(module, (allowed[0],))[0]
    return (f'sort: ONE column, ascending ("{example}"), or descending with a .desc '
            f'suffix ("{example}.desc"; "-{example}" is read the same). {cols} '
            "A column CDESK cannot sort by is refused (CDESK itself would ignore it "
            "without a word). Without sort, CDESK's default order applies.")
