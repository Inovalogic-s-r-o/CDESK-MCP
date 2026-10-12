"""Which tool parameters a tenant's CDESK form allows: off / optional / required.

CDESK decides per tenant, through module settings, whether a form field is
hidden (off), optional or mandatory. The MCP write tools follow the same rules
as the CDESK web form: an OFF field is refused when given, a REQUIRED field is
refused when missing (create) or cleared (update), an OPTIONAL one may be set
or left out. The same tables feed the `fields` block of the get_*_enums tools,
so what the model is shown and what the write tools enforce cannot drift.

What lives here is CDESK's form logic — which setting controls which field and
what each value means — read from the backend source (FormConfig.php ngIf /
ngRequired and the models' validate()) and the same on every tenant. The
setting VALUES are tenant data and are always read live by the caller.

Not covered, by design: fields without a setting (fixed rules, enforced by the
tools as before), the calling user's rights (not readable through the API —
CDESK drops a value the user may not write, and the write tools warn), and
per-record rules (service area per customer/category, SLA-computed deadlines),
which CDESK checks itself. A setting the tenant does not report leaves its
field unconstrained.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

State = Literal["off", "optional", "required"]


@dataclass(frozen=True)
class FieldRule:
    """One form field: the tool parameter(s) that set it (alternatives — any
    one satisfies a requirement) and how the tenant settings decide its state.
    `decide` returns (state, the setting that decided it), or None when the
    tenant does not report that setting."""

    params: tuple[str, ...]
    decide: Callable[[Mapping[str, Any]], tuple[State, str] | None]

    @property
    def label(self) -> str:
        return " or ".join(self.params)


def _get(settings: Mapping[str, Any], path: str) -> Any:
    node: Any = settings
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, str)):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def levels(path: str, *, hidable: bool = True) -> Callable[[Mapping[str, Any]], tuple[State, str] | None]:
    """The common 0 off / 1 optional / 2 required setting. `hidable=False` for
    fields the form never hides (0 then means optional, not off)."""
    def decide(s: Mapping[str, Any]) -> tuple[State, str] | None:
        level = _int(_get(s, path))
        if level is None:
            return None
        where = f"{path} = {level}"
        if level == 2:
            return "required", where
        if level == 0 and hidable:
            return "off", where
        return "optional", where
    return decide


def switch(path: str) -> Callable[[Mapping[str, Any]], tuple[State, str] | None]:
    """A field shown only while a setting is on (truthy / > 0)."""
    def decide(s: Mapping[str, Any]) -> tuple[State, str] | None:
        value = _get(s, path)
        if value is None:
            return None
        level = _int(value)
        on = bool(value) if level is None else level > 0
        return ("optional" if on else "off"), f"{path} = {value!r}"
    return decide


# --- requests (v3/request/enums `config`; Model/Request.php getFields/validate,
#     Model/Request/RequestColumnsTrait.php, Module/Request/Module.php) ---------

def _request_solver(s: Mapping[str, Any]) -> tuple[State, str] | None:
    group = _get(s, "use_solver_group")
    if group is None:
        return None
    return ("optional" if group else "required"), f"use_solver_group = {group!r}"


def _request_solver_group(s: Mapping[str, Any]) -> tuple[State, str] | None:
    group = _get(s, "use_solver_group")
    if group is None:
        return None
    return ("required" if group else "off"), f"use_solver_group = {group!r}"


def _request_title(s: Mapping[str, Any]) -> tuple[State, str] | None:
    auto = _get(s, "auto_name_enabled")
    if auto is None:
        return None
    return ("optional" if auto else "required"), f"auto_name_enabled = {auto!r}"


def _request_cat_type(s: Mapping[str, Any]) -> tuple[State, str] | None:
    # The category type is also kept while service areas are on — the form
    # needs it to pick an area (RequestColumnsTrait.php cat_type block).
    level = _int(_get(s, "enumCatTypes.status"))
    if level is None:
        return None
    areas = _int(_get(s, "enumCatAreas.status")) or 0
    if level == 2:
        return "required", f"enumCatTypes.status = {level}"
    if level == 0 and areas == 0:
        return "off", f"enumCatTypes.status = 0, enumCatAreas.status = {areas}"
    return "optional", f"enumCatTypes.status = {level}"


def _request_desired_date(s: Mapping[str, Any]) -> tuple[State, str] | None:
    value = _get(s, "desiredDate.enabled")
    if value is None:
        return None
    level = _int(value) or 0
    where = f"desiredDate.enabled = {value!r}"
    if level == 0:
        return "off", where
    if level == 2:
        return "required", where
    return "optional", where  # 1, or 3 = required per request type (CDESK checks)


def _request_due_date(s: Mapping[str, Any]) -> tuple[State, str] | None:
    # Request/FormConfig.php due_date: shown while dueDate.enabled; the form
    # marks it required by dueDate.required, but it prefills the date from the
    # SLA, so a create without one is not refused here — CDESK fills or refuses
    # it. The fixed-due-date checkbox only exists with fixedDueDate.enabled
    # (FE edit/basic.twig), which the backend checks only on create.
    enabled = _get(s, "dueDate.enabled")
    if enabled is None:
        return None
    if not enabled:
        return "off", "dueDate.enabled = False"
    return "optional", "dueDate.enabled = True (CDESK computes it from the SLA when left out)"


REQUEST_FIELDS: tuple[FieldRule, ...] = (
    FieldRule(("solver_id",), _request_solver),
    FieldRule(("solver_group_id",), _request_solver_group),
    FieldRule(("title",), _request_title),
    FieldRule(("code",), levels("code.status")),
    FieldRule(("cat_type_name",), _request_cat_type),
    FieldRule(("cat_area_name",), switch("enumCatAreas.status")),
    FieldRule(("cat_area_2nd_name",), switch("enumCatAreas.status")),
    FieldRule(("desired_date",), _request_desired_date),
    FieldRule(("due_date",), _request_due_date),
    FieldRule(("due_date_fixed",), switch("fixedDueDate.enabled")),
    FieldRule(("close_reason_name",), switch("close_reason.enabled")),
    FieldRule(("solver_method_name",), switch("solver_method.enabled")),
    FieldRule(("root_cause_name",), switch("root_cause.enabled")),
    FieldRule(("solution",), switch("solution.enabled")),
    FieldRule(("urgency_name",), switch("priorityUrgencyImpact.enabled")),
    FieldRule(("impact_name",), switch("priorityUrgencyImpact.enabled")),
    FieldRule(("catalog_id",), switch("catalog.enabled")),
    FieldRule(("place_name",), switch("place.enabled")),
    FieldRule(("branch_id",), switch("branch.enabled")),
)


# --- deals (v3/contract/enums `settings`; Module/Contract/FormConfig.php,
#     Module/Contract/Module.php checkRequiredFields, Model/Contract.php) -------

def _deal_code(s: Mapping[str, Any]) -> tuple[State, str] | None:
    level = _int(_get(s, "rc_code"))
    if level is None:
        return None
    if level == 2 and not _get(s, "codeAutomatic"):
        return "required", "rc_code = 2"
    if level == 2:
        return "optional", "rc_code = 2 but codeAutomatic is on (CDESK generates it)"
    return "optional", f"rc_code = {level}"  # the form never hides the code


def _deal_cat_type(s: Mapping[str, Any]) -> tuple[State, str] | None:
    # The form shows it only while types.status > 0. `types` is written only
    # when the "Typy zákazky" settings page is saved
    # (Contract/Controller/Settings.php postTypeSettings), so a tenant that
    # never saved it sends no `types` at all — and its form hides the field
    # (confirmed in the cmpp UI 2026-10-12: disabled there until switched on).
    level = _int(_get(s, "types.status")) or 0
    where = f"types.status = {level}" if "types" in s else "types not set (never saved — form hides it)"
    if level == 2:
        return "required", where
    return ("optional" if level > 0 else "off"), where


def _deal_phase(s: Mapping[str, Any]) -> tuple[State, str] | None:
    enabled = _get(s, "phases.enabled")
    if enabled is None:
        return None
    return ("required" if enabled else "off"), f"phases.enabled = {enabled!r}"


def _deal_responsible(s: Mapping[str, Any]) -> tuple[State, str] | None:
    level = _int(_get(s, "order.responsiblePerson.status"))
    if level is None:
        return None
    where = f"order.responsiblePerson.status = {level}"
    if level == 2:
        return "required", where
    return ("off" if level == 0 else "optional"), where


DEAL_FIELDS: tuple[FieldRule, ...] = (
    FieldRule(("code",), _deal_code),
    FieldRule(("cat_type_name",), _deal_cat_type),
    FieldRule(("phase_name",), _deal_phase),
    FieldRule(("responsible_person_ids",), _deal_responsible),
    FieldRule(("coordinator_contact_ids",), levels("coordinator_contact_id", hidable=False)),
    FieldRule(("orderer_contact_id",), levels("orderer_contact_id")),
    FieldRule(("est_opening_date",), levels("est_opening_date", hidable=False)),
    FieldRule(("opening_date",), levels("opening_date", hidable=False)),
    FieldRule(("est_due_date",), levels("est_due_date", hidable=False)),
    FieldRule(("due_date",), levels("due_date", hidable=False)),
)


# --- projects (v3/project?include_metadata `config`;
#     Module/Project/FormConfig.php — CDESK's API itself enforces none) -------

PROJECT_FIELDS: tuple[FieldRule, ...] = (
    FieldRule(("type_id", "type_name"), levels("type")),
    FieldRule(("leader_id",), levels("leaderId")),
    FieldRule(("manager_id",), levels("managerId")),
    FieldRule(("team_member_ids",), levels("projectTeam")),
    FieldRule(("watcher_ids",), levels("projectWatchers")),
    FieldRule(("date_start",), levels("dateStart")),
    FieldRule(("date_end",), levels("dateEnd")),
    FieldRule(("unit_count",), switch("unitCount")),
    FieldRule(("percent_done_manual",), levels("percent_done", hidable=False)),
)


# --- work orders (v1 workorder/settings `general`; Module/WorkOrder/FormConfig.php
#     188-201, Module/WorkOrder/Module.php 782-788) -----------------------------

def _wo_assign_due(s: Mapping[str, Any]) -> tuple[State, str] | None:
    enabled = _get(s, "assignDueDate.enabled")
    if enabled is None:
        return None
    if not enabled:
        return "off", f"assignDueDate.enabled = {enabled!r}"
    required = _get(s, "assignDueDate.required") is True
    return ("required" if required else "optional"), f"assignDueDate.required = {required!r}"


def _wo_due(s: Mapping[str, Any]) -> tuple[State, str] | None:
    required = _get(s, "dueDate.required")
    if required is None:
        return None
    return ("required" if required is True else "optional"), f"dueDate.required = {required!r}"


WORKORDER_FIELDS: tuple[FieldRule, ...] = (
    FieldRule(("assign_due_date",), _wo_assign_due),
    FieldRule(("due_date",), _wo_due),
)


# --- tasks, fills, customers: core/base-config ---------------------------------
# These modules expose no settings through v3, and their v1 settings endpoints
# are admin-only; the source is core/base-config, the payload the CDESK web app
# loads for every user. Each tool builds the mapping these rules read (see
# task_settings / fill_settings / customer_settings below).

def _task_type_2nd(s: Mapping[str, Any]) -> tuple[State, str] | None:
    # Task/FormConfig.php: shown while type.status is on (and a type is chosen),
    # required when type2nd.status is 2.
    level = _int(_get(s, "type.status"))
    if level is None:
        return None
    if level == 0:
        return "off", "type.status = 0"
    second = _int(_get(s, "type2nd.status"))
    if second == 2:
        return "required", "type2nd.status = 2"
    return "optional", f"type2nd.status = {second}" if second is not None else "type.status on"


TASK_FIELDS: tuple[FieldRule, ...] = (
    FieldRule(("type_name",), levels("type.status")),
    FieldRule(("type_2nd_name",), _task_type_2nd),
    FieldRule(("watcher_ids",), levels("taskWatchers")),
)


def task_settings(base: Any) -> dict[str, Any] | None:
    """The mapping TASK_FIELDS reads: modules.task.settings (what the task
    form's `editCtrl.config` is) plus `task.taskWatchers` (BASE_CONFIG.task)."""
    if not isinstance(base, Mapping):
        return None
    settings = _get(base, "modules.task.settings")
    if not isinstance(settings, Mapping):
        return None
    return {**settings, "taskWatchers": _get(base, "task.taskWatchers")}


def _fill_description(s: Mapping[str, Any]) -> tuple[State, str] | None:
    simplified = _get(s, "simplifiedEntering")
    if simplified is None:
        return None
    return ("optional" if simplified is True else "required"), f"simplifiedEntering = {simplified!r}"


def _fill_place(s: Mapping[str, Any]) -> tuple[State, str] | None:
    # Fill/FE/template/edit/basic/core.twig: shown only while places are on
    # system-wide AND fill.place.enabled > 0; FormConfig: required at 2.
    level = _int(_get(s, "place.enabled"))
    if level is None:
        return None
    system = _get(s, "systemPlaceEnabled")
    if level == 0 or system is False:
        return "off", f"place.enabled = {level}, places enabled system-wide = {system!r}"
    return ("required" if level == 2 else "optional"), f"place.enabled = {level}"


def _fill_billing_item(s: Mapping[str, Any]) -> tuple[State, str] | None:
    # Fill/FE/js/editController.js showBillingItemSelect / isRequiredBillingItem:
    # hidden while the deal module is off; required for every fill when both
    # billingItemRequired and billingItemRequiredStrict are on (project-contract
    # fills only with billingItemForProjectContract). The non-strict case
    # depends on the record's parent — CDESK decides that one.
    deals_on = _get(s, "contractEnabled")
    if deals_on is None:
        return None
    if deals_on is not True:
        return "off", "the deal module is off"
    strict = _get(s, "billingItemRequired") is True and _get(s, "billingItemRequiredStrict") is True
    if strict and _get(s, "billingItemForProjectContract") is True:
        return "required", "billingItemRequired + billingItemRequiredStrict"
    return "optional", "billingItemRequired / billingItemRequiredStrict"


FILL_FIELDS: tuple[FieldRule, ...] = (
    FieldRule(("description",), _fill_description),
    FieldRule(("place_id", "place_custom"), _fill_place),
    FieldRule(("billing_item_id",), _fill_billing_item),
)


def fill_settings(base: Any) -> dict[str, Any] | None:
    """The mapping FILL_FIELDS reads: modules.fill.settings plus the two
    outside switches the fill form also checks."""
    if not isinstance(base, Mapping):
        return None
    settings = _get(base, "modules.fill.settings")
    if not isinstance(settings, Mapping):
        return None
    return {
        **settings,
        "systemPlaceEnabled": _get(base, "user.systemPlaceEnabled"),
        "contractEnabled": _get(base, "modules.contract.settings.enabled"),
    }


def _customer_type(s: Mapping[str, Any]) -> tuple[State, str] | None:
    # Customer/Module.php:959-967 — on CREATE a company needs a customer-type
    # tag when customer.customer_type is 2. The type tags are the ones
    # core/base-config lists as defaultCompanyTags.
    level = _int(_get(s, "customer_type"))
    if level is None:
        return None
    tags = _get(s, "defaultCompanyTags")
    where = f"customer_type = {level}; customer-type tag ids: {tags}"
    return ("required" if level == 2 else "optional"), where


# Address/contact fields: the customer form only ever REQUIRES them (ngRequired
# at 2, never hidden — Customer/FormConfig.php). Apply on create and update.
CUSTOMER_UPDATE_FIELDS: tuple[FieldRule, ...] = (
    *(
        FieldRule((param,), levels(key, hidable=False))
        for param, key in (
            ("email_general", "email_general"), ("phone_general", "phone_general"),
            ("mobile_general", "mobile_general"), ("web_page", "web_page"),
            ("street", "street"), ("city", "city"), ("postal_code", "zip"),
            ("country", "country"), ("ico", "ico"), ("dic", "dic"), ("icdph", "icdph"),
        )
    ),
)

# The customer type is checked by CDESK on create only; `tag_ids` must then
# carry a customer-type tag — the caller passes only those (customer_type_tags).
CUSTOMER_FIELDS: tuple[FieldRule, ...] = (
    FieldRule(("tag_ids",), _customer_type),
    *CUSTOMER_UPDATE_FIELDS,
)


def customer_settings(base: Any) -> dict[str, Any] | None:
    """The mapping CUSTOMER_FIELDS reads: the `customer` block plus the
    customer-type tag ids."""
    if not isinstance(base, Mapping):
        return None
    settings = _get(base, "customer")
    if not isinstance(settings, Mapping):
        return None
    return {**settings, "defaultCompanyTags": _get(base, "defaultCompanyTags")}


def customer_type_tags(given_tag_ids: Any, settings: Any) -> list[int]:
    """The customer-type tags among the given tag ids — the `tag_ids` value the
    customer-type rule checks (empty = no type chosen)."""
    types = _get(settings, "defaultCompanyTags") if isinstance(settings, Mapping) else None
    if not isinstance(given_tag_ids, list) or not isinstance(types, list):
        return []
    return [t for t in given_tag_ids if t in types]


# --- CMDB configuration items -----------------------------------------------
# The CI form's own config, read live: for a new CI the v1 `cmdb/type/{id}/template`
# the web form loads (TypeController::__getTemplate, no ACL check), for an
# existing one `v3/cmdb/ci/{id}?include_metadata=true` → `config`. Both already
# apply the type's relations (1 off / 2 optional / 3 required) and the tenant's
# switches (branches, places, cost centres, security attributes, BCM), so the
# rules below only read enabled/required. What the config does not carry is the
# category's form template (Module/CMDB/FE/template/<category>/tab/basic.twig):
# a field missing from that template is off whatever the type says. Rules from
# Module/CMDB/FormConfig.php `ci/edit` ngIf / ngRequired.

_CI_CATEGORIES = {
    3: "device", 4: "place", 5: "branch", 9: "service", 10: "general",
    11: "supply", 12: "person", 13: "vehicle", 14: "IT monitoring",
    15: "cost centre", 16: "property area",
}
_ALL_CI = frozenset(_CI_CATEGORIES)
_CI_WITH_BRANCH = frozenset({3, 4, 9, 10, 13, 14, 15, 16})
_CI_WITH_PLACE = frozenset({3, 9, 10, 13, 14, 15, 16})
_CI_WITH_COSTCENTER = frozenset({3, 10, 12, 13, 14})  # also the backend's list
_CI_WITH_OWNER = frozenset({3, 9, 10, 13, 14, 15, 16})
_CI_WITH_COMPANY = _ALL_CI - {11}  # approver/manager/owner also need a company
_CI_SERVICE = frozenset({9})
_CI_WITH_ASSET = frozenset({3, 10, 13, 14})
_CI_WITH_SLA = _ALL_CI - {11, 12}


def _ci_category(s: Mapping[str, Any], allowed: frozenset[int]) -> tuple[State, str] | None:
    """Off when the CI's category form has no such field; None when it has."""
    category = _int(_get(s, "category_id"))
    if category is None or category in allowed:
        return None
    return "off", f"the {_CI_CATEGORIES.get(category, str(category))} CI form has no such field"


def _ci_block(key: str, allowed: frozenset[int]) -> Callable[[Mapping[str, Any]], tuple[State, str] | None]:
    """A config block {enabled, required, readonly}: the type relations and the
    branch / place / cost-centre / property-area blocks. A relation the type
    never set is left out of the config, and the form then hides the field."""
    def decide(s: Mapping[str, Any]) -> tuple[State, str] | None:
        no_field = _ci_category(s, allowed)
        if no_field is not None:
            return no_field
        config = _get(s, "config")
        if not isinstance(config, Mapping):
            return None
        block = config.get(key)
        if not isinstance(block, Mapping):
            return "off", f"{key} is not set on the CI type"
        if block.get("enabled") is not True:
            return "off", f"{key}.enabled = {block.get('enabled')!r}"
        if block.get("readonly") is True:
            return "off", f"{key}.readonly = true (CDESK fills it from IP matching)"
        if block.get("required") is True:
            return "required", f"{key}.required = true"
        return "optional", f"{key}.enabled = true"
    return decide


def _ci_name(s: Mapping[str, Any]) -> tuple[State, str] | None:
    mode = _int(_get(s, "config.name_generate"))
    if mode is None:
        return None
    if mode == 1:
        return "off", "name_generate = 1 (CDESK builds the name from a property; a given name is ignored)"
    return "required", f"name_generate = {mode}"  # 2 = prefix: the backend still requires it


def _ci_company(s: Mapping[str, Any]) -> tuple[State, str] | None:
    return _ci_category(s, _CI_WITH_COMPANY) or (
        ("required", "always on the CI form") if _get(s, "category_id") is not None else None
    )


def _ci_branch(s: Mapping[str, Any]) -> tuple[State, str] | None:
    decided = _ci_block("branch", _CI_WITH_BRANCH)(s)
    if decided is not None and decided[0] == "optional" and _int(_get(s, "category_id")) == 4:
        return "required", "a place always needs its branch"
    return decided


def _ci_place(s: Mapping[str, Any]) -> tuple[State, str] | None:
    # Shown once a branch is chosen (or always, without branches); required
    # only then — CDESK checks the branch-dependent case itself.
    decided = _ci_block("place", _CI_WITH_PLACE)(s)
    if decided is not None and decided[0] == "required" and _get(s, "config.branch.enabled") is True:
        return "optional", "place.required = true, applies once a branch is chosen (CDESK checks it)"
    return decided


def _ci_statuses(s: Mapping[str, Any]) -> tuple[State, str] | None:
    return levels("config.ci_statuses")(s)


def _ci_solver(relation: str, *, group: bool) -> Callable[[Mapping[str, Any]], tuple[State, str] | None]:
    # use_solver_group picks the solver GROUP field; the solver itself is then
    # only offered once a group is chosen (solverSection.twig) — optional.
    def decide(s: Mapping[str, Any]) -> tuple[State, str] | None:
        decided = _ci_block(relation, _ALL_CI)(s)
        if decided is None or decided[0] == "off":
            return decided
        mode = _get(s, "use_solver_group")
        if mode is None:  # mode unreadable: either field may carry the requirement
            return "optional", f"{decided[1]}; solver-group mode unknown"
        use_group = mode is True
        if group and not use_group:
            return "off", "use_solver_group = false (the form offers a solver, not a group)"
        if not group and use_group:
            return "optional", "use_solver_group = true (the group is the field the form requires)"
        return decided
    return decide


def _ci_switch(key: str, allowed: frozenset[int] = _ALL_CI) -> Callable[[Mapping[str, Any]], tuple[State, str] | None]:
    def decide(s: Mapping[str, Any]) -> tuple[State, str] | None:
        no_field = _ci_category(s, allowed)
        if no_field is not None:
            return no_field
        value = _get(s, key)
        if value is None:
            return None
        return ("optional" if value else "off"), f"{key.removeprefix('config.')} = {value!r}"
    return decide


def _ci_only(allowed: frozenset[int]) -> Callable[[Mapping[str, Any]], tuple[State, str] | None]:
    """A field with no setting that only some category forms have."""
    def decide(s: Mapping[str, Any]) -> tuple[State, str] | None:
        return _ci_category(s, allowed)
    return decide


def _ci_sla(s: Mapping[str, Any]) -> tuple[State, str] | None:
    no_field = _ci_category(s, _CI_WITH_SLA)
    if no_field is not None:
        return no_field
    if _get(s, "request_enabled") is False:
        return "off", "the request module is off (the CI form shows no SLA tab)"
    return None


_SECURITY_ATTRIBUTES = (
    "availability", "confidentiality", "criticality", "integrity",
    "security_classification", "sensitivity",
)

CI_FIELDS: tuple[FieldRule, ...] = (
    FieldRule(("name",), _ci_name),
    FieldRule(("company_id",), _ci_company),
    FieldRule(("branch_ci_id",), _ci_branch),
    FieldRule(("place_ci_id",), _ci_place),
    FieldRule(("costcenter_ci_id",), _ci_block("costcenter", _CI_WITH_COSTCENTER)),
    FieldRule(("propertyarea_ci_id",), _ci_block("propertyarea", _CI_WITH_COSTCENTER)),
    FieldRule(("status_id",), _ci_statuses),
    FieldRule(("item_approver_id",), _ci_block("item_approver_relation", _CI_WITH_COMPANY)),
    FieldRule(("item_manager_id",), _ci_block("item_manager_relation", _CI_WITH_COMPANY)),
    FieldRule(("owner_id",), _ci_block("owner_relation", _CI_WITH_OWNER)),
    FieldRule(("service_owner_id",), _ci_block("service_owner_relation", _CI_SERVICE)),
    FieldRule(("service_it_manager_id",), _ci_block("service_it_manager_relation", _CI_SERVICE)),
    FieldRule(("technical_owner_id",), _ci_block("technical_owner_relation", _CI_SERVICE)),
    FieldRule(("key_user_ids",), _ci_block("key_user_relation", _CI_SERVICE)),
    FieldRule(("admin_solver_id",), _ci_solver("admin_solver_relation", group=False)),
    FieldRule(("admin_solver_group_id",), _ci_solver("admin_solver_relation", group=True)),
    FieldRule(("technical_solver_id",), _ci_solver("technical_solver_relation", group=False)),
    FieldRule(("technical_solver_group_id",), _ci_solver("technical_solver_relation", group=True)),
    *(
        FieldRule((f"security_attribute_{a}_id",), _ci_switch(f"config.security_attribute_{a}.enabled"))
        for a in _SECURITY_ATTRIBUTES
    ),
    *(
        FieldRule((p,), _ci_switch("config.bcm"))
        for p in ("bcm_rpo_value", "bcm_rpo_unit_type", "bcm_rto_value", "bcm_rto_unit_type")
    ),
    FieldRule(("inventory_number",), _ci_switch("include_in_asset_management", _CI_WITH_ASSET)),
    FieldRule(("sla_id",), _ci_sla),
    FieldRule(("slo_id",), _ci_only(_CI_SERVICE)),
)


def ci_settings(form: Any, base: Any) -> dict[str, Any] | None:
    """The mapping CI_FIELDS reads, from a CI-form response (the type template
    or an include_metadata CI read: `config` plus `data.category_id` and
    `data.include_in_asset_management`) and core/base-config (solver-group
    mode, request module). None when the form config could not be read."""
    if not isinstance(form, Mapping):
        return None
    config = form.get("config")
    data = form.get("data")
    if not isinstance(config, Mapping) or not isinstance(data, Mapping):
        return None
    request = _get(base, "request") if isinstance(base, Mapping) else None
    asset = data.get("include_in_asset_management")
    return {
        "config": config,
        "category_id": data.get("category_id"),
        "include_in_asset_management": None if asset is None else bool(asset),
        # base-config carries a `request` block only while requests are on
        # (Core/Module.php); without it the CI form runs in solver mode.
        "use_solver_group": (
            None if not isinstance(base, Mapping)
            else isinstance(request, Mapping) and request.get("use_solver_group") is True
        ),
        "request_enabled": None if not isinstance(base, Mapping) else isinstance(request, Mapping),
    }


# --- request catalogs (v3/request-catalog/enums `config` = the request settings,
#     plus core/base-config `request` / `project` / `fill`, as BASE_CONFIG in the
#     form; Module/RequestCatalog/FormConfig.php `edit` and
#     FE/template/edit/basic.twig) ---------------------------------------------
# `base_params.<key>` stands for one entry of the tools' base_params (its
# `value`); the filing fields also have their own parameters. A hyperlink
# template (type "X") shows only the link and its target (basic.twig 8-13).

_HYPERLINK = "X"
CHANGE_MANAGER_CUSTOM = 1  # ChangeManagement::TYPE_CUSTOM: a named change manager


def _catalog_hyperlink(s: Mapping[str, Any]) -> bool:
    return bool(_get(s, "type") == _HYPERLINK)


def _not_on_hyperlink(
    decide: Callable[[Mapping[str, Any]], tuple[State, str] | None],
) -> Callable[[Mapping[str, Any]], tuple[State, str] | None]:
    def wrapped(s: Mapping[str, Any]) -> tuple[State, str] | None:
        if _catalog_hyperlink(s):
            return "off", "type = X (a hyperlink template has only the link and its target)"
        return decide(s)
    return wrapped


def _optional(s: Mapping[str, Any]) -> tuple[State, str] | None:
    return None


def _catalog_hyperlink_field(s: Mapping[str, Any]) -> tuple[State, str] | None:
    if _get(s, "type") is None:
        return None
    if _catalog_hyperlink(s):
        return "required", "type = X (hyperlink template)"
    return "off", f"type = {_get(s, 'type')!r} (only a hyperlink template, type X, has a link)"


def _catalog_code(s: Mapping[str, Any]) -> tuple[State, str] | None:
    # Catalog.php getFields: `code` is a writable field only while request
    # codes are on; the form never requires it on a catalog.
    level = _int(_get(s, "code.status"))
    if level is None:
        return None
    return ("off" if level == 0 else "optional"), f"code.status = {level}"


def _catalog_area_2nd(s: Mapping[str, Any]) -> tuple[State, str] | None:
    first, second = _int(_get(s, "enumCatAreas.status")), _int(_get(s, "enumCatAreas2nd.status"))
    if first is None and second is None:
        return None
    on = (first or 0) > 0 and (second or 0) > 0
    return ("optional" if on else "off"), f"enumCatAreas.status = {first}, enumCatAreas2nd.status = {second}"


def _catalog_hide_name(s: Mapping[str, Any]) -> tuple[State, str] | None:
    use = _get(s, "use_catalog_item_name")
    if use is None:
        return None
    return ("optional" if use else "off"), f"use_catalog_item_name = {bool(use)!r}"


def _catalog_change_manager_type(s: Mapping[str, Any]) -> tuple[State, str] | None:
    on = _get(s, "change_management")
    if on is None:
        return None
    return ("required" if on else "off"), f"change_management = {bool(on)!r}"


def _catalog_change_manager(s: Mapping[str, Any]) -> tuple[State, str] | None:
    on = _get(s, "change_management")
    if on is None:
        return None
    if not on:
        return "off", "change_management = False"
    kind = _int(_get(s, "change_manager_type"))
    if kind == CHANGE_MANAGER_CUSTOM:
        return "required", f"change_manager_type = {kind} (a named change manager)"
    return "off", f"change_manager_type = {kind} (no named change manager)"


def _catalog_approval(s: Mapping[str, Any]) -> tuple[State, str] | None:
    on = _get(s, "catalog.approval_enabled")
    if on is None:
        return None
    return ("optional" if on else "off"), f"catalog.approval_enabled = {on!r}"


def _catalog_fill_hours(s: Mapping[str, Any]) -> tuple[State, str] | None:
    project, fill = _get(s, "project_enabled"), _get(s, "fill_enabled")
    if project is None and fill is None:
        return None
    on = bool(project) and bool(fill)
    return ("optional" if on else "off"), f"project module on = {bool(project)}, fill module on = {bool(fill)}"


def _catalog_matrix(s: Mapping[str, Any]) -> int | None:
    """The priority matrix in force for the catalog's request type (the form's
    priorityByMatrix): >0 when urgency × impact set the priority."""
    if _get(s, "priorityUrgencyImpact.enabled") is not True:
        return 0 if _get(s, "priorityUrgencyImpact") is not None else None
    types = _get(s, "priorityUrgencyImpact.reqTypes")
    cat_type = _get(s, "cat_type_id")
    if not isinstance(types, Mapping) or cat_type is None:
        return 0
    return _int(types.get(str(cat_type))) or 0


def _catalog_priority(s: Mapping[str, Any]) -> tuple[State, str] | None:
    matrix = _catalog_matrix(s)
    if matrix is None:
        return None
    if matrix > 0:
        return "off", "the urgency × impact matrix sets the priority for this request type"
    return "required", "always on the catalog form (no priority matrix for this request type)"


def _catalog_matrix_field(required: bool) -> Callable[[Mapping[str, Any]], tuple[State, str] | None]:
    def decide(s: Mapping[str, Any]) -> tuple[State, str] | None:
        matrix = _catalog_matrix(s)
        if matrix is None:
            return None
        if matrix > 0:
            return ("required" if required else "optional"), "the urgency × impact matrix applies to this request type"
        return "off", "no urgency × impact matrix for this request type"
    return decide


def _catalog_solver_group(s: Mapping[str, Any]) -> tuple[State, str] | None:
    request = _get(s, "base_request")
    if not isinstance(request, Mapping):
        return None
    group = request.get("use_solver_group") is True
    return ("required" if group else "off"), f"use_solver_group = {group!r}"


def _catalog_base_switch(key: str) -> Callable[[Mapping[str, Any]], tuple[State, str] | None]:
    """A BASE_CONFIG.request switch (> 0 shows the field)."""
    def decide(s: Mapping[str, Any]) -> tuple[State, str] | None:
        request = _get(s, "base_request")
        if not isinstance(request, Mapping):
            return None
        value = request.get(key)
        on = (_int(value) or 0) > 0
        return ("optional" if on else "off"), f"request {key} = {value!r}"
    return decide


def _catalog_project_contract(s: Mapping[str, Any]) -> tuple[State, str] | None:
    project = _get(s, "project_enabled")
    if project is None:
        return None
    return ("optional" if project else "off"), f"project module on = {bool(project)}"


CATALOG_FIELDS: tuple[FieldRule, ...] = (
    FieldRule(("code",), _catalog_code),
    FieldRule(("hide_name",), _catalog_hide_name),
    FieldRule(("cat_type", "base_params.cat_type_id"), _not_on_hyperlink(switch("enumCatTypes.status"))),
    FieldRule(("cat_area", "base_params.cat_area_id"), _not_on_hyperlink(switch("enumCatAreas.status"))),
    FieldRule(("cat_area_2nd", "base_params.cat_area_2nd_id"), _not_on_hyperlink(_catalog_area_2nd)),
    FieldRule(("change_manager_type",), _catalog_change_manager_type),
    FieldRule(("change_manager_id",), _catalog_change_manager),
    FieldRule(("approval_rule_id",), _catalog_approval),
    FieldRule(("simple_approval_rules",), _catalog_approval),
    FieldRule(("approval_rules",), _catalog_approval),
    FieldRule(("include_fill_hours",), _catalog_fill_hours),
    FieldRule(("base_params.priority",), _not_on_hyperlink(_catalog_priority)),
    FieldRule(("base_params.urgency_id",), _not_on_hyperlink(_catalog_matrix_field(required=False))),
    FieldRule(("base_params.impact_id",), _not_on_hyperlink(_catalog_matrix_field(required=True))),
    FieldRule(("base_params.solver_group_id",), _not_on_hyperlink(_catalog_solver_group)),
    FieldRule(("base_params.id_solver",), _not_on_hyperlink(_optional)),
    FieldRule(("base_params.reaction_missed_id_solver",), _not_on_hyperlink(switch("reactionMissed.enabled"))),
    FieldRule(("base_params.other_solvers",), _not_on_hyperlink(levels("otherSolvers.status"))),
    FieldRule(("base_params.request_watchers",), _not_on_hyperlink(switch("request_watchers_ids"))),
    FieldRule(("base_params.responsible_person",), _not_on_hyperlink(levels("responsiblePerson.status"))),
    FieldRule(("base_params.cat_branch_id",), _not_on_hyperlink(_catalog_base_switch("branch"))),
    FieldRule(("base_params.cat_place_id",), _not_on_hyperlink(_catalog_base_switch("place"))),
    FieldRule(("base_params.cat_close_reason_id",), _not_on_hyperlink(_catalog_base_switch("close_reason"))),
    FieldRule(("base_params.cat_solver_method_id",), _not_on_hyperlink(_catalog_base_switch("solver_method"))),
    FieldRule(("base_params.cat_root_cause_id",), _not_on_hyperlink(_catalog_base_switch("root_cause"))),
    FieldRule(("base_params.project_contract_id",), _not_on_hyperlink(_catalog_project_contract)),
    FieldRule(("base_params.hyperlink",), _catalog_hyperlink_field),
    FieldRule(("base_params.hyperlink_target",), _catalog_hyperlink_field),
)


_CATALOG_RECORD_KEYS = (
    "type", "change_management", "change_manager_type", "use_catalog_item_name", "cat_type_id",
)


def catalog_settings(config: Any, base: Any, record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The mapping CATALOG_FIELDS reads: the request settings from the catalog
    enums `config`, base-config's `request` block and module switches, and the
    catalog's own values the form keys on (`record`: type, change management,
    use_catalog_item_name, the request type — this write's or the stored one;
    None = unknown)."""
    if not isinstance(config, Mapping) or not config:
        return None
    # The catalog's own values always win, None (unknown) included: the request
    # config has a `type` of its own, which is not the catalog's type.
    out: dict[str, Any] = {**config, **{k: record.get(k) for k in _CATALOG_RECORD_KEYS}}
    if isinstance(base, Mapping):
        request = base.get("request")
        # base-config has a `request` block only while requests are on.
        out["base_request"] = request if isinstance(request, Mapping) else {}
        out["project_enabled"] = (_int(_get(base, "project.enabled")) or 0) > 0
        out["fill_enabled"] = (_int(_get(base, "fill.enabled")) or 0) > 0
    return out


def base_params_given(base_params: Any) -> dict[str, Any]:
    """The tools' `base_params` as rule parameters: `base_params.<key>` → the
    entry's value (the entry itself when it sets only rules, so an off field
    is still refused)."""
    if not isinstance(base_params, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, entry in base_params.items():
        value = entry.get("value", entry) if isinstance(entry, Mapping) else entry
        out[f"base_params.{key}"] = value
    return out


# --- evaluation ---------------------------------------------------------------

def _blank(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def describe(rules: Iterable[FieldRule], settings: Any) -> dict[str, dict[str, str]]:
    """{param label: {state, setting}} for every field the tenant reports — the
    `fields` block of a get_*_enums tool."""
    if not isinstance(settings, Mapping) or not settings:
        return {}
    out: dict[str, dict[str, str]] = {}
    for rule in rules:
        decided = rule.decide(settings)
        if decided is not None:
            out[rule.label] = {"state": decided[0], "setting": decided[1]}
    return out


def violations(
    rules: Iterable[FieldRule],
    settings: Any,
    given: Mapping[str, Any],
    *,
    update: bool = False,
    cleared: Iterable[str] = (),
) -> list[str]:
    """Why a write breaks the tenant's form rules (empty when it does not).

    `given` maps tool parameter → value as passed (None = not given). On create
    a required field must be given; on update it may be left out (the stored
    value stays) but not cleared — an empty value or a name in `cleared`. An
    off field may never be set. Unreadable settings constrain nothing."""
    # Empty counts as unreadable: a tenant always sends its settings object, so
    # {} means none arrived — and a rule reading a missing key as "off" (the
    # deal type) must not fire on that.
    if not isinstance(settings, Mapping) or not settings:
        return []
    cleared_set = set(cleared)
    problems: list[str] = []
    for rule in rules:
        decided = rule.decide(settings)
        if decided is None:
            continue
        state, where = decided
        values = [given.get(p) for p in rule.params]
        is_set = any(not _blank(v) for v in values)
        is_cleared = any(p in cleared_set for p in rule.params) or any(
            v is not None and _blank(v) for v in values
        )
        if state == "off" and is_set:
            problems.append(f"{rule.label} is off on this tenant ({where}) — leave it out")
        elif state == "required" and not update and not is_set:
            problems.append(f"{rule.label} is required on this tenant ({where})")
        elif state == "required" and update and is_cleared and not is_set:
            problems.append(f"{rule.label} is required on this tenant ({where}) and cannot be cleared")
    return problems


def refuse_if_violated(problems: list[str]) -> None:
    """Raise the tool-input error listing every rule a write breaks."""
    if problems:
        raise ValueError(
            "the tenant's form rules (read live from its settings) do not allow "
            "this call:\n" + "\n".join(f"  - {p}" for p in problems)
        )
