# cdesk-mcp

**Talk to your CDESK in plain language.** This connector lets an AI assistant
(such as Claude) work with your real CDESK data — helpdesk tickets, tasks,
customers, users, deals ("zákazky"), fulfillments and the CMDB — so you can ask
for what you need instead of clicking through screens and filters.

## What it's for

- **Ask about what's there.** *"Which tickets are still open for this customer?"*,
  *"How many hours did we log last month?"*, *"Find the user with this e-mail."*
  Answers come from your live CDESK, not from anything the AI remembers.
- **Get work done.** Open a ticket, log a task and assign a solver, add a
  customer, note a fulfillment, change a status, or post into a ticket's
  discussion — visible to the customer or internal to your technicians.
- **In your own words.** Statuses, types and categories can be named the way
  your team names them, in Slovak, Czech or English, with or without diacritics.
- **Answers you can check.** For factual questions the assistant must quote real
  records, and the server verifies every quote against the record before the
  answer reaches you. *"I found nothing about that"* is a proper answer.
- **It only sees what you see.** It works through a normal CDESK login and
  inherits exactly that account's permissions. Nothing is copied into a separate
  database.

## 📘 Manuals

> **The setup and user manual for this connector lives in the CDESK SharePoint.**

Installation, configuration, connecting an AI client, hosting it for your team,
and day-to-day usage are all documented there — not in this repository.

## Technical summary

A thin translation layer between an AI client and the CDESK v3 API — **65 tools
across 10 modules**, no datastore of its own:

```
┌─────────────────┐   MCP (JSON-RPC)   ┌──────────────────┐   HTTPS   ┌──────────────┐
│   AI client     │ ←────────────────→ │    cdesk-mcp     │ ←───────→ │ CDESK tenant │
│ (Claude, …)     │  stdio  or  HTTP   │  (this project)  │           │   /api/v3    │
└─────────────────┘                    └──────────────────┘           └──────────────┘
```

Two run modes: **stdio** for a single local user (no inbound network surface),
and **http** for a shared remote connector where the server is also its own
OAuth 2.1 authorization server, so each user signs in with their own CDESK
account. Python 3.11+, `uv`-managed.

Each tool's own description, visible in the client's tool list, is the
authoritative reference for its parameters and behavior.

> **Release-branch scope (`rc`).** Five modules — knowledge base, calendar,
> approval, project and work order — are not part of this build and remain on
> `develop`. The CNB CMDB export is included, but runs only against the
> production CNB tenant.

## Acknowledgments

Built against the CDESK v3 API. CDESK is © [InovaLogic s.r.o.](https://www.inovalogic.com/).
The MCP protocol is © [Anthropic](https://modelcontextprotocol.io/).
