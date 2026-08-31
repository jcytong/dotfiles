---
name: hubspot
description: Read and write HubSpot CRM data through the HubSpot CLI (`hs`). Use when the user asks about HubSpot contacts, companies, lists, properties, segments, imports, or a portal/account. Covers searching and paginating CRM records, inspecting properties, and running guarded imports that cannot silently overwrite contacts. Triggers on "hubspot", "hs api", "CRM", "contact record", "portal", "import into hubspot", "add to list", "lp_source_list", "personal access key".
---

# HubSpot

Work the user's HubSpot portal through the official CLI. Authentication, account
selection and config are the CLI's job, so this skill never handles a token.

Two layers:

- **`hs`** is HubSpot's own CLI. `hs api <endpoint>` is an authenticated passthrough
  to any REST endpoint that accepts a personal access key, which is how all CRM work
  gets done. The rest of `hs` is developer tooling (projects, HubDB, secrets).
- **`hubspot`** is this skill's wrapper. It shells out to `hs api` and adds the three
  things bare `hs api` lacks: pagination, a pre-import hazard scan, and imports that
  verify themselves afterwards.

This skill is shareable: it holds no PII and hardcodes no portal.

## Setup

**Preflight (run first, every session):**

```bash
test -f ~/.config/hubspot/.env && command -v hs >/dev/null && test -x ~/.local/bin/hubspot && echo OK || echo MISSING
```

Anything other than `OK` means stop and fix the missing piece below before any read or
write.

Both HubSpot credentials live in one file, `~/.config/hubspot/.env` (mode 0600), because
they do different jobs and you will need both:

```bash
HUBSPOT_ACCOUNT_ID=1234567
HUBSPOT_PERSONAL_ACCESS_KEY=...   # reads  — consumed by `hs --use-env`
HUBSPOT_SERVICE_KEY=...           # writes — Bearer token, direct REST
```

The `hubspot` wrapper sources this file, and the Python script appends `--use-env` when
`HUBSPOT_PERSONAL_ACCESS_KEY` is present. If the file is absent the flag is omitted and
`hs` falls back to its own `~/.hscli/config.yml`, so an unmigrated machine still works.

### 1. Install the CLI

```bash
npm install -g @hubspot/cli    # needs Node 20+; verify with `hs --version` (8.12.0+)
```

### 2. Authenticate with a personal access key

```bash
hs account auth --pak "$(pbpaste)"    # after copying the key from the PAK page
hs account auth                       # or interactive, pastes at a prompt
```

Use `hs account auth`, not `hs auth`. Under the global config `hs auth` errors out with
"not compatible with this command".

**A PAK cannot write CRM data.** Its permission picker offers only 16 checkboxes, and
the CRM-relevant ones are `CRM Objects` (read only) and `Custom Objects`. There is no
Lists permission and no write for standard objects, so `crm.lists.read`,
`crm.lists.write` and `crm.objects.contacts.write` are **not grantable to a PAK at
all**. Regenerating the key does not help; the picker is the same every time.

So the auth model splits:

| Operation | Credential |
|---|---|
| Any CRM read, schema read, CMS and developer tooling | PAK, via `hs api --use-env` |
| Writing objects, lists, or schemas | **Service key** |

For writes, create a **Service key** (Settings → Integrations → **Service Keys** → Create
a service key; needs Super Admin or Developer tools access), grant only the scopes the
job needs — e.g. `crm.objects.deals.read`/`.write`,
`crm.objects.contacts.read`/`.write`, `crm.schemas.contacts.write` — and call the REST
API with `Authorization: Bearer <key>` directly. `hs api` cannot use a service key.

Service keys entered public beta in Feb 2026 and are HubSpot's recommended credential for
data-only integrations; **UI-created private apps are now legacy** for this purpose. Two
practical wins over a private app: the key is account-level rather than tied to a user
(so it survives people changing roles), and it can be rotated without rebuilding the
integration. They cannot do webhook subscriptions or UI extensions — those still require
a project-based app.

When a write 403s, HubSpot names the exact scopes it wanted in
`errors[].context.requiredGranularScopes`. Read that list rather than guessing. To probe
whether a credential can write at all *without touching data*, aim a `PATCH` at an id
that cannot exist — `403` means the scope is missing, `404` means auth is fine:

```bash
hs api /crm/v3/objects/deals/999999999999 -X PATCH --data '{"properties":{"dealstage":"x"}}'
```

A 403 from `hs api` on a write is not a bad key, it is the PAK doing what it is designed
to do. Do not try to fix it by regenerating.

Scopes are fixed when a key is generated, and deactivating is the only way to change
them. The old key keeps working for up to six hours after deactivation, but a
replacement is issuable immediately.

### 3. Config must be global, not per-directory

`hs init` writes `hubspot.config.yml` into the **current working directory**. That is
the deprecated location and it makes every command fail from anywhere else. Migrate:

```bash
printf 'y\n' | hs config migrate -f     # -f alone does not skip the confirm prompt
chmod 600 ~/.hscli/config.yml           # it contains the key
```

Then delete the leftover `archived.hubspot.config.yml`, which still holds a copy of the
key. Verify portability by running `hs account list` from `~`.

### 4. Shell wrapper

```bash
mkdir -p ~/.local/bin
cat > ~/.local/bin/hubspot <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
command -v hs >/dev/null || { echo "hubspot: HubSpot CLI not found — npm install -g @hubspot/cli" >&2; exit 1; }
ENV_FILE="$HOME/.config/hubspot/.env"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; fi
SCRIPT="$HOME/.agents/skills/hubspot/scripts/hubspot.py"
[ "$#" -eq 0 ] && exec python3 "$SCRIPT" --help
case "$1" in
  hs)   shift; exec hs "$@" ;;   # escape hatch: `hubspot hs project upload`
  exec) shift; exec "$@" ;;      # escape hatch: `hubspot exec env | grep HUBSPOT`
  *)    exec python3 "$SCRIPT" "$@" ;;
esac
EOF
chmod +x ~/.local/bin/hubspot
```

Sourcing the env file is what makes the consolidated credential store work; without it
`hs` falls back to `~/.hscli/config.yml` and the service key is never in scope.

`~/.local/bin` must be on `PATH`. Verify with `hubspot whoami`.

## Guardrails (IMPORTANT)

Two failure modes have already corrupted data in a live portal. Both are silent: the
records look correct and only the totals betray the problem.

**1. Upsert resolves against the full email graph.** `batch/upsert` with
`idProperty: email` also matches `hs_additional_emails`, so importing someone who
happens to own an address listed as a *secondary* email on an unrelated contact
overwrites that contact's name, email and company.

Always scan before writing. `hubspot import` does this automatically and refuses to
proceed on a hit; `hubspot check-emails` runs it standalone. Treat a secondary-email
match as a weak signal and require surname corroboration before calling it the same
person.

**2. Batch write results do not come back in input order.** Zipping
`response["results"]` with the input batch produces a scrambled row to ID map. This is
how 50 wrong people once landed in a segment.

Never derive membership from batch results. Write an identifying property during the
import, then re-query by that property for the authoritative ID set. `hubspot import`
and `hubspot list-add` both do exactly this, which is why `--tag-property` is required.

**Default to `--mode create`.** It skips emails that already exist rather than touching
them. Reach for `--mode upsert` deliberately; it demands
`--i-understand-upsert-risk`.

## The tool

All subcommands route through `hs api` and therefore through the PAK. The read-only ones
(`whoami`, `search`, `props`, `api` GET, `check-emails`) work as documented. **`import`
and `list-add` perform writes and will 403 under a PAK**, including the `--dry-run`
paths' final step; their scan and re-query logic is still correct and useful. To
actually write, drive the same endpoints with the service key
(`Authorization: Bearer $HUBSPOT_SERVICE_KEY`), or run them against a portal where the
CLI is authed with OAuth carrying write scopes.

### `whoami` — confirm which portal you are pointed at

```bash
hubspot whoami
```

Run before any write. Every subcommand takes `-a <account>` to target a non-default
portal from `hs account list`.

### `search` — CRM search, fully paginated

```bash
hubspot search contacts \
  --filter lp_qualify_verdict=qualified \
  --filter 'createdate:GT=2026-01-01' \
  --props email,firstname,lastname,company \
  [--query "free text"] [--limit 500]
```

- Filters are repeatable and ANDed. Forms: `prop=value` (EQ), `prop:OP=value`
  (`GT`, `LT`, `GTE`, `LTE`, `NEQ`, `CONTAINS_TOKEN`, `IN`), `prop:HAS_PROPERTY`.
- Pages automatically. HubSpot caps `/search` at 10,000 results total and warns on
  stderr when you hit it; narrow the filters rather than trusting a truncated set.
- Output: `{"count": N, "results": [...]}`.

### `props` — inspect the schema

```bash
hubspot props contacts [--grep lp_] [--json]
```

Use this before filtering or importing. Filtering on a property that does not exist
returns an empty result set rather than an error, which reads as "no matches".

### `api` — raw passthrough

```bash
hubspot api /crm/v3/objects/companies --all --limit 500
hubspot api /crm/v3/objects/contacts/215547405958
hubspot api /crm/v3/lists/253 
```

`--all` follows `paging.next.after` on GET collections. For writes use `-X POST` with
`--data '{...}'`, or go straight to `hs api`, which this wraps.

### `check-emails` — pre-import hazard scan

```bash
hubspot check-emails prospects.jsonl [--email-property email]
```

Resolves every input email the same way an upsert would and reports three buckets:
`existing_primary`, `secondary_matches` (the dangerous one) and `unmatched`.

### `import` — guarded batch import

```bash
hubspot import prospects.jsonl \
  --tag-property lp_source_list --tag-value anchor-lp-2026-08 \
  [--mode create|upsert] [--dry-run] [--out report.json]
```

Accepts JSONL (one flat object of properties per line) or CSV (header row of property
names). Sequence: scan emails, abort on secondary matches, write in batches of 100,
then re-query by tag and diff `createdate` against the run start.

The report separates `created_this_run` from `pre_existing_touched`. **Any entry in
`pre_existing_touched` is a contact that already existed and was modified, not
created.** Check those before treating the import as clean. A mismatch between input
count and `tagged_total` prints a warning; investigate it before building anything
downstream.

Start with `--dry-run`.

### `list-add` — add a tagged cohort to a list

```bash
hubspot list-add 253 --tag-property lp_source_list --tag-value anchor-lp-2026-08 [--dry-run]
```

Re-queries the tag for the authoritative ID set, then adds in batches of 100.

## Workflow patterns

- **"How many contacts match X?"** → `props contacts --grep X` to confirm the property
  exists, then `search`.
- **"Import this list of prospects"** → `check-emails` → `import --dry-run` → `import`
  → read `pre_existing_touched` → `list-add`.
- **"Build a segment from the import"** → `list-add` with the same tag. Never from
  batch output.
- **"What happened in that import?"** → `search contacts --filter <tag>=<value>
  --props email,createdate` and histogram `createdate`. Rows predating the run are
  pre-existing contacts that were swept in.
- **"Something in the portal looks wrong"** → `hs doctor` for local config, `hs open`
  to jump to the UI.

## What this skill doesn't do

- **Bulk delete or merge.** Deliberately absent. Use `hs api -X DELETE` explicitly, one
  record at a time, after confirming with the user.
- **Marketing email, workflows, forms.** Separate scopes, not wired up.
- **Developer tooling.** `hs project`, `hs hubdb`, `hs secret`, `hs custom-object` and
  `hs app` are the CLI's own domain. Reach them through `hubspot hs <args>` or just run
  `hs` directly.
- **HubSpot Dev MCP.** `hs mcp setup` exists and targets app development rather than
  CRM data. Not enabled.

## Raw API access

Anything the wrapper omits goes through the CLI directly:

```bash
hs api /crm/v3/objects/deals --json
hs api /crm/v3/objects/contacts -X POST --data '{"properties":{"email":"a@b.com"}}'
```

Endpoint reference: https://developers.hubspot.com/docs/api-reference/latest/overview
