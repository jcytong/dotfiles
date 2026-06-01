---
name: n8n
description: Interact with n8n Cloud via the REST API. Use when the user asks about n8n workflows, executions, credentials, tags, or variables — listing, inspecting, activating/deactivating workflows, checking execution history, or any other n8n Cloud API operation. Triggers on "n8n", "workflow", "execution", "n8n trigger", "activate workflow", "check executions", "n8n credentials", or similar.
---

# n8n Cloud API

## Setup (one-time, per user)

Credentials live in `~/.config/n8n/.env` (mode 0600):

```bash
N8N_API_KEY="your-api-key-here"
N8N_BASE_URL="https://myinstance.app.n8n.cloud/api/v1"
```

Find your API key: n8n Cloud → Settings → API → Create an API key.
Your base URL is `https://<instance-name>.app.n8n.cloud/api/v1`.

Create the wrapper at `~/.local/bin/n8n-api` (chmod +x):

```bash
#!/usr/bin/env bash
set -euo pipefail
ENV_FILE="$HOME/.config/n8n/.env"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE" >&2; exit 1; }
set -a; source "$ENV_FILE"; set +a
SCRIPT="$HOME/.claude/skills/n8n/scripts/n8n.py"
[ "$#" -eq 0 ] && exec python3 "$SCRIPT" --help
case "$1" in
  exec) shift; exec "$@" ;;
  *)    exec python3 "$SCRIPT" "$@" ;;
esac
```

Never print, log, or copy the API key.

## The tool

### `workflows list` — find workflows

```bash
n8n-api workflows list \
  [--active true|false] \
  [--tags "production,critical"] \
  [--limit 100] \
  [--cursor <cursor>]
```

Output: `{ "data": [...], "nextCursor": "..." }`. Each workflow has `id`, `name`, `active`, `createdAt`, `updatedAt`, `tags`.

### `workflows get` — inspect a workflow

```bash
n8n-api workflows get <id>
```

Returns the full workflow including `nodes`, `connections`, `settings`.

### `workflows activate` / `deactivate`

```bash
n8n-api workflows activate <id>
n8n-api workflows deactivate <id>
```

### `executions list` — check run history

```bash
n8n-api executions list \
  [--workflow-id <id>] \
  [--status error|success|waiting|running] \
  [--limit 20] \
  [--cursor <cursor>] \
  [--include-data]
```

- `--include-data`: adds full node input/output data — can be very large, use only when debugging a specific execution.
- Default limit is 20. Newest executions first.

### `executions get` — inspect one execution

```bash
n8n-api executions get <id> [--include-data]
```

Use `--include-data` to see what data flowed through the nodes.

### `credentials list`

```bash
n8n-api credentials list
```

Secret values are redacted — only names and types are returned.

### `tags list` / `variables list` / `users list`

```bash
n8n-api tags list
n8n-api variables list
n8n-api users list      # admin only
```

## Workflow

For "show me all active workflows":
1. `n8n-api workflows list --active true`
2. Format the `data` array: id, name, last updated.

For "why did the X workflow fail?":
1. `n8n-api workflows list` to find the workflow ID by name.
2. `n8n-api executions list --workflow-id <id> --status error --limit 5`
3. `n8n-api executions get <id> --include-data` to see which node failed and what data it received.

For "activate workflow X":
1. Find the ID via `workflows list` if not already known.
2. `n8n-api workflows activate <id>`

## Pagination

When results include `nextCursor`, pass it with `--cursor` to fetch the next page. Keep paginating until `nextCursor` is absent.

## Raw API access

For operations not covered by the script (create/update workflows, source control pull, credential schema, user role changes):

```bash
n8n-api exec bash -c 'curl -s "$N8N_BASE_URL/workflows" \
  -H "X-N8N-API-KEY: $N8N_API_KEY" \
  -H "Accept: application/json"'
```

See [api.md](references/api.md) for all endpoints and field schemas.

## What this skill doesn't do

- **Create/update/delete workflows**: the script is read-focused; use `exec` + curl for write operations.
- **Trigger executions manually**: use `POST /workflows/{id}/run` via the exec escape hatch.
- **Source control**: use `POST /source-control/pull` via exec.
