---
name: clickup
description: Read and write ClickUp tasks via the REST API v2 with a personal token. Use when the user asks about ClickUp — "what tasks are assigned to me", "find ClickUp tasks about X", "show task <id>", "create a ClickUp task", "move task to in progress", "comment on task", "list my spaces/lists". Defaults to tasks assigned to the user (assignees=me) unless they broaden scope. Reads are immediate; writes (create/update/comment) preview first and require explicit confirmation. Triggers on "clickup", "task", "list", "space", "assigned to me", "move to <status>", "due", "subtask".
---

# ClickUp

Search, read, create, and update tasks in the user's ClickUp workspace through the REST API v2. ClickUp has no official CLI; this skill is a thin self-owned wrapper (curated JSON, "assigned to me" default, preview-before-write) — the same shape as the `slack` and `fireflies` skills.

This skill is shareable: it holds no PII. Every user installs the same two artifacts the first time they use it — an env file with their token, and a shell wrapper that loads it and dispatches to the skill's Python script.

## Setup — Claude verifies this on first invocation

**Preflight (run this first, every session):**

```bash
test -f ~/.config/clickup/.env && test -x ~/.local/bin/clickup && echo OK || echo MISSING
```

If the output is anything other than `OK`, **stop** and walk the user through whichever piece is missing below before any other operation. Never proceed half-configured.

### 1. Personal API token

ClickUp personal tokens start with `pk_`. Get one at **ClickUp → Settings → Apps → API Token → Generate**. The token acts as the user with their full permissions — treat it like a password.

### 2. Env file — `~/.config/clickup/.env`

```bash
mkdir -p ~/.config/clickup && chmod 700 ~/.config/clickup
cat > ~/.config/clickup/.env <<'EOF'
CLICKUP_TOKEN="pk_xxxxxxxxxxxxxxxxxxxxxxxx"
# Optional: default workspace id so commands don't need --team.
# Find it with `clickup workspaces` after the wrapper is installed.
CLICKUP_TEAM_ID=""
EOF
chmod 600 ~/.config/clickup/.env
```

`CLICKUP_TOKEN` is required; `CLICKUP_TEAM_ID` is optional (auto-detected when the token sees exactly one workspace). Never print, log, or copy the token.

### 3. Shell wrapper — `~/.local/bin/clickup`

```bash
mkdir -p ~/.local/bin
cat > ~/.local/bin/clickup <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ENV_FILE="$HOME/.config/clickup/.env"
[ -f "$ENV_FILE" ] || { echo "clickup: missing $ENV_FILE" >&2; exit 1; }
set -a; source "$ENV_FILE"; set +a
exec python3 "$HOME/.claude/skills/clickup/scripts/clickup.py" "$@"
EOF
chmod +x ~/.local/bin/clickup
```

Make sure `~/.local/bin` is on the user's `PATH`. Verify with `clickup whoami` — should print the authed user id + username. Then run `clickup workspaces` and offer to write the id into `CLICKUP_TEAM_ID`.

## Default "me" filter (IMPORTANT)

`tasks` auto-filters to tasks assigned to the authed user (resolved once via `GET /user`). The user opts out with one of:

- "everyone's tasks" / "all assignees" / "don't filter to me" → pass `--no-default-me`
- "tasks assigned to <person>" → look up their user id (`clickup workspaces` lists members, or grab it from a task), then `--no-default-me --assignee <id>`

If a default-filtered query returns empty and a task clearly should exist, mention the filter and offer `--no-default-me`.

## Writes require confirmation (IMPORTANT)

`create`, `update`, and `comment` mutate the user's real workspace. The script is built so they **preview by default and only execute with `--yes`**: without `--yes` they print the intended endpoint + JSON body and send nothing.

Always: build the command, run it once **without `--yes`** to show the user the preview, confirm in plain language (*"Create this task / move #abc to In Progress? (yes/no)"*), then re-run **with `--yes`**. Never pass `--yes` on the first call. `--dry-run` forces preview even if `--yes` is present.

## The tool

The `clickup` wrapper loads the token and runs `scripts/clickup.py`. Invoke subcommands directly. All output is JSON.

### Read commands

```bash
clickup whoami                       # authed user — sanity check
clickup workspaces                   # list workspaces (id, name, member count)
clickup tree [--team <id>]           # spaces → folders → lists (find a list id)

clickup tasks \                      # filtered search, defaults to assignees=me
  [--status "in progress"] [--status review] \   # repeatable
  [--tag urgent] \                               # repeatable
  [--list <list_id>] \                           # restrict to list(s), repeatable
  [--due-before 2026-06-20] [--due-after 2026-06-01] \  # epoch ms or ISO
  [--name-contains "invoice"] \                  # client-side name filter
  [--assignee me] [--no-default-me] \
  [--include-closed] [--subtasks] \
  [--order-by updated|created|due_date] [--reverse] \
  [--raw] [--limit 100]

clickup get <task_id> [--custom-task-id --team <id>]   # one task, full detail
clickup comments <task_id>                             # task comments
```

- `tasks` returns a **slim** projection by default (`id`, `name`, `status`, `assignees`, `due_date`, `list`, `url`). Pass `--raw` for the full task objects (custom fields, descriptions, etc.).
- `due_date` values in output are epoch **milliseconds** (ClickUp's native unit).
- `--list` takes a list **id** — get it from `clickup tree`.

### Write commands (preview first, then `--yes`)

```bash
clickup create --list <list_id> --name "Title" \
  [--description "..."] [--status open] [--priority 1..4] \
  [--due 2026-06-15] [--assignee me] [--tag x] [--yes]

clickup update <task_id> \
  [--name "..."] [--description "..."] [--status "in progress"] \
  [--priority 1..4] [--due 2026-06-20] \
  [--add-assignee me] [--rem-assignee <id>] [--yes]

clickup comment <task_id> --text "..." [--assignee me] [--notify-all] [--yes]
```

- `--priority`: `1` urgent, `2` high, `3` normal, `4` low.
- `--due`: epoch ms or an ISO date (`2026-06-15`, converted to ms automatically).
- `--status` must match a status that exists in the task's list (case-insensitive name, e.g. `"in progress"`). If unsure, `clickup get <id>` shows the current status; `clickup tree` won't list statuses — read one task in the list.

## Workflow patterns

- **"What's assigned to me?"** → `clickup tasks` (me-filter applies). Add `--status` to narrow.
- **"What's overdue?"** → `clickup tasks --due-before <today-ms> --order-by due_date`.
- **"Find the task about X"** → `clickup tasks --name-contains "X"` (add `--no-default-me` if it might be someone else's).
- **"Show me task abc123"** → `clickup get abc123` (+ `clickup comments abc123` if they want discussion).
- **"Where do tasks live / what lists exist?"** → `clickup tree`.
- **"Create a task in <list>"** → `clickup tree` to find the list id → `clickup create ... ` (preview) → confirm → `--yes`.
- **"Move abc123 to In Progress" / "reassign to me"** → `clickup update abc123 --status "in progress"` / `--add-assignee me` (preview → confirm → `--yes`).

## Time / dates

ClickUp stores dates as Unix **epoch milliseconds**. The script accepts ISO dates (`YYYY-MM-DD` or full ISO 8601) and converts them; it also accepts raw ms. When the user says "due Friday," resolve the calendar date first, then pass it as an ISO date. Output `due_date` fields are ms — convert to the user's timezone when presenting.

## Raw API access

For endpoints the script doesn't cover (custom fields write, time tracking, checklists, attachments, webhooks, docs v3), hit the API directly with the token loaded:

```bash
clickup whoami >/dev/null   # confirms auth works, then:
# the wrapper sources the env, so within a shell you launch from it, $CLICKUP_TOKEN is set
curl -s https://api.clickup.com/api/v2/team \
  -H "Authorization: $CLICKUP_TOKEN"
```

Note the auth header is the **raw personal token**, not `Bearer <token>`. See [api.md](references/api.md) for the endpoint map, query-param reference, and curl recipes.

## What this skill doesn't do

- **Bulk/destructive ops** — no task delete, no bulk status moves. Add deliberately; deletion crosses a line that needs its own guardrails.
- **Custom fields / time tracking / checklists / attachments** — read them via `get --raw`; writing them isn't wired up. Use raw API access if needed.
- **ClickUp Docs (v3 API)** — out of scope; this skill is tasks-only.
- **Webhooks / real-time** — each invocation is one HTTP round-trip.
