---
name: fireflies
description: Find and extract Fireflies.ai meeting transcripts via the GraphQL API. Use when the user asks about a Fireflies meeting, transcript, summary, action items, or who said what in a meeting. By default narrows results to meetings where the user's own email(s) appear as a participant (configured via FIREFLIES_DEFAULT_PARTICIPANTS), unless the user explicitly broadens the scope. Triggers on "fireflies", "transcript for", "meeting on [date]", "what did [name] say in", "action items from", "summary of [meeting]".
---

# Fireflies

This skill is shareable: it holds no PII. Every user installs the same two artifacts the first time they use it — an env file with their API key, and a shell wrapper that loads it and dispatches to the skill's Python script.

## Setup — Claude verifies this on first invocation

**Preflight (run this first, every session):**

```bash
test -f ~/.config/fireflies/.env && test -x ~/.local/bin/fireflies && echo OK || echo MISSING
```

If the output is anything other than `OK`, **stop** and walk the user through whichever piece is missing below before any other operation. Never proceed to `list`/`get`/`live` half-configured.

### 1. Env file — `~/.config/fireflies/.env`

```bash
mkdir -p ~/.config/fireflies && chmod 700 ~/.config/fireflies
cat > ~/.config/fireflies/.env <<'EOF'
FIREFLIES_API_KEY="ff_xxxxxxxxxxxxxxxxxxxxxxxx"
# Optional: comma-separated emails for the default participant filter.
# Omit or leave empty to search across all meetings the API key can see.
FIREFLIES_DEFAULT_PARTICIPANTS="me@company.com,me@personal.com"
EOF
chmod 600 ~/.config/fireflies/.env
```

`FIREFLIES_API_KEY` is required (get it at https://app.fireflies.ai/integrations/custom/fireflies); `FIREFLIES_DEFAULT_PARTICIPANTS` is optional. Never print, log, or copy the API key.

### 2. Shell wrapper — `~/.local/bin/fireflies`

```bash
mkdir -p ~/.local/bin
cat > ~/.local/bin/fireflies <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ENV_FILE="$HOME/.config/fireflies/.env"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE" >&2; exit 1; }
set -a; source "$ENV_FILE"; set +a
SCRIPT="$HOME/.claude/skills/fireflies/scripts/fireflies.py"
[ "$#" -eq 0 ] && exec python3 "$SCRIPT" --help
case "$1" in
  exec) shift; exec "$@" ;;        # escape hatch: `fireflies exec curl ...`
  *)    exec python3 "$SCRIPT" "$@" ;;
esac
EOF
chmod +x ~/.local/bin/fireflies
```

Make sure `~/.local/bin` is on the user's `PATH`. Verify with `fireflies --help`.

## Default participant filter (IMPORTANT)

The `list` command automatically filters to meetings where one of the emails in `FIREFLIES_DEFAULT_PARTICIPANTS` is an attendee, so the user only sees their own meetings. The user opts out with one of:

- "search all meetings" / "include everyone" / "ignore my participant filter" → pass `--no-default-filter`
- "meetings with [other email]" → pass `--participant [email]` (replaces the default for that call)

If `FIREFLIES_DEFAULT_PARTICIPANTS` is unset or empty, no participant filter is applied — equivalent to passing `--no-default-filter` on every call.

If a default-filtered query returns empty and a meeting clearly *should* exist at that time, mention the filter to the user and offer to retry with `--no-default-filter`.

## The tool

The `fireflies` wrapper loads credentials and dispatches to `scripts/fireflies.py`. Use it like any normal CLI:

### `list` — find meetings

```bash
fireflies list \
  [--from 2026-05-25T13:00:00Z] \
  [--to   2026-05-25T16:00:00Z] \
  [--title "venture"] \
  [--participant alice@example.com] \
  [--no-default-filter] \
  [--limit 50]
```

- `--from` / `--to`: ISO 8601 UTC timestamps. **Convert from the user's timezone first** (e.g., 10am ET in May → `14:00Z`; 10am ET in December → `15:00Z`).
- `--title`: server-side substring match.
- `--participant`: overrides the default participant filter with the specified email for this call.
- `--no-default-filter`: disables the default participant filter entirely.
- Output: JSON `{ "count": N, "transcripts": [...] }`, newest first. Each entry has `id`, `title`, `dateString`, `date`, `duration` (minutes), `participants`, `host_email`, `organizer_email`.

### `get` — fetch a specific transcript

```bash
fireflies get [transcript_id] [--no-sentences] [--text-only]
```

- Default: full JSON including `summary` and `sentences`.
- `--no-sentences`: summary + metadata only. **Use this first** when the user just wants a summary or action items — the sentences array can be 1000+ entries.
- `--text-only`: prints sentences as `[start_time s] Speaker: text` lines. Use when the user wants the transcript itself rather than JSON.

### `live` — list active (in-progress) meetings

```bash
fireflies live
```

- Backed by the `active_meetings` GraphQL query. Returns meetings happening *right now*.
- Output: JSON `{ "count": N, "active_meetings": [...] }`. Each entry has `id`, `title`, `organizer_email`, `meeting_link`, `start_time`, `end_time` (scheduled, not actual), `state` (`active` | `paused`), `privacy`.
- **Metadata only** — no live transcript, summary, participants, or speaker content. Those populate only after the meeting ends and Fireflies finishes processing.
- Permission scope: a regular API key sees only its own active meetings; admin keys can see any team member's.
- The `end_time` is the *scheduled* end, not when the meeting actually wraps.

## Workflow

For a request like "get me the transcript for my 10am meeting yesterday":

1. Convert the user's local time window to UTC (ET in May = UTC-4 EDT; ET in winter = UTC-5 EST). When in doubt, widen the window by ±30 min.
2. Run `list --from ... --to ...` (default participant filter applies).
3. If exactly one result: proceed to `get`. If multiple: show the user titles + times and ask which one (or pick the best title match). If zero: confirm the filter, then offer `--no-default-filter`.
4. For "summary / action items / what happened": `get [id] --no-sentences`.
5. For "what did X say" / "full transcript" / "quote me a section": `get [id] --text-only` and grep/scan.

## Time zone conversion cheat sheet

The Fireflies API uses UTC. The user usually speaks in ET.

- ET → UTC: add 4 hours in EDT (~mid-Mar to early Nov), add 5 hours in EST (rest of year).
- PT → UTC: add 7 hours PDT, 8 hours PST.

When a user gives a single time (not a window), search a window of ±30 min around it to handle clock skew between scheduled and recorded start.

## Raw API access

For queries the script doesn't cover (analytics, sentiment, host_email filtering, pagination beyond 50), use the `exec` escape hatch to run arbitrary shell with the env loaded:

```bash
fireflies exec bash -c 'curl -s -X POST https://api.fireflies.ai/graphql \
  -H "Authorization: Bearer $FIREFLIES_API_KEY" -H "Content-Type: application/json" \
  -d "{\"query\":\"...\"}"'
```

See [api.md](references/api.md) for the full GraphQL schema and curl recipes.

## What this skill doesn't do

- **Live transcript content**: `fireflies live` lists in-progress meetings (metadata only). There is no public endpoint for the *live transcript text* of an ongoing meeting — sentences and summary appear only after the recording is processed.
- **Modifying transcripts**: this skill is read-only. Fireflies does have mutations (delete, update title, "add to live meeting", etc.) — add them deliberately, not by reflex.
