---
name: slack-messages
description: Read/search Slack messages via the Slack Web API with a user token. Use when the user asks about Slack — "what did I say in #channel", "find my Slack messages about X", "list my channels", "show the thread", "search Slack for Y". Defaults to the user's own messages (from:@me) unless they explicitly broaden the scope. Triggers on "slack", "channel", "thread", "DM", "#channel-name". Read-only; sending is intentionally out of scope.
---

# Slack messages

Search and read the user's Slack workspace through the Slack Web API. The user is a workspace owner; this skill operates as them via a user token. Defaults bias toward "just my messages" so results stay relevant.

## Setup (one-time, per user)

This skill is shareable. Credentials live outside the repo in `~/.config/slack/.env` (mode 0600). The skill itself contains no PII.

### 1. Create a Slack app + user token

Slack user tokens (`xoxp-…`) come from a Slack app installed in the workspace. As a workspace owner:

1. Go to https://api.slack.com/apps → **Create New App** → **From scratch** → name "Personal CLI" → pick the target workspace.
2. **OAuth & Permissions** → **User Token Scopes**, add at minimum:
   - `search:read` — for `search.messages`
   - `channels:read`, `groups:read`, `mpim:read`, `im:read` — list channels/DMs
   - `channels:history`, `groups:history`, `mpim:history`, `im:history` — read messages
   - `users:read` — resolve user ids (optional but useful)
3. **Install to Workspace**, copy the **User OAuth Token** (starts with `xoxp-`).

Bot tokens (`xoxb-`) won't work for `from:@me` — search.messages requires a user token.

### 2. Write the env file

`~/.config/slack/.env` (mode 0600):

```bash
SLACK_USER_TOKEN="xoxp-..."
```

```bash
mkdir -p ~/.config/slack
chmod 700 ~/.config/slack
# write the file, then:
chmod 600 ~/.config/slack/.env
```

### 3. Install the `slack` wrapper

Create `~/.local/bin/slack` (chmod +x). It loads the token, then runs the script — so callers just type `slack <subcommand> ...`.

```bash
#!/usr/bin/env bash
set -euo pipefail
ENV_FILE="$HOME/.config/slack/.env"
[ -f "$ENV_FILE" ] || { echo "slack: missing $ENV_FILE" >&2; exit 1; }
set -a; source "$ENV_FILE"; set +a
exec python3 "$HOME/.claude/skills/slack-messages/scripts/slack.py" "$@"
```

Verify: `slack whoami` should print the user + team.

Never print, log, or copy the token.

## Default "me" filter (IMPORTANT)

`search` auto-prepends `from:<@USER_ID>` to the query (resolved once per run via `auth.test`) so only the user's own messages return. They opt out with one of:

- "search everyone" / "include all messages" / "don't filter to me" → pass `--no-default-me`
- "messages from @alice" → look up alice's id, then `--no-default-me --query 'from:<@U…> ...'`

Note: `from:@me` and `from:@username` don't reliably work in Slack's search API. Always use the `<@USER_ID>` mention form. The script handles this automatically for "me"; for other people, look up their id via `users.info` or grab it from a permalink.

If a default-filtered query returns empty and a message clearly should exist, mention the filter and offer `--no-default-me`.

`history` does **not** apply a me-filter by default (channel history is naturally narrow). Pass `--mine-only` to filter to the user's messages.

## The tool

The `slack` wrapper loads the token and runs `scripts/slack.py`. Invoke subcommands directly:

```bash
slack <subcommand> [args...]
```

### `whoami`

Sanity check — prints authed user id + team. Run this once after setup.

```bash
slack whoami
```

### `channels` — list channels you're in

```bash
slack channels [--all] [--include-archived] [--types public_channel,private_channel,mpim,im]
```

- Defaults to **channels you're a member of** via `users.conversations`.
- `--all` switches to `conversations.list` (all workspace channels). Use sparingly — large workspaces page a lot.

### `search` — find messages (defaults to from:@me)

```bash
slack search \
  --query "deploy" \
  [--in-channel eng-platform] \
  [--after 2026-05-01] [--before 2026-05-27] \
  [--no-default-me] \
  [--limit 50] [--sort timestamp|score]
```

- Slack search supports operators: `from:`, `to:`, `in:`, `before:`, `after:`, `has:link`, `has:pin`, `is:thread`, etc.
- Output: JSON with `total`, `count`, `messages[]` (each has `ts`, `user`, `channel_name`, `permalink`, `text`).
- The query you can build manually wins — only use the `--in-channel/--after/--before` shortcuts when convenient.

### `history` — recent messages in a channel

```bash
slack history eng-platform [--limit 50] [--mine-only] [--oldest 1700000000.000000] [--latest ...]
```

Accepts channel name (with or without `#`) or id (`C0…`).

### `replies` — fetch a thread

```bash
slack replies eng-platform 1716800000.123456 [--limit 200]
```

`ts` is the parent message timestamp (visible in permalinks as `p1716800000123456` — insert a `.` six from the right).

### `recap` — full conversations from a given day

The right tool for "what did I talk about in Slack today?" — finds every channel/DM the user posted in on a given day, then pulls **all** messages (both sides) in those conversations within that day. DM partners are resolved to real names.

```bash
slack recap                              # today, system local tz
slack recap --date 2026-05-26
slack recap --tz America/New_York
slack recap --json                       # JSON instead of pretty text
```

Output: each conversation block shows `[HH:MM] Name ← you: text` for the user's messages and `[HH:MM] Name: text` for others, with threaded replies flagged.

Why this exists: `search.messages` only matches messages the user *sent*. Replies from others to those messages don't appear in search — `recap` joins those back in via `conversations.history`.

## Workflow patterns

- **"What conversations did I have in Slack today/yesterday?"** → `recap [--date YYYY-MM-DD]`. Don't use `search` for this — it misses replies from others.
- **"What have I said about X in Slack?"** → `search --query "X"` (me-filter applies automatically).
- **"What did the team say about X?"** → `search --query "X" --no-default-me`.
- **"My recent activity in #eng?"** → `history eng --mine-only --limit 100`.
- **"Show me that thread"** → derive `channel` + `ts` from the permalink, then `replies`.
- **"What channels am I in?"** → `channels` (defaults to member-only).

## What this skill doesn't do

- **Sending messages** — read-only by design. Add a `post` subcommand deliberately, not by reflex; it requires the `chat:write` scope and crosses the "talk back" line.
- **Real-time / streaming** — no Socket Mode or RTM. Each invocation is one HTTP call.
- **Admin / audit** — workspace admin APIs are a separate scope set and aren't wired up here.
- **File downloads** — not in scope yet; add if needed.

## Raw API access

For endpoints the script doesn't cover, hit `https://slack.com/api/<method>` directly with `Authorization: Bearer $SLACK_USER_TOKEN`. Method docs: https://api.slack.com/methods.
