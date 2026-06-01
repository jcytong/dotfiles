---
name: daily-brief
description: Synthesize the user's day across Slack, Fireflies, Gmail, Google Drive, Google Calendar, and Fizzy. Two modes — `recap` (EOD, backward-looking) and `prep` (AM, forward-looking). Surfaces a structured brief in the conversation, archives it to an Obsidian vault, and files Tier 1 follow-ups as Fizzy cards on a configured board (deduping by source-anchored tags). Also includes `sync-sheet` for on-demand import of a Google Sheet's rows owned by the user as Fizzy cards. Use when the user says "recap my day," "what did I do today," "EOD summary," "morning brief," "prep me for today," "what's coming today/tomorrow," "GSD me," "sync my fundraising sheet," "import my todos from the sheet," or similar.
---

# daily-brief

Cross-source daily synthesis for the user. The skill is mostly prose: it tells Claude which CLIs to run in parallel, how to synthesize the output, where to archive it, and how to file follow-ups. A small Python helper (`scripts/daily_brief.py`) handles the new logic that doesn't already exist as a CLI — day-window math, Fizzy filing with dedup, Obsidian archive.

This skill is shareable: it holds no PII or workspace-specific ids. Per-user values (Fizzy board, Obsidian vault, timezone, sync-sheet target) live in `~/.config/daily-brief/.env` — see **Configuration** below.

## Modes

Two modes — pick based on what the user asked for:

| User phrase contains… | Mode |
|---|---|
| "recap," "EOD," "what did I do," "summarize my day," "today's debrief" | `recap` |
| "prep," "morning brief," "what's coming," "what's on today/tomorrow," "today's plan" | `prep` |
| "sync my sheet," "import the fundraising sheet," "pull my todos from the sheet" | `sync-sheet` |
| ambiguous | Ask which |

For `sync-sheet`: just run the subcommand. It's not a synthesis pass — no brief, no Obsidian archive. Just import. Show the user the resulting `created`/`skipped` summary.

## Setup the skill assumes is in place

The user already has these wrappers on PATH and authed:

- `slack` — Slack user-token wrapper (see `slack-messages` skill)
- `fireflies` — Fireflies wrapper (see `fireflies-transcripts` skill)
- `gws` — Google Workspace CLI (Gmail, Drive, Calendar)
- `fizzy` — Fizzy CLI

Plus this skill's helper at `~/.claude/skills/daily-brief/scripts/daily_brief.py`.

If any wrapper is missing or unauthed, surface the specific failure and stop — don't proceed half-blind.

## Configuration

Per-user values live in `~/.config/daily-brief/.env` (mode 0600), loaded automatically by the helper. Copy `.env.example` from this skill to get started. Vars:

- `DAILY_BRIEF_TZ` — timezone for day-window math + archive filenames. Default `America/New_York`.
- `DAILY_BRIEF_FIZZY_BOARD_ID` — board Tier 1 cards are filed to. **Required** for `open-cards`/`file-cards`/`sync-sheet`. (`fizzy board list` to find it.)
- `DAILY_BRIEF_FIZZY_IN_PROGRESS_COLUMN_ID` — that board's "In Progress" column id. Optional; falls back to the pseudo id `in-progress`.
- `DAILY_BRIEF_VAULT_DIR` — Obsidian vault dir briefs archive to. **Required** for `archive`. `~` and `$HOME` are expanded. File pattern: `YYYY-MM-DD-{recap,prep}.md`.
- `DAILY_BRIEF_SHEET_ID` / `_TAB` / `_GID` / `_OWNER` — `sync-sheet` defaults. Optional; without them, pass `--spreadsheet-id`/`--owner` per call.

If a required var is missing, the helper exits with a message naming the var. The gather/synthesis steps themselves need no config.

---

# Mode: recap

End-of-day, backward-looking. The pattern from the conversation that birthed this skill.

## Step 1: Compute the day window

```bash
python3 ~/.claude/skills/daily-brief/scripts/daily_brief.py day-window recap
```

This returns the start/end timestamps in multiple formats. Use them for the downstream queries.

## Step 2: Gather, in parallel

Make all of these calls in a **single message with parallel tool uses**. They're independent.

**1. Slack — today's conversations (both sides)**
```bash
slack recap --tz America/New_York --json
```

**2. Fireflies — today's meetings**
```bash
fireflies python3 ~/.claude/skills/fireflies-transcripts/scripts/fireflies.py list \
  --from <start_iso_utc> --to <end_iso_utc>
```

Use `start_iso_utc` / `end_iso_utc` from step 1.

**3. Gmail — today's signal (noise pre-filtered)**

Use this query — it strips promotional/social noise:
```bash
gws gmail users messages list \
  --params '{"userId": "me", "q": "newer_than:1d -category:promotions -category:social", "maxResults": 100}' \
  --format json
```

Then fetch headers for each ID with `format: metadata, metadataHeaders: [From, To, Subject, Date]`. Drop anything from `noreply@`, `notifications@`, `donotreply@`, `no-reply@` after fetch.

**4. Drive — today's modifications in user's orbit**
```bash
gws drive files list --params "{\"q\": \"modifiedTime > '<start_iso_utc>'\", \"orderBy\": \"modifiedTime desc\", \"pageSize\": 50, \"fields\": \"files(id,name,mimeType,modifiedTime,lastModifyingUser,webViewLink,owners)\"}" --format json
```

**5. Fizzy — open cards (roll-forward)**
```bash
python3 ~/.claude/skills/daily-brief/scripts/daily_brief.py open-cards
```

Then sequentially fetch summaries for each Fireflies meeting (one call per meeting):
```bash
fireflies python3 ~/.claude/skills/fireflies-transcripts/scripts/fireflies.py get <id> --no-sentences
```

(Parallelize across meetings if there are >2.)

## Step 3: Synthesize the brief

Structure (always in this order):

1. **Header:** `# Recap — YYYY-MM-DD`
2. **Still open from prior days** — render any cards from `open-cards` whose `created_at` is < today. Format: `- [[card title]] (filed YYYY-MM-DD, [link](url))`. Skip this section if empty.
3. **Shape of the day** — 2–4 sentences naming the dominant threads.
4. **Meetings** — markdown table (time, title, duration), then a 2–3 bullet sub-section per meeting summarizing substance + action items relevant to the user.
5. **Slack** — only the *consequential* messages. The 4-message DM logistics → ignore. The 10:11 directive to Sam → mention. Show counterparties' replies (slack recap returns both sides).
6. **Drive** — table of files modified in the user's orbit (`name`, `lastModifyingUser`, link).
7. **Gmail signal** — bullets of the genuinely actionable items only. Calendar invites, RSVPs requested, real human reply needed, financial/security notices with deadlines. Newsletters/promos already filtered at gather; if any sneak through, drop them here.
8. **Top things to work on (Tier 1)** — 3–5 items max, each with:
   - One-line headline (action-imperative).
   - 1–3 lines of reasoning (*why this matters now*).
   - The underlying source link (Fireflies / Slack / Drive / Calendar).
9. **Follow-ups (Tier 2)** — markdown checklist, 4–10 items. Smaller things. Do NOT file these to Fizzy.
10. **What didn't happen but probably should** — Claude's observation about gaps in the day (e.g., "no deep-work block, no movement on X"). One short paragraph.

### Wikilink rules (apply throughout the brief)

Wrap in `[[…]]`:
- People at Acme (anyone whose email is `*@company.com` or whose name appears in a known internal context).
- External people who've had ≥1 prior interaction with the user (judge from the day's data; if uncertain, link).
- Active Acme portfolio companies / ventures (Northwind, Globex, etc.).
- Internal projects (Atlas, Beacon, scout, etc.).

Do **not** wikilink:
- SaaS vendors (Supabase, Linear, Gemini, Zapier, Fireflies, GCP, FINTRX, etc.).
- Generic concepts (EIR, CTO, IC Meeting, Town Hall).
- One-off external senders on first mention.

When in doubt about a person, prefer linking — Obsidian creates a stub page, which is fine.

## Step 4: Identify Tier 1 → file to Fizzy

For each Tier 1 item, build a JSON object:

```json
{
  "title": "Send Northwind investment view to Alice + Bob",
  "description_md": "**Context:** ...\n\n**Source:** <url>\n\n**Next action:** ...\n\n**Filed:** daily-brief recap on YYYY-MM-DD",
  "source_type": "fireflies",
  "source_id": "01KS2Q9ZHA0NYNPQEX4B7W239V",
  "semantic_slug": "northwind-investment-view"
}
```

- `source_type` ∈ `fireflies | slack | calendar | gmail | drive | sheet | other`
- `source_id`: the meeting ID / Slack channel id / event id / Gmail thread id / Drive file id / spreadsheet id. Used for dedup; choose the most stable source for this action.
- `semantic_slug`: short kebab-case, ≤ 40 chars. Distinguishes multiple Tier 1 items that came from the same source meeting.
- `column` (optional): `maybe` (default) | `in_progress` | `not_now`. Use `in_progress` when the item is already actively underway (e.g., synced from a status="In Progress" sheet row); use `maybe` for new commitments that need triage; use `not_now` to park.

**Dedup gate (you, not the script):**
Before filing, scan `open-cards` output. If a Tier 1 item's source-id substring matches the `db_tags` of an existing open card, **do not file it again** — instead, in the brief, surface that item as "↻ still open from card #N" and link to it. The script also refuses on exact-tag collision, but Claude is the smart layer.

**Show the user the proposed Tier 1 list before filing.** Ask: *"File these N cards? (yes / drop #2 / change titles / no)"* Wait for confirmation, then run:

```bash
echo '<JSON array>' | python3 ~/.claude/skills/daily-brief/scripts/daily_brief.py file-cards
```

If the user wants to preview without filing, append `--dry-run`.

## Step 5: Archive to Obsidian

Pipe the final brief content (everything below the `# Recap — …` header is fine — frontmatter is added by the helper) to:

```bash
python3 ~/.claude/skills/daily-brief/scripts/daily_brief.py archive recap \
  --people "Jane Doe, John Smith, ..." \
  --projects "Northwind, Atlas, ..."
```

Pass the wikilinked people and projects to `--people` / `--projects` (comma-separated, without `[[]]` brackets — they go into YAML frontmatter for Dataview/grep). Overwrites if same date already exists.

## Step 6: Wrap up

End with one line in the conversation:

> Archived to `<path>`. Filed N Fizzy cards: <card numbers and titles>.

---

# Mode: prep

Morning, forward-looking. Window: yesterday 6pm → end of today.

## Step 1: Compute the window

```bash
python3 ~/.claude/skills/daily-brief/scripts/daily_brief.py day-window prep
```

## Step 2: Gather, in parallel

**1. Calendar — today's events**
```bash
gws calendar events list --params '{"calendarId": "primary", "timeMin": "<start_iso>", "timeMax": "<end_iso>", "singleEvents": true, "orderBy": "startTime", "maxResults": 50}' --format json
```

Filter out: declined events, all-day OOO/blocking, attendee-less holds.

**2. Gmail — overnight + unread**
```bash
gws gmail users messages list \
  --params '{"userId": "me", "q": "(newer_than:1d -category:promotions -category:social) OR (is:unread is:starred)", "maxResults": 100}' \
  --format json
```

Then headers per id, same filter rules as recap.

**3. Drive — shared with me + attachments to today's meetings**
```bash
gws drive files list --params "{\"q\": \"sharedWithMe and viewedByMeTime > '<start_iso_utc>'\", \"orderBy\": \"sharedWithMeTime desc\", \"pageSize\": 30, \"fields\": \"files(id,name,mimeType,modifiedTime,sharingUser,webViewLink)\"}" --format json
```

For meeting prereads, walk each calendar event's `attachments` (if present in event payload) — those are Drive file refs.

**4. Slack — unread + mentions**
```bash
slack search --query "is:unread"
slack search --query "to:@me" --no-default-me
```

**5. Fizzy — open cards**
```bash
python3 ~/.claude/skills/daily-brief/scripts/daily_brief.py open-cards
```

## Step 3: Deep-fetch external-meeting attachments

For each calendar event today that has **at least one non-`company.com` attendee**, and that has Drive attachments, fetch the attachment contents as text:

```bash
gws drive files export --params '{"fileId": "<id>", "mimeType": "text/plain"}'
```

Cap at the first 3 such meetings — don't go nuts. Use the content to compose a 2-sentence pre-read in the brief.

## Step 4: Synthesize

Structure:

1. **Header:** `# Prep — YYYY-MM-DD`
2. **Still open from prior days** — same as recap.
3. **The day's shape** — 1–2 sentences. Total meeting count + first/last meeting times.
4. **Today's meetings (with pre-reads)** — chronological. For each:
   - `## HH:MM — Title (duration · attendees)`
   - 1 sentence of who/why.
   - For external meetings with attachments: a `### Pre-read` sub-section with the 2-sentence summary you composed.
5. **Inbox state** — what came in after EOD yesterday that genuinely needs action before noon. Bullets.
6. **Slack — overnight signal** — unread + mentions. Brief.
7. **Top things to work on (Tier 1)** — same shape as recap. **Important:** prep mode rarely files NEW Tier 1 cards (those are end-of-day insights). Surface existing open cards prominently; only file genuinely new commitments.
8. **Follow-ups (Tier 2)** — small things, checklist form.

## Step 5: Archive + (maybe) file

Same archive step as recap, but with `prep` mode:

```bash
python3 ~/.claude/skills/daily-brief/scripts/daily_brief.py archive prep \
  --people "..." --projects "..."
```

Only file Tier 1 cards if you've identified genuinely new commitments (rare in prep mode). When in doubt, don't file — the user will refile during the EOD recap if it's still relevant.

---

# Helper script reference

`scripts/daily_brief.py` subcommands:

| Subcommand | Purpose |
|---|---|
| `day-window {recap,prep} [--date YYYY-MM-DD]` | Returns the time window as JSON (multiple formats). |
| `open-cards` | Lists open cards (`Maybe?` + `In Progress`) on the configured board with their `db:` tags. |
| `file-cards [--dry-run] [--force]` | Reads JSON array on stdin; creates Fizzy cards with dedup by `db:<source>:<id>:<slug>` tag. |
| `archive {recap,prep} [--date] [--people] [--projects] [--content-file]` | Writes the brief markdown to the Obsidian vault with frontmatter. |
| `sync-sheet [--spreadsheet-id …] [--tab …] [--gid …] [--owner …] [--dry-run] [--print-json]` | One-shot sync of a Google Sheet's rows-owned-by-user into Fizzy cards. Sheet/tab/owner come from `DAILY_BRIEF_SHEET_*` config or the `--spreadsheet-id`/`--tab`/`--gid`/`--owner` flags. Status="In Progress" rows land directly in the In Progress column; everything else lands in Maybe?. Skips Complete rows with no follow-up note. Use on demand (after a planning session, not on every recap). |

# What this skill does NOT do

- Schedule itself (Architecture A — conversation-driven).
- Synthesize without a Claude conversation present.
- Push to Linear/Notion/Todoist.
- Auto-close cards when actions appear complete.
- Track Tier 2 items long-term (they live in the brief markdown only).
- Enrich senders with LinkedIn / interaction history.

If the user asks for any of the above, name what's not in scope and offer to add it as v2.
