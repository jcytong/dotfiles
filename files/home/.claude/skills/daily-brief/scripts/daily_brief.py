#!/usr/bin/env python3
"""daily-brief helpers.

Subcommands:
  day-window  - Print the day window (start/end ISO + unix) for a mode in the
                fixed America/New_York timezone. Used by SKILL.md to compute
                source query ranges.
  open-cards  - List open Fizzy cards (Maybe? + In Progress) on the Acme
                board, in a flat shape with their daily-brief tags surfaced.
                Claude uses this to dedup before filing new ones.
  file-cards  - Read a JSON array of Tier 1 items from stdin and create Fizzy
                cards for each, applying the two-part dedup tag and skipping
                items whose source-anchored tag already exists on any card.
  archive     - Write the markdown brief to the Obsidian vault with frontmatter.

Reads SLACK/FIREFLIES/FIZZY/etc. tokens from the wrappers' env files. This
script itself only touches Fizzy directly (via the `fizzy` CLI) and the
filesystem; everything else is invoked by Claude.

This is a thin helper. The heavy lifting — synthesis, source aggregation,
wikilinking — happens in the Claude conversation per SKILL.md.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------- per-user config ----------
# This skill is shareable: nothing here is hardcoded to one person or workspace.
# Each user keeps their own values in ~/.config/daily-brief/.env (see .env.example
# and SKILL.md → Configuration). The file is loaded below before the constants
# are read, so the script works whether it's invoked directly or via a wrapper.

def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file (no overwrite)."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, val = line.partition("=")
        if not sep:
            continue
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(Path.home() / ".config/daily-brief/.env")

TZ = ZoneInfo(os.environ.get("DAILY_BRIEF_TZ", "America/New_York"))
BOARD_ID = os.environ.get("DAILY_BRIEF_FIZZY_BOARD_ID", "")
_vault = os.environ.get("DAILY_BRIEF_VAULT_DIR", "")
VAULT_DIR = Path(os.path.expanduser(os.path.expandvars(_vault))) if _vault else None
DAILY_BRIEF_MARKER_TAG = "daily-brief"

# Friendly column-name → fizzy column id. Pseudo columns ("maybe", "not-now",
# "done") work as ids in fizzy as well, so callers can use either. The
# In Progress column id is board-specific, so it comes from config (falling back
# to the pseudo id, which fizzy also accepts).
_IN_PROGRESS_COLUMN = os.environ.get("DAILY_BRIEF_FIZZY_IN_PROGRESS_COLUMN_ID", "in-progress")
COLUMN_IDS = {
    "maybe": "maybe",
    "not-now": "not-now",
    "not_now": "not-now",
    "in-progress": _IN_PROGRESS_COLUMN,
    "in_progress": _IN_PROGRESS_COLUMN,
    "done": "done",
}

# sync-sheet defaults (all optional; only used by the sync-sheet subcommand).
SHEET_ID_DEFAULT = os.environ.get("DAILY_BRIEF_SHEET_ID", "")
SHEET_TAB_DEFAULT = os.environ.get("DAILY_BRIEF_SHEET_TAB", "")
SHEET_GID_DEFAULT = os.environ.get("DAILY_BRIEF_SHEET_GID", "")
SHEET_OWNER_DEFAULT = os.environ.get("DAILY_BRIEF_SHEET_OWNER", "")


def _require(value, env_name: str, what: str) -> None:
    """Exit with a clear message when a required config value is missing."""
    if not value:
        sys.exit(
            f"daily-brief: {env_name} is not set. Add it to "
            f"~/.config/daily-brief/.env ({what}). See SKILL.md → Configuration."
        )
# Sheet status → target fizzy column (anything else → Maybe? for triage)
STATUS_TO_COLUMN = {
    "in progress": "in_progress",
}


# ---------- shared helpers ----------

def run_fizzy(args: list[str], stdin: str | None = None) -> dict:
    """Run `fizzy <args>` and parse JSON. Exits on error."""
    proc = subprocess.run(
        ["fizzy", *args],
        capture_output=True,
        text=True,
        input=stdin,
    )
    if proc.returncode != 0:
        sys.exit(f"fizzy {' '.join(args)} failed:\n{proc.stderr}")
    try:
        body = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit(f"fizzy {' '.join(args)} returned non-JSON:\n{proc.stdout[:500]}")
    if not body.get("success", True):
        sys.exit(f"fizzy {' '.join(args)} reported failure: {body}")
    return body


def today_date(date_str: str | None = None) -> _dt.date:
    if date_str:
        return _dt.date.fromisoformat(date_str)
    return _dt.datetime.now(TZ).date()


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-")


def md_to_html(text: str) -> str:
    """Minimal markdown→HTML for Fizzy card descriptions.

    Supports paragraphs, bold (**x**), italic (*x*), inline code (`x`), and
    line breaks. Anything fancier (lists, headers) should be plain prose since
    Fizzy renders the HTML inline.
    """
    if not text:
        return ""
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    paragraphs = [p.strip() for p in escaped.split("\n\n") if p.strip()]
    return "\n".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs)


# ---------- day-window ----------

def cmd_day_window(args: argparse.Namespace) -> None:
    d = today_date(args.date)
    if args.mode == "recap":
        start = _dt.datetime.combine(d, _dt.time(0, 0), tzinfo=TZ)
        end = _dt.datetime.now(TZ)
    elif args.mode == "prep":
        # Yesterday 6pm → end of today
        yesterday = d - _dt.timedelta(days=1)
        start = _dt.datetime.combine(yesterday, _dt.time(18, 0), tzinfo=TZ)
        end = _dt.datetime.combine(d + _dt.timedelta(days=1), _dt.time(0, 0), tzinfo=TZ)
    else:
        sys.exit(f"unknown mode: {args.mode}")

    out = {
        "mode": args.mode,
        "tz": str(TZ),
        "date": d.isoformat(),
        "start_iso": start.isoformat(),
        "end_iso": end.isoformat(),
        "start_iso_utc": start.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_iso_utc": end.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_unix": int(start.timestamp()),
        "end_unix": int(end.timestamp()),
        # Slack and Gmail conveniences
        "gmail_after_yyyy_mm_dd": d.strftime("%Y/%m/%d") if args.mode == "recap" else (d - _dt.timedelta(days=1)).strftime("%Y/%m/%d"),
        "slack_after_yyyy_mm_dd": d.strftime("%Y-%m-%d") if args.mode == "recap" else (d - _dt.timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    print(json.dumps(out, indent=2))


# ---------- open-cards ----------

def fetch_tag_index() -> dict[str, dict]:
    """Map tag title → tag record. Uses --all because Fizzy paginates
    tag list at ~20 by default and dedup needs every tag visible."""
    body = run_fizzy(["tag", "list", "--all"])
    return {t["title"]: t for t in (body.get("data") or [])}


def cmd_open_cards(args: argparse.Namespace) -> None:
    """Fetch open cards on the configured board with their daily-brief tags surfaced.

    Returns cards in column `Maybe?` and `In Progress`, omitting closed cards.
    """
    _require(BOARD_ID, "DAILY_BRIEF_FIZZY_BOARD_ID", "the Fizzy board cards are filed to")
    body = run_fizzy([
        "card", "list",
        "--board", BOARD_ID,
        "--all",
    ])
    cards = body.get("data") or []

    results = []
    for c in cards:
        if c.get("closed"):
            continue
        col = c.get("column") or {}
        col_name = col.get("name") if isinstance(col, dict) else None
        col_kind = col.get("kind") if isinstance(col, dict) else None
        # Default (untriaged) cards have no column → treat as Maybe?
        if col_kind == "closed":
            continue
        # Fizzy returns tags as plain strings on card list/show responses
        # (e.g. ["daily-brief", "db:fireflies:..."]).
        tags = [t for t in (c.get("tags") or []) if isinstance(t, str)]
        db_tags = [t for t in tags if t.startswith("db:")]
        results.append({
            "number": c.get("number"),
            "title": c.get("title"),
            "column": col_name or "Maybe?",
            "tags": tags,
            "db_tags": db_tags,
            "created_at": c.get("created_at"),
            "url": c.get("url"),
        })
    results.sort(key=lambda r: (r["column"] != "In Progress", r.get("created_at") or ""))
    print(json.dumps({"count": len(results), "cards": results}, indent=2))


# ---------- file-cards ----------

def cmd_file_cards(args: argparse.Namespace) -> None:
    """Read a JSON array of Tier 1 items from stdin and create Fizzy cards.

    Item shape:
      {
        "title":           "Send Northwind investment view to Alice + Bob",
        "description_md":  "Markdown body with **bold**, *italic*, `code`...",
        "source_type":     "fireflies" | "slack" | "calendar" | "gmail" | "drive" | "sheet" | "other",
        "source_id":       "01KS2Q9ZHA0NYNPQEX4B7W239V",
        "semantic_slug":   "northwind-investment-view",  # short kebab-case
        "column":          "maybe" | "in_progress" | "not_now"  # optional, default maybe
      }

    For each item:
      1. Build tag string: db:<source_type>:<source_id_short>:<semantic_slug>
      2. If a tag with that title already exists AND has cards attached → skip
      3. Else: create card, then tag with [daily-brief, full-tag]
      4. If `column` ≠ default, move the card to that column
    """
    _require(BOARD_ID, "DAILY_BRIEF_FIZZY_BOARD_ID", "the Fizzy board cards are filed to")
    raw = sys.stdin.read().strip()
    if not raw:
        sys.exit("file-cards: no JSON on stdin")
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"file-cards: bad JSON on stdin: {e}")
    if not isinstance(items, list):
        sys.exit("file-cards: stdin must be a JSON array")

    tag_index = fetch_tag_index()
    actions = []

    for item in items:
        title = (item.get("title") or "").strip()
        desc_md = item.get("description_md") or ""
        source_type = item.get("source_type") or "other"
        source_id = (item.get("source_id") or "").strip()
        semantic_slug = slugify(item.get("semantic_slug") or title)

        if not title:
            actions.append({"status": "skipped", "reason": "missing title", "item": item})
            continue
        if not semantic_slug:
            actions.append({"status": "skipped", "reason": "empty semantic_slug", "item": item})
            continue

        # Short source id (first 12 chars of UUID/ULID for tag brevity).
        # Lowercased because Fizzy stores tag titles lowercased on the server.
        short_id = re.sub(r"[^a-zA-Z0-9]", "", source_id)[:12].lower() if source_id else "noid"
        full_tag = f"db:{source_type}:{short_id}:{semantic_slug}".lower()

        # Dedup: if tag exists and any card holds it, skip.
        existing_tag = tag_index.get(full_tag)
        if existing_tag and not args.force:
            cards_with_tag = run_fizzy(
                ["card", "list", "--board", BOARD_ID, "--tag", existing_tag["id"], "--all"]
            ).get("data") or []
            if cards_with_tag:
                actions.append({
                    "status": "skipped",
                    "reason": "duplicate tag",
                    "tag": full_tag,
                    "existing_card_numbers": [c.get("number") for c in cards_with_tag],
                })
                continue

        # Create the card.
        desc_html = md_to_html(desc_md)
        create_args = [
            "card", "create",
            "--board", BOARD_ID,
            "--title", title[:200],
        ]
        if desc_html:
            create_args += ["--description", desc_html]

        if args.dry_run:
            actions.append({
                "status": "would_create",
                "title": title,
                "tag": full_tag,
                "description_html_preview": desc_html[:140],
            })
            continue

        created = run_fizzy(create_args).get("data") or {}
        number = created.get("number")
        if not number:
            actions.append({"status": "error", "reason": "no card number returned", "item": item})
            continue

        # Tag the card with both the marker and the dedup tag.
        for tag_name in (DAILY_BRIEF_MARKER_TAG, full_tag):
            run_fizzy(["card", "tag", str(number), "--tag", tag_name])

        # Optionally move out of the default triage column.
        col = (item.get("column") or "").strip().lower()
        target_col = ""
        if col and col not in ("maybe", "triage", ""):
            target_col = COLUMN_IDS.get(col, "")
            if not target_col:
                actions.append({
                    "status": "warning",
                    "number": number,
                    "title": title,
                    "msg": f"unknown column {col!r}; card left in Maybe?",
                })
            else:
                run_fizzy(["card", "column", str(number), "--column", target_col])

        actions.append({
            "status": "created",
            "number": number,
            "title": title,
            "tag": full_tag,
            "column": col or "maybe",
            "url": created.get("url"),
        })

    print(json.dumps({"count": len(actions), "actions": actions}, indent=2))


# ---------- sync-sheet ----------

def _gws_sheet_values(spreadsheet_id: str, sheet_range: str) -> list[list[str]]:
    """Fetch the values of a sheet range via the gws CLI."""
    params = json.dumps({"spreadsheetId": spreadsheet_id, "range": sheet_range})
    proc = subprocess.run(
        ["gws", "sheets", "spreadsheets", "values", "get", "--params", params, "--format", "json"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"gws sheets get failed:\n{proc.stderr}")
    out = proc.stdout
    # gws prints "Using keyring backend: keyring" before the JSON; strip everything before first '{'.
    idx = out.find("{")
    if idx < 0:
        sys.exit(f"gws returned no JSON:\n{out[:300]}")
    data = json.loads(out[idx:])
    return data.get("values", [])


def cmd_sync_sheet(args: argparse.Namespace) -> None:
    """Sync a Google Sheet's rows-owned-by-user into Fizzy cards.

    Defaults target the Fundraising Channels → Marketing tab, Owner=<your-name>.
    Cards are placed by sheet status: 'In Progress' → In Progress column;
    everything else → Maybe? for triage. Completed rows with no follow-up
    note are skipped.
    """
    _require(BOARD_ID, "DAILY_BRIEF_FIZZY_BOARD_ID", "the Fizzy board cards are filed to")
    _require(args.spreadsheet_id, "DAILY_BRIEF_SHEET_ID", "the Google Sheet to sync, or pass --spreadsheet-id")
    _require(args.owner, "DAILY_BRIEF_SHEET_OWNER", "the Owner-column value to match, or pass --owner")
    spreadsheet_id = args.spreadsheet_id
    tab = args.tab
    gid = args.gid
    owner_value = args.owner.strip().lower()

    rows = _gws_sheet_values(spreadsheet_id, tab)
    if not rows:
        sys.exit("sheet returned no rows")

    headers = rows[0]
    header_lc = [h.strip().lower() for h in headers]

    def col_index(*candidates: str) -> int:
        for c in candidates:
            if c.lower() in header_lc:
                return header_lc.index(c.lower())
        return -1

    owner_idx = col_index("owner")
    channel_idx = col_index("channel")
    project_idx = col_index("experiment / project", "experiment/project", "project", "experiment")
    status_idx = col_index("status")
    # "Status / Notes" — find the rightmost match since the sheet has 2 "Status" columns
    notes_idx = -1
    for i, h in enumerate(header_lc):
        if "notes" in h:
            notes_idx = i

    if owner_idx < 0:
        sys.exit(f"no 'Owner' column found in headers: {headers}")

    items = []
    debug_rows = []
    for row_num, row in enumerate(rows[1:], start=2):
        owner = (row[owner_idx] if len(row) > owner_idx else "").strip()
        if owner.lower() != owner_value:
            continue

        channel = (row[channel_idx] if channel_idx >= 0 and len(row) > channel_idx else "").strip()
        project = (row[project_idx] if project_idx >= 0 and len(row) > project_idx else "").strip()
        status = (row[status_idx] if status_idx >= 0 and len(row) > status_idx else "").strip()
        notes = (row[notes_idx] if notes_idx >= 0 and len(row) > notes_idx else "").strip()

        # Skip completed rows with no follow-up note.
        if status.lower() == "complete" and not notes:
            debug_rows.append({"row": row_num, "skipped": "complete-no-followup"})
            continue

        # Build title. Prefer "Channel: Project" when both are present and
        # distinct, so the channel context isn't lost on rows where the
        # project description is opaque on its own.
        if channel and project and project.lower() not in channel.lower():
            title = f"{channel}: {project}"
        elif project:
            title = project
        elif channel:
            title = channel
        else:
            debug_rows.append({"row": row_num, "skipped": "no-title-source"})
            continue

        # Build semantic slug. Deterministic from channel+project so future
        # sync runs dedup correctly against existing cards.
        slug_source = f"{channel}-{project}" if channel and project else (project or channel)
        semantic_slug = slugify(slug_source, max_len=50)

        # Build description.
        desc_lines = []
        if channel:
            desc_lines.append(f"**Channel:** {channel}")
        if project and channel:
            desc_lines.append(f"**Project:** {project}")
        if status:
            desc_lines.append(f"**Sheet status:** {status}")
        if notes:
            desc_lines.append(f"**Notes:** {notes}")
        sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit?gid={gid}"
        desc_lines.append(f"**Source:** [{tab} sheet → row {row_num}]({sheet_url})")
        desc_lines.append(f"**Filed:** sync-sheet on {today_date().isoformat()}")
        description_md = "\n\n".join(desc_lines)

        # Column placement.
        column = STATUS_TO_COLUMN.get(status.lower(), "maybe")

        items.append({
            "title": title[:200],
            "description_md": description_md,
            "source_type": "sheet",
            "source_id": spreadsheet_id,
            "semantic_slug": semantic_slug,
            "column": column,
        })

    if args.print_json:
        print(json.dumps(items, indent=2))
        return

    # Reuse file-cards logic: write items to stdin of a sub-invocation.
    fc_args = argparse.Namespace(dry_run=args.dry_run, force=False)
    sys.stdin = _ListAsStdin(items)
    cmd_file_cards(fc_args)


class _ListAsStdin:
    """Adapter so cmd_file_cards can read from a Python list."""

    def __init__(self, items: list[dict]) -> None:
        self._payload = json.dumps(items)

    def read(self) -> str:
        return self._payload


# ---------- archive ----------

def cmd_archive(args: argparse.Namespace) -> None:
    """Write the brief markdown to the Obsidian vault.

    Reads content from stdin or --content-file. Frontmatter is built from args.
    Path: <VAULT_DIR>/<YYYY-MM-DD>-<mode>.md
    """
    _require(VAULT_DIR, "DAILY_BRIEF_VAULT_DIR", "the Obsidian vault dir briefs are archived to")
    if args.content_file:
        body = Path(args.content_file).read_text()
    else:
        body = sys.stdin.read()

    d = today_date(args.date)
    people = [p.strip() for p in (args.people or "").split(",") if p.strip()]
    projects = [p.strip() for p in (args.projects or "").split(",") if p.strip()]

    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    path = VAULT_DIR / f"{d.isoformat()}-{args.mode}.md"

    # Build YAML frontmatter
    def yaml_list(items: list[str]) -> str:
        if not items:
            return "[]"
        return "[" + ", ".join(items) + "]"

    frontmatter = (
        "---\n"
        f"date: {d.isoformat()}\n"
        f"mode: {args.mode}\n"
        f"people: {yaml_list(people)}\n"
        f"projects: {yaml_list(projects)}\n"
        "---\n\n"
    )

    path.write_text(frontmatter + body.strip() + "\n")
    print(json.dumps({"path": str(path), "bytes": len(frontmatter) + len(body)}, indent=2))


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser(description="daily-brief helpers")
    sub = p.add_subparsers(dest="cmd", required=True)

    pdw = sub.add_parser("day-window", help="Compute the date/time window for a mode")
    pdw.add_argument("mode", choices=["recap", "prep"])
    pdw.add_argument("--date", help="YYYY-MM-DD (default: today in America/New_York)")
    pdw.set_defaults(func=cmd_day_window)

    poc = sub.add_parser("open-cards", help="List open Fizzy cards on the Acme board")
    poc.set_defaults(func=cmd_open_cards)

    pfc = sub.add_parser("file-cards", help="Create Fizzy cards from JSON array on stdin (Tier 1)")
    pfc.add_argument("--dry-run", action="store_true", help="Print intended actions; don't create")
    pfc.add_argument("--force", action="store_true", help="Ignore dedup tag collisions")
    pfc.set_defaults(func=cmd_file_cards)

    pss = sub.add_parser("sync-sheet", help="Sync rows owned by you in a Google Sheet to Fizzy cards")
    pss.add_argument("--spreadsheet-id", default=SHEET_ID_DEFAULT)
    pss.add_argument("--tab", default=SHEET_TAB_DEFAULT, help="Sheet tab name")
    pss.add_argument("--gid", default=SHEET_GID_DEFAULT, help="Sheet gid for URL link in card description")
    pss.add_argument("--owner", default=SHEET_OWNER_DEFAULT, help="Owner column value to match (case-insensitive)")
    pss.add_argument("--dry-run", action="store_true", help="Print intended actions; don't create")
    pss.add_argument("--print-json", action="store_true", help="Print the items JSON and exit (don't file)")
    pss.set_defaults(func=cmd_sync_sheet)

    pa = sub.add_parser("archive", help="Write the brief markdown into the Obsidian vault")
    pa.add_argument("mode", choices=["recap", "prep"])
    pa.add_argument("--date", help="YYYY-MM-DD (default: today)")
    pa.add_argument("--people", help="Comma-separated people names for frontmatter")
    pa.add_argument("--projects", help="Comma-separated project names for frontmatter")
    pa.add_argument("--content-file", help="Read body from file (else stdin)")
    pa.set_defaults(func=cmd_archive)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
