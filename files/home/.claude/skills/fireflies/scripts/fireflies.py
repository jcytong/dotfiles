#!/usr/bin/env python3
"""Fireflies transcript extraction tool.

Subcommands:
  list  - Find meetings by date range / title / participant. By default
          filters to meetings where any email in
          FIREFLIES_DEFAULT_PARTICIPANTS (comma-separated env var) is an
          attendee. If that env var is unset/empty, no default filter is
          applied.
  get   - Fetch a transcript by ID (summary, action items, sentences).
  live  - List currently active (in-progress) meetings. Metadata only;
          no live transcript content is available via the API.

Auth: reads FIREFLIES_API_KEY from env. Invoke via the `fireflies` bash
wrapper, which loads the key (and FIREFLIES_DEFAULT_PARTICIPANTS) from
~/.config/fireflies/.env:

  fireflies list --from 2026-05-25T13:00:00Z ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API_URL = "https://api.fireflies.ai/graphql"


def default_emails() -> list[str]:
    """Read default participant emails from FIREFLIES_DEFAULT_PARTICIPANTS.

    Comma-separated list of emails. If unset or empty, no default participant
    filter is applied (the skill behaves as `--no-default-filter`).
    Configured in ~/.config/fireflies/.env so the skill is shareable.
    """
    raw = os.environ.get("FIREFLIES_DEFAULT_PARTICIPANTS", "")
    return [e.strip() for e in raw.split(",") if e.strip()]

LIST_FIELDS = "id title dateString date duration host_email organizer_email participants"
SUMMARY_FIELDS = "overview short_summary bullet_gist action_items keywords topics_discussed shorthand_bullet"
GET_FIELDS_NO_SENTENCES = f"id title dateString date duration host_email organizer_email participants summary {{ {SUMMARY_FIELDS} }}"
GET_FIELDS_FULL = f"{GET_FIELDS_NO_SENTENCES} sentences {{ index speaker_name speaker_id text start_time end_time }}"
LIVE_FIELDS = "id title organizer_email meeting_link start_time end_time state privacy"


def graphql(query: str, variables: dict | None = None) -> dict:
    key = os.environ.get("FIREFLIES_API_KEY")
    if not key:
        sys.exit("FIREFLIES_API_KEY not set. Run via the `fireflies` wrapper.")
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode(errors='replace')}")
    if "errors" in body:
        sys.exit(f"GraphQL errors: {json.dumps(body['errors'], indent=2)}")
    return body["data"]


def build_filter_block(
    from_date: str | None,
    to_date: str | None,
    title: str | None,
    participant: str | None,
    limit: int,
) -> str:
    args = []
    if from_date:
        args.append(f'fromDate: "{from_date}"')
    if to_date:
        args.append(f'toDate: "{to_date}"')
    if title:
        args.append(f'title: "{title}"')
    if participant:
        args.append(f'participant_email: "{participant}"')
    args.append(f"limit: {limit}")
    return ", ".join(args)


def cmd_list(args: argparse.Namespace) -> None:
    if args.no_default_filter or args.participant:
        emails = [args.participant] if args.participant else [None]
    else:
        emails = default_emails() or [None]

    seen: dict[str, dict] = {}
    for email in emails:
        block = build_filter_block(
            args.from_date, args.to_date, args.title, email, args.limit
        )
        query = f"query {{ transcripts({block}) {{ {LIST_FIELDS} }} }}"
        data = graphql(query)
        for t in data.get("transcripts") or []:
            seen[t["id"]] = t

    results = sorted(seen.values(), key=lambda t: t.get("date") or 0, reverse=True)
    print(json.dumps({"count": len(results), "transcripts": results}, indent=2))


def cmd_get(args: argparse.Namespace) -> None:
    fields = GET_FIELDS_NO_SENTENCES if args.no_sentences else GET_FIELDS_FULL
    query = f'query {{ transcript(id: "{args.id}") {{ {fields} }} }}'
    data = graphql(query)
    transcript = data.get("transcript")
    if not transcript:
        sys.exit(f"No transcript found with id={args.id}")

    if args.text_only:
        sentences = transcript.get("sentences") or []
        for s in sentences:
            speaker = s.get("speaker_name") or f"Speaker {s.get('speaker_id')}"
            print(f"[{s.get('start_time'):>7.2f}s] {speaker}: {s.get('text')}")
        return

    print(json.dumps(transcript, indent=2))


def cmd_live(args: argparse.Namespace) -> None:
    query = f"query {{ active_meetings {{ {LIVE_FIELDS} }} }}"
    data = graphql(query)
    meetings = data.get("active_meetings") or []
    print(json.dumps({"count": len(meetings), "active_meetings": meetings}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="fireflies", description="Fireflies transcript extraction")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="List/search transcripts")
    pl.add_argument("--from", dest="from_date", help="ISO timestamp, e.g. 2026-05-25T13:00:00Z")
    pl.add_argument("--to", dest="to_date", help="ISO timestamp, e.g. 2026-05-25T16:00:00Z")
    pl.add_argument("--title", help="Title substring (server-side filter)")
    pl.add_argument(
        "--participant",
        help="Override default participant filter with a single email",
    )
    pl.add_argument(
        "--no-default-filter",
        action="store_true",
        help="Disable the FIREFLIES_DEFAULT_PARTICIPANTS filter (search all transcripts)",
    )
    pl.add_argument("--limit", type=int, default=50)
    pl.set_defaults(func=cmd_list)

    pg = sub.add_parser("get", help="Fetch full transcript by id")
    pg.add_argument("id")
    pg.add_argument(
        "--no-sentences",
        action="store_true",
        help="Skip sentences (summary + metadata only). Faster, smaller payload.",
    )
    pg.add_argument(
        "--text-only",
        action="store_true",
        help="Print sentence-by-sentence text only (implies sentences are fetched)",
    )
    pg.set_defaults(func=cmd_get)

    pv = sub.add_parser("live", help="List active (in-progress) meetings")
    pv.set_defaults(func=cmd_live)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
