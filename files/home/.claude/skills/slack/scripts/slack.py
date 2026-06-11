#!/usr/bin/env python3
"""Slack workspace search/read tool.

Subcommands:
  search    - Search messages. Defaults to messages from you (rewrites to
              `from:<@USER_ID>`). Override with --no-default-me.
  channels  - List channels the user is a member of.
  history   - Fetch recent messages from a channel (by id or name).
  replies   - Fetch a thread by channel + parent ts.
  recap     - Full conversations (both sides) for a given day — every channel
              or DM you posted in, with names resolved.
  whoami    - Print authed user id + workspace.

Auth: reads SLACK_USER_TOKEN from env (must start with xoxp-). Invoke via the
`slack` bash wrapper, which loads the token from ~/.config/slack/.env.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

API = "https://slack.com/api"

_ME_CACHE: dict | None = None
_USER_CACHE: dict[str, dict] = {}
_CHANNEL_CACHE: dict[str, dict] = {}


def me() -> dict:
    """Return cached auth.test result (user_id, user, team)."""
    global _ME_CACHE
    if _ME_CACHE is None:
        _ME_CACHE = call("auth.test")
    return _ME_CACHE


def user_info(user_id: str) -> dict:
    if user_id not in _USER_CACHE:
        try:
            _USER_CACHE[user_id] = (call("users.info", {"user": user_id}).get("user") or {})
        except SystemExit:
            _USER_CACHE[user_id] = {}
    return _USER_CACHE[user_id]


def user_label(user_id: str | None) -> str:
    if not user_id:
        return "(unknown)"
    u = user_info(user_id)
    profile = u.get("profile") or {}
    return (
        profile.get("display_name")
        or u.get("real_name")
        or u.get("name")
        or user_id
    )


def conversation_info(channel_id: str) -> dict:
    if channel_id not in _CHANNEL_CACHE:
        try:
            _CHANNEL_CACHE[channel_id] = (call("conversations.info", {"channel": channel_id}).get("channel") or {})
        except SystemExit:
            _CHANNEL_CACHE[channel_id] = {}
    return _CHANNEL_CACHE[channel_id]


def channel_label(channel_id: str) -> str:
    info = conversation_info(channel_id)
    if info.get("is_im"):
        return f"DM with {user_label(info.get('user'))}"
    if info.get("is_mpim"):
        return f"Group DM: {info.get('name') or channel_id}"
    name = info.get("name")
    return f"#{name}" if name else channel_id


def call(method: str, params: dict | None = None, post: bool = False) -> dict:
    token = os.environ.get("SLACK_USER_TOKEN")
    if not token:
        sys.exit("SLACK_USER_TOKEN not set. Run via the `slack` wrapper.")
    params = {k: v for k, v in (params or {}).items() if v is not None}
    url = f"{API}/{method}"
    headers = {"Authorization": f"Bearer {token}"}
    if post:
        data = urllib.parse.urlencode(params).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    else:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {method}: {e.read().decode(errors='replace')}")
    if not body.get("ok"):
        err = body.get("error", "unknown_error")
        needed = body.get("needed")
        provided = body.get("provided")
        extra = ""
        if needed:
            extra = f" (needed scope: {needed}; provided: {provided})"
        sys.exit(f"Slack API error on {method}: {err}{extra}")
    return body


def paginate(method: str, params: dict, key: str, max_pages: int = 20):
    cursor = None
    pages = 0
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        body = call(method, p)
        for item in body.get(key, []):
            yield item
        cursor = (body.get("response_metadata") or {}).get("next_cursor") or ""
        pages += 1
        if not cursor or pages >= max_pages:
            return


def resolve_channel(name_or_id: str) -> str:
    """Accept #channel-name, channel-name, or C0XXXX id. Returns channel id."""
    # Slack ids: start with C/D/G, length >= 9, all uppercase alnum.
    if (
        len(name_or_id) >= 9
        and name_or_id[0] in ("C", "D", "G")
        and all(c.isdigit() or (c.isalpha() and c.isupper()) for c in name_or_id)
    ):
        return name_or_id
    needle = name_or_id.lstrip("#").lower()
    for ch in paginate(
        "conversations.list",
        {"types": "public_channel,private_channel,mpim,im", "limit": 1000, "exclude_archived": "true"},
        "channels",
    ):
        if (ch.get("name") or "").lower() == needle:
            return ch["id"]
    sys.exit(f"channel not found: {name_or_id}")


def cmd_whoami(args: argparse.Namespace) -> None:
    body = call("auth.test")
    print(json.dumps({
        "user": body.get("user"),
        "user_id": body.get("user_id"),
        "team": body.get("team"),
        "team_id": body.get("team_id"),
        "url": body.get("url"),
    }, indent=2))


def cmd_channels(args: argparse.Namespace) -> None:
    types = args.types
    out = []
    for ch in paginate(
        "users.conversations" if args.member_only else "conversations.list",
        {"types": types, "limit": 1000, "exclude_archived": "true" if not args.include_archived else "false"},
        "channels",
    ):
        out.append({
            "id": ch.get("id"),
            "name": ch.get("name") or ch.get("user"),
            "is_private": ch.get("is_private"),
            "is_member": ch.get("is_member"),
            "is_archived": ch.get("is_archived"),
            "num_members": ch.get("num_members"),
            "topic": (ch.get("topic") or {}).get("value"),
        })
    out.sort(key=lambda c: (c.get("name") or "").lower())
    print(json.dumps({"count": len(out), "channels": out}, indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    parts = []
    if args.query:
        parts.append(args.query)
    if not args.no_default_me:
        parts.append(f"from:<@{me()['user_id']}>")
    if args.in_channel:
        parts.append(f"in:#{args.in_channel.lstrip('#')}")
    if args.after:
        parts.append(f"after:{args.after}")
    if args.before:
        parts.append(f"before:{args.before}")
    q = " ".join(parts).strip()
    if not q:
        sys.exit("search requires --query or a filter (--in-channel/--after/--before), or use --no-default-me with a query.")

    body = call("search.messages", {
        "query": q,
        "count": str(args.limit),
        "sort": args.sort,
        "sort_dir": args.sort_dir,
        "highlight": "false",
    })
    msgs = (body.get("messages") or {}).get("matches") or []
    total = (body.get("messages") or {}).get("total")
    results = []
    for m in msgs:
        ch = m.get("channel") or {}
        results.append({
            "ts": m.get("ts"),
            "user": m.get("user"),
            "username": m.get("username"),
            "channel_id": ch.get("id"),
            "channel_name": ch.get("name"),
            "permalink": m.get("permalink"),
            "text": m.get("text"),
        })
    print(json.dumps({"query": q, "total": total, "count": len(results), "messages": results}, indent=2))


def cmd_history(args: argparse.Namespace) -> None:
    channel_id = resolve_channel(args.channel)
    params = {"channel": channel_id, "limit": str(args.limit)}
    if args.oldest:
        params["oldest"] = args.oldest
    if args.latest:
        params["latest"] = args.latest
    body = call("conversations.history", params)
    msgs = body.get("messages") or []
    if args.mine_only:
        me = call("auth.test").get("user_id")
        msgs = [m for m in msgs if m.get("user") == me]
    print(json.dumps({"channel": channel_id, "count": len(msgs), "messages": msgs}, indent=2))


def cmd_replies(args: argparse.Namespace) -> None:
    channel_id = resolve_channel(args.channel)
    body = call("conversations.replies", {"channel": channel_id, "ts": args.ts, "limit": str(args.limit)})
    msgs = body.get("messages") or []
    print(json.dumps({"channel": channel_id, "thread_ts": args.ts, "count": len(msgs), "messages": msgs}, indent=2))


def _day_bounds(date_str: str | None, tz_name: str | None) -> tuple[_dt.datetime, _dt.datetime, str]:
    """Return (start_dt, end_dt, iso_date) for the given day in the given tz.

    Defaults: today, system local tz.
    """
    if tz_name:
        tz = ZoneInfo(tz_name)
    else:
        tz = _dt.datetime.now().astimezone().tzinfo
    if date_str:
        d = _dt.date.fromisoformat(date_str)
    else:
        d = _dt.datetime.now(tz).date()
    start = _dt.datetime.combine(d, _dt.time(0, 0), tzinfo=tz)
    end = start + _dt.timedelta(days=1)
    return start, end, d.isoformat()


def _fmt_ts(ts_str: str, tz) -> str:
    secs = float(ts_str)
    return _dt.datetime.fromtimestamp(secs, tz).strftime("%H:%M")


def cmd_recap(args: argparse.Namespace) -> None:
    start, end, iso = _day_bounds(args.date, args.tz)
    tz = start.tzinfo
    my_id = me()["user_id"]

    # Step 1: find every channel I posted in that day, via search.
    channel_ids: set[str] = set()
    cursor_query = f"from:<@{my_id}> on:{iso}"
    body = call("search.messages", {
        "query": cursor_query,
        "count": "100",
        "sort": "timestamp",
        "sort_dir": "asc",
        "highlight": "false",
    })
    for m in (body.get("messages") or {}).get("matches") or []:
        cid = (m.get("channel") or {}).get("id")
        if cid:
            channel_ids.add(cid)

    if not channel_ids:
        print(json.dumps({"date": iso, "tz": str(tz), "conversations": []}, indent=2))
        if not args.json:
            print(f"\nNo messages sent on {iso}.")
        return

    # Step 2: pull conversations.history for each channel within the day window.
    oldest = f"{start.timestamp():.6f}"
    latest = f"{end.timestamp():.6f}"
    conversations = []
    for cid in channel_ids:
        msgs: list[dict] = []
        cursor = None
        while True:
            params = {"channel": cid, "limit": "200", "oldest": oldest, "latest": latest, "inclusive": "true"}
            if cursor:
                params["cursor"] = cursor
            resp = call("conversations.history", params)
            msgs.extend(resp.get("messages") or [])
            cursor = (resp.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor:
                break
        msgs.sort(key=lambda m: float(m.get("ts", "0")))
        conversations.append({
            "channel_id": cid,
            "label": channel_label(cid),
            "message_count": len(msgs),
            "messages": [
                {
                    "ts": m.get("ts"),
                    "time": _fmt_ts(m.get("ts", "0"), tz),
                    "user_id": m.get("user"),
                    "user": user_label(m.get("user")),
                    "text": m.get("text", ""),
                    "thread_ts": m.get("thread_ts"),
                    "reply_count": m.get("reply_count"),
                }
                for m in msgs
            ],
        })
    conversations.sort(key=lambda c: c["label"])

    if args.json:
        print(json.dumps({"date": iso, "tz": str(tz), "conversations": conversations}, indent=2))
        return

    # Text output
    print(f"Recap for {iso} ({tz})")
    print(f"{len(conversations)} conversation(s)\n")
    for conv in conversations:
        print(f"━━━ {conv['label']}  ({conv['message_count']} messages) ━━━")
        for m in conv["messages"]:
            mark = " ← you" if m["user_id"] == my_id else ""
            text = m["text"].replace("\n", "\n        ")
            print(f"  [{m['time']}] {m['user']}{mark}: {text}")
            if m.get("reply_count"):
                print(f"        ↳ thread with {m['reply_count']} replies (use: slack replies {conv['channel_id']} {m['ts']})")
        print()


def main() -> None:
    p = argparse.ArgumentParser(description="Slack workspace read/search tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("whoami", help="Show authed user + workspace")
    pw.set_defaults(func=cmd_whoami)

    pc = sub.add_parser("channels", help="List channels (defaults to channels you're a member of)")
    pc.add_argument("--types", default="public_channel,private_channel,mpim,im")
    pc.add_argument("--include-archived", action="store_true")
    pc.add_argument("--all", dest="member_only", action="store_false",
                    help="List all workspace channels, not just ones you're a member of")
    pc.set_defaults(member_only=True, func=cmd_channels)

    ps = sub.add_parser("search", help="Search messages (defaults to from:@me)")
    ps.add_argument("--query", "-q", default="", help="Slack search query (e.g. 'deploy', 'has:link')")
    ps.add_argument("--no-default-me", action="store_true",
                    help="Don't auto-prepend from:@me")
    ps.add_argument("--in-channel", help="Restrict to a channel (name without #)")
    ps.add_argument("--after", help="YYYY-MM-DD")
    ps.add_argument("--before", help="YYYY-MM-DD")
    ps.add_argument("--limit", type=int, default=50)
    ps.add_argument("--sort", choices=["score", "timestamp"], default="timestamp")
    ps.add_argument("--sort-dir", choices=["asc", "desc"], default="desc")
    ps.set_defaults(func=cmd_search)

    ph = sub.add_parser("history", help="Recent messages in a channel")
    ph.add_argument("channel", help="Channel id (C…) or name")
    ph.add_argument("--limit", type=int, default=50)
    ph.add_argument("--oldest", help="Unix ts (e.g. 1700000000.000000)")
    ph.add_argument("--latest", help="Unix ts")
    ph.add_argument("--mine-only", action="store_true", help="Filter to messages from you")
    ph.set_defaults(func=cmd_history)

    pr = sub.add_parser("replies", help="Fetch a thread")
    pr.add_argument("channel", help="Channel id or name")
    pr.add_argument("ts", help="Parent message ts")
    pr.add_argument("--limit", type=int, default=200)
    pr.set_defaults(func=cmd_replies)

    prc = sub.add_parser("recap", help="Full conversations from a given day (every channel/DM you posted in)")
    prc.add_argument("--date", help="YYYY-MM-DD (default: today, in --tz)")
    prc.add_argument("--tz", help="IANA tz name, e.g. America/New_York (default: system local)")
    prc.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    prc.set_defaults(func=cmd_recap)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
