#!/usr/bin/env python3
"""ClickUp task tool (REST API v2).

Subcommands:
  whoami     - Authed user (id, username, email). Sanity check.
  workspaces - List workspaces ("teams" in the API) the token can see.
  tree       - List spaces / folders / lists under a workspace (navigation).
  tasks      - Filtered task search. By default narrows to tasks assigned to
               the authed user (assignees=me). Opt out with --no-default-me.
  get        - Fetch one task in full (description, subtasks, custom fields).
  comments   - List comments on a task.
  create     - Create a task in a list.            (write; needs --yes)
  update     - Update a task (status/assignee/...). (write; needs --yes)
  comment    - Post a comment on a task.            (write; needs --yes)

Writes are guarded: create/update/comment print the intended endpoint +
payload and do nothing unless --yes is passed (--dry-run forces preview).

Auth: reads CLICKUP_TOKEN (personal token, starts with pk_) from env, plus
optional CLICKUP_TEAM_ID default. Invoke via the `clickup` bash wrapper,
which loads them from ~/.config/clickup/.env:

  clickup tasks --status "in progress"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API_BASE = "https://api.clickup.com/api/v2"


def _token() -> str:
    tok = os.environ.get("CLICKUP_TOKEN")
    if not tok:
        sys.exit("CLICKUP_TOKEN not set. Run via the `clickup` wrapper.")
    return tok


def request(method: str, path: str, params: list[tuple] | None = None,
            body: dict | None = None) -> dict:
    """Call the ClickUp API. `params` is a list of (key, value) tuples so that
    repeated array keys like assignees[] survive. Returns parsed JSON."""
    url = API_BASE + path
    if params:
        # Build the query manually: ClickUp wants literal `key[]=v`, not the
        # percent-encoded brackets urlencode would emit.
        qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params)
        url = f"{url}?{qs}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": _token(),       # personal token: raw, no "Bearer"
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode(errors='replace')}")


# --- helpers ---------------------------------------------------------------

def my_user_id() -> int:
    return int(request("GET", "/user")["user"]["id"])


def resolve_team(arg: str | None) -> str:
    """--team flag > CLICKUP_TEAM_ID env > auto (only if exactly one)."""
    if arg:
        return arg
    env = os.environ.get("CLICKUP_TEAM_ID")
    if env:
        return env
    teams = request("GET", "/team").get("teams") or []
    if len(teams) == 1:
        return teams[0]["id"]
    names = ", ".join(f'{t["id"]}={t["name"]}' for t in teams)
    sys.exit(
        "Multiple workspaces; pass --team <id> or set CLICKUP_TEAM_ID. "
        f"Options: {names}"
    )


def to_epoch_ms(value: str) -> int:
    """Accept epoch-ms (all digits) or an ISO 8601 / YYYY-MM-DD date."""
    if value.isdigit():
        return int(value)
    s = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def slim_task(t: dict) -> dict:
    return {
        "id": t.get("id"),
        "name": t.get("name"),
        "status": (t.get("status") or {}).get("status"),
        "assignees": [a.get("username") for a in t.get("assignees") or []],
        "due_date": t.get("due_date"),
        "list": (t.get("list") or {}).get("name"),
        "url": t.get("url"),
    }


def preview_or_go(args, method: str, path: str, body: dict, label: str) -> None:
    """Writes print their payload and stop unless --yes is set."""
    if getattr(args, "dry_run", False) or not getattr(args, "yes", False):
        print(json.dumps(
            {"would": label, "method": method, "path": path, "body": body},
            indent=2,
        ))
        if not getattr(args, "dry_run", False):
            print("\n# not sent — re-run with --yes to execute", file=sys.stderr)
        return
    print(json.dumps(request(method, path, body=body), indent=2))


# --- read commands ---------------------------------------------------------

def cmd_whoami(args) -> None:
    print(json.dumps(request("GET", "/user")["user"], indent=2))


def cmd_workspaces(args) -> None:
    teams = request("GET", "/team").get("teams") or []
    out = [{"id": t["id"], "name": t["name"],
            "members": len(t.get("members") or [])} for t in teams]
    print(json.dumps({"count": len(out), "workspaces": out}, indent=2))


def cmd_tree(args) -> None:
    team = resolve_team(args.team)
    spaces = request("GET", f"/team/{team}/space",
                     [("archived", "false")]).get("spaces") or []
    tree = []
    for sp in spaces:
        node = {"space_id": sp["id"], "space": sp["name"],
                "folders": [], "folderless_lists": []}
        folders = request("GET", f"/space/{sp['id']}/folder",
                          [("archived", "false")]).get("folders") or []
        for f in folders:
            node["folders"].append({
                "folder_id": f["id"], "folder": f["name"],
                "lists": [{"list_id": l["id"], "list": l["name"]}
                          for l in f.get("lists") or []],
            })
        floose = request("GET", f"/space/{sp['id']}/list",
                         [("archived", "false")]).get("lists") or []
        node["folderless_lists"] = [{"list_id": l["id"], "list": l["name"]}
                                    for l in floose]
        tree.append(node)
    print(json.dumps({"workspace": team, "spaces": tree}, indent=2))


def cmd_tasks(args) -> None:
    team = resolve_team(args.team)
    params: list[tuple] = []

    if not args.no_default_me:
        ids = []
        for a in (args.assignee or ["me"]):
            ids.append(my_user_id() if a == "me" else a)
        for i in ids:
            params.append(("assignees[]", i))
    elif args.assignee:
        for a in args.assignee:
            params.append(("assignees[]", my_user_id() if a == "me" else a))

    for s in args.status or []:
        params.append(("statuses[]", s))
    for t in args.tag or []:
        params.append(("tags[]", t))
    for lid in args.list or []:
        params.append(("list_ids[]", lid))
    if args.due_before:
        params.append(("due_date_lt", to_epoch_ms(args.due_before)))
    if args.due_after:
        params.append(("due_date_gt", to_epoch_ms(args.due_after)))
    if args.include_closed:
        params.append(("include_closed", "true"))
    if args.subtasks:
        params.append(("subtasks", "true"))
    params.append(("order_by", args.order_by))
    if args.reverse:
        params.append(("reverse", "true"))

    collected: list[dict] = []
    page = 0
    while len(collected) < args.limit:
        page_params = params + [("page", page)]
        tasks = request("GET", f"/team/{team}/task",
                        page_params).get("tasks") or []
        if not tasks:
            break
        collected.extend(tasks)
        if len(tasks) < 100:      # last page
            break
        page += 1

    collected = collected[:args.limit]
    if args.name_contains:
        needle = args.name_contains.lower()
        collected = [t for t in collected
                     if needle in (t.get("name") or "").lower()]
    out = collected if args.raw else [slim_task(t) for t in collected]
    print(json.dumps({"count": len(out), "tasks": out}, indent=2))


def cmd_get(args) -> None:
    params = [("include_subtasks", "true")]
    if args.custom_task_id:
        params += [("custom_task_ids", "true"),
                   ("team_id", resolve_team(args.team))]
    print(json.dumps(request("GET", f"/task/{args.task_id}", params), indent=2))


def cmd_comments(args) -> None:
    data = request("GET", f"/task/{args.task_id}/comment")
    print(json.dumps(data, indent=2))


# --- write commands --------------------------------------------------------

def cmd_create(args) -> None:
    body: dict = {"name": args.name}
    if args.description:
        body["description"] = args.description
    if args.status:
        body["status"] = args.status
    if args.priority:
        body["priority"] = args.priority
    if args.due:
        body["due_date"] = to_epoch_ms(args.due)
    if args.assignee:
        body["assignees"] = [my_user_id() if a == "me" else int(a)
                             for a in args.assignee]
    if args.tag:
        body["tags"] = args.tag
    preview_or_go(args, "POST", f"/list/{args.list}/task", body, "create task")


def cmd_update(args) -> None:
    body: dict = {}
    if args.name:
        body["name"] = args.name
    if args.description:
        body["description"] = args.description
    if args.status:
        body["status"] = args.status
    if args.priority:
        body["priority"] = args.priority
    if args.due:
        body["due_date"] = to_epoch_ms(args.due)
    add = [my_user_id() if a == "me" else int(a) for a in args.add_assignee or []]
    rem = [my_user_id() if a == "me" else int(a) for a in args.rem_assignee or []]
    if add or rem:
        body["assignees"] = {"add": add, "rem": rem}
    if not body:
        sys.exit("update: nothing to change. Pass at least one field.")
    preview_or_go(args, "PUT", f"/task/{args.task_id}", body, "update task")


def cmd_comment(args) -> None:
    body: dict = {"comment_text": args.text, "notify_all": args.notify_all}
    if args.assignee:
        body["assignee"] = my_user_id() if args.assignee == "me" else int(args.assignee)
    preview_or_go(args, "POST", f"/task/{args.task_id}/comment", body,
                  "post comment")


# --- argparse --------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(prog="clickup", description="ClickUp task tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("whoami", help="Authed user").set_defaults(func=cmd_whoami)
    sub.add_parser("workspaces", help="List workspaces").set_defaults(func=cmd_workspaces)

    pt = sub.add_parser("tree", help="Spaces / folders / lists for a workspace")
    pt.add_argument("--team", help="Workspace id (else CLICKUP_TEAM_ID / auto)")
    pt.set_defaults(func=cmd_tree)

    pk = sub.add_parser("tasks", help="Filtered task search (defaults to me)")
    pk.add_argument("--team", help="Workspace id (else CLICKUP_TEAM_ID / auto)")
    pk.add_argument("--assignee", action="append",
                    help="User id or 'me' (repeatable). Overrides the default.")
    pk.add_argument("--no-default-me", action="store_true",
                    help="Don't filter to the authed user (search all assignees)")
    pk.add_argument("--status", action="append", help="Status name (repeatable)")
    pk.add_argument("--tag", action="append", help="Tag name (repeatable)")
    pk.add_argument("--list", action="append", help="Restrict to list id (repeatable)")
    pk.add_argument("--due-before", help="Epoch ms or ISO date")
    pk.add_argument("--due-after", help="Epoch ms or ISO date")
    pk.add_argument("--name-contains", help="Client-side filter on task name")
    pk.add_argument("--include-closed", action="store_true")
    pk.add_argument("--subtasks", action="store_true", help="Include subtasks")
    pk.add_argument("--order-by", default="updated",
                    help="created | updated | due_date (default updated)")
    pk.add_argument("--reverse", action="store_true")
    pk.add_argument("--raw", action="store_true", help="Full task JSON, not slim")
    pk.add_argument("--limit", type=int, default=100)
    pk.set_defaults(func=cmd_tasks)

    pg = sub.add_parser("get", help="Fetch one task in full")
    pg.add_argument("task_id")
    pg.add_argument("--custom-task-id", action="store_true",
                    help="task_id is a custom id (requires team)")
    pg.add_argument("--team", help="Workspace id (for --custom-task-id)")
    pg.set_defaults(func=cmd_get)

    pc = sub.add_parser("comments", help="List comments on a task")
    pc.add_argument("task_id")
    pc.set_defaults(func=cmd_comments)

    pcr = sub.add_parser("create", help="Create a task (write)")
    pcr.add_argument("--list", required=True, help="List id to create in")
    pcr.add_argument("--name", required=True)
    pcr.add_argument("--description")
    pcr.add_argument("--status")
    pcr.add_argument("--priority", type=int, choices=[1, 2, 3, 4],
                     help="1 urgent .. 4 low")
    pcr.add_argument("--due", help="Epoch ms or ISO date")
    pcr.add_argument("--assignee", action="append", help="User id or 'me' (repeatable)")
    pcr.add_argument("--tag", action="append")
    pcr.add_argument("--yes", action="store_true", help="Actually send (else preview)")
    pcr.add_argument("--dry-run", action="store_true", help="Force preview only")
    pcr.set_defaults(func=cmd_create)

    pu = sub.add_parser("update", help="Update a task (write)")
    pu.add_argument("task_id")
    pu.add_argument("--name")
    pu.add_argument("--description")
    pu.add_argument("--status")
    pu.add_argument("--priority", type=int, choices=[1, 2, 3, 4])
    pu.add_argument("--due", help="Epoch ms or ISO date")
    pu.add_argument("--add-assignee", action="append", help="User id or 'me'")
    pu.add_argument("--rem-assignee", action="append", help="User id or 'me'")
    pu.add_argument("--yes", action="store_true", help="Actually send (else preview)")
    pu.add_argument("--dry-run", action="store_true", help="Force preview only")
    pu.set_defaults(func=cmd_update)

    pcm = sub.add_parser("comment", help="Post a comment on a task (write)")
    pcm.add_argument("task_id")
    pcm.add_argument("--text", required=True)
    pcm.add_argument("--assignee", help="Assign the comment to a user id or 'me'")
    pcm.add_argument("--notify-all", action="store_true")
    pcm.add_argument("--yes", action="store_true", help="Actually send (else preview)")
    pcm.add_argument("--dry-run", action="store_true", help="Force preview only")
    pcm.set_defaults(func=cmd_comment)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
