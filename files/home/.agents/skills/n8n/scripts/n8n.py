#!/usr/bin/env python3
"""n8n Cloud API CLI.

Subcommands:
  workflows list      - List/search workflows
  workflows get       - Get a workflow by ID
  workflows activate  - Activate a workflow
  workflows deactivate - Deactivate a workflow
  executions list     - List executions (optionally filtered by workflow)
  executions get      - Get a single execution
  credentials list    - List available credentials
  tags list           - List all tags
  variables list      - List all variables

Auth: reads N8N_API_KEY and N8N_BASE_URL from env. Invoke via the `n8n-api`
bash wrapper, which loads credentials from ~/.config/n8n/.env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def base_url() -> str:
    url = os.environ.get("N8N_BASE_URL", "").rstrip("/")
    if not url:
        sys.exit("N8N_BASE_URL not set. Run via the `n8n-api` wrapper.")
    return url


def api_request(
    method: str,
    path: str,
    params: dict | None = None,
    body: dict | None = None,
) -> dict | list:
    key = os.environ.get("N8N_API_KEY")
    if not key:
        sys.exit("N8N_API_KEY not set. Run via the `n8n-api` wrapper.")

    url = base_url() + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})

    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "X-N8N-API-KEY": key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        sys.exit(f"HTTP {e.code}: {body_text}")


# ── workflows ────────────────────────────────────────────────────────────────

def cmd_workflows_list(args: argparse.Namespace) -> None:
    params: dict = {}
    if args.active is not None:
        params["active"] = str(args.active).lower()
    if args.tags:
        params["tags"] = args.tags
    if args.limit:
        params["limit"] = args.limit
    if args.cursor:
        params["cursor"] = args.cursor

    data = api_request("GET", "/workflows", params=params)
    print(json.dumps(data, indent=2))


def cmd_workflows_get(args: argparse.Namespace) -> None:
    data = api_request("GET", f"/workflows/{args.id}")
    print(json.dumps(data, indent=2))


def cmd_workflows_activate(args: argparse.Namespace) -> None:
    data = api_request("POST", f"/workflows/{args.id}/activate")
    print(json.dumps(data, indent=2))


def cmd_workflows_deactivate(args: argparse.Namespace) -> None:
    data = api_request("POST", f"/workflows/{args.id}/deactivate")
    print(json.dumps(data, indent=2))


# ── executions ───────────────────────────────────────────────────────────────

def cmd_executions_list(args: argparse.Namespace) -> None:
    params: dict = {}
    if args.workflow_id:
        params["workflowId"] = args.workflow_id
    if args.status:
        params["status"] = args.status
    if args.limit:
        params["limit"] = args.limit
    if args.cursor:
        params["cursor"] = args.cursor
    if args.include_data:
        params["includeData"] = "true"

    data = api_request("GET", "/executions", params=params)
    print(json.dumps(data, indent=2))


def cmd_executions_get(args: argparse.Namespace) -> None:
    params: dict = {}
    if args.include_data:
        params["includeData"] = "true"
    data = api_request("GET", f"/executions/{args.id}", params=params or None)
    print(json.dumps(data, indent=2))


def cmd_executions_delete(args: argparse.Namespace) -> None:
    data = api_request("DELETE", f"/executions/{args.id}")
    print(json.dumps(data, indent=2))


# ── credentials ──────────────────────────────────────────────────────────────

def cmd_credentials_list(args: argparse.Namespace) -> None:
    data = api_request("GET", "/credentials")
    print(json.dumps(data, indent=2))


# ── tags ─────────────────────────────────────────────────────────────────────

def cmd_tags_list(args: argparse.Namespace) -> None:
    data = api_request("GET", "/tags")
    print(json.dumps(data, indent=2))


# ── variables ────────────────────────────────────────────────────────────────

def cmd_variables_list(args: argparse.Namespace) -> None:
    data = api_request("GET", "/variables")
    print(json.dumps(data, indent=2))


# ── users ────────────────────────────────────────────────────────────────────

def cmd_users_list(args: argparse.Namespace) -> None:
    data = api_request("GET", "/users")
    print(json.dumps(data, indent=2))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(prog="n8n-api", description="n8n Cloud API CLI")
    sub = p.add_subparsers(dest="resource", required=True)

    # workflows
    pw = sub.add_parser("workflows", help="Workflow operations")
    wsub = pw.add_subparsers(dest="action", required=True)

    pwl = wsub.add_parser("list", help="List workflows")
    pwl.add_argument("--active", type=lambda x: x.lower() == "true",
                     metavar="true|false", help="Filter by active status")
    pwl.add_argument("--tags", help="Comma-separated tag names to filter by")
    pwl.add_argument("--limit", type=int, default=100)
    pwl.add_argument("--cursor", help="Pagination cursor")
    pwl.set_defaults(func=cmd_workflows_list)

    pwg = wsub.add_parser("get", help="Get workflow by ID")
    pwg.add_argument("id")
    pwg.set_defaults(func=cmd_workflows_get)

    pwa = wsub.add_parser("activate", help="Activate a workflow")
    pwa.add_argument("id")
    pwa.set_defaults(func=cmd_workflows_activate)

    pwd = wsub.add_parser("deactivate", help="Deactivate a workflow")
    pwd.add_argument("id")
    pwd.set_defaults(func=cmd_workflows_deactivate)

    # executions
    pe = sub.add_parser("executions", help="Execution operations")
    esub = pe.add_subparsers(dest="action", required=True)

    pel = esub.add_parser("list", help="List executions")
    pel.add_argument("--workflow-id", help="Filter by workflow ID")
    pel.add_argument("--status", choices=["error", "success", "waiting", "running"],
                     help="Filter by execution status")
    pel.add_argument("--limit", type=int, default=20)
    pel.add_argument("--cursor", help="Pagination cursor")
    pel.add_argument("--include-data", action="store_true",
                     help="Include full execution data (can be large)")
    pel.set_defaults(func=cmd_executions_list)

    peg = esub.add_parser("get", help="Get execution by ID")
    peg.add_argument("id")
    peg.add_argument("--include-data", action="store_true",
                     help="Include full execution data")
    peg.set_defaults(func=cmd_executions_get)

    ped = esub.add_parser("delete", help="Delete execution by ID")
    ped.add_argument("id")
    ped.set_defaults(func=cmd_executions_delete)

    # credentials
    pc = sub.add_parser("credentials", help="Credential operations")
    csub = pc.add_subparsers(dest="action", required=True)
    pcl = csub.add_parser("list", help="List credentials")
    pcl.set_defaults(func=cmd_credentials_list)

    # tags
    pt = sub.add_parser("tags", help="Tag operations")
    tsub = pt.add_subparsers(dest="action", required=True)
    ptl = tsub.add_parser("list", help="List tags")
    ptl.set_defaults(func=cmd_tags_list)

    # variables
    pv = sub.add_parser("variables", help="Variable operations")
    vsub = pv.add_subparsers(dest="action", required=True)
    pvl = vsub.add_parser("list", help="List variables")
    pvl.set_defaults(func=cmd_variables_list)

    # users
    pu = sub.add_parser("users", help="User operations")
    usub = pu.add_subparsers(dest="action", required=True)
    pul = usub.add_parser("list", help="List users")
    pul.set_defaults(func=cmd_users_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
