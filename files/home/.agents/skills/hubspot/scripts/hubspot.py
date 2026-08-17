#!/usr/bin/env python3
"""
hubspot.py - CRM helpers layered on the HubSpot CLI.

Every HubSpot call here shells out to `hs api`, so authentication, account
selection and config live entirely in the CLI. No tokens are read, stored or
printed by this script.

What this adds over bare `hs api`:
  - pagination (`hs api` returns one page; the CRM search cap is 10k results)
  - a pre-import scan that catches secondary-email upsert hijacks
  - imports that tag every record, then re-query by that tag for the
    authoritative ID set instead of trusting batch result order
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Iterator

BATCH = 100          # HubSpot batch endpoints cap at 100 inputs
SEARCH_PAGE = 100    # max page size for the CRM search API
SEARCH_CAP = 10_000  # hard ceiling on total results from /search


class HsError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# CLI transport
# --------------------------------------------------------------------------

def hs_api(endpoint: str, method: str = "GET", data: Any = None,
           account: str | None = None) -> Any:
    cmd = ["hs", "api", endpoint, "--json"]
    if method.upper() != "GET":
        cmd += ["-X", method.upper()]
    if data is not None:
        cmd += ["--data", json.dumps(data)]
    if account:
        cmd += ["-a", account]

    env = {**os.environ, "HS_DISABLE_AUTOUPDATE": "1", "NO_UPDATE_NOTIFIER": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise HsError(f"hs api {endpoint} failed (exit {proc.returncode}):\n{detail}")
    return _extract_json(proc.stdout, endpoint)


def _extract_json(stdout: str, endpoint: str) -> Any:
    """`hs` may prefix output with beta/update banners - find the first JSON value."""
    text = stdout.strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except ValueError:
                continue
    raise HsError(f"no JSON in `hs api {endpoint}` output:\n{text[:500]}")


def chunked(items: list, size: int = BATCH) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


# --------------------------------------------------------------------------
# Pagination
# --------------------------------------------------------------------------

def paginate_get(endpoint: str, account: str | None = None,
                 limit: int | None = None) -> list[dict]:
    """Follow paging.next.after on a GET collection endpoint."""
    out: list[dict] = []
    after = None
    sep = "&" if "?" in endpoint else "?"
    while True:
        url = endpoint if after is None else f"{endpoint}{sep}after={after}"
        page = hs_api(url, account=account) or {}
        out.extend(page.get("results", []))
        if limit and len(out) >= limit:
            return out[:limit]
        after = (page.get("paging") or {}).get("next", {}).get("after")
        if not after:
            return out


def search_all(obj: str, body: dict, account: str | None = None,
               limit: int | None = None) -> list[dict]:
    """Page through /crm/v3/objects/{obj}/search."""
    out: list[dict] = []
    after = None
    while True:
        payload = {**body, "limit": SEARCH_PAGE}
        if after:
            payload["after"] = after
        page = hs_api(f"/crm/v3/objects/{obj}/search", "POST", payload, account) or {}
        results = page.get("results", [])
        out.extend(results)
        if limit and len(out) >= limit:
            return out[:limit]
        after = (page.get("paging") or {}).get("next", {}).get("after")
        if not after or not results:
            return out
        if len(out) >= SEARCH_CAP:
            print(f"warning: hit the {SEARCH_CAP} result ceiling on /search; "
                  f"narrow the filters to see the rest", file=sys.stderr)
            return out


# --------------------------------------------------------------------------
# Input parsing
# --------------------------------------------------------------------------

def load_records(path: str) -> list[dict]:
    """Read JSONL or CSV into a list of flat property dicts."""
    if path.endswith((".csv", ".tsv")):
        delim = "\t" if path.endswith(".tsv") else ","
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [{k: v for k, v in row.items() if v not in (None, "")}
                    for row in csv.DictReader(fh, delimiter=delim)]
    else:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise HsError(f"{path} contained no records")
    return rows


def parse_filter(spec: str) -> dict:
    """`prop=value`, `prop:GT=value`, or `prop:HAS_PROPERTY`."""
    if "=" not in spec:
        name, _, op = spec.partition(":")
        return {"propertyName": name, "operator": op or "HAS_PROPERTY"}
    left, _, value = spec.partition("=")
    name, _, op = left.partition(":")
    return {"propertyName": name, "operator": op or "EQ", "value": value}


# --------------------------------------------------------------------------
# Email pre-scan: the secondary-email hijack check
# --------------------------------------------------------------------------

def scan_emails(emails: list[str], account: str | None = None) -> dict:
    """
    Resolve each email through batch/read with idProperty=email - the same
    resolution an upsert performs. An input that matches a contact whose *primary*
    email is different matched via hs_additional_emails; upserting it would
    overwrite that unrelated person.
    """
    matches: dict[str, dict] = {}
    props = ["email", "firstname", "lastname", "createdate"]
    for batch in chunked(emails):
        body = {"idProperty": "email", "properties": props,
                "inputs": [{"id": e} for e in batch]}
        try:
            resp = hs_api("/crm/v3/objects/contacts/batch/read", "POST", body, account) or {}
        except HsError as exc:
            # A batch where every input is new can 404 as a whole; fall back per record.
            if "404" not in str(exc):
                raise
            resp = {"results": []}
            for one in batch:
                single = {"idProperty": "email", "properties": props,
                          "inputs": [{"id": one}]}
                try:
                    got = hs_api("/crm/v3/objects/contacts/batch/read", "POST",
                                 single, account) or {}
                    resp["results"].extend(got.get("results", []))
                except HsError:
                    continue
        for rec in resp.get("results", []):
            primary = (rec.get("properties") or {}).get("email") or ""
            matches[rec["id"]] = {"id": rec["id"], "primary_email": primary,
                                  "properties": rec.get("properties", {})}

    lowered = {e.lower() for e in emails}
    primary_hits, secondary_hits = {}, []
    for rec in matches.values():
        primary = (rec["primary_email"] or "").lower()
        if primary in lowered:
            primary_hits[primary] = rec
        else:
            secondary_hits.append(rec)

    new = sorted(e for e in emails if e.lower() not in primary_hits)
    # An input resolving only via a secondary email is not genuinely "new".
    return {
        "input_count": len(emails),
        "existing_primary": list(primary_hits.values()),
        "secondary_matches": secondary_hits,
        "unmatched": new,
    }


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_whoami(args) -> int:
    info = hs_api("/account-info/v3/details", account=args.account) or {}
    print(json.dumps({k: info.get(k) for k in
                      ("portalId", "accountType", "timeZone", "companyName", "uiDomain")},
                     indent=2))
    return 0


def cmd_api(args) -> int:
    if args.all:
        results = paginate_get(args.endpoint, args.account, args.limit)
        print(json.dumps({"count": len(results), "results": results}, indent=2))
    else:
        data = json.loads(args.data) if args.data else None
        print(json.dumps(hs_api(args.endpoint, args.method, data, args.account), indent=2))
    return 0


def cmd_props(args) -> int:
    resp = hs_api(f"/crm/v3/properties/{args.object}", account=args.account) or {}
    rows = [{"name": p["name"], "label": p.get("label"), "type": p.get("type"),
             "fieldType": p.get("fieldType")} for p in resp.get("results", [])]
    rows.sort(key=lambda r: r["name"])
    if args.grep:
        needle = args.grep.lower()
        rows = [r for r in rows
                if needle in r["name"].lower() or needle in (r["label"] or "").lower()]
    if args.json:
        print(json.dumps({"count": len(rows), "properties": rows}, indent=2))
    else:
        for r in rows:
            print(f"{r['name']:<45} {r['type']:<12} {r['label']}")
        sys.stdout.flush()   # keep the summary after the rows when stdout is piped
        print(f"\n{len(rows)} properties", file=sys.stderr)
    return 0


def cmd_search(args) -> int:
    body: dict[str, Any] = {"properties": args.props.split(",") if args.props else
                            ["email", "firstname", "lastname", "company"]}
    if args.filter:
        body["filterGroups"] = [{"filters": [parse_filter(f) for f in args.filter]}]
    if args.query:
        body["query"] = args.query
    results = search_all(args.object, body, args.account, args.limit)
    print(json.dumps({"count": len(results), "results": results}, indent=2))
    return 0


def cmd_check_emails(args) -> int:
    records = load_records(args.file)
    emails = [r[args.email_property] for r in records if r.get(args.email_property)]
    report = scan_emails(emails, args.account)
    print(json.dumps(report, indent=2))
    if report["secondary_matches"]:
        print(f"\n{len(report['secondary_matches'])} input(s) resolve to a contact whose "
              f"PRIMARY email differs. An email-keyed upsert would overwrite those "
              f"contacts. Corroborate by surname before treating them as matches.",
              file=sys.stderr)
    return 0


def cmd_import(args) -> int:
    records = load_records(args.file)
    for rec in records:
        rec[args.tag_property] = args.tag_value

    emails = [r[args.email_property] for r in records if r.get(args.email_property)]
    if len(emails) != len(records):
        raise HsError(f"{len(records) - len(emails)} record(s) lack "
                      f"'{args.email_property}'")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"scanning {len(emails)} email(s) against the portal...", file=sys.stderr)
    scan = scan_emails(emails, args.account)
    existing = {r["primary_email"].lower() for r in scan["existing_primary"]}

    if scan["secondary_matches"] and not args.allow_secondary_matches:
        print(json.dumps({"aborted": "secondary_email_matches",
                          "matches": scan["secondary_matches"]}, indent=2))
        print(f"\n{len(scan['secondary_matches'])} input email(s) resolve only via "
              f"hs_additional_emails. Writing would overwrite unrelated contacts. "
              f"Review them, then re-run with --allow-secondary-matches if they are "
              f"genuinely the same people.", file=sys.stderr)
        return 2

    if args.mode == "upsert" and not args.i_understand_upsert_risk:
        print("upsert mode resolves against the full email graph, not just primary "
              "emails, and can silently overwrite contacts. Pass "
              "--i-understand-upsert-risk to proceed, or use --mode create.",
              file=sys.stderr)
        return 2

    to_write = records
    skipped: list[str] = []
    if args.mode == "create":
        to_write = [r for r in records
                    if r[args.email_property].lower() not in existing]
        skipped = [r[args.email_property] for r in records
                   if r[args.email_property].lower() in existing]

    if args.dry_run:
        print(json.dumps({"dry_run": True, "would_write": len(to_write),
                          "would_skip_existing": skipped,
                          "secondary_matches": scan["secondary_matches"]}, indent=2))
        return 0

    written = 0
    for batch in chunked(to_write):
        if args.mode == "create":
            body = {"inputs": [{"properties": r} for r in batch]}
            endpoint = "/crm/v3/objects/contacts/batch/create"
        else:
            body = {"inputs": [{"idProperty": args.email_property,
                                "id": r[args.email_property],
                                "properties": r} for r in batch]}
            endpoint = "/crm/v3/objects/contacts/batch/upsert"
        hs_api(endpoint, "POST", body, args.account)
        written += len(batch)
        print(f"  wrote {written}/{len(to_write)}", file=sys.stderr)

    # Authoritative set: re-query by tag. Never zip batch results with input order.
    print("re-querying by tag for the authoritative ID set...", file=sys.stderr)
    tagged = search_all("contacts", {
        "filterGroups": [{"filters": [{"propertyName": args.tag_property,
                                       "operator": "EQ", "value": args.tag_value}]}],
        "properties": ["email", "createdate", args.tag_property],
    }, args.account)

    pre_existing = [t for t in tagged
                    if (t.get("properties") or {}).get("createdate", "") < started]
    report = {
        "tag": {args.tag_property: args.tag_value},
        "input_records": len(records),
        "written": written,
        "skipped_existing": skipped,
        "tagged_total": len(tagged),
        "created_this_run": len(tagged) - len(pre_existing),
        "pre_existing_touched": [{"id": t["id"],
                                  "email": (t.get("properties") or {}).get("email"),
                                  "createdate": (t.get("properties") or {}).get("createdate")}
                                 for t in pre_existing],
        "ids": [t["id"] for t in tagged],
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.out}", file=sys.stderr)
    print(json.dumps(report, indent=2))

    if len(tagged) != len(records):
        print(f"\ncount mismatch: {len(records)} input records but {len(tagged)} "
              f"tagged contacts. Investigate before building anything downstream.",
              file=sys.stderr)
    return 0


def cmd_list_add(args) -> int:
    """Add contacts to a list by re-querying a tag - never by batch result order."""
    tagged = search_all("contacts", {
        "filterGroups": [{"filters": [{"propertyName": args.tag_property,
                                       "operator": "EQ", "value": args.tag_value}]}],
        "properties": ["email", args.tag_property],
    }, args.account)
    ids = [t["id"] for t in tagged]
    if not ids:
        print(f"no contacts carry {args.tag_property}={args.tag_value}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps({"dry_run": True, "list_id": args.list_id,
                          "would_add": len(ids), "ids": ids}, indent=2))
        return 0
    added = 0
    for batch in chunked(ids):
        hs_api(f"/crm/v3/lists/{args.list_id}/memberships/add", "PUT", batch, args.account)
        added += len(batch)
        print(f"  added {added}/{len(ids)}", file=sys.stderr)
    print(json.dumps({"list_id": args.list_id, "added": added, "ids": ids}, indent=2))
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hubspot", description="CRM helpers layered on the HubSpot CLI (`hs api`).")
    p.add_argument("-a", "--account", help="account id or name from the hs config")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("whoami", help="print the authenticated portal")
    s.set_defaults(func=cmd_whoami)

    s = sub.add_parser("api", help="raw `hs api` passthrough, with optional pagination")
    s.add_argument("endpoint")
    s.add_argument("-X", "--method", default="GET")
    s.add_argument("--data", help="JSON request body")
    s.add_argument("--all", action="store_true", help="follow paging.next.after (GET only)")
    s.add_argument("--limit", type=int)
    s.set_defaults(func=cmd_api)

    s = sub.add_parser("props", help="list properties on an object")
    s.add_argument("object", nargs="?", default="contacts")
    s.add_argument("--grep", help="filter by substring of name or label")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_props)

    s = sub.add_parser("search", help="CRM search, fully paginated")
    s.add_argument("object", nargs="?", default="contacts")
    s.add_argument("--filter", action="append",
                   help="prop=value | prop:GT=value | prop:HAS_PROPERTY (repeatable, ANDed)")
    s.add_argument("--query", help="free-text search string")
    s.add_argument("--props", help="comma-separated properties to return")
    s.add_argument("--limit", type=int)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("check-emails",
                       help="pre-import scan for secondary-email upsert hijacks")
    s.add_argument("file", help="JSONL or CSV of records")
    s.add_argument("--email-property", default="email")
    s.set_defaults(func=cmd_check_emails)

    s = sub.add_parser("import", help="guarded batch import with tag and post-verification")
    s.add_argument("file", help="JSONL or CSV of records")
    s.add_argument("--tag-property", required=True,
                   help="property written on every record, e.g. lp_source_list")
    s.add_argument("--tag-value", required=True)
    s.add_argument("--mode", choices=["create", "upsert"], default="create")
    s.add_argument("--email-property", default="email")
    s.add_argument("--allow-secondary-matches", action="store_true")
    s.add_argument("--i-understand-upsert-risk", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--out", help="write the report JSON here")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("list-add", help="add tagged contacts to a list by re-query")
    s.add_argument("list_id")
    s.add_argument("--tag-property", required=True)
    s.add_argument("--tag-value", required=True)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_list_add)

    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except HsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
