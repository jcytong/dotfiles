#!/usr/bin/env bash
# Shared helpers for delegate scripts. Source, don't run.
set -euo pipefail

DENY_FILE="$HOME/.pi/agent/delegate-deny.txt"

die() { echo "delegate: $*" >&2; exit 1; }

# Refuse cwd under any deny-listed path.
check_cwd() {
  local cwd; cwd=$(cd "$1" 2>/dev/null && pwd -P) || die "cwd not found: $1"
  [ -f "$DENY_FILE" ] || return 0
  while IFS= read -r line; do
    line="${line%%#*}"; line="${line// /}"; [ -n "$line" ] || continue
    local d="${line/#\~/$HOME}"; d=$(cd "$d" 2>/dev/null && pwd -P) || continue
    case "$cwd/" in "$d"/*) die "refusing cwd $cwd: under deny-listed $d" ;; esac
  done < "$DENY_FILE"
}

# parse_args BRIEF [--cwd DIR] [--edit] [--dry-run] [--model ID] [-- extra...]
parse_args() {
  BRIEF=""; CWD=""; EDIT=0; DRY=0; MODEL=""; EXTRA=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --cwd) CWD="$2"; shift 2 ;;
      --edit) EDIT=1; shift ;;
      --dry-run) DRY=1; shift ;;
      --model) MODEL="$2"; shift 2 ;;
      --) shift; EXTRA=("$@"); break ;;
      -*) die "unknown flag $1" ;;
      *) [ -z "$BRIEF" ] || die "unexpected arg $1"; BRIEF="$1"; shift ;;
    esac
  done
  [ -n "$BRIEF" ] && [ -f "$BRIEF" ] || die "usage: $(basename "$0") BRIEF.md [--cwd DIR] [--edit] [--dry-run] [-- extra args]"
  BRIEF=$(cd "$(dirname "$BRIEF")" && pwd -P)/$(basename "$BRIEF")
  OUT="$(dirname "$BRIEF")/reply.md"
  CWD="${CWD:-$(dirname "$BRIEF")}"
  check_cwd "$CWD"
}

show_payload() {
  echo "=== DRY RUN — nothing sent ==="
  echo "target : $1"
  echo "cwd    : $CWD"
  echo "mode   : $([ "$EDIT" = 1 ] && echo edit || echo read-only)"
  echo "command: $2"
  echo "--- payload ($(wc -w < "$BRIEF" | tr -d ' ') words) ---"
  cat "$BRIEF"
  echo "--- end payload ---"
}
