#!/usr/bin/env bash
# Flag likely PII / secrets in a brief. Exit 1 if anything is flagged.
set -euo pipefail
f="${1:?usage: scan.sh BRIEF.md}"; [ -f "$f" ] || { echo "no such file: $f" >&2; exit 2; }
TERMS="$HOME/.pi/agent/private-terms.txt"
hits=0
flag() { # label regex [grep-flags]
  local out; out=$(grep -nE ${3:-} -- "$2" "$f" || true)
  [ -n "$out" ] || return 0
  echo "[$1]"; echo "$out" | sed 's/^/  /'; hits=1
}
flag email    '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
flag phone    '(\+?[0-9]{1,3}[ .-]?)?\(?[0-9]{3}\)?[ .-][0-9]{3}[ .-][0-9]{4}'
flag ssn/id   '\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b'
flag card/acct '\b[0-9]{4}[ -]?[0-9]{4}[ -]?[0-9]{4}[ -]?[0-9]{1,7}\b'
flag secret   '(sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,}|xox[abp]-[A-Za-z0-9-]+|fw_[A-Za-z0-9]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY)'
flag money    '(\$|USD|€|£)[ ]?[0-9][0-9,]{3,}(\.[0-9]+)?'
flag url-priv '(docs\.google\.com|drive\.google\.com|notion\.so|slack\.com/archives)'
flag address  '\b[0-9]{1,5} [A-Z][a-z]+ (St|Ave|Rd|Blvd|Dr|Ln|Way|Ct)\b'
flag health   '\b(diagnos|prescri|mg\b|dosage|therap|clinic|hospital|symptom)' -i
if [ -f "$TERMS" ]; then
  pat=$(grep -v '^[[:space:]]*#' "$TERMS" | sed '/^[[:space:]]*$/d' | paste -sd'|' - || true)
  [ -n "$pat" ] && flag private-term "($pat)" -i
fi
[ "$hits" = 0 ] && echo "scan: clean" || { echo "scan: review the hits above (redact or get explicit approval)"; exit 1; }
