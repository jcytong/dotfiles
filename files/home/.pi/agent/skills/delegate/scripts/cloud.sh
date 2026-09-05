#!/usr/bin/env bash
# Delegate a brief to a Fireworks-hosted model (chat completion, no tools).
# Key: $FIREWORKS_API_KEY or ~/.config/fireworks/api_key. Model: --model or $FIREWORKS_MODEL.
source "$(dirname "$0")/common.sh"
BASE="https://api.fireworks.ai/inference/v1"
key() { echo "${FIREWORKS_API_KEY:-$(cat "$HOME/.config/fireworks/api_key" 2>/dev/null || true)}"; }
if [ "${1:-}" = "--list" ]; then
  k=$(key); [ -n "$k" ] || die "no Fireworks key"
  curl -sS "$BASE/models" -H "Authorization: Bearer $k" | jq -r '.data[].id' | sort; exit 0
fi
parse_args "$@"
MODEL="${MODEL:-${FIREWORKS_MODEL:-accounts/fireworks/models/deepseek-v4-pro}}"
SYSTEM='You are a reasoning delegate for a local assistant. The user has replaced private details with placeholders like [PERSON_A] or [AMOUNT] on purpose. Keep every placeholder exactly as written in your answer; do not guess what it stands for. Be precise and complete.'
body=$(jq -n --arg m "$MODEL" --arg s "$SYSTEM" --rawfile u "$BRIEF" \
  '{model:$m, max_tokens:16384, messages:[{role:"system",content:$s},{role:"user",content:$u}]}')
[ "$DRY" = 1 ] && { show_payload "fireworks:$MODEL" "POST $BASE/chat/completions"; exit 0; }
k=$(key); [ -n "$k" ] || die "no Fireworks key: export FIREWORKS_API_KEY or write ~/.config/fireworks/api_key"
resp=$(curl -sS "$BASE/chat/completions" -H "Authorization: Bearer $k" -H "Content-Type: application/json" -d "$body")
echo "$resp" | jq -e '.choices[0].message.content' >/dev/null 2>&1 || die "API error: $(echo "$resp" | head -c 500)"
echo "$resp" | jq -r '.choices[0].message.content' > "$OUT"
echo "reply: $OUT ($(echo "$resp" | jq -r '.usage | "\(.prompt_tokens) in / \(.completion_tokens) out"'))"
