#!/usr/bin/env bash
# Delegate a brief to Codex CLI. Reply -> reply.md beside the brief.
source "$(dirname "$0")/common.sh"; parse_args "$@"
sandbox=$([ "$EDIT" = 1 ] && echo workspace-write || echo read-only)
cmd=(codex exec -C "$CWD" -s "$sandbox" --skip-git-repo-check -o "$OUT" ${EXTRA[@]+"${EXTRA[@]}"})
[ -n "$MODEL" ] && cmd+=(-m "$MODEL")
[ "$DRY" = 1 ] && { show_payload codex "${cmd[*]} - < brief"; exit 0; }
"${cmd[@]}" - < "$BRIEF" > /dev/null
echo "reply: $OUT"
