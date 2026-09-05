#!/usr/bin/env bash
# Delegate a brief to Claude Code CLI. Reply -> reply.md beside the brief.
source "$(dirname "$0")/common.sh"; parse_args "$@"
if [ "$EDIT" = 1 ]; then
  perm=(--permission-mode acceptEdits --allowedTools "Read,Grep,Glob,Edit,Write,MultiEdit,Bash(git diff:*),Bash(git status:*)")
else
  perm=(--allowedTools "Read,Grep,Glob,Bash(git diff:*),Bash(git log:*)")
fi
cmd=(claude -p --output-format text ${perm[@]+"${perm[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})
[ -n "$MODEL" ] && cmd+=(--model "$MODEL")
[ "$DRY" = 1 ] && { show_payload claude "${cmd[*]} < brief"; exit 0; }
(cd "$CWD" && "${cmd[@]}" < "$BRIEF") > "$OUT"
echo "reply: $OUT"
