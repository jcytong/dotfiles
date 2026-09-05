---
name: delegate
description: Delegate a task to a cloud model — Claude Code CLI (repo/code changes), Codex CLI (debugging, review, second opinion), or Fireworks (pure reasoning/drafting). Use whenever work exceeds local capability. Enforces the disclosure protocol before anything leaves the machine.
---

# Delegate

All three targets are cloud. AGENTS.md defines data tiers and the disclosure protocol; this skill gives the mechanics. Scripts live in `scripts/` relative to this file.

## Steps
1. **Brief** — `D=$(scripts/new.sh)`; write `$D/brief.md`: goal, constraints, expected output format. Only what the delegate needs. Placeholders for T1/T2. Keep placeholders literal so replies map back.
2. **Scan** — `scripts/scan.sh $D/brief.md`. Lists suspected PII (emails, phones, IDs, keys, amounts, terms from `~/.pi/agent/private-terms.txt`). Exit 1 = hits. Resolve each: redact, or get an explicit yes.
3. **Preview** — run the target with `--dry-run`. Show Johnny the payload, the redaction list, and what each redaction costs. Wait for: send / reveal <items> / stop.
4. **Send** (only after approval). Reply lands in `$D/reply.md`.
   - `scripts/claude.sh $D/brief.md [--cwd REPO] [--edit]`
   - `scripts/codex.sh $D/brief.md [--cwd REPO] [--edit]`
   - `scripts/cloud.sh $D/brief.md [--model ID]` (`--list` shows Fireworks models)
   Default: read-only, cwd = `$D` (holds only the brief). `--edit` permits file edits. `--cwd` runs inside a repo — the delegate will read that repo.
5. **Return** — read `$D/reply.md`, map placeholders back, summarize. State which delegate ran and what was disclosed.

## Rules
- Scripts refuse `--cwd` under any path in `~/.pi/agent/delegate-deny.txt`. Do not work around it.
- Never paste notes, transcripts, emails, or chat history wholesale. Extract the needed facts.
- One brief per delegation, under ~2k words. Chain briefs rather than growing one.
- If the delegate's reply contains a placeholder it needed filled, that is a new disclosure request — go back to step 3.
