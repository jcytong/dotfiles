---
name: ship
description: Commit, push, and optionally create a PR in one command. Use when the user says "ship it", "ship", or "/ship". Streamlines the most common git workflow.
---

# Ship

Commit, push, and optionally create a PR.

## Usage

```
/ship                    - Commit and push (auto-detect branch, default target)
/ship pr                 - Commit, push, and create PR
/ship pr main            - Commit, push, and create PR targeting main
/ship amend              - Amend last commit and force-push (ask confirmation first)
/ship squash             - Interactive squash of unpushed commits, then push
```

---

## Default Flow

### Step 1: Assess

1. Run `git status` (never use `-uall`)
2. Run `git diff --staged` and `git diff` to see all changes
3. Run `git log --oneline -5` to match commit message style
4. Check current branch name

**Guard rails:**
- If on `main` or `master`, STOP and warn the user. Ask if they want to create a branch first.
- If `.env`, credentials, or secret files are staged, STOP and warn.
- If there are no changes, say so and stop.

### Step 2: Commit

1. Stage relevant files (prefer specific files over `git add -A`)
2. Draft a concise commit message (1-2 sentences, "why" not "what")
3. Present the commit message to the user for approval
4. Commit with:

```bash
git commit -m "$(cat <<'EOF'
<message>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

### Step 3: Push

1. Check if branch has a remote tracking branch
2. If not, push with `-u origin <branch>`
3. If yes, `git push`
4. Report push result

### Step 4: PR (only if `pr` subcommand)

1. Determine target branch (default: `dev`, or user-specified)
2. Run `git log` and `git diff <target>...HEAD` to understand all changes since divergence
3. Create PR:

```bash
gh pr create --title "<short title>" --body "$(cat <<'EOF'
## Summary
<1-3 bullet points covering ALL commits in the PR, not just the latest>

## Test plan
- [ ] <verification steps>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

4. Return the PR URL

---

## Subcommands

### amend

1. Confirm with user first: "This will amend the last commit and force-push. Continue?"
2. Stage changes
3. `git commit --amend --no-edit`
4. `git push --force-with-lease`

### squash

1. Show `git log --oneline <target>..HEAD` to display commits to squash
2. Confirm with user
3. `git reset --soft <target>`
4. Create a single new commit with combined message
5. Push with `--force-with-lease`

---

## Key Rules

- NEVER skip hooks (`--no-verify`)
- NEVER force-push to `main` or `master`
- ALWAYS show the commit message before committing
- ALWAYS check for secrets/credentials before staging
- Default PR target is `dev` unless the project only has `main`
- Commit message style should match the repo's existing convention
