---
name: review-pr
description: Review a GitHub pull request and provide feedback
argument-hint: "[pr-number or url]"
---

# PR Review

Review pull request $ARGUMENTS thoroughly and provide constructive feedback.

## Step 0: Check Prerequisites

First, verify the GitHub CLI is installed and authenticated:

```bash
which gh && gh auth status
```

If `gh` is not found, install it with `brew install gh`. If not authenticated, run `gh auth login`.

## Step 1: Check for Previous Reviews and New Commits

**IMPORTANT:** Before doing a full review, check if Claude has already reviewed this PR and if there are new commits since then.

```bash
# Get PR metadata including commits and comments in one call
gh pr view $ARGUMENTS --json number,title,state,updatedAt,commits,comments
```

From the output:
1. Look for comments containing "*Reviewed by Claude*" - note the timestamp if found
2. Check the commits array - get the latest commit's `committedDate`
3. Compare: If the latest commit is NEWER than the last Claude review, there are new changes to review

**If a previous Claude review exists:**
- Check if any commits were pushed AFTER that review's timestamp
- If YES: Do a fresh review of the CURRENT diff (the old review may be outdated)
- If NO: Inform the user the review is still current and summarize the key findings

**Always fetch the current diff regardless** - it reflects the latest state:
```bash
gh pr diff $ARGUMENTS
```

## Step 2: Gather Full PR Context

```bash
gh pr checks $ARGUMENTS
```

## Step 3: Analyze Changes

Review the diff for:
- **Correctness**: Logic errors, edge cases, null handling
- **Security**: Injection risks, auth issues, data exposure
- **Performance**: N+1 queries, memory leaks, unnecessary computation
- **Style**: Consistency with codebase patterns
- **Tests**: Coverage of new/changed code
- **Documentation**: Updated comments, README, API docs

If this is a re-review after new commits:
- Note which previous issues have been fixed
- Identify any new issues introduced
- Acknowledge improvements made

## Step 4: Provide Feedback

Structure your review:
1. Summary of what the PR does
2. If re-review: What changed since last review / what was fixed
3. What's good about the changes
4. Specific issues or concerns (with file:line references)
5. Suggestions for improvement
6. Questions for the author

## Step 5: Post Comment

Post your review as a comment on the PR:

```bash
gh pr comment $ARGUMENTS --body "## Code Review

[Your structured review here]

---
*Reviewed by Claude*"
```

For re-reviews, consider starting with:
```
## Code Review (Updated)

**Note:** This is an updated review following new commits pushed since [previous review date].

### Changes Since Last Review
- [List what was fixed/changed]

### Current Status
[Rest of review...]
```
