---
name: maintain-verification-skill
description: Audit or repair a project's verification skill, executable checks, and feature map against the current app. Use when verification instructions drift or a verification audit is requested.
---

# Maintain a verification skill

Keep a project's verification recipe usable without changing the definition of success to hide a
product defect. The canonical location is `.agents/skills/verify-*/`; inspect existing host adapters
as links to that source, not separate copies to update independently.

## Establish scope

Locate the target skill and read its feature index, harness, and affected source. If several skills
could fit, use the user's feature or project to narrow the target; ask only if ambiguity remains.
If none exists, report that and recommend `create-verification-skill`.

For a targeted change, cover the changed feature and affected entry points. For a requested full
audit, account for every indexed feature. Record coverage explicitly; a sample is not a full audit.
Honor the caller's authority and attempt limit. Without a supplied limit, allow one repair and
recheck per failing recipe, then report the remaining failure.

## Check and repair

1. Compare the map to source and the accepted product requirements. Identify stale commands,
   selectors, prerequisites, missing entry points, and broken index links.
2. Run the skill's health check and relevant recipes on an instance owned by this run. Preserve
   evidence outside disposable application state. Check actual effects as well as visible results.
3. Classify each mismatch: instruction/harness drift, product regression, changed requirement, or
   unavailable environment. An unavailable environment is blocked, not passed. A requirement change
   needs an authoritative decision; source behavior alone does not establish intended behavior.
4. Repair drift in the skill, its owned helpers, or feature map. Do not edit product code under this
   maintenance scope. Report regressions to the caller for a separate implementation task. Keep the
   expected behavior visible rather than weakening an assertion to make the run green.
5. Rerun the repaired recipe. Confirm stable helper assertions detect incorrect output as well as
   accept correct output where a controlled negative case is practical. Clean up only resources
   created by this run, including failed attempts; retain receipts and artifacts.

## Handoff

Report **clean**, **changed**, or **blocked**, with covered and unverified features, actual commands
and results, artifact locations, defects requiring implementation, and any changes made. Identify
the source state including uncommitted changes and new files. Leave edits uncommitted and propose a
commit message; publication follows the caller's policy. Do not claim a host scheduled future audits.
