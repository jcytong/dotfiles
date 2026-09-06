---
name: evaluate-agent-workflow
description: Evaluate a skill or agent workflow against realistic tasks before adopting a change. Use when testing a CoS workflow, assessing a skill revision, or turning repeated failures into verified improvements.
---

# Evaluate an agent workflow

Test whether a proposed instruction or helper change improves observable behavior. Skill formatting
validation does not demonstrate workflow correctness. Keep evaluation tasks separate from live work.

## Define the comparison

Read the target workflow and the caller's objective. Freeze the baseline and candidate versions,
including their referenced resources. Select a small representative set of tasks and define expected
outcomes before running either version. Include a normal task and the failure or edge case motivating
the change. CoS changes should exercise stage handoff, failed verification, exhausted retries, and
resuming interrupted work when those behaviors are affected.

For each case, record the fixture, user request, permitted actions, observable acceptance checks, and
stop condition. Use isolated repositories, disposable data, and inert service doubles at external
boundaries. Do not use a real board, send messages, or publish changes merely to evaluate routing.
If live services are essential, obtain the required authorization through the caller's normal policy.

## Run and judge

- Use the current host's supported executor; do not assume a vendor-specific subagent API or model.
  Keep executor/model settings and fixtures comparable between baseline and candidate. Record any
  difference that limits comparison. Do not choose premium models or larger budgets implicitly.
- Run the actual workflow when available. A paper walkthrough is useful for contract inspection but
  must be labeled as such; it cannot establish real agent behavior, cost, or reliability.
- Give executors the request, workflow, and raw fixtures, not the expected verdict or suspected fix.
  Respect the supplied run budget. Without one, run each selected case once per version, with no
  automatic retry; report this as a smoke evaluation, not a reliability estimate.
- Judge saved artifacts against the predeclared criteria. Prefer deterministic checks for facts such
  as changed files, transitions, preserved evidence, and duplicate work. Use a separate evaluator for
  judgment-dependent results where available, concealing which version produced each result.
- Record success, missed defects, unauthorized actions, unnecessary human interruptions, elapsed time,
  and token/cost measurements if the executor exposes them. Missing usage data is unknown, not zero.
  Variability requires repeated runs before making repeatability claims.

## Improve only from evidence

Recommend adoption only when acceptance checks pass without violating authority or scope. A single
successful example does not prove general superiority. Explain regressions and untested cases.
For a recurring defect, prefer an executable check, clearer data contract, or narrow helper over
adding broad instructions. Propose the smallest change, then rerun the affected cases within budget.
Do not silently change the active workflow or its authorization policy during an evaluation.

## Report

Return a comparison table with case, version, verdict, evidence, interventions, and measured usage.
Name the frozen versions, execution environment, skipped cases, and whether the run was actual or
simulated. End with adopt / revise / inconclusive and its evidence. Keep receipts in the caller's
run folder, or a gitignored `.verify/` directory in the evaluation workspace when none is supplied.
