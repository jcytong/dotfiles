---
name: create-verification-skill
description: "Create a project-local verification skill and executable checks for a real app. Use when asked to make a verification skill or establish repeatable user-facing verification in a repository."
---

# Create a verification skill

A repo needs a scripted way to drive the real app and prove behavior: launch it, exercise a feature the
way a user would, capture evidence. This skill generates that as a project-local skill,
`.agents/skills/verify-<app>/`, tailored to the repo. Write the output for the next agent, not for a
human: it will be read cold, mid-task, by an agent that has never seen the app. Keep instructions
independent of a model vendor; discover the executor and available tools in the target environment.

Passing unit tests are a claim about code. A verification run is evidence about the product. Both are
required before "done"; this skill produces the second.

## 1. Interview the repo, not the user

Answer these from the codebase and only ask the user what you cannot observe:

- **Surface:** what does a user actually touch? Web UI, CLI/TUI, desktop app, API, mobile app, library?
  A repo can have several; pick the primary one and note the rest.
- **Run:** how does the app start locally? Prefer the repo's own documented dev command (package
  scripts, `Makefile`, `README` quickstart, `uv run …`, `wrangler dev`, `docker compose up`). Note ports,
  env vars, seed data, auth. A repo `CLAUDE.md`/`AGENTS.md` often already says — read it first.
- **Drive:** how can an agent interact with it programmatically? Existing harnesses first — Playwright
  or Cypress specs, expect scripts, PTY helpers, curl-able endpoints, a debug port. Only then pick a
  generic recipe: an available browser/CDP driver for web and Electron, a PTY for interactive
  CLI/TUI, or plain HTTP for services. Use Herdr only when requested and available; otherwise
  discover a suitable local harness. Noninteractive CLIs generally need only a shell.
- **Observe:** what evidence can be captured? Screenshots, ARIA snapshots, terminal transcripts,
  response bodies, logs, exit codes, DB rows, files on disk.
- **Isolate:** can two instances run side by side (ports, data dirs, profiles)? If not, say so in the
  generated skill: refusing to double-drive a shared instance beats corrupting the user's session.

If the checkout does not build or start as-is, fix that first (or report it precisely) before
generating; a skill written against a broken base teaches wrong steps. When an irrelevant missing asset
blocks startup (a static dir the API never serves, a sample config), the generated skill may create it,
clearly marked as verification scaffolding, and remove it in cleanup.

## 2. Generate the skill

Write `.agents/skills/verify-<app>/SKILL.md` with YAML frontmatter — `name: verify-<app>` and a
`description` that names the app, the surface, and when to reach for it; without frontmatter the skill
cannot be discovered — and these sections, each grounded in what the interview found (no placeholders left).
Keep this directory canonical. If a chosen host needs a different discovery path, use its documented
adapter or a symlink to this directory; do not maintain duplicate skill bodies. Do not overwrite an
existing verification skill: inspect it and extend it when it already covers this app.

- **Launch:** the exact command that starts the app for verification and how to tell it is ready (a
  log line, a port answering, a prompt). Include teardown. For a short-lived CLI or TUI there is no
  server to keep alive: build the binary (or install deps) once, then run each drive with isolated
  state. Use a PTY only when the program needs an interactive terminal.
- **Doctor:** one read-only check that answers "is this instance worth driving?" — process up, right
  version/build, port owned by us, auth valid. An agent runs this first whenever anything looks off.
- **Drive:** the harness recipe with real selectors and commands from this repo, not examples. Prefer
  stable handles (ARIA roles and names, data attributes, prompt strings, route paths) over coordinates
  and tab order.
- **Evidence:** what to capture for a proof and where it goes. Default location:
  `$VERIFY_EVIDENCE_DIR` if set (a CoS brief sets it to the run folder), else `.verify/<run-id>/` in
  the repo — add `.verify/` to `.gitignore` when you create it. Proof standards: exercise the real user
  path, not internal setters or test-only endpoints; capture the action and the resulting state, not
  just the final screen; verify side effects (files written, rows inserted, messages sent) alongside
  what is visible; mocks only where a production boundary already isolates the external system. When
  the safe path is a dry-run or test mode, verify what it actually skips by observing (files, network,
  git refs) rather than trusting its name: some dry-runs still touch the network or open a browser.
- **Cleanup:** how to tear down instances the run created. Never kill by process name; kill what you
  started (record the PID or pane id at launch). Cleanup removes instances and scratch state, never the
  evidence: proof artifacts survive the teardown, at the location the skill names.
- **Repeatable checks:** reuse existing tests and harnesses. Put stable setup, fixture creation,
  actions, and objective assertions into executable helpers or existing tests when they otherwise
  require repeated interpretation. Checks return nonzero on failure; screenshots alone are not
  assertions. Keep judgment-dependent evaluation in the recipe, with explicit criteria. Integrate
  unattended checks into existing CI when they fit its environment and task scope; report any
  missing credentials or infrastructure instead of claiming CI coverage.
- **Helpers:** scripts are executable, their invocation is shown, and prerequisites are explicit.
  Python helpers run under `uv run`. Each run uses its own scratch state and can clean up after failure.
- **Receipt:** record each acceptance criterion, command/action, observed result, exit code where
  applicable, and artifact path. Identify the checkout and base commit plus the reviewed diff and
  new-file contents (or a content digest); a commit hash alone does not identify uncommitted work.
  Mark skipped, blocked, and failed checks separately. New source changes invalidate affected proof.

## 3. Seed the feature map

Create `.agents/skills/verify-<app>/features/README.md` plus one file per user-facing feature you can
identify (aim for the top 3–5 to start, from routes, commands, menus, or docs). Follow the shape in
[`references/feature-map-example/`](references/feature-map-example/): a README index and one file per
feature. Each file answers, from the user's point of view: what the feature is, how to reach it, how
to drive it with the harness, and what observable end state proves it works. The four H2s are
`Sub-features`, `How to get to it (user POV)`, `Driving it with <harness>`, and `Gotchas`. The map is
the repo's maintained verification source; a proof that drives one convenient entry point is
incomplete when the map lists others.

## 4. Prove the generated skill before handing it over

Run its own instructions end to end once: launch, doctor, drive ONE mapped feature (one is enough; the
map exists so later runs can cover the rest), capture evidence, clean up. After cleanup, confirm the
evidence still exists at the named location — a cleanup that eats the proof fails this step. Fix what
fails, and run the generated cleanup after every failed iteration too, so broken attempts do not strand
processes, panes, and ports. Honor the caller's attempt/time limits; without an attempt limit, allow
one repair and recheck, then report blocked with evidence. A generated skill that was never executed
is a draft, not a deliverable. Mark the other mapped features as unverified until actually exercised.

Leave the changes uncommitted on a branch and propose the commit message; never commit or push.

## 5. Hand over

Report: the skill path, the surface and harness chosen, the features mapped, the one feature proven and
where its evidence lives, executable checks added/reused, CI integration if any, and anything the
interview could not settle. A coordinator can name the canonical skill path and acceptance criteria
in implementer and reviewer briefs. Do not assume any host-specific command automatically loads it.

## Keeping it honest

Use the shared `maintain-verification-skill` workflow for an explicit audit or when drift is found.

The map rots the moment the app changes. When a verification run finds the map wrong, fix the map or
the harness in the same change — never product code — and say which. A behavior the map describes that
the app no longer does is either doc drift (fix the map) or a product regression (report it, do not
paper over it in docs).

---
Adapted from `create-verification-skill` in github.com/cursor/plugins (pstack, MIT, © 2026 Lauren Tan;
commit 7314f72) for vendor-neutral project skills, executable checks, and evidence-based handoffs.
