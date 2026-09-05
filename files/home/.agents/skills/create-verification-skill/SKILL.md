---
name: create-verification-skill
description: "Generate a project-local verification skill (.claude/skills/verify-<app>/) that drives the real app the way a user does and captures proof — web, CLI/TUI, service, desktop, mobile, library. Use for /create-verification-skill, \"make a verify skill for this repo\", or when a repo has no scripted way to prove behavior. The built-in /run skill and CoS implementer/reviewer delegates pick the generated skill up."
disable-model-invocation: true
---

# Create a verification skill

A repo needs a scripted way to drive the real app and prove behavior: launch it, exercise a feature the
way a user would, capture evidence. This skill generates that as a project-local skill,
`.claude/skills/verify-<app>/`, tailored to the repo. Write the output for the next agent, not for a
human: it will be read cold, mid-task, by an agent that has never seen the app — a Claude Code session
here, a Codex reviewer, or a CoS delegate that has only its brief.

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
  generic recipe: browser/CDP (the Claude in Chrome tools or Playwright) for web and Electron; a PTY
  for CLI/TUI — a Herdr pane when `HERDR_ENV=1` (`herdr pane split` / `send-text` / `read`), otherwise
  `tmux` or `script`; plain HTTP for services.
- **Observe:** what evidence can be captured? Screenshots, ARIA snapshots, terminal transcripts,
  response bodies, logs, exit codes, DB rows, files on disk.
- **Isolate:** can two instances run side by side (ports, data dirs, profiles)? If not, say so in the
  generated skill: refusing to double-drive a shared instance beats corrupting the user's session.

If the checkout does not build or start as-is, fix that first (or report it precisely) before
generating; a skill written against a broken base teaches wrong steps. When an irrelevant missing asset
blocks startup (a static dir the API never serves, a sample config), the generated skill may create it,
clearly marked as verification scaffolding, and remove it in cleanup.

## 2. Generate the skill

Write `.claude/skills/verify-<app>/SKILL.md` with YAML frontmatter — `name: verify-<app>` and a
`description` that names the app, the surface, and when to reach for it; without frontmatter the skill
never registers — and these sections, each grounded in what the interview found (no placeholders left):

- **Launch:** the exact command that starts the app for verification and how to tell it is ready (a
  log line, a port answering, a prompt). Include teardown. For a short-lived CLI or TUI there is no
  server to keep alive: launch means build the binary (or install deps) once, then start each drive in
  its own isolated PTY.
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
- **Helpers:** any script the skill ships is executable and its invocation is shown in the skill body.
  A helper the reader has to reverse-engineer is not a helper. Python helpers run under `uv run`.

## 3. Seed the feature map

Create `.claude/skills/verify-<app>/features/README.md` plus one file per user-facing feature you can
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
processes, panes, and ports. A generated skill that was never executed is a draft, not a deliverable.

Leave the changes uncommitted on a branch and propose the commit message; never commit or push.

## 5. Hand over

Report: the skill path, the surface and harness chosen, the features mapped, the one feature proven and
where its evidence lives, and anything the interview could not settle. Point out that `/run` will now
use this skill to launch the app, and that a CoS implementer brief can name it as the "done" check.

## Keeping it honest

The map rots the moment the app changes. When a verification run finds the map wrong, fix the map or
the harness in the same change — never product code — and say which. A behavior the map describes that
the app no longer does is either doc drift (fix the map) or a product regression (report it, do not
paper over it in docs).

---
Adapted from `create-verification-skill` in github.com/cursor/plugins (pstack, MIT, © 2026 Lauren Tan; commit 7314f72), rewritten for Claude Code project
skills, Herdr, uv, and the CoS delegate contract.
