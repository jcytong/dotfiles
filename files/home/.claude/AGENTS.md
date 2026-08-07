# Instructions

## Python

- `uv` is the package manager — not pip, pip-tools, or poetry
- `pyproject.toml` for config — not requirements.txt or setup.py
- `uv run <cmd>`, `uv add <pkg>`, `uv add --dev <pkg>`, `uv init` for new projects

## Tools

- **ast-grep** — available for AST-pattern search and rewrite; prefer it for large-scale refactors
- **gh** — for anything GitHub. Given an issue/PR URL or `/pull/5`, use `gh`, never web search.
  `gh pr view <url> --comments --files -R owner/repo`
- **gws** — CLI for my Google Workspace (Drive, Docs, Sheets, Gmail, Calendar, Slides, Tasks,
  People, Chat, Forms, Keep, Meet, Apps Script). When I reference anything in my Workspace,
  fetch it with `gws` — don't ask me to paste what you can retrieve.
  - Shape: `gws <service> <resource> [sub-resource] <method> [flags]`
  - Discover: `gws <service> --help`, then `gws schema <service.resource.method>` for params
  - Flags: `--params '<JSON>'` (query/URL), `--json '<JSON>'` (body), `--format json|table|csv`,
    `--page-all`, `--output <PATH>`
  - Given a Google link, extract the ID from the URL (`/document/d/<ID>/edit`) and call `gws`

## Git

- Stage files individually: `git add <file1> <file2>`. Never `git add .`, `git add -A`, or
  `git commit -am`. Only stage changes you made yourself.
- Single-quote paths containing `$`: `git add 'app/routes/_protected.foo.$bar.tsx'`
- If my prompt was a compiler or linter error, use a `fixup!` commit message
- Otherwise commit messages: present-tense verb (Fix, Add, Implement), single line, 60–120 chars,
  reads like the title of the issue we resolved. No implementation details discovered along the
  way. No praise adjectives (comprehensive, essential, best practices).
- No attribution footer — no `🤖 Generated with…`, no `Co-Authored-By: Claude`
- Echo exactly `Ready to commit: git commit --message "<message>"`, confirm with me, then run
  that same command

## Code style

- Refactor large files and functions into smaller focused units as you go
- Ruby / OO design → the `sandi-metz-rules` skill (or `/sandi`)
- Outside-in TDD, walking skeletons, mock-roles-not-objects → the `goos` skill
- Frontend visual design → the `frontend-design` skill
