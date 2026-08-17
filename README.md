# dotfiles

Personal dotfiles managed with symlinks, shared across several coding agents.

## Setup

```bash
git clone git@github.com:jcytong/dotfiles.git ~/.dotfiles
cd ~/.dotfiles
./bin/dotf link
```

`./bin/dotf --dry-run` shows what would change; `./bin/dotf verify` checks the
managed links without touching anything.

The installer is safe to rerun. It only ever replaces symlinks it manages, and
never removes an existing non-symlink file or directory — an unmanaged file in
the way is reported and skipped, not clobbered.

Two submodules are private. Without access, `git submodule update` fails and the
skills that link into them stay dangling; everything else still works.

## Skills

`.agents/skills/` is the single source of truth. It is vendor-neutral: skills
live there once, and each agent gets its own view of the same set.

```
files/home/.agents/skills/          canonical store
files/home/.agents/claude-skills/       submodule (team tier)
files/home/.agents/agent-skills-local/  submodule (personal tier)
        │
        ├── ~/.claude/skills/   one symlink per skill   (hardcoded path)
        └── ~/.codex/skills/    one symlink per skill   (hardcoded path)
```

Add an agent to `SHARED_SKILL_LINK_TARGETS` in `bin/dotf` only when its skills
directory is hardcoded. An agent that can be *pointed* at a directory should
read the canonical root directly instead — pi does this via `~/.pi/settings.json`:

```json
{ "skills": ["~/.agents/skills"] }
```

Giving such an agent a mirror as well makes it load every skill twice.

A skill that only makes sense for one agent can be dropped into that agent's
skills directory as a real directory; the installer walks the canonical set and
leaves anything else alone. Set the bar high — provenance is not portability.
Of 43 skills collected from three different agents, only one contained anything
agent-specific.

## Instruction files

Agents share one core of guidance, plus a per-agent overlay:

```
files/home/.agents/AGENTS.md              shared core — every agent
files/home/.agents/overlays/claude.md     appended for Claude only
files/home/.agents/overlays/codex.md      appended for codex only
        │
        ├── ~/.claude/CLAUDE.md    core + claude overlay
        └── ~/.codex/AGENTS.md     core + codex overlay
```

These two are **generated** rather than symlinked, which is the one exception to
how everything else here is deployed. No agent can compose them at read time:

- codex has no import mechanism — an `@path` line is read as literal text.
- Claude's `@path` import expands only when `CLAUDE.md` is a real file. It
  silently stops expanding when the file is a symlink, which is how every other
  dotfile here is deployed. Same-directory, parent-relative, `~/`-relative and
  absolute forms were all tested.

Composing at link time also keeps agent-specific guidance out of the other
agents' context windows, which a shared file would not: instruction files load
in full on every turn, unlike skills, where only the name and description are
read until one is invoked.

Edit the core or an overlay, then rerun `./bin/dotf link`. Do not edit the
generated files — they carry a header saying so, and `verify` reports them as
stale. pi is deliberately absent: it discovers `AGENTS.md`/`CLAUDE.md` by walking
up from the working directory and ignores `~/.pi/AGENTS.md`, so a global file
would be inert.

## herdr

`~/.config/herdr/config.toml` is tracked. The Claude integration is not —
herdr generates and version-stamps `~/.claude/hooks/herdr-agent-state.sh`,
so vendoring it would pin a stale copy. Regenerate it after linking:

```bash
herdr integration install claude
```

That writes the hook script and a `SessionStart` entry into the untracked
`~/.claude/settings.json`. To keep the hook across machines, move that entry
into `.claude/settings.shared.json`, which is the tracked, durable config.

## Adding dotfiles

Put files in `files/home/` mirroring your home directory structure:

```
files/home/.gitconfig        → ~/.gitconfig
files/home/.config/nvim/     → ~/.config/nvim/
files/home/.zshrc            → ~/.zshrc
```

Then run `./bin/dotf link`. Files are linked individually so that unrelated
contents of a directory are left alone; add a path to `DIR_LINKS` to link a
whole directory instead.

Submodule working trees are never linked into `~` file-by-file — they reach the
agents only through the per-skill links in `.agents/skills`. `dotf` reads their
paths from `.gitmodules`, so adding a submodule needs no change to the installer.

## Structure

```
bin/dotf                     Installer: links, composes, verifies
files/home/.agents/          Vendor-neutral agent config (skills, instructions)
files/home/                  Everything else, mirroring ~/
```
