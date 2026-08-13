# dotfiles

Personal dotfiles managed with symlinks.

## Setup

```bash
git clone git@github.com:jcytong/dotfiles.git ~/.dotfiles
cd ~/.dotfiles
./bin/dotf link
```

### herdr

`~/.config/herdr/config.toml` is tracked. The Claude integration is not —
herdr generates and version-stamps `~/.claude/hooks/herdr-agent-state.sh`,
so vendoring it would pin a stale copy. Regenerate it after linking:

```bash
herdr integration install claude
```

That writes the hook script and a `SessionStart` entry into the untracked
`~/.claude/settings.json`. To keep the hook across machines, move that entry
into `.claude/settings.shared.json`, which is the tracked, durable config.

## Adding Dotfiles

Put files in `files/home/` mirroring your home directory structure:

```
files/home/.gitconfig        → ~/.gitconfig
files/home/.config/nvim/     → ~/.config/nvim/
files/home/.zshrc            → ~/.zshrc
```

Then run `./bin/dotf link` to create symlinks.

## Structure

```
bin/dotf       Symlink script
files/home/    Dotfiles (mirrors ~/)
```
