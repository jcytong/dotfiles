# AGENTS.md

This file provides guidance to AI coding assistants when working with code in this repository.

## Repository Purpose

Personal dotfiles repository. Files in `files/home/` get symlinked to `~/`.

## Commands

```bash
./bin/dotf link    # Symlink files/home/* to ~/
```

## Adding Dotfiles

Place files in `files/home/` with the same path they should have relative to `~`:

- `files/home/.gitconfig` → `~/.gitconfig`
- `files/home/.config/fish/config.fish` → `~/.config/fish/config.fish`

The script creates parent directories automatically and skips existing non-symlink files.
