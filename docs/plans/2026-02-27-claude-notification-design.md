# Claude Code Done Notification

## Problem

Claude Code runs on a VPS. When it finishes a response, there's no way to
know without watching the terminal. We want an audible chime and a macOS
notification banner on the MacBook connected via SSH + iTerm2.

## Approach

Use terminal escape sequences emitted by a Claude Code hook script. No
external services or dependencies.

- **BEL character** (`\a`) — iTerm2 plays a bell sound
- **OSC 9** (`\e]9;message\a`) — iTerm2 shows a Notification Center banner

Both work over SSH. OSC 9 needs DCS passthrough wrapping when inside tmux.

## Components

### 1. Hook script: `~/.claude/hooks/notify-done.sh`

Emits OSC 9 (with tmux DCS passthrough if needed) then BEL. ~15 lines of bash.

### 2. Claude Code settings.json

Add `Stop` and `Notification` hooks pointing to the script.

### 3. tmux.conf

Add `set -g allow-passthrough on` so tmux forwards DCS-wrapped OSC 9.

### 4. iTerm2 (manual, on Mac)

Enable in Preferences > Profile > Terminal:
- "Notification center alerts" (bell triggers macOS notification)
- "Bell" (bell plays sound)

## Files changed

| File | Change |
|------|--------|
| `files/home/.claude/hooks/notify-done.sh` | New — hook script |
| `files/home/.claude/settings.json` | Add Stop + Notification hooks |
| `files/home/.tmux.conf` | Add allow-passthrough on |
