#!/usr/bin/env bash
# Notify when Claude Code finishes — plays a chime and shows a notification
# in iTerm2 via terminal escape sequences (works over SSH).

message="Claude is done"

# Pick an output target for the terminal escape sequences.
# /dev/tty only exists when the process has a controlling terminal (e.g. an
# interactive session). When Claude Code runs this hook without one (common on
# a VPS), opening /dev/tty fails, so fall back to stderr, which is normally
# still connected to the terminal.
if { : > /dev/tty; } 2>/dev/null; then
    out=/dev/tty
else
    out=/dev/stderr
fi

# OSC 9 triggers an iTerm2 Notification Center banner
if [ -n "$TMUX" ]; then
    printf '\ePtmux;\e\e]9;%s\a\e\\' "$message" > "$out"
else
    printf '\e]9;%s\a' "$message" > "$out"
fi

# BEL triggers iTerm2 bell sound
printf '\a' > "$out"
