#!/usr/bin/env bash
# Notify when Claude Code finishes — plays a chime and shows a notification
# in iTerm2 via terminal escape sequences (works over SSH).

message="Claude is done"

# OSC 9 triggers an iTerm2 Notification Center banner
if [ -n "$TMUX" ]; then
    printf '\ePtmux;\e\e]9;%s\a\e\\' "$message" > /dev/tty
else
    printf '\e]9;%s\a' "$message" > /dev/tty
fi

# BEL triggers iTerm2 bell sound
printf '\a' > /dev/tty
