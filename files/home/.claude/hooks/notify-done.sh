#!/usr/bin/env bash
# Notify when Claude Code finishes — plays a chime and shows a notification
# in iTerm2 via terminal escape sequences (works over SSH).

message="Claude is done"

# Bind fd 3 to wherever the escape sequences should go.
#
# /dev/tty only exists when the process has a controlling terminal (e.g. an
# interactive session). When Claude Code runs this hook without one (common on
# a VPS), opening /dev/tty fails and we fall back to stderr.
#
# That fallback must dup fd 2 (`>&2`), not open the /dev/stderr *path*.
# /dev/stderr symlinks to /proc/self/fd/2, and Claude Code hands hooks a socket
# as stderr — opening a socket by path fails with ENXIO ("No such device or
# address"), which spammed that error on every hook run. Duplicating the
# descriptor sidesteps the reopen entirely.
if { : > /dev/tty; } 2>/dev/null; then
    exec 3>/dev/tty
else
    exec 3>&2
fi

# OSC 9 triggers an iTerm2 Notification Center banner
if [ -n "$TMUX" ]; then
    printf '\ePtmux;\e\e]9;%s\a\e\\' "$message" >&3
else
    printf '\e]9;%s\a' "$message" >&3
fi

# BEL triggers iTerm2 bell sound
printf '\a' >&3

exec 3>&-
