#!/usr/bin/env bash
# Create a fresh scratch dir for one delegation and print its path.
set -euo pipefail
d="$HOME/.cache/pi-delegate/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$d" && echo "$d"
