#!/bin/bash

# Read JSON input from stdin
input=$(cat)

# Extract data from JSON
model_name=$(echo "$input" | jq -r '.model.display_name // "Claude"')
project_dir=$(echo "$input" | jq -r '.workspace.current_dir // ""' | xargs basename)
used=$(echo "$input" | jq -r '.context_window.used_percentage // 0')
session_id=$(echo "$input" | jq -r '.session_id // ""')

# Get git branch (skip git locks for performance)
git_branch=$(cd "$(echo "$input" | jq -r '.workspace.current_dir // "."')" 2>/dev/null && git -c core.fileMode=false -c advice.detachedHead=false rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

# Per-session cost & duration via baseline subtraction
# cost.total_cost_usd and total_duration_ms are cumulative across ALL sessions,
# so we save the initial values at session start and subtract to get per-session.
cumulative_cost=$(echo "$input" | jq -r '.cost.total_cost_usd // 0')
cumulative_dur=$(echo "$input" | jq -r '.cost.total_duration_ms // 0')

baseline_file="/tmp/claude-baseline-${session_id}"
if [ -n "$session_id" ] && [ ! -f "$baseline_file" ]; then
    echo "${cumulative_cost} ${cumulative_dur}" > "$baseline_file"
fi

if [ -f "$baseline_file" ]; then
    read -r base_cost base_dur < "$baseline_file"
    session_cost=$(echo "scale=4; $cumulative_cost - $base_cost" | bc -l 2>/dev/null || echo "0")
    session_dur_ms=$(echo "$cumulative_dur - $base_dur" | bc 2>/dev/null || echo "0")
else
    session_cost="$cumulative_cost"
    session_dur_ms="$cumulative_dur"
fi

total_cost=$(printf "%.2f" "$session_cost" 2>/dev/null || echo "0.00")
duration_s=$((session_dur_ms / 1000))
minutes=$((duration_s / 60))
seconds=$((duration_s % 60))
duration_str="${minutes}m ${seconds}s"

# Abbreviate model name and assign color (darker = more intelligent)
case "$model_name" in
    *"Opus"*)
        model_abbr="Opus"
        color="\033[38;5;54m"  # Dark purple (most intelligent)
        ;;
    *"Sonnet"*)
        model_abbr="Sonnet"
        color="\033[38;5;135m"  # Medium purple
        ;;
    *"Haiku"*)
        model_abbr="Haiku"
        color="\033[38;5;183m"  # Light purple (least intelligent)
        ;;
    *)
        model_abbr="$model_name"
        color="\033[35m"  # Default purple
        ;;
esac
reset="\033[0m"  # Reset color

# Effective context usage accounting for autocompact buffer (~17% of context window)
# Raw "used" can show 76% but only 8% is actually free because the buffer eats the rest.
# effective_used shows how full the USABLE portion is — 100% means autocompact is imminent.
buffer_pct=17
usable_pct=$((100 - buffer_pct))
effective_used=$(echo "scale=0; $used * 100 / $usable_pct" | bc 2>/dev/null || echo "0")
[ "$effective_used" -gt 100 ] && effective_used=100

# Context bar color: green → yellow → orange → red as effective usage increases
ctx_colors=(46 82 118 154 190 226 220 214 208 202 196)
ctx_idx=$((effective_used / 10))
[ "$ctx_idx" -gt 10 ] && ctx_idx=10
ctx_color="\033[38;5;${ctx_colors[$ctx_idx]}m"

bar_length=10
filled=$(echo "scale=0; $effective_used * $bar_length / 100" | bc 2>/dev/null || echo "0")
empty=$((bar_length - filled))

bar="${ctx_color}"
for ((i=0; i<filled; i++)); do bar="${bar}█"; done
bar="${bar}${reset}"
for ((i=0; i<empty; i++)); do bar="${bar}░"; done

context_str="${bar} ${ctx_color}${effective_used}%${reset}"

# Build status line
# Line 1: model | folder | branch
printf "${color}[%s]${reset} | %s" "$model_abbr" "$project_dir"
[ -n "$git_branch" ] && printf " | %s" "$git_branch"
printf "\n"

# Line 2: context used | cost | duration
printf "%b | \$%s | %s\n" "$context_str" "$total_cost" "$duration_str"
