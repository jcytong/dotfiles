# Managed by dotfiles — keep machine-specific config in ~/.zshrc.local

DISABLE_AUTO_UPDATE="true"
DISABLE_MAGIC_FUNCTIONS="true"
DISABLE_COMPFIX="true"

# Enable Powerlevel10k instant prompt. Should stay close to the top of ~/.zshrc.
# Initialization code that may require console input (password prompts, [y/n]
# confirmations, etc.) must go above this block; everything else may go below.
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# Path to your oh-my-zsh configuration.
ZSH=$HOME/.oh-my-zsh

# Set name of the theme to load.
# Look in ~/.oh-my-zsh/themes/
# Optionally, if you set this to "random", it'll load a random theme each
# time that oh-my-zsh is loaded.
ZSH_THEME="powerlevel10k/powerlevel10k"

# Set to this to use case-sensitive completion
CASE_SENSITIVE="true"

# Which plugins would you like to load? (plugins can be found in ~/.oh-my-zsh/plugins/*)
# Custom plugins may be added to ~/.oh-my-zsh/custom/plugins/
# Example format: plugins=(rails git textmate ruby lighthouse)
plugins=(git fzf-tab fzf-zsh-plugin)

source $ZSH/oh-my-zsh.sh

unsetopt SHARE_HISTORY
# Filter zsh's completion candidates - http://superuser.com/questions/427985/how-can-i-filter-zshs-completion-candidates
setopt GLOB_COMPLETE

bindkey -v
bindkey '^R' history-incremental-search-backward

# set up variables for development environment
export EDITOR=vim
export GIT_EDITOR=vim

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# ── Machine-specific config (API keys, PATH, tool inits) ──
[[ -f ~/.zshrc.local ]] && source ~/.zshrc.local

portopen() {
  lsof -nP -iTCP:$1 | grep LISTEN
}

br-name() {
  hub issue show --format '%I-%t' $1 | tr '[:upper:]' '[:lower:]' | sed 's/[[:space:]]\{2,\}/~/g' | sed 's/ /-/g';
}

alias ls='gls --color=auto'
alias dcom='docker compose'
alias top10='du -sh * | sort -rh | head -n 10'
alias glod='git log --graph --pretty="%Cgreen%h%Creset%Cblue%d%Creset %Cred%an%Creset: %s"'
alias r=rails
alias hubr="hub issue |  sed 's/[[:space:]]\{2,\}/~/g' | cut -d '~' -f 2,3 | sed 's/#//g' | sed 's/ /-/g' | sed 's/~/-/g' | tr '[:upper:]' '[:lower:]'"
alias speedtest="networkQuality -v"
alias start_open_webui="docker run -d -p 11414:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main"
alias upgrade_webui='CURRENT_WEBUI_ID=$(docker ps -f "name=open-webui" -q) && \
                       if [ ! -z "$CURRENT_WEBUI_ID" ]; then \
                         echo "Stopping container: $CURRENT_WEBUI_ID"; \
                         docker stop "$CURRENT_WEBUI_ID"; \

                         echo "Removing container: $CURRENT_WEBUI_ID"; \
                         docker rm "$CURRENT_WEBUI_ID"; \

                         echo "Removing dangling open-webui images"; \
                         sleep 1
                         echo "."
                         sleep 1
                         echo "."
                         docker images -f "dangling=true" --format "{{.ID}} {{.Repository}}:{{.Tag}}" | grep -E "open-webui:<none>" |  awk "{print $1}" | xargs -r docker rmi; \

                         echo "Pulling ghcr.io/open-webui/open-webui:main"; \
                         docker pull ghcr.io/open-webui/open-webui:main; \
                         start_open_webui; \
                       else \
                         echo "No container found"; \
                       fi'
alias start_caddy="Caddy run --config ${HOME}/localhost/Caddyfile"
alias start_n8n="docker run -it --rm --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n"
alias python="uv run python"
alias python3="uv run python"

HISTTIMEFORMAT="%d/%m/%y %T "

# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh.
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh
export FZF_DEFAULT_OPS="--extended"

wt() {
  # Use local error handling instead of set -e
  local project_dir
  local project_name
  local branch_name="$1"
  local feature_name

  # Get the current Git project directory (must be inside a Git repo)
  if ! project_dir=$(git rev-parse --show-toplevel 2>/dev/null); then
    echo "Error: Not in a Git repository"
    return 1
  fi

  # Get the base name of the current project folder
  project_name=$(basename "$project_dir")

  # Fail fast if no feature name was provided
  if [ -z "$branch_name" ]; then
    echo "Usage: wt <feature-name>"
    return 1
  fi

  # Convert branch name to safe directory name by replacing / with -
  feature_name="${branch_name//\//-}"

  # Define the parent folder where all worktrees go, beside the main repo
  local worktree_parent="$(dirname "$project_dir")/${project_name}-worktrees"
  # Define the full path of the new worktree folder
  local worktree_path="${worktree_parent}/${feature_name}"

  # Create the parent worktrees folder if it doesn't exist
  mkdir -p "$worktree_parent" || { echo "Failed to create worktree parent directory"; return 1; }

  # Check if branch exists (locally or as remote tracking branch)
  local branch_exists=false
  local branch_ref=""

  # Check for local branch
  if git -C "$project_dir" show-ref --verify --quiet "refs/heads/$branch_name"; then
    branch_exists=true
    branch_ref="$branch_name"
    echo "Branch '$branch_name' already exists locally."
  # Check for remote branch (tries all remotes)
  elif git -C "$project_dir" ls-remote --heads --quiet | grep -q "[[:space:]]refs/heads/$branch_name$"; then
    branch_exists=true
    # Find which remote has this branch
    local remote=$(git -C "$project_dir" ls-remote --heads | grep "[[:space:]]refs/heads/$branch_name$" | cut -f2 | cut -d'/' -f3- | head -1)
    branch_ref="remotes/origin/$branch_name"  # Assuming origin, but could be enhanced
    echo "Branch '$branch_name' exists on remote."
  fi

  # Handle existing vs new branch
  if [ "$branch_exists" = true ]; then
    # Ask user if they want to create worktree with existing branch
    echo -n "Do you want to create a worktree with the existing branch '$branch_name'? (y/n): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
      echo "Aborted."
      return 1
    fi

    # Create worktree with existing branch
    if ! git -C "$project_dir" worktree add "$worktree_path" "$branch_ref"; then
      echo "Failed to create worktree with existing branch"
      return 1
    fi
  else
    # Create the worktree and the branch (use original branch_name for git)
    if ! git -C "$project_dir" worktree add -b "$branch_name" "$worktree_path"; then
      echo "Failed to create worktree"
      return 1
    fi
  fi

  # Copy env if it exists
  if [ -f "$project_dir/.env" ]; then
    cp "$project_dir/.env" "$worktree_path/.env"
    cp $project_dir/.env.* "$worktree_path/" 2>/dev/null
    echo "Copied .env into worktree."
  fi

  # Copy if it exists
  if [ -f "$project_dir/.mcp.json" ]; then
    cp "$project_dir/.mcp.json" "$worktree_path/.mcp.json"
    echo "Copied .mcp.json into worktree."
  fi

  # List of hidden folders to copy if they exist and don't already exist in worktree
  local hidden_dirs=(.claude .cursor)
  for dir in "${hidden_dirs[@]}"; do
    if [ -d "$project_dir/$dir" ]; then
      if [ -d "$worktree_path/$dir" ]; then
        echo "Skipped $dir (already exists in worktree)."
      else
        cp -R "$project_dir/$dir" "$worktree_path/$dir"
        echo "Copied $dir into worktree."
      fi
    fi
  done

  # Open the worktree in Cursor
  cursor "$worktree_path" &

  # Confirm success
  echo "Worktree '$branch_name' created at $worktree_path and checked out."
}
