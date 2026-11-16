#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dotf-test.XXXXXX")"
TEST_DIR="$(cd "$TEST_DIR" && pwd)"
trap 'rm -rf "$TEST_DIR"' EXIT

fixture() {
  repo="$TEST_DIR/$1/repo"
  home_dir="$TEST_DIR/$1/home"
  mkdir -p "$repo/bin" "$repo/files/home/.agents/skills/shared" "$home_dir"
  cp "$REPO_DIR/bin/dotf" "$repo/bin/dotf"
  printf 'shared skill\n' > "$repo/files/home/.agents/skills/shared/SKILL.md"
  printf 'shared instructions\n' > "$repo/files/home/.agents/AGENTS.md"
  printf 'managed shell config\n' > "$repo/files/home/.zshrc"
  # The private submodule may not be initialized yet.
  ln -s ../agent-skills/skills/team "$repo/files/home/.agents/skills/team"
}

dotf() {
  HOME="$home_dir" "$repo/bin/dotf" "$@"
}

assert_link() {
  [ -L "$1" ] && [ "$(readlink "$1")" = "$2" ] || {
    echo "FAIL: $1 should link to $2" >&2
    exit 1
  }
}

fixture fresh
dotf --dry-run > "$TEST_DIR/dry-run.log"
[ -z "$(ls -A "$home_dir")" ]
dotf link > "$TEST_DIR/link.log"
dotf verify > "$TEST_DIR/verify.log"
assert_link "$home_dir/.agents/skills" "$repo/files/home/.agents/skills"
assert_link "$home_dir/.codex/skills/team" "$repo/files/home/.agents/agent-skills/skills/team"
dotf link > "$TEST_DIR/repeat.log"
[ ! -s "$TEST_DIR/repeat.log" ]
echo "PASS: fresh install, dry-run, verification, and repeat install"

fixture local-skills
mkdir -p "$home_dir/.agents/skills/local"
printf 'local skill\n' > "$home_dir/.agents/skills/local/SKILL.md"
dotf --dry-run > "$TEST_DIR/dry-run.log"
[ ! -e "$home_dir/.agents/skills/shared" ]
[ ! -L "$home_dir/.agents/skills/team" ]
dotf link > "$TEST_DIR/link.log"
dotf verify > "$TEST_DIR/verify.log"
[ ! -L "$home_dir/.agents/skills" ]
[ "$(cat "$home_dir/.agents/skills/local/SKILL.md")" = 'local skill' ]
assert_link "$home_dir/.agents/skills/shared" "$repo/files/home/.agents/skills/shared"
assert_link "$home_dir/.agents/skills/team" "$repo/files/home/.agents/agent-skills/skills/team"
dotf link > "$TEST_DIR/repeat.log"
[ ! -s "$TEST_DIR/repeat.log" ]
echo "PASS: existing canonical directory preserves local skills and resolves relative links"

fixture legacy
mkdir -p "$home_dir/.claude" "$home_dir/.codex" "$repo/files/home/.codex/skills"
ln -s "$TEST_DIR/old-skill" "$repo/files/home/.codex/skills/old"
# Exercise both a broken absolute link and a populated relative link.
ln -s "$repo/files/home/.claude/skills" "$home_dir/.claude/skills"
ln -s ../../repo/files/home/.codex/skills "$home_dir/.codex/skills"
dotf --dry-run > "$TEST_DIR/dry-run.log"
assert_link "$home_dir/.claude/skills" "$repo/files/home/.claude/skills"
assert_link "$home_dir/.codex/skills" ../../repo/files/home/.codex/skills
dotf link > "$TEST_DIR/link.log"
dotf verify > "$TEST_DIR/verify.log"
[ -d "$home_dir/.claude/skills" ] && [ ! -L "$home_dir/.claude/skills" ]
[ -d "$home_dir/.codex/skills" ] && [ ! -L "$home_dir/.codex/skills" ]
assert_link "$home_dir/.claude/skills/shared" "$repo/files/home/.agents/skills/shared"
assert_link "$repo/files/home/.codex/skills/old" "$TEST_DIR/old-skill"
echo "PASS: legacy managed skill roots migrate without removing source contents"

fixture unrelated-link
mkdir -p "$home_dir/.claude" "$TEST_DIR/unrelated-skills"
printf 'keep me\n' > "$TEST_DIR/unrelated-skills/local.txt"
ln -s "$TEST_DIR/unrelated-skills" "$home_dir/.claude/skills"
dotf link > "$TEST_DIR/link.log"
assert_link "$home_dir/.claude/skills" "$TEST_DIR/unrelated-skills"
[ "$(cat "$TEST_DIR/unrelated-skills/local.txt")" = 'keep me' ]
[ ! -L "$TEST_DIR/unrelated-skills/shared" ]
if dotf verify > "$TEST_DIR/verify.log"; then
  echo 'FAIL: verify should report the unrelated skills root' >&2
  exit 1
fi
echo "PASS: unrelated root symlinks stay unchanged and fail verification"

fixture conflicts
mkdir -p "$home_dir/.agents/skills/shared"
printf 'local shared skill\n' > "$home_dir/.agents/skills/shared/SKILL.md"
printf 'local shell config\n' > "$home_dir/.zshrc"
dotf link > "$TEST_DIR/link.log"
[ "$(cat "$home_dir/.agents/skills/shared/SKILL.md")" = 'local shared skill' ]
[ "$(cat "$home_dir/.zshrc")" = 'local shell config' ]
if dotf verify > "$TEST_DIR/verify.log"; then
  echo 'FAIL: verify should report unmanaged file and skill conflicts' >&2
  exit 1
fi
echo "PASS: conflicting regular configs and skill directories stay unchanged"
