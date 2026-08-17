---
name: opencode
description: CLI commands and usage for OpenCode, an open-source AI coding agent.
---

# OpenCode CLI

OpenCode is an open-source AI coding agent available as a TUI, desktop app, IDE extension, and CLI. This skill covers CLI usage, especially for non-interactive scripting and file context injection.

## Quick Reference

### Run non-interactively
```bash
opencode run "your prompt here"
```

### Attach files to a prompt
Use `-f` or `--file` to include file contents in the context:
```bash
opencode run "explain this function" -f src/utils.ts
opencode run "compare these" -f file1.ts -f file2.ts
```

### Common Flags
| Flag | Short | Description |
|------|-------|-------------|
| `--model` | `-m` | Model to use (e.g., `anthropic/claude-sonnet-4-20250514`) |
| `--continue` | `-c` | Continue the last session |
| `--session` | `-s` | Continue a specific session ID |
| `--fork` | | Fork the session when continuing |
| `--share` | | Share the session and print URL |
| `--format json` | | Output raw JSON events for programmatic parsing |
| `--dangerously-skip-permissions` | | Auto-approve all tool calls (use with caution) |
| `--agent` | | Use a specific agent |
| `--title` | | Set session title |
| `--attach` | | Attach to a running `opencode serve` instance |

### Session Management
```bash
opencode session list          # List sessions
opencode session delete <id>   # Delete a session
```

### Other Commands
- `opencode` or `opencode tui`: Start the interactive TUI
- `opencode serve`: Start a headless HTTP server
- `opencode web`: Start web interface
- `opencode models`: List available models
- `opencode stats`: Show token usage and costs
- `opencode auth login`: Configure provider API keys

## Best Practices
- Always use absolute paths or paths relative to the project root when passing files.
- For scripting, combine `--format json` with `jq` to extract responses.
- Use `--continue` to maintain conversation history across multiple `run` invocations.
- Use `--attach` with a persistent `opencode serve` process to avoid MCP cold-start delays.
