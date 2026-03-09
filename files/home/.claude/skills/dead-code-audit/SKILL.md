---
name: dead-code-audit
description: Systematically identify and remove dead code with evidence. Use when asked to find unused code, clean up, remove dead paths, or audit for dead code. Triggers on "dead code", "unused code", "clean up", "dormant code".
---

# Dead Code Audit

Systematically find, prove, and remove dead code from a codebase.

## Usage

```
/dead-code-audit                  - Full audit of the project
/dead-code-audit src/module/      - Audit a specific directory
/dead-code-audit --report-only    - Generate report without removing anything
```

---

## Phase 1: Catalog

Scan the codebase and build a catalog of suspected dead code.

### 1a. Static analysis

Use available tools to find unreferenced code:

- **JS/TS:** Check for unused exports, unreferenced files
  ```bash
  # Find files not imported anywhere
  # Check for exported symbols with zero importers
  ```
- **Python:** Check for unused imports, unreferenced modules
- **General:** Search for TODO/FIXME/DEPRECATED/HACK comments that indicate abandoned code

### 1b. Pattern-based detection

Search for these dead code patterns:

| Pattern | How to detect |
|---|---|
| **Unreachable functions** | Exported but never imported; defined but never called |
| **Dead feature flags** | `if (false)`, `if (FEATURE_X)` where FEATURE_X is always false |
| **Commented-out code** | Large blocks of `//` or `/* */` commented code |
| **Deprecated paths** | Functions/routes marked deprecated with no callers |
| **Orphaned tests** | Tests for functions/modules that no longer exist |
| **Unused dependencies** | Packages in package.json/pyproject.toml never imported |
| **Dead config** | Config keys that are read but never used in logic |
| **Legacy adapters** | Adapters for systems that have been replaced (e.g., old DB layer) |
| **Unused CLI commands** | Registered commands/scripts that serve no current purpose |

### 1c. Build the catalog

For each suspect, record:
- File path and line range
- Type (function, class, module, config, dependency)
- Confidence level: HIGH / MEDIUM / LOW
- Evidence (no callers found, deprecated comment, etc.)

---

## Phase 2: Prove

**Do not remove anything without evidence.** For each catalog entry:

### Evidence levels (strongest to weakest):

1. **Static proof** - Zero references in the entire codebase (grep for all possible call patterns)
2. **Git evidence** - `git log --follow -p <file>` shows when it was last meaningfully used
3. **Runtime evidence** - Logs show the code path was never hit (if available via `gcloud logs` or similar)
4. **Deployment evidence** - Not deployed, not in any build target, not in any route table
5. **Comment evidence** - Marked as deprecated/unused by the author

### Verification checklist for each item:

- [ ] Searched for all references (imports, string references, dynamic requires)
- [ ] Checked if referenced in config files, deploy scripts, CI
- [ ] Checked if referenced in tests (if only tests reference it, flag the tests too)
- [ ] Checked for dynamic/computed references (e.g., `require(variable)`, reflection)
- [ ] Checked git history for when it was last actively used

**Present the proven catalog to the user for approval before removing anything.**

```
DEAD CODE CATALOG
=================

HIGH CONFIDENCE (safe to remove):
1. src/adapters/firestore.ts - entire file
   Evidence: Zero imports. Replaced by Supabase adapter in PR #34.

2. src/utils/legacy_formatter.ts:45-120 - formatV1() function
   Evidence: Zero callers. Last called in commit abc123 (2025-08-15).

MEDIUM CONFIDENCE (likely dead, verify):
3. src/config/feature_flags.ts - ENABLE_SMS_V2 flag
   Evidence: Always set to true. Flag check in 2 locations.

LOW CONFIDENCE (needs investigation):
4. src/handlers/webhook_backup.ts
   Evidence: No direct imports, but registered dynamically in routes.ts
```

---

## Phase 3: Remove

**Only after user approval.** Process in order of confidence.

### For each removal:

1. Remove the dead code
2. Remove any orphaned imports/dependencies created by the removal
3. Remove orphaned tests (tests that only tested the removed code)
4. Run full test suite to verify nothing broke
5. If tests fail, REVERT immediately and downgrade confidence level

### Commit strategy:

- Group removals by logical unit (e.g., "remove firestore adapter" not individual functions)
- Use descriptive commit messages:
  ```
  chore: remove dead firestore adapter code

  Replaced by Supabase in PR #34. Zero references remaining.
  ```
- One commit per logical group so each removal is independently revertable

---

## Phase 4: Cleanup

After removals:

1. **Check for newly orphaned code** - removing one thing may make others dead
2. **Update dependency list** - remove packages that are no longer imported
3. **Update docs** - remove references to deleted modules/features
4. **Run full test suite** one final time
5. **Run coverage** - coverage % should go UP (less code, same tests)

---

## Final Report

```
Dead Code Audit Complete
========================
Removed:
- N files deleted entirely
- N functions/classes removed from existing files
- N dependencies removed
- N orphaned tests removed

Impact:
- Lines removed: N
- Coverage change: X% -> Y%
- Dependencies reduced by: N

Preserved (low confidence, needs manual review):
- path/to/suspicious_file.ts - reason
```

---

## Key Rules

- NEVER remove code without evidence of it being dead
- NEVER remove code without running the full test suite after
- ALWAYS present the catalog for user approval before removing
- ALWAYS make removals independently revertable (separate commits)
- Check for dynamic references (string-based imports, reflection, config-driven routing)
- If unsure, move to a `dormant/` directory instead of deleting
- Treat deploy scripts, CI configs, and route registrations as potential reference sources
