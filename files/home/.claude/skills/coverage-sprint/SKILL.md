---
name: coverage-sprint
description: Systematically increase test coverage and mutation score. Use when asked to bump coverage, improve test quality, or reach a coverage target. Triggers on "coverage", "bump tests", "increase coverage", "test gaps".
---

# Coverage Sprint

Systematically increase test coverage and mutation test scores toward a target.

## Usage

```
/coverage-sprint                     - Analyze current state, propose plan
/coverage-sprint 80                  - Sprint toward 80% coverage target
/coverage-sprint mutate              - Focus on mutation score improvement
/coverage-sprint mutate 75           - Sprint toward 75% mutation score
```

---

## Phase 1: Baseline

Gather current metrics before making any changes.

1. **Find the test and coverage commands** by checking `package.json` scripts, `Makefile`, `pyproject.toml`, or equivalent
2. **Run coverage report:**
   - JS/TS: `npx vitest run --coverage` or `npm run test:coverage`
   - Python: `pytest --cov --cov-report=term-missing`
3. **Run mutation tests if available:**
   - JS/TS: `npx stryker run` or `npm run test:mutate`
   - Python: `mutmut run`
4. **Record baseline:**
   - Overall coverage %
   - Mutation score % (if available)
   - List of files with lowest coverage

**Present to user:**
```
Baseline:
- Coverage: X% (target: Y%)
- Mutation score: X% (target: Y%)
- Gap: Z percentage points

Lowest coverage files:
1. path/to/file.ts (12%)
2. path/to/other.ts (34%)
...
```

---

## Phase 2: Prioritize

Rank files by impact. Not all coverage gaps are equal.

**Priority order:**
1. **Business logic with 0% coverage** - highest risk
2. **Files with survived mutants** - tests exist but don't catch bugs
3. **Core domain files with low coverage** - high business value
4. **Utility/helper files** - lower priority
5. **Wiring/config code** - lowest priority, often exclude from targets

**Skip these entirely (add to exclusions if not already):**
- Generated files
- Type definitions only
- Migration files
- Config/wiring with no logic
- Third-party adapters with trivial delegation

**Present the prioritized plan to user as batches of 3-5 files each.**

---

## Phase 3: Execute (per batch)

For each batch of files:

### 3a. Analyze gaps

Read the file. Identify:
- Uncovered branches (if/else, switch, error paths)
- Survived mutants (if mutation report available)
- Edge cases not tested (null, empty, boundary values)

### 3b. Write tests

**Quality principles:**
- Test behavior, not implementation
- Prefer real backends (Supabase local, test DB) over mocks
- Only mock external services (APIs, email, etc.)
- Each test should catch at least one potential bug
- Use Given/When/Then structure
- Name tests after the behavior they verify

**Anti-patterns to avoid:**
- Tests that only verify mocks were called
- Tests that duplicate existing coverage
- Tests that test framework/library behavior
- Snapshot tests for logic (use assertions)

### 3c. Verify

1. Run full test suite - all must pass
2. Run coverage - verify improvement
3. Run mutation tests on changed files if available:
   ```
   npx stryker run --mutate "path/to/file.ts"
   ```

### 3d. Report batch progress

```
Batch N complete:
- Coverage: X% -> Y% (+Z%)
- Files covered: file1.ts, file2.ts
- Tests added: N
- Remaining to target: Z%
```

### 3e. Commit the batch

Use descriptive commit message: `test: add coverage for <module/behavior>`

---

## Phase 4: Final Report

After all batches or when target reached:

```
Coverage Sprint Complete:
- Coverage: X% -> Y% (target was Z%)
- Mutation score: X% -> Y%
- Tests added: N
- Files improved: N
- Exclusions added: [list]
```

**If target not reached**, explain why:
- Remaining uncovered code is wiring/config
- Would require integration test infrastructure not available
- Diminishing returns on remaining files

---

## Key Rules

- NEVER delete existing tests to improve coverage %
- NEVER add tests that test nothing (empty assertions, mock-only tests)
- ALWAYS run the full test suite after each batch to catch regressions
- ALWAYS prefer testing against real backends when available
- Commit after each batch so progress is saved
- If mutation testing is set up, use survived mutants to guide test writing
- Set `ignoreStatic: true` in Stryker config if not already set
