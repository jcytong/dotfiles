# /green - Make Test Pass

Make the failing test pass with the simplest possible code.

## Usage

```
/green
```

No arguments needed - works on the current failing test.

## Process

Follow @agents/tdd/green-phase.md to:

1. **Read the failure** - What exactly is expected?
2. **Write minimal code** - Simplest thing that works
3. **Allow shortcuts** - Hardcoding, duplication OK
4. **Verify it passes** - All tests green
5. **Note refactoring opportunities** - For later

## Shameless Green

These are acceptable:

- **Hardcoded values**: Return exactly what test expects
- **Duplication**: Copy-paste similar code
- **Simple conditionals**: if/elif chains
- **No abstractions**: Save that for refactor phase

## References

- @agents/tdd/green-phase.md - Full green phase instructions

## Output

1. Implementation code (minimal)
2. Test run output showing pass
3. Notes for refactor phase (if any)

## Rules

- Make test pass with LEAST code
- All existing tests must still pass
- No refactoring
- No premature abstractions
- No new tests
