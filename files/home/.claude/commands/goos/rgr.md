# /rgr - Red-Green-Refactor Cycle

Execute a complete TDD cycle for a behavior: write failing test, make it pass, clean up.

## Usage

```
/rgr [behavior description]
```

## Examples

```
/rgr validates-email-format
/rgr calculates-order-total
/rgr rejects-expired-tokens
/rgr sends-confirmation-email
```

## Process

This orchestrates the full TDD cycle:

### Phase 1: Red
Follow @agents/tdd/red-phase.md

- Write exactly ONE failing test
- Ensure clear diagnostic message
- Test behavior, not implementation

### Phase 2: Green
Follow @agents/tdd/green-phase.md

- Write simplest code that passes
- Hardcoding allowed
- Duplication allowed
- Note debt for refactor

### Phase 3: Refactor
Follow @agents/tdd/refactor-phase.md

- Apply GOOS techniques (Break Out, Bud Off, Bundle Up)
- Run tests after every change
- Stop when no clear pattern

## Workflow

```
/rgr validates-email-format
       │
       ├─→ RED: Write test_rejects_invalid_email_format()
       │        Run → FAIL (expected)
       │
       ├─→ GREEN: Add minimal validation
       │          Run → PASS
       │
       └─→ REFACTOR: Extract EmailValidator if pattern emerges
                     Run → PASS
```

## References

- @agents/tdd/red-phase.md - Red phase details
- @agents/tdd/green-phase.md - Green phase details
- @agents/tdd/refactor-phase.md - Refactor phase details
- @prompts/decision-trees/refactor-or-not.md - When to refactor
- @prompts/goos.md - GOOS principles

## Output

After each phase:
1. Code changes
2. Test results
3. Phase completion status

Final output:
1. Test code
2. Implementation code
3. Refactorings applied (if any)
4. All tests passing

## Rules

- One behavior per /rgr invocation
- Red must fail before green
- Green must pass before refactor
- Tests must pass after every refactor step
- Stop refactoring when no clear improvement
