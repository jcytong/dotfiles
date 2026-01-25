# /red - Write Failing Test

Write exactly one failing test for the specified behavior.

## Usage

```
/red [behavior description]
```

## Examples

```
/red validates user email format
/red calculates shipping cost for international orders
/red rejects expired authentication tokens
/red sends welcome email after registration
```

## Process

Follow @agents/tdd/red-phase.md to:

1. **Understand the behavior** - What should happen?
2. **Choose test level** - Unit, integration, or acceptance?
3. **Write ONE test** - Arrange/Act/Assert structure
4. **Verify it fails** - With informative diagnostic
5. **Report the failure** - Show the test and error message

## References

- @agents/tdd/red-phase.md - Full red phase instructions
- @prompts/decision-trees/test-granularity.md - Choosing test level
- @prompts/decision-trees/when-to-mock.md - Mock decisions

## Output

1. The test code (single test)
2. Test run output showing failure
3. Clear failure message

## Rules

- Write exactly ONE test
- Test must FAIL (not pass, not syntax error)
- Test name describes behavior
- No implementation code
