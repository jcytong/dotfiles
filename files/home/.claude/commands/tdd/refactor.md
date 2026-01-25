# /refactor - Clean Up Code

Improve code structure while keeping all tests green.

## Usage

```
/refactor [optional: specific area or concern]
```

## Examples

```
/refactor
/refactor the payment processing logic
/refactor extract Card value object
/refactor reduce duplication in validators
```

## Process

Follow @agents/tdd/refactor-phase.md to:

1. **Verify all tests pass** - Must start green
2. **Identify opportunities** - Duplication, long methods, primitives
3. **Make small changes** - One refactoring at a time
4. **Run tests after each change** - Stay green
5. **Stop when clear** - No obvious improvements left

## Techniques

- **Breaking Out**: Extract method in same class
- **Budding Off**: Extract to new class
- **Bundling Up**: Group related primitives into object

## References

- @agents/tdd/refactor-phase.md - Full refactor phase instructions
- @prompts/decision-trees/refactor-or-not.md - When to refactor
- @prompts/goos.md - GOOS refactoring patterns

## Output

1. Refactored code
2. List of refactorings applied
3. Test run showing all green

## Rules

- Tests must be GREEN before starting
- One small change at a time
- Run tests after every change
- If red, revert immediately
- Stop when no clear pattern
- Rule of Three for abstractions
