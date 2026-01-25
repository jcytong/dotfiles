---
name: goos
description: GOOS workflow commands - skeleton, outside-in, discover
---

# GOOS Workflow

Execute GOOS (Growing Object-Oriented Software) workflows.

## Usage

```
/goos skeleton [feature]    - Start with walking skeleton
/goos outside-in [feature]  - Develop feature outside-in
/goos discover [object]     - Discover collaborator interfaces
```

## Commands

### skeleton [feature]

Start a new feature with the thinnest possible end-to-end slice.

**Process:**
1. Identify E2E path: Entry → Process → Exit
2. Write one acceptance test in domain language
3. Implement thinnest deployable slice (hardcoded OK)
4. Document deferred decisions

**Output:**
- Acceptance test file
- Minimal implementation
- List of deferred items

See [workflow.md](workflow.md) for full example.

---

### outside-in [feature]

Develop a feature working from user-facing layer inward.

**Process:**
1. Write acceptance test (Given/When/Then)
2. Run it - watch it fail
3. Identify needed collaborators
4. Use `/goos discover` for each collaborator
5. Use `/tdd` to implement each behavior
6. Acceptance test passes when complete

See [principles.md](principles.md#outside-in-development) for details.

---

### discover [object]

Discover interfaces for an object's collaborators.

**Process:**
1. Identify what the object needs to do its job
2. Name each role (not implementation)
3. Categorize: Dependency / Notification / Adjustment
4. Define minimal interface from caller's perspective
5. Determine stub vs expect for each method

**Output format:**
```markdown
## [RoleName]
**Category**: Dependency | Notification | Adjustment
**Purpose**: [One sentence]
**Interface**: [Method signatures]
**Test Strategy**: Stub | Expect for each method
```

See [when-to-mock.md](when-to-mock.md) for decision guidance.

---

## References

- [principles.md](principles.md) - Core GOOS principles
- [workflow.md](workflow.md) - Step-by-step example
- [when-to-mock.md](when-to-mock.md) - Mocking decisions
- [test-granularity.md](test-granularity.md) - Choosing test level
- [refactor-or-not.md](refactor-or-not.md) - When to refactor
