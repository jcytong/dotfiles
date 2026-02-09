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

See [workflow.md](references/workflow.md) for full example.

---

### outside-in [feature]

Develop a feature using double-loop TDD, working from user-facing layer inward.

```
┌─────────────────────────────────────────────────────────────┐
│  OUTER LOOP (Acceptance)                                    │
│                                                             │
│  1. Write a failing acceptance test                         │
│                                                             │
│     ┌─────────────────────────────────────────────────┐     │
│     │  INNER LOOP (Unit)                              │     │
│     │                                                 │     │
│     │  2. Write a failing unit test                   │     │
│     │  3. Make the test pass (minimum code)           │     │
│     │  4. Refactor                                    │     │
│     │                                                 │     │
│     │  Repeat 2-4 until acceptance test passes        │     │
│     └─────────────────────────────────────────────────┘     │
│                                                             │
│  5. Acceptance test passes → next feature slice             │
│                                                             │
│  Repeat 1-5 for each feature slice                          │
└─────────────────────────────────────────────────────────────┘
```

**Phase 1 — Outer Loop Entry:**
1. Identify the thinnest feature slice to implement
2. Write a failing acceptance test describing the slice from the outside
   - See [acceptance-test-writer.md](references/acceptance-test-writer.md) for procedure
3. Run it — confirm it fails for the RIGHT reason (missing implementation, not broken test)

**Phase 2 — Inner Loop (repeat until acceptance test passes):**
4. Analyze the acceptance test failure — what's the next smallest piece needed?
5. Write a failing unit test for that piece
   - See [unit-test-writer.md](references/unit-test-writer.md) for procedure
6. Make it pass with minimum code
   - See [implementer.md](references/implementer.md) for procedure
7. Refactor while green
   - See [refactorer.md](references/refactorer.md) for procedure
8. Run the acceptance test — if still failing, go to step 4

**Phase 3 — Completion:**
9. Acceptance test passes. Report the result.
10. Use `/goos discover` if collaborator interfaces need refinement.

**Communicate at each phase transition:**
- Entering outer loop: "Writing acceptance test for: [scenario]"
- Acceptance test fails: "Fails: [reason]. Starting inner loop."
- Each inner iteration: "Unit test: [what] → Green → Refactored"
- Acceptance test passes: "Acceptance test passes. Slice complete."

**Watch for problems:**
- **Slice too thick** — if inner loop takes >5-7 unit test cycles, suggest splitting
- **Test hard to write** — surface as design feedback, not just a testing problem
- **Implementation ahead of tests** — if writing code no test demands, stop
- **Refactoring breaks things** — revert and take smaller steps

See [principles.md](references/1-principles.md#outside-in-development) for details.

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

See [when-to-mock.md](references/when-to-mock.md) for decision guidance.

---

## References

- [principles.md](references/1-principles.md) - Core GOOS principles
- [what-is-goos.md](references/0-what-is-goos.md) - What is GOOS
- [workflow.md](references/workflow.md) - Step-by-step example
- [when-to-mock.md](references/when-to-mock.md) - Mocking decisions
- [test-granularity.md](references/test-granularity.md) - Choosing test level
- [refactor-or-not.md](references/refactor-or-not.md) - When to refactor
- [acceptance-test-writer.md](references/acceptance-test-writer.md) - Writing failing acceptance tests
- [unit-test-writer.md](references/unit-test-writer.md) - Writing failing unit tests
- [implementer.md](references/implementer.md) - Making tests pass with minimum code
- [refactorer.md](references/refactorer.md) - Improving design while green
