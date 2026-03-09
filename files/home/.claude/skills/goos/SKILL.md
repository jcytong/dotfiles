---
name: goos
description: GOOS workflow commands - skeleton, outside-in, discover, review. Use for building software using Growing Object-Oriented Software principles or reviewing code against GOOS principles. Triggers on "goos", "outside-in", "walking skeleton", "goos review", "goos-review".
---

# GOOS Workflow

Execute GOOS (Growing Object-Oriented Software) workflows.

## Usage

```
/goos skeleton [feature]       - Start with walking skeleton
/goos outside-in [feature]     - Develop feature outside-in
/goos discover [object]        - Discover collaborator interfaces
/goos review [file or glob]    - Review code against GOOS principles
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

### review [file or glob]

Review existing code against GOOS principles.

```
/goos review src/services/auth.py
/goos review src/domain/*.py
/goos review src/**/*.ts
```

**Check each principle:**

#### 1. Mock Roles Not Objects

Are mocks used for roles you own, with adapters for third-party code?

**Red flags**:
- Mocking third-party libraries directly (Stripe, AWS SDK, etc.)
- Mocking concrete classes instead of interfaces
- Mock names matching implementations (e.g., `mock_stripe_client`)

**Should be**: Mocking role interfaces you define (e.g., `PaymentGateway`), adapters wrapping third-party code, stub queries / expect actions.

#### 2. Tell Don't Ask

Do objects tell collaborators what to do, or ask for data and decide?

**Red flags**:
```python
# Ask (bad)
if order.customer.balance >= order.total:
    order.customer.debit(order.total)
```

**Should be**:
```python
# Tell (good)
order.customer.charge(order.total)
```

#### 3. Context Independence

Do objects depend on context they shouldn't know about?

**Red flags**:
- References to "the application" or "the system"
- Long chains of object navigation (`a.b.c.d.method()`)
- Assumptions about calling context

**Should be**: Objects receive what they need directly. Dependencies injected, not discovered.

#### 4. Single Responsibility

Can each object be described without conjunctions?

**Red flags**:
- Class does X *and* Y *and* Z
- Methods belonging to conceptually different domains
- Vague names: Manager, Handler, Processor, Helper

**Should be**: One clear responsibility per class. Specific, intention-revealing names.

#### 5. Test Behavior Not Methods

Do tests describe scenarios/behaviors, or test implementation details?

**Red flags**:
```python
def test_validate():  # Method name
def test_process_returns_true():  # Implementation detail
```

**Should be**:
```python
def test_rejects_invalid_email_format():  # Behavior
def test_applies_loyalty_discount():  # Scenario
```

#### 6. Interface Segregation

Are interfaces narrow and role-focused?

**Red flags**: Large interfaces with many methods, clients using only subset, god interfaces.

**Should be**: Small focused interfaces, clients depend only on what they use, role-based naming.

**Output format:**

```markdown
## GOOS Review: [file(s)]

### Findings

#### [PRINCIPLE]: [Issue Title]
**Location**: path/to/file.ext:line
**Current**: [Code snippet]
**Issue**: [What violates GOOS]
**Suggested**: [Code snippet showing fix]

---

### Summary
- X total issues found
- Y high priority
- Top priority: [Most impactful fix]

### Recommendations
1. [First thing to address]
2. [Second thing to address]
3. [Third thing to address]
```

**Severity levels:**

| Level | Description |
|-------|-------------|
| **High** | Violates core GOOS principle, should fix |
| **Medium** | Suboptimal but not critical |
| **Low** | Minor improvement opportunity |

**Process:**
1. Read specified file(s)
2. Analyze against each GOOS principle
3. Identify violations with specific line numbers
4. Suggest concrete improvements
5. Prioritize findings by impact
6. Don't nitpick style issues (that's for linters)

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
