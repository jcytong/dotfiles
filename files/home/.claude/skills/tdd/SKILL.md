---
name: tdd
description: TDD cycle commands - red, green, refactor, rgr
---

# TDD Cycle

Execute Test-Driven Development workflows.

## Usage

```
/tdd red [behavior]     - Write one failing test
/tdd green              - Make test pass minimally
/tdd refactor [area]    - Clean up, stay green
/tdd rgr [behavior]     - Full red-green-refactor cycle
```

---

## red [behavior]

Write exactly ONE failing test for a behavior.

**Process:**
1. Name the test after the behavior (not the method)
2. Write test using Given/When/Then structure
3. Run test - must fail
4. Ensure failure message is clear and diagnostic

**Example:**
```python
def test_rejects_expired_tokens():
    # Given an expired token
    token = Token(expires_at=yesterday)

    # When validating
    result = validator.validate(token)

    # Then it's rejected
    assert result.valid is False
    assert "expired" in result.reason
```

**Checklist:**
- [ ] Test name describes behavior
- [ ] Failure message explains what went wrong
- [ ] Only one logical assertion
- [ ] Test actually fails

---

## green

Make the current failing test pass with MINIMAL code.

**Allowed (shameless green):**
- Hardcoded return values
- Duplication
- Simple conditionals
- Obvious implementations

**Not allowed:**
- Premature abstraction
- Speculative generality
- "While I'm here" improvements
- Refactoring

**Example:**
```python
# Test expects 6-char code
def generate(self) -> str:
    return "abc123"  # Hardcoded is fine!
```

**Checklist:**
- [ ] Test passes
- [ ] Code is minimal
- [ ] No refactoring done
- [ ] Noted any obvious debt for refactor phase

---

## refactor [area]

Improve code structure while keeping all tests green.

**Process:**
1. Verify all tests pass
2. Make ONE small change
3. Run tests
4. If green → commit, continue
5. If red → revert immediately
6. Stop when no clear pattern

**Techniques:**
- **Breaking Out**: Extract method in same class
- **Budding Off**: Extract to new class
- **Bundling Up**: Group primitives into object

**Checklist:**
- [ ] Started with all tests green
- [ ] Each step was small and reversible
- [ ] Tests stayed green throughout
- [ ] Stopped when no clear improvement visible

**When to stop:**
- No obvious duplication
- No clear pattern for abstraction
- Methods under ~10 lines
- Rule of Three not met

---

## rgr [behavior]

Execute complete Red-Green-Refactor cycle for a behavior.

**Flow:**
```
/tdd rgr validates-email-format
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

**Process:**
1. **Red**: Write failing test for behavior
2. **Green**: Minimal code to pass
3. **Refactor**: Clean up if patterns visible

**Output:**
- Test file with new test
- Implementation code
- List of refactorings (if any)
- All tests passing

---

## Key Principles

### Test Behavior, Not Methods

```python
# Bad: Testing method names
def test_validate():
def test_process():

# Good: Testing behaviors
def test_rejects_invalid_email_format():
def test_applies_loyalty_discount():
```

### Shameless Green

In green phase, don't optimize. Just pass the test.

```python
# Acceptable in green phase
def calculate_discount(order):
    if order.total > 100:
        return 10
    return 0

# NOT acceptable in green phase (premature abstraction)
def calculate_discount(order):
    return DiscountCalculator(
        RuleEngine(self.discount_rules)
    ).calculate(order)
```

### Rule of Three

Wait for 3 examples before abstracting:

```
1st occurrence: Just write it
2nd occurrence: Note duplication, resist abstracting
3rd occurrence: Now refactor
```

### Small Steps

Each refactoring step should:
- Take under 2 minutes
- Be easily reversible
- Keep tests green

If stuck, revert and try a smaller step.

---

## References

For decision guidance, see the goos skill reference files:
- When to refactor: `/goos` skill's refactor-or-not.md
- Test levels: `/goos` skill's test-granularity.md
- What to mock: `/goos` skill's when-to-mock.md
