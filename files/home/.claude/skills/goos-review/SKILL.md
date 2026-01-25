# /goos-review - GOOS Principles Review

Review code files against Growing Object-Oriented Software principles.

## Usage

```
/goos-review [file or glob pattern]
```

## Examples

```
/goos-review src/services/auth.py
/goos-review src/domain/*.py
/goos-review app/models/order.rb
/goos-review src/**/*.ts
```

## What It Checks

### 1. Mock Roles Not Objects

**Check**: Are mocks used for roles you own, with adapters for third-party code?

**Red flags**:
- Mocking third-party libraries directly (Stripe, AWS SDK, etc.)
- Mocking concrete classes instead of interfaces
- Mock names matching implementations (e.g., `mock_stripe_client`)

**Should be**:
- Mocking role interfaces you define (e.g., `PaymentGateway`)
- Adapters wrapping third-party code
- Stub queries, expect actions

Reference: @prompts/decision-trees/when-to-mock.md

### 2. Tell Don't Ask

**Check**: Do objects tell collaborators what to do, or ask for data and decide?

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

### 3. Context Independence

**Check**: Do objects depend on context they shouldn't know about?

**Red flags**:
- References to "the application" or "the system"
- Long chains of object navigation (`a.b.c.d.method()`)
- Assumptions about calling context

**Should be**:
- Objects receive what they need directly
- Dependencies injected, not discovered

### 4. Single Responsibility

**Check**: Can each object be described without conjunctions?

**Red flags**:
- Class does X *and* Y *and* Z
- Methods belonging to conceptually different domains
- Vague names: Manager, Handler, Processor, Helper

**Should be**:
- One clear responsibility per class
- Specific, intention-revealing names

### 5. Test Behavior Not Methods

**Check**: Do tests describe scenarios/behaviors, or test implementation details?

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

### 6. Interface Segregation

**Check**: Are interfaces narrow and role-focused?

**Red flags**:
- Large interfaces with many methods
- Clients using only subset of interface
- God interfaces doing everything

**Should be**:
- Small, focused interfaces
- Clients depend only on what they use
- Role-based naming

## Output Format

```markdown
## GOOS Review: [file(s)]

### Findings

#### [PRINCIPLE]: [Issue Title]
**Location**: path/to/file.ext:line
**Current**:
[Code snippet showing issue]

**Issue**: [Explanation of what violates GOOS]

**Suggested**:
[Code snippet showing fix]

**Reference**: @prompts/goos.md#section

---

### Summary
- X total issues found
- Y high priority
- Top priority: [Most impactful fix to make]

### Recommendations
1. [First thing to address]
2. [Second thing to address]
3. [Third thing to address]
```

## Severity Levels

| Level | Description |
|-------|-------------|
| **High** | Violates core GOOS principle, should fix |
| **Medium** | Suboptimal but not critical |
| **Low** | Minor improvement opportunity |

## Process

1. Read specified file(s)
2. Analyze against each GOOS principle
3. Identify violations with specific line numbers
4. Suggest concrete improvements
5. Prioritize findings by impact

## References

- @prompts/goos.md - Full GOOS principles reference
- @prompts/decision-trees/when-to-mock.md - Mocking decisions
- @prompts/decision-trees/refactor-or-not.md - Refactoring guidance

## Notes

- Focus on actionable improvements
- Prioritize high-impact issues
- Include code snippets for context
- Reference specific GOOS principles
- Don't nitpick style issues (that's for linters)
