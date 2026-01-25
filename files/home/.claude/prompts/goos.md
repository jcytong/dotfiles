# GOOS Principles Reference

Growing Object-Oriented Software Guided by Tests - core principles for building maintainable, well-tested software.

## Fractal Nature

GOOS principles apply at every level of abstraction:

| Level | Walking Skeleton | Outside-In | Mock Roles | Red-Green-Refactor |
|-------|------------------|------------|------------|-------------------|
| **System** | Deploy thin slice E2E | Start from acceptance | Service contracts | Iterate on system |
| **Feature** | One user flow | Start from UI/API | Collaborator interfaces | Iterate on feature |
| **Class** | Minimal API | Start from public interface | Dependency interfaces | Iterate on design |
| **Method** | Return simplest value | Start from caller's needs | Parameter contracts | Iterate on impl |

---

## Core Principles

### 1. Walking Skeleton

**Definition**: The thinnest possible implementation that exercises the full technical stack end-to-end.

**Purpose**:
- Proves architecture works before adding complexity
- Reveals integration issues early
- Provides foundation for iterative development

**Structure**: Entry Point → Processing → Exit Point

**Rules**:
- Must be deployable
- Must exercise real infrastructure
- Can return hardcoded or trivial responses
- Document what's deferred

---

### 2. Outside-In Development

**Definition**: Start from the outermost layer (user-facing) and work inward, letting tests drive interface discovery.

**Workflow**:
1. Write acceptance test in domain language
2. Run test - watch it fail
3. Identify needed collaborators
4. Define collaborator interfaces from caller's perspective
5. Implement collaborators (recursively apply outside-in)
6. Acceptance test passes

**Benefits**:
- Interfaces designed for callers, not implementers
- No speculative design
- Every component has a clear purpose

---

### 3. Mock Roles, Not Objects

**Key Insight**: Mock interfaces you own that represent roles, not concrete implementations or third-party code.

**Rules**:
- **Do mock**: Roles you define (e.g., `PaymentGateway`, `UserRepository`)
- **Don't mock**: Third-party libraries, concrete classes, value objects
- **For third-party**: Create adapter interface, mock the adapter

**Role Categories**:
| Category | Description | Test Strategy |
|----------|-------------|---------------|
| Dependency | Object uses to do its work | Stub returns |
| Notification | Object informs of events | Expect calls |
| Adjustment | Object uses to observe/configure | Either |

**Stub vs Expect**:
- **Stub queries**: Methods that return data (no side effects)
- **Expect actions**: Methods that cause observable effects

---

### 4. Tell, Don't Ask

**Principle**: Objects should tell collaborators what to do, not ask for data and make decisions.

**Anti-pattern** (Ask):
```python
if order.customer.balance >= order.total:
    order.customer.debit(order.total)
```

**Pattern** (Tell):
```python
order.customer.charge(order.total)  # Customer decides if allowed
```

**Benefits**:
- Encapsulates decisions where data lives
- Reduces coupling
- Makes testing simpler

---

### 5. Context Independence

**Principle**: Objects should not know about their position in the larger system.

**Signs of violation**:
- Object references "the application" or "the system"
- Object navigates through other objects to find data
- Object makes assumptions about calling context

**Remedy**: Pass in what the object needs directly.

---

### 6. Single Responsibility

**Test**: Can you describe what the object does without using conjunctions (and, or, but)?

**Signs of violation**:
- Class name includes "Manager", "Handler", "Processor" (vague)
- Methods belong to conceptually different domains
- Testing requires complex setup for unrelated scenarios

---

## TDD Mechanics

### Red Phase

**Goal**: Write exactly one failing test with a clear diagnostic.

**Checklist**:
- [ ] Test describes behavior, not implementation
- [ ] Failure message explains what went wrong
- [ ] Test name reads as documentation
- [ ] Only one assertion per test (or one logical concept)

### Green Phase

**Goal**: Shameless green - make test pass with minimal code.

**Allowed**:
- Hardcoded return values
- Duplication
- Simple conditionals
- Obvious implementations

**Not allowed**:
- Premature abstraction
- Speculative generality
- "While I'm here" improvements

### Refactor Phase

**Goal**: Clean up without changing behavior.

**Techniques**:
| Technique | When | What |
|-----------|------|------|
| Breaking Out | Method too long | Extract to private method |
| Budding Off | New responsibility emerging | Extract to new class |
| Bundling Up | Related concepts scattered | Group into cohesive unit |

**Rules**:
- Run tests after every change
- Commit after each successful refactor
- Stop when no clear pattern emerges
- Rule of Three: Wait for 3 examples before abstracting

---

## Test Organization

### Acceptance Tests
- Written in domain language
- Test complete user scenarios
- Slow to run, few in number
- Drive the walking skeleton

### Integration Tests
- Test component collaboration
- May use real infrastructure
- Medium speed and quantity

### Unit Tests
- Test isolated behavior
- All dependencies stubbed/mocked
- Fast to run, many in number
- Drive implementation details

---

## Interface Discovery

When tests reveal the need for a collaborator:

1. **Name the role** from caller's perspective (not implementation)
2. **Define minimal interface** - only what caller needs
3. **Categorize** as Dependency, Notification, or Adjustment
4. **Decide stub vs expect** based on query vs action

**Interface Naming**:
- Good: `PaymentGateway`, `EmailSender`, `InventoryChecker`
- Bad: `StripeService`, `SMTPClient`, `DatabaseRepo`

---

## Common Patterns

### Adapter Pattern for Third-Party Code

```
Your Code → YourInterface → YourAdapter → ThirdPartyLibrary
              (mock this)   (don't mock)   (never mock)
```

### Ports and Adapters

- **Port**: Interface defining what your domain needs
- **Adapter**: Implementation connecting to external system
- Test domain logic with port mocks
- Test adapters with integration tests

### Notification Pattern

When an object needs to inform others without knowing who:
1. Define notification interface
2. Accept notifier as dependency
3. Call notification method
4. Test with expectation
