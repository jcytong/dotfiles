# What is GOOS?

**Growing Object-Oriented Software Guided by Tests** by Steve Freeman and Nat Pryce.

A methodology for building maintainable software through test-driven development as a *design technique*, not just a testing technique.

---

## Core Philosophy

### Software Development as Learning

- Development is incremental and iterative
- Expose uncertainty early by testing assumptions as soon as possible
- Use empirical feedback to learn about the system, then apply it back
- Expect unexpected changes

### TDD as Design

> "If we try to practise TDD as only a testing technique, we eventually notice 'this code is annoying to test', which points to design problems."

- Tests reveal the communication between objects
- Something difficult to test is probably poorly designed
- Mock objects identify essential interactions, leading to better abstractions

---

## The Walking Skeleton

The thinnest possible slice of functionality that can be automatically built, deployed, and tested end-to-end.

```
Understand Problem → Broad Design → Automate Build/Deploy/Test → Deployable System
```

**Why it matters:**
- Proves architecture works before adding complexity
- Flushes out operational risks early (database setup, deployment signatures)
- Forces the team to understand how the system fits into the world

**A system is deployable when acceptance tests pass.**

---

## Outside-In Development

Work from the outside (user-facing) toward the inside:

1. Start each feature with an **acceptance test** in domain language
2. Develop from **inputs to outputs** - consider triggering events first
3. Discover supporting objects as you work inward
4. Fill in implementations as the system grows

> "When writing a test, ask: 'If this worked, who would know?' If the answer isn't in the target object, introduce a new collaborator."

---

## Mock Roles, Not Objects

### The Key Insight

Mock interfaces you own that represent **roles**, not concrete implementations or third-party code.

### Only Mock Types You Own

- Don't mock third-party code (you don't control it)
- Create an **adapter layer** for external dependencies
- Test adapters with focused integration tests
- Mock your adapter interface, not the library

### Object Peer Stereotypes

| Type | Description | Test Strategy |
|------|-------------|---------------|
| **Dependencies** | Services the object needs | Cannot create object without them |
| **Notifications** | Peers kept up to date | Fire and forget, decouples objects |
| **Adjustments** | Policy objects, configuration | Can use safe defaults |

### Stub Queries, Expect Actions

- **Stub**: Methods that return data (no side effects)
- **Expect**: Methods that cause observable effects

---

## Tell, Don't Ask

Objects should tell collaborators what to do, not ask for data and make decisions.

**Benefits:**
- More flexible code (easy to swap objects with same role)
- Caller sees nothing of internal structure
- Objects naturally evolve reusable public interfaces

### Law of Demeter

> "Only talk to your immediate neighbors"

**Violations are clues of missing objects** whose public interface you haven't discovered yet.

```python
# Violation - reaching through objects
customer.bicycle.wheel.tire

# Better - ask yourself what behavior you actually need
# and put it where the data lives
```

---

## Design Principles

### Single Responsibility

> "Describe what an object does without using conjunctions (and, or, but)."

- Interacting with a composite should be simpler than its components
- If an object becomes complex, break out coherent units

### Context Independence

> "A system is easier to change if its objects have no built-in knowledge about the system."

- Makes relationships explicit, defined separately from objects
- Write code that depends as little as possible on its context

### Information Hiding

- Define immutable value types
- Avoid global variables and singletons
- Don't share references to mutable objects
- Copy collections when passing between objects

---

## Techniques for Growing Objects

### Breaking Out

When an object becomes too large to test easily:
- Extract coherent units of behavior into helper types
- Unit test the new parts separately

### Budding Off

Mark new domain concepts with placeholder types:
- Create an interface for the service the object needs
- Write tests as if the service exists (using mocks)
- Fill in implementation later

### Bundling Up

When test setup becomes too complicated:
- Bundle collaborating objects into a new abstraction
- Give it a name that helps understand the domain
- Scope dependencies more clearly

---

## Two-Layer Architecture

### Implementation Layer
- Graph of objects
- Behavior is combined result of how objects respond to events
- Follow conventional OO style guidelines

### Declarative Layer
- Domain-specific language describing *what* code does
- Implementation layer defines *how*
- Emerges from continual refactoring

> "Most of the time, a declarative layer emerges from continual merciless refactoring."

---

## Quality

### External Quality
How well the system meets customer/user needs. Part of the contract.

### Internal Quality
How well the system meets developer/administrator needs. Allows safe, predictable modification.

> "Running end-to-end tests tells us about external quality. Writing them tells us about how well we understand the domain."

---

## Key Quotes

> "Object-oriented design focuses more on the communication between objects than on the objects themselves."

> "The best we can get from TDD is confidence that we can change the code without breaking it. Fear kills progress."

> "Nothing forces us to understand a process better than trying to automate it."

> "As the legends say, if we have something's true name, we can control it."

---

## References

- [GOOS Book Code](http://www.growing-object-oriented-software.com/code.html)
- [Mock Roles Not Objects (Paper)](http://jmock.org/oopsla2004.pdf)
- [Testing Without Mocks](https://www.jamesshore.com/v2/blog/2018/testing-without-mocks)
- [Domain-Oriented Observability](https://martinfowler.com/articles/domain-oriented-observability.html)
