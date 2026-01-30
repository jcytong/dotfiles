# Refactor or Not - Decision Tree

Guide for deciding when and how to refactor during TDD.

## The Primary Question

```
Are all tests passing?
│
├─ NO → DO NOT REFACTOR
│       Get to green first
│
└─ YES → Do you see clear duplication or a clear pattern?
         │
         ├─ NO → STOP
         │       Don't refactor speculatively
         │
         └─ YES → Have you seen this pattern 3+ times?
                  │
                  ├─ NO → WAIT
                  │       Note it, but don't act yet
                  │       (Rule of Three)
                  │
                  └─ YES → REFACTOR
                           But keep it small and incremental
```

---

## Rule of Three

**Before abstracting, wait for three examples.**

```
1st occurrence: Just write it
2nd occurrence: Note the duplication, resist abstracting
3rd occurrence: Now refactor - you have enough examples to generalize correctly
```

**Why wait?**
- Two examples rarely reveal the true abstraction
- Premature abstractions are often wrong
- Wrong abstractions are expensive to fix
- Better to have a bit of duplication than the wrong abstraction

---

## Signs You SHOULD Refactor

### Clear Duplication

```python
# Before: Obvious duplication
def process_order(order):
    log.info(f"Processing order {order.id}")
    validate_order(order)
    log.info(f"Validated order {order.id}")

def process_return(ret):
    log.info(f"Processing return {ret.id}")
    validate_return(ret)
    log.info(f"Validated return {ret.id}")
```

### Long Methods (>10 lines of logic)

The code tells a story with too many details at once.

### Primitive Obsession

Passing around raw strings, numbers, dicts instead of domain objects.

### Feature Envy

Method uses more of another object's data than its own.

### Deep Nesting (>3 levels)

```python
# This screams for extraction
if condition1:
    if condition2:
        for item in items:
            if condition3:
                # actual work here
```

---

## Signs You Should NOT Refactor

### "While I'm Here" Syndrome

```
❌ "I'm fixing this bug, might as well clean up this other code"
✓ Fix the bug. Commit. Consider cleanup separately.
```

### Speculative Generality

```
❌ "We might need to support multiple payment providers someday"
✓ Support the one you need now. Generalize when you need two.
```

### Tests Are Red

Never refactor while tests are failing. Get to green first.

### No Clear Pattern

If you can't name the pattern or explain the abstraction simply, wait.

### Last Day Before Deadline

Refactoring introduces risk. Ship what works.

---

## Refactoring Techniques

### Breaking Out

Extract code into a new **method** within the same class.

**When**: Method is too long, or a section has a distinct purpose.

```python
# Before
def process_order(order):
    # 20 lines of validation
    # 10 lines of pricing
    # 15 lines of persistence

# After
def process_order(order):
    self._validate(order)
    self._calculate_pricing(order)
    self._persist(order)
```

### Budding Off

Extract code into a **new class**.

**When**: Responsibility is emerging that doesn't belong here.

```python
# Before: Order is doing too much
class Order:
    def calculate_shipping(self): ...
    def calculate_tax(self): ...
    def calculate_discount(self): ...

# After: New class for pricing concerns
class OrderPricer:
    def calculate_total(self, order): ...
```

### Bundling Up

Group related concepts into a **new abstraction**.

**When**: Multiple related items travel together.

```python
# Before: Related params always together
def charge(card_number, expiry, cvv, amount, currency):
    ...

# After: Bundled into objects
def charge(card: Card, money: Money):
    ...
```

---

## The Refactoring Workflow

1. **Ensure tests are green**
2. **Make one small change**
3. **Run tests**
4. **If green**: Commit, continue
5. **If red**: Revert immediately, try smaller step
6. **Stop when**: No clear next improvement

**Key principle**: Each step should take under 2 minutes. If stuck, revert.

---

## "Good Enough" Checklist

Before declaring refactoring complete, verify:

- [ ] All tests pass
- [ ] No obvious duplication remains
- [ ] Methods are under ~10 lines
- [ ] Classes have one clear responsibility
- [ ] Names reflect domain concepts
- [ ] No deep nesting (3+ levels)
- [ ] You can explain the code to a colleague

If most boxes are checked, stop. Diminishing returns ahead.

---

## The Wrong Abstraction

**A wrong abstraction is worse than duplication.**

Signs of wrong abstraction:
- Adding parameters/flags to handle variations
- Methods with boolean parameters changing behavior
- "This doesn't quite fit but I'll make it work"
- Inheritance hierarchy that doesn't match domain

**Fix**:
1. Inline the abstraction (go back to duplication)
2. Look at the concrete cases
3. Find the right abstraction (or accept duplication)

---

## Refactoring Debt Notes

During green phase, note refactoring opportunities:

```
# TODO(refactor): Third time seeing this validation pattern
# TODO(refactor): This method is getting long
# TODO(refactor): card_number + expiry + cvv should be Card object
```

Address in refactor phase, or in dedicated refactoring sessions.

---

## Timebox Guidance

- **Small refactoring** (rename, extract method): Just do it
- **Medium refactoring** (extract class, introduce pattern): 15-30 min max
- **Large refactoring** (restructure module): Plan it, don't do it "while you're here"

If a refactoring is taking longer than expected:
1. Commit what you have (or revert if incomplete)
2. Create a task for the larger refactoring
3. Continue with the original task
