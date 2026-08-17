# Test Granularity - Decision Tree

Guide for choosing the right test level for different scenarios.

## The Flow

```
What are you testing?
│
├─ Complete user journey / business scenario?
│  └─ ACCEPTANCE TEST
│     - Full stack, real(ish) infrastructure
│     - Domain language (Given/When/Then)
│     - Slow, few tests
│
├─ How components work together?
│  └─ INTEGRATION TEST
│     - Real collaborators, possibly fake infrastructure
│     - Medium speed and quantity
│
└─ Isolated behavior of single unit?
   └─ UNIT TEST
      - All dependencies mocked/stubbed
      - Fast, many tests
```

---

## Test Pyramid

```
        /\
       /  \  Acceptance (E2E)
      /    \  - Few tests
     /------\  - Slow
    /        \  - High confidence per test
   / Integr.  \  - Expensive to maintain
  /            \
 /--------------\ Unit
/                \  - Many tests
──────────────────  - Fast
                    - Low confidence per test
                    - Cheap to maintain
```

---

## Decision Matrix

| Question | Acceptance | Integration | Unit |
|----------|------------|-------------|------|
| Tests real user value? | ✓ Primary | ✓ Supports | Indirectly |
| Runs in CI quickly? | ✗ | ✓ | ✓✓ |
| Pinpoints failures? | ✗ | ✓ | ✓✓ |
| Tests infrastructure? | ✓ | ✓ | ✗ |
| Tests business logic? | ✓ | ✓ | ✓✓ |
| Survives refactoring? | ✓✓ | ✓ | ✗ |

---

## When to Use Each Level

### Acceptance Tests

**Use for**:
- Critical user journeys (login, checkout, signup)
- Walking skeleton verification
- Regression on key business flows
- Contractual requirements ("user can...")

**Characteristics**:
- Written in domain language
- Test from user's perspective
- May use browser automation, API clients
- Touch real database, possibly stubbed externals

**Example scenarios**:
- "User can log in with valid credentials"
- "Customer can complete purchase"
- "Admin can export reports"

### Integration Tests

**Use for**:
- Database interactions
- API endpoint behavior
- Message queue processing
- Cache behavior
- Third-party service adapters

**Characteristics**:
- Test component boundaries
- May use test database, in-memory queues
- Faster than acceptance, slower than unit
- Focus on collaboration, not UI

**Example scenarios**:
- "Repository correctly saves and retrieves entities"
- "API returns 404 for unknown resources"
- "Event handler updates read model"

### Unit Tests

**Use for**:
- Business logic / domain rules
- Algorithms and calculations
- State machines
- Value object behavior
- Error handling paths

**Characteristics**:
- All dependencies mocked/stubbed
- Millisecond execution
- One concept per test
- Heavy use of mocks for collaborators

**Example scenarios**:
- "Order calculates total with discounts"
- "Password meets strength requirements"
- "State transitions correctly on events"

---

## Language-Specific Organization

### Python (pytest)

```
tests/
├── acceptance/           # E2E tests
│   ├── conftest.py       # Fixtures (app client, db)
│   └── test_user_flows.py
├── integration/          # Component tests
│   ├── conftest.py       # Fixtures (db session)
│   └── test_repositories.py
└── unit/                 # Isolated tests
    ├── conftest.py       # Mock factories
    └── domain/
        └── test_order.py
```

**Markers**:
```python
@pytest.mark.acceptance
@pytest.mark.integration
@pytest.mark.unit
```

### Ruby (RSpec)

```
spec/
├── acceptance/           # Feature specs
│   └── user_login_spec.rb
├── requests/             # API integration
│   └── orders_spec.rb
├── models/               # ActiveRecord integration
│   └── order_spec.rb
└── lib/                  # Unit specs
    └── domain/
        └── order_calculator_spec.rb
```

**Tags**:
```ruby
describe "Login", :acceptance do
describe "Order API", :integration do
describe OrderCalculator, :unit do
```

### TypeScript/Jest

```
tests/
├── e2e/                  # End-to-end
│   └── user-flows.test.ts
├── integration/          # API/DB tests
│   └── order-repository.test.ts
└── unit/                 # Isolated
    └── domain/
        └── order.test.ts
```

**Configuration**:
```javascript
// jest.config.js
projects: [
  { displayName: 'unit', testMatch: ['**/unit/**/*.test.ts'] },
  { displayName: 'integration', testMatch: ['**/integration/**/*.test.ts'] },
  { displayName: 'e2e', testMatch: ['**/e2e/**/*.test.ts'] },
]
```

---

## Common Mistakes

### Testing Too High

**Problem**: Everything is an acceptance test
- Slow CI pipeline
- Flaky tests
- Hard to diagnose failures

**Fix**: Push testing down the pyramid. If logic can be tested in isolation, do it.

### Testing Too Low

**Problem**: Thousands of unit tests, but system breaks in production
- Mocks don't match reality
- Integration points untested
- False confidence

**Fix**: Add integration tests at boundaries. Test real collaborations.

### Wrong Level for the Code

| Code Type | Wrong Level | Right Level |
|-----------|-------------|-------------|
| Pure calculation | Integration | Unit |
| Database query | Unit (mocked) | Integration |
| API contract | Unit | Integration + Contract |
| User flow | Unit | Acceptance |
| Domain rule | Acceptance | Unit |

---

## Balanced Strategy

For most features:

1. **One acceptance test** - Proves the feature works end-to-end
2. **Few integration tests** - Verify critical boundaries
3. **Many unit tests** - Cover logic variations, edge cases

**Ratio guideline**: ~70% unit, ~20% integration, ~10% acceptance

---

## When in Doubt

Ask yourself:

1. "What breaks if this test fails?"
   - User-visible feature → Acceptance
   - System boundary → Integration
   - Internal logic → Unit

2. "How fast does this need to run?"
   - Every keystroke → Unit
   - Every commit → Unit + Integration
   - Pre-deploy → All levels

3. "What survives refactoring?"
   - Always survives → Acceptance
   - Usually survives → Integration
   - Often changes → Unit (and that's OK)
