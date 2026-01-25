# When to Mock - Decision Tree

Quick reference for deciding what to mock in tests.

## The Flow

```
Is it a type you own?
│
├─ NO → Don't mock it directly
│       │
│       └─ Create an adapter interface you own
│          Then mock the adapter
│
└─ YES → Is it a value object?
         │
         ├─ YES → Don't mock
         │        Use real instances
         │
         └─ NO → Is it a query or action?
                 │
                 ├─ QUERY (returns data, no side effects)
                 │  └─ STUB it
                 │     Set up return values
                 │
                 └─ ACTION (causes side effects)
                    └─ EXPECT it
                       Verify it was called
```

---

## Quick Reference Table

| Situation | Mock? | Strategy |
|-----------|-------|----------|
| Third-party SDK (Stripe, AWS) | No | Create adapter, mock adapter |
| Standard library | No | Use real implementation |
| Your domain interfaces | Yes | Stub queries, expect actions |
| Value objects | No | Use real instances |
| Entity objects | Maybe | Stub if complex, real if simple |
| Database/File system | No | Adapter pattern or test doubles |
| HTTP clients | No | Adapter pattern |
| Time/Randomness | Yes | Inject and stub |

---

## Stub vs Expect

### Stub Queries

Use stubs for methods that **return data** without side effects.

```python
# Stub: we care about the return value
user_repo.find_by_id.return_value = User(id=1, name="Alice")

# The test then uses this return value
result = service.get_user_profile(1)
assert result.name == "Alice"
```

### Expect Actions

Use expectations for methods that **cause effects** we want to verify.

```python
# Expect: we care that this was called
email_sender = Mock(spec=EmailSender)

service.register_user(email="alice@example.com")

# Verify the action happened
email_sender.send_welcome.assert_called_once_with("alice@example.com")
```

---

## The Adapter Pattern

When you need to mock third-party code:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Your Domain    │────▶│  Your Interface  │◀────│  Your Adapter   │
│  (uses mock)    │     │  (mockable)      │     │  (wraps SDK)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Third-Party    │
                                                 │  SDK/Library    │
                                                 └─────────────────┘
```

**Example**:

```python
# Bad: Mocking Stripe directly
stripe.Charge.create = Mock()  # Fragile, couples to Stripe API

# Good: Adapter pattern
class PaymentGateway(Protocol):  # Your interface
    def charge(self, amount: Money, card: CardToken) -> PaymentResult: ...

class StripePaymentGateway:  # Your adapter
    def charge(self, amount: Money, card: CardToken) -> PaymentResult:
        result = stripe.Charge.create(...)  # Real Stripe call
        return PaymentResult(...)

# In tests: mock PaymentGateway, not Stripe
gateway = Mock(spec=PaymentGateway)
gateway.charge.return_value = PaymentResult(success=True)
```

---

## Role Categories

When designing mockable interfaces, categorize by role:

### Dependencies
Objects the system **uses** to do its work.

```python
class OrderProcessor:
    def __init__(self, inventory: InventoryChecker):  # Dependency
        self.inventory = inventory

    def process(self, order):
        if self.inventory.is_available(order.item):  # Query - stub
            ...
```

### Notifications
Objects the system **informs** of events.

```python
class OrderProcessor:
    def __init__(self, events: OrderEvents):  # Notification
        self.events = events

    def process(self, order):
        ...
        self.events.order_completed(order)  # Action - expect
```

### Adjustments
Objects used to **configure** or **observe** the system.

```python
class OrderProcessor:
    def __init__(self, clock: Clock):  # Adjustment
        self.clock = clock

    def process(self, order):
        order.processed_at = self.clock.now()  # Query - stub
```

---

## Anti-Patterns

### Mocking What You Don't Own

```python
# Bad: Mocking requests library
requests.get = Mock(return_value=...)

# Good: Abstract behind your interface
class HttpClient(Protocol):
    def get(self, url: str) -> Response: ...
```

### Mocking Value Objects

```python
# Bad: Mocking a simple value
mock_money = Mock(amount=100, currency="USD")

# Good: Use real value object
money = Money(100, "USD")
```

### Over-Specifying Interactions

```python
# Bad: Testing implementation details
mock.method.assert_called_once_with(exact_args)
mock.other_method.assert_called_after(mock.method)

# Good: Test observable behavior
assert result == expected_output
mock.important_action.assert_called()  # Only critical interactions
```

---

## Language-Specific Notes

### Python
- Use `unittest.mock.Mock` with `spec=` for type safety
- Consider `Protocol` for interfaces
- `MagicMock` for dunder methods

### Ruby
- Use `rspec-mocks` with `instance_double`
- Verify partial doubles for safety

### TypeScript/JavaScript
- Use `jest.mock` for modules
- Prefer dependency injection over module mocking
- Consider `ts-mockito` for typed mocks
