# /discover - Interface Discovery

Discover and define interfaces for collaborators based on what callers need.

## Usage

```
/discover [object or component name]
```

## Examples

```
/discover OrderProcessor
/discover AuthenticationService
/discover ReportGenerator
/discover PaymentHandler
```

## Process

Follow @agents/goos/interface-discoverer.md to:

1. **Identify needed collaborators** - What does this object need?
2. **Name the roles** - From caller's perspective
3. **Categorize each role** - Dependency, Notification, Adjustment
4. **Define minimal interface** - Only what caller needs
5. **Determine test strategy** - Stub queries, expect actions

## Output Format

For each interface discovered:

```markdown
## [RoleName]

**Category**: Dependency | Notification | Adjustment
**Purpose**: [One sentence]

**Interface**:
class RoleName(Protocol):
    def method(self, param: Type) -> Return: ...

**Test Strategy**:
- method: Stub | Expect

**Example Setup**:
role = Mock(spec=RoleName)
role.method.return_value = value
```

## Role Categories

| Category | Description | Strategy |
|----------|-------------|----------|
| Dependency | Object uses for work | Stub returns |
| Notification | Object informs of events | Expect calls |
| Adjustment | Object configures/observes | Either |

## Naming Rules

| Bad (Implementation) | Good (Role) |
|---------------------|-------------|
| StripeService | PaymentGateway |
| PostgresRepo | OrderRepository |
| SendGridClient | NotificationSender |
| RedisCache | InventoryChecker |

## References

- @agents/goos/interface-discoverer.md - Full discovery instructions
- @prompts/decision-trees/when-to-mock.md - Mock decisions
- @prompts/goos.md - Mock Roles Not Objects

## Rules

- Name for roles, not implementations
- Minimal interface (only needed methods)
- Define from caller's perspective
- Categorize as Dependency/Notification/Adjustment
- Specify stub vs expect for each method
