# /skeleton - Create Walking Skeleton

Build the thinnest possible end-to-end implementation that exercises the full stack.

## Usage

```
/skeleton [feature or system name]
```

## Examples

```
/skeleton user-authentication
/skeleton order-processing
/skeleton notification-service
/skeleton payment-integration
```

## Process

Follow @agents/goos/walking-skeleton.md to:

1. **Identify E2E path** - Entry → Process → Exit
2. **Write acceptance test** - Domain language
3. **Implement thin slice** - Touch all layers
4. **Make it deployable** - Production-like environment
5. **Document deferred decisions** - What comes later

## What Gets Built

| Include | Example |
|---------|---------|
| One API endpoint | POST /orders |
| Service layer | OrderService.create() |
| Domain model | Order class |
| Repository | OrderRepository.save() |
| Database schema | orders table |
| Config | Environment setup |
| Deploy config | Dockerfile |

## What Gets Deferred

| Exclude | Why |
|---------|-----|
| Authentication | Separate concern |
| Validation | Add incrementally |
| Error handling | Add incrementally |
| Multiple endpoints | One path only |
| Complex logic | Hardcode for now |

## References

- @agents/goos/walking-skeleton.md - Full skeleton instructions
- @prompts/goos.md - GOOS principles

## Output

1. Acceptance test for the skeleton
2. Minimal implementation (all layers)
3. Deployment configuration
4. DEFERRED.md documenting what comes later
5. Instructions to run

## Rules

- Must be deployable
- Must touch every layer
- Can return hardcoded data
- Document all deferred decisions
- One E2E path only
