# GOOS Workflow Guide

Practical guide for using the GOOS tooling to build software from a PRD.

## The Composition

```
/goos skeleton [feature]      ← Start with walking skeleton
    │
    ▼
/goos outside-in [feature]    ← Develop each feature
    │
    ├──▶ /goos discover [object]   ← Find collaborator interfaces
    │         │
    │         ▼
    └──▶ /tdd rgr [behavior]       ← Implement each behavior
              │
              ├──▶ /tdd red
              ├──▶ /tdd green
              └──▶ /tdd refactor

/goos-review [files]          ← Review code against GOOS principles
```

---

## Step-by-Step Example: Link Shortener

**PRD says:**
- User submits a URL, gets a short code
- Short code redirects to original URL
- Track click counts

---

### Step 1: Walking Skeleton

```
/goos skeleton link-shortener
```

Creates the thinnest possible E2E slice:

```
HTTP Request → App → Response
     │                  │
     └── POST /shorten ─┘  (returns hardcoded "abc123")
```

**What you get:**
- One acceptance test: "user can shorten a URL"
- Minimal app with one route
- Deployed (even locally) and working

**What's deferred:**
- Real persistence
- Real short code generation
- Redirects
- Click tracking

**Why this matters:** Proves the architecture works before adding complexity.

---

### Step 2: First Real Feature

```
/goos outside-in shorten-url
```

Starts from acceptance test in domain language:

```python
def test_shortening_a_url():
    # Given a valid URL
    url = "https://example.com/very/long/path"

    # When I submit it for shortening
    response = client.post("/shorten", json={"url": url})

    # Then I get a short code
    assert response.json["short_code"] is not None
    assert len(response.json["short_code"]) == 6
```

The test fails. Now we work inward.

---

### Step 3: Discover Interfaces

```
/goos discover UrlShortener
```

Ask: "What collaborators does UrlShortener need to do its job?"

**Output:**

```markdown
## ShortCodeGenerator
**Category**: Dependency
**Purpose**: Generate unique short codes
**Interface**:
    def generate() -> str
**Test Strategy**: Stub (returns data, no side effects)

## LinkRepository
**Category**: Dependency
**Purpose**: Persist shortened links
**Interface**:
    def save(short_code: str, original_url: str) -> None
    def find_by_code(short_code: str) -> str | None
**Test Strategy**: Expect save(), Stub find()
```

Interfaces are named for **roles** (what they do), not implementations (how).

---

### Step 4: Implement Behaviors with TDD

For each behavior, run the full red-green-refactor cycle:

```
/tdd rgr generates-unique-short-code
```

**Red phase:**
```python
def test_generates_six_character_code():
    generator = ShortCodeGenerator()
    code = generator.generate()
    assert len(code) == 6
    assert code.isalnum()
```

**Green phase (shameless!):**
```python
class ShortCodeGenerator:
    def generate(self) -> str:
        return "abc123"  # Hardcoded is fine!
```

**Refactor phase:** Nothing yet—wait for more examples.

---

```
/tdd rgr saves-shortened-url
```

**Red phase:**
```python
def test_saves_url_with_short_code():
    repo = Mock(spec=LinkRepository)
    generator = Mock(spec=ShortCodeGenerator)
    generator.generate.return_value = "abc123"

    shortener = UrlShortener(repo, generator)
    result = shortener.shorten("https://example.com")

    repo.save.assert_called_with("abc123", "https://example.com")
    assert result == "abc123"
```

**Green phase:**
```python
class UrlShortener:
    def __init__(self, repo, generator):
        self.repo = repo
        self.generator = generator

    def shorten(self, url: str) -> str:
        code = self.generator.generate()
        self.repo.save(code, url)
        return code
```

---

### Step 5: Next Feature

```
/goos outside-in redirect-to-original
```

Same pattern:
1. Write acceptance test
2. Discover needed interfaces
3. `/tdd rgr` each behavior

---

### Step 6: Periodic Review

```
/goos-review src/
```

Checks your code against GOOS principles:
- Mocking roles you own? ✓
- Tell don't ask? ✓
- Single responsibility? ✓
- Tests describe behavior? ✓

---

## The Rhythm

```
PRD Feature List
     │
     ├─▶ /goos skeleton           (once, at project start)
     │
     └─▶ For each feature:
            │
            ├─▶ /goos outside-in [feature]
            │        │
            │        ├─▶ /goos discover [collaborator]  (as needed)
            │        │
            │        └─▶ /tdd rgr [behavior]  (repeat for each behavior)
            │
            └─▶ /goos-review   (periodically)
```

---

## Quick Reference

| Command | When to Use | What It Does |
|---------|-------------|--------------|
| `/goos skeleton [feature]` | Project start | Thinnest E2E slice, proves architecture |
| `/goos outside-in [feature]` | Each feature | Acceptance test → work inward |
| `/goos discover [object]` | Need collaborators | Define interfaces from caller's perspective |
| `/tdd rgr [behavior]` | Each behavior | Full red-green-refactor cycle |
| `/tdd red [behavior]` | Just the test | Write one failing test |
| `/tdd green` | Just pass it | Minimal code to pass |
| `/tdd refactor [area]` | Clean up | Improve structure, stay green |
| `/goos-review [files]` | Periodic check | Review against GOOS principles |

---

## Key Principles

1. **Never write code without a failing test**
2. **Interfaces emerge from caller needs**, not implementation imagination
3. **Mock roles you own**, adapt third-party code
4. **Shameless green is okay**—refactor when patterns emerge
5. **Walking skeleton first**—prove architecture before complexity

---

## References

- [principles.md](principles.md) - Core GOOS principles
- [when-to-mock.md](when-to-mock.md) - Mocking decisions
- [test-granularity.md](test-granularity.md) - Test level selection
- [refactor-or-not.md](refactor-or-not.md) - When to refactor
