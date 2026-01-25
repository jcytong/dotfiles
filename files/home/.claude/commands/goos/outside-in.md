# /outside-in - Feature Development

Develop a feature using outside-in TDD, starting from acceptance test and working inward.

## Usage

```
/outside-in [feature description]
```

## Examples

```
/outside-in user-login
/outside-in checkout-with-discount
/outside-in password-reset
/outside-in export-report-as-csv
```

## Process

This orchestrates the full outside-in workflow:

### 1. Write Acceptance Test
Follow @agents/goos/acceptance-writer.md

```
Given [context]
When [action]
Then [outcome]
```

### 2. Run and Watch It Fail
The test drives what we need to build.

### 3. Discover Interfaces
Follow @agents/goos/interface-discoverer.md

Identify collaborators needed, define their roles.

### 4. Implement Outside Layer
Start from the entry point (API, UI, etc.)

### 5. TDD Inner Layers
For each collaborator, use /rgr cycle:
- /red - failing test
- /green - make pass
- /refactor - clean up

### 6. Acceptance Test Passes
Feature is complete when outer test goes green.

## Workflow Diagram

```
Acceptance Test (RED)
       ↓
  Entry Point (API/UI)
       ↓
  Discover Interfaces (/discover)
       ↓
  For each collaborator:
    └─→ /rgr [behavior]
          ↓
  Acceptance Test (GREEN)
```

## References

- @agents/goos/acceptance-writer.md - Writing acceptance tests
- @agents/goos/interface-discoverer.md - Discovering interfaces
- @prompts/goos.md - GOOS principles
- @prompts/decision-trees/when-to-mock.md - Mock decisions

## Output

1. Acceptance test
2. All discovered interfaces
3. Implementations (driven by unit tests)
4. Passing acceptance test

## Rules

- Start from user perspective
- Let tests drive interface discovery
- Mock roles you own
- Each layer tested before moving inward
- Done when acceptance test passes
