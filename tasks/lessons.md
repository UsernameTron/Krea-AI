# Lessons

## Active Rules

### Seed Rules
- [2026-03-22] [Config]: Never modify shared config files without checking downstream consumers.
- [2026-03-22] [Scope]: If a "quick fix" requires 3+ files, it is not quick. Re-plan.
- [2026-03-22] [Testing]: Run the full test suite, not just tests for the changed module.
- [2026-03-22] [Dependencies]: Never add dependencies without explicit user approval.
- [2026-03-22] [Data]: Never delete production data, migrations, or seed data without approval.

### Learned Rules
- [2026-03-22] [Types]: When creating dicts with None-initialized keys that will later hold strings, annotate as `Dict[str, Any]` upfront — mypy infers narrow types from dict literals and will reject later assignments.
- [2026-03-22] [CI]: Always run `mypy . --ignore-missing-imports` locally before pushing. The CI typecheck job catches real errors that ruff doesn't.
- [2026-03-22] [Merge]: When merging multiple feature branches, check for file-level conflicts first (both branches creating the same new file) — these need manual resolution strategy.

## Archived
<!-- Rules that no longer apply -->
