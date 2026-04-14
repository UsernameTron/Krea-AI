---
status: testing
phase: 02-post-brief-generator
source: integration test (mirror-post/test/integration.js)
started: 2026-04-14T21:30:00Z
updated: 2026-04-14T21:30:00Z
---

## Current Test

number: 5
name: Generation latency under 15s + system prompt under 8K
expected: |
  All 5 integration scenarios complete in <15s each, system prompt assembled at <8K tokens.
awaiting: user decision on criterion update

## Tests

### 1. Input classifier — archetype exact, fuzzy, domain match
expected: |
  Integration test verifies all 3 routes:
  - "resume-padder" (exact archetype match)
  - "The Demo Magician" (fuzzy archetype match by name)
  - "contact-center" (domain-only routing)
result: pass
evidence: |
  PASS: archetype exact (resume-padder) (48657ms)
  PASS: archetype by name (The Demo Magician) (44578ms)
  PASS: domain match (contact-center) (39011ms)

### 2. Freeform scenarios produce original characters (not archetype copies)
expected: |
  "HR soul-stripper" and "PowerPoint architect" generate original characters with coherent props.
result: pass
evidence: |
  PASS: freeform (HR soul-stripper) (44850ms)
  PASS: freeform (PowerPoint architect) (48578ms)

### 3. Brief Validator hard-fail checks (schema, safety, banned phrases, props, engagement)
expected: |
  All 5 generated briefs pass semantic validator (schema compliance, safety rails REQ-X-020..024,
  banned corporate jargon, prop completeness, engagement sanity).
result: pass
evidence: |
  All 5 scenarios returned PASS — semantic validator accepted every brief after 3 rounds of
  prompt fixes (8aef5d8: timeout/safety/jargon, 618e457: transform callout, f0afcb1: deterministic
  substitutions + archetype trait rename).

### 4. Reference fixture inputs produce structurally similar briefs
expected: |
  Generated briefs match the structural shape of the 4 fixtures from Phase 1.
result: pass
evidence: |
  Schema compliance + structural completeness verified by validator on all 5 live runs.
  Direct fixture-by-fixture diff not executed but covered by 111 unit/mock tests in test/brief.js.

### 5. Generation latency <15s + system prompt <8K tokens
expected: |
  Each brief generates in <15 seconds with assembled system prompt under 8,192 tokens.
result: issue
reported: |
  System prompt size: PASS — uncached input tokens 3,825-3,931 (well under 8K, matches D-09 budget).
  Latency: FAIL — observed 39,011-48,657ms per brief (criterion was 15,000ms).
  Cause: D-22 chose claude-opus-4-6 over Sonnet for voice fidelity. The 15s ceiling in ROADMAP
  predates that decision.
severity: criterion-mismatch

## Summary

total: 5
passed: 4
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "Generation completes in under 15 seconds per brief"
  status: criterion_obsolete
  reason: |
    ROADMAP success criterion was authored assuming Sonnet. Phase 2 D-22 explicitly selected
    claude-opus-4-6 for the brief generator, citing persona complexity and voice fidelity as
    justification for the cost/latency premium. Observed latency (39-48s) is consistent with
    Opus 4.6 + ~3,900 input tokens + structured output. No code defect — the criterion needs
    updating to reflect the architectural decision.
  options:
    - update ROADMAP success criterion to "<60s per brief" (aligns with Opus reality)
    - revisit D-22 and downgrade to Sonnet (rejected when chosen — would regress voice quality)
    - add streaming + progress UI in Phase 6 to mask latency (D-23 deferred this)
  recommendation: "Update criterion to <60s. D-22 stands."
  severity: criterion-mismatch
  test: 5
