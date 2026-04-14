# TODO

## Phase 0 Bootstrap — COMPLETE

- [x] Task 1: Codebase mapping (7 docs in .planning/codebase/)
- [x] Task 2: Initialize git repo for Krea-AI
- [x] Task 3: Create CLAUDE.md, README.md, docs/DEVOPS-HANDOFF.md
- [x] Task 4: Write PROJECT.md from plan documents
- [x] Task 5: Commit Mirror Post source assets
- [x] Task 6: Collect workflow preferences and create config.json
- [x] Task 7: Define REQUIREMENTS.md from plans + pattern context (71 REQs)
- [x] Task 8: Create ROADMAP.md via gsd-roadmapper (7+1 phases)
- [x] Task 9: Create STATE.md and finalize CLAUDE.md

## Phase 1 Scaffolding — COMPLETE

- [x] /gsd:discuss-phase 1 — 6 decisions captured in 01-CONTEXT.md
- [x] /gsd:plan-phase 1 — 3 PLAN.md files materialized (commit 7424143)
- [x] PLAN-01-01: Directory structure, config, foundational files (7 tasks) — commit 77af7b5
- [x] PLAN-01-02: Asset import and conversion (5 tasks) — commit a32a8a7
- [x] PLAN-01-03: Post Brief schema + 4 reference fixtures (5 tasks) — commit 69b1903
- [x] Verification: 9/9 items passed
- [x] State close: commit 507a5bc

## Phase 2 Post Brief Generator — CODE COMPLETE

- [x] /gsd:discuss-phase 2 — 23 decisions captured in 02-CONTEXT.md (commit 21d056f)
- [x] /gsd:plan-phase 2 — 8 PLAN.md files in 5 waves (commits a5ea09f, 0ec6c1d)
- [x] Wave 1 (02-01): Structured Outputs schema audit gate — PASS (19/24 optional params)
- [x] Wave 2 (02-02/03/04): Persona distillate, input classifier, comedy selector — 37 tests green
- [x] Wave 3 (02-05): System prompt builder, Pattern 5, cache markers — 31 tests (SHA 93e4666)
- [x] Wave 4 (02-06): LLM generation client, Structured Outputs + fallback — 12 mock tests (SHA 4d2bb67)
- [x] Wave 5 (02-07/08): Semantic validator + barrel export + integration test — 30 tests (SHA 07e3dce)
- [ ] Run integration test with real API key (5 scenarios, `node test/integration.js`)
- [ ] /gsd:verify-work — validate against Phase 2 success criteria

## Backlog

- [ ] Review CONCERNS.md and prioritize tech debt items
- [x] Update REQUIREMENTS.md REQ-M-013 model from claude-sonnet-4 to claude-opus-4-6 (aligns with D-22) — done in preliminary commit 1bf7776

## Session Handoff

**Last session:** 2026-04-14
**Branch:** main (clean after commit below)
**State:** Phase 2 code complete — 8/8 plans done. 111 unit/mock tests green across 8 suites. Integration test created but needs ANTHROPIC_API_KEY to run.
**Pick up with:** Run `node test/integration.js` with API key, then `/gsd:verify-work` against Phase 2 success criteria, then close Phase 2.
