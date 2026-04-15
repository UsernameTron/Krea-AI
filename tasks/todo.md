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
- [x] Fix integration test failures: timeout (15s→90s), safety prompt (no fictional contacts), jargon prompt (require quotes) — commit 8aef5d8
- [ ] Run integration test with real API key (5 scenarios, `node test/integration.js`)
- [ ] /gsd:verify-work — validate against Phase 2 success criteria

## Backlog

- [ ] Review CONCERNS.md and prioritize tech debt items
- [x] Update REQUIREMENTS.md REQ-M-013 model from claude-sonnet-4 to claude-opus-4-6 (aligns with D-22) — done in preliminary commit 1bf7776

## Phase 3 Visual Grammar + Image Prompt Builder — COMPLETE

- [x] /gsd:discuss-phase 3 — 14 decisions captured in 03-CONTEXT.md
- [x] /gsd:plan-phase 3 --skip-research — 2 plans in 1 wave, verified 10/10
- [x] /gsd:execute-phase 3 — 2 plans parallel (Wave 1), 43 new tests, 153 total green
- [x] /gsd:verify-work — 14/14 must-haves verified, 19/19 REQs accounted for

## Phase 4 Compositor — NOT STARTED

- [ ] /gsd:discuss-phase 4 — gather context for compositor (LinkedIn chrome, text overlay, tweet embed, engagement bar)
- [ ] /gsd:plan-phase 4 — create plans
- [ ] /gsd:execute-phase 4 — build compositor
- [ ] /gsd:verify-work — validate

## Session Handoff

**Last session:** 2026-04-15 (discuss-phase 04.2 — decisions captured, writeup deferred)
**Branch:** main (commit 8567444 — wip handoff)
**State:** Phase 04.2 discussion complete. Two architecture amendments captured in .continue-here.md but not yet written to 04-CONTEXT.md. D-NEW-01 REVISED (chrome ownership: Phase 5 → Phase 4). D-NEW-06 RESOLVED (Post Brief v2 bump).
**Pick up with:**
1. `/clear` then read `.planning/phases/04-compositor/.continue-here.md` + `04-CONTEXT-RESET.md` + original `04-CONTEXT.md` + `$HOME/.claude/get-shit-done/templates/context.md`
2. Write `.planning/phases/04-compositor/04-CONTEXT.md` (supersedes both prior CONTEXT files; annotate D-NEW-01 as REVISED and D-NEW-06 as RESOLVED)
3. Write `.planning/phases/04-compositor/04-DISCUSSION-LOG.md` (advisor A/B/C summaries + collision + user picks)
4. Commit both, update STATE.md to `context_captured_ready_for_plan`
5. Before `/gsd:plan-phase 04.2`: get operator confirmation on D-NEW-01 walk-back and chrome PNG authoring contract (4 human-authored PNGs required)
6. Still-open carry-over gaps: ROADMAP renumbering, Phase 5 split (5a/5b), deprecated module deletion timing, Phase 2 v2 migration scope

## Variant Asset Delivery (Path 2)
- [ ] Generate 3-5 sub-variants per letter (A/B/C/D), 12-20 total
- [ ] Same prompt per letter, vary seed only (keeps layout stable)
- [ ] 1920x1080 RGB, full opaque, no transparency
- [ ] Commit to mirror-post/src/compositor/variant-assets/{a,b,c,d}/
- [ ] Measure VARIANT_SLOTS coords from one reference per letter, update mirror-post/src/compositor/constants.js

## Deferred follow-ups (Path 2 amendment, 2026-04-15)
- [ ] REQUIREMENTS.md REQ-X-052a retarget, ROADMAP Phase 5 collapse, STATE.md note, DISCUSSION-LOG appendix, D-NEW-09 revision (tracked as one item — next session)

## Phase 5 Preconditions (captured 2026-04-15)
- [ ] Update flux-krea/CLAUDE.md Python requirement to "3.10-3.12" (currently says "3.10+" which is misleading — torch 2.7 wheels don't exist for Python 3.14)
- [ ] First task of Phase 5 execution: onboarding runbook (homebrew python@3.11, venv creation, pip install -r requirements.txt, HF token setup, one-shot verification render). Saves 30+ min of rediscovery.
