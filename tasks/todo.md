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

## Phase 3 Prompt Artifact Build (Product Reset 2, 2026-04-15 evening)

Build happens in a separate Claude Desktop chat with artifact support. Do NOT run `/gsd:execute-phase`.

- [ ] Build React artifact in Claude Desktop chat: idea textarea, 4 voice sliders (sarcasm, cynicism, warmth, satirical_intensity), archetype dropdown (17 archetypes + auto-assign), generate button, copy-to-clipboard on output
- [ ] Artifact calls `window.claude.complete` with persona distillate + archetype + sliders + idea
- [ ] Parse response into Post Brief JSON
- [ ] Map archetype → variant (A/B/C/D) via D-NEW-05 table; load variant template from `.planning/phases/05-image-prompt-engine/templates/`
- [ ] Fill variant template by substituting `{{placeholders}}` with Post Brief fields
- [ ] Display: filled prompt text (copy button), Post Brief JSON preview, archetype match confidence
- [ ] Obsidian dark-mode aesthetic (deep navy, gold accents, cream text)
- [ ] When artifact satisfies, commit to `mirror-post/artifact/MirrorPoster.jsx`
- [ ] Optional later: promote to deployed web app (Vercel)

## Session Handoff

**Last session:** 2026-04-15 evening (Product Reset 2 — eb96874)
**Branch:** docs/path-c-pivot (clean)
**State:** Scope collapsed. Compositor/chrome/scene/FLUX-integration work abandoned. New Phase 3 = single Claude Desktop React artifact (prompt generator). Archive branches preserved in mirror-post. See .planning/PRODUCT-RESET-2.md.
**Pick up with:** Build React artifact in a separate Claude Desktop chat with artifact support. When satisfied, commit to `mirror-post/artifact/MirrorPoster.jsx`. Do NOT run `/gsd:execute-phase`.

**Prior session:** 2026-04-15 evening (Path 2 doc hygiene — 6c16097)

**Prior session:** 2026-04-15 PM (Path 2 critical-path amendment — c9ca02a)

---

**Prior session:** 2026-04-15 (discuss-phase 04.2 — decisions captured, writeup deferred)
**Branch:** main (commit 8567444 — wip handoff)
**State:** Phase 04.2 discussion complete. Two architecture amendments captured in .continue-here.md but not yet written to 04-CONTEXT.md. D-NEW-01 REVISED (chrome ownership: Phase 5 → Phase 4). D-NEW-06 RESOLVED (Post Brief v2 bump).
**Pick up with:**
1. `/clear` then read `.planning/phases/04-compositor/.continue-here.md` + `04-CONTEXT-RESET.md` + original `04-CONTEXT.md` + `$HOME/.claude/get-shit-done/templates/context.md`
2. Write `.planning/phases/04-compositor/04-CONTEXT.md` (supersedes both prior CONTEXT files; annotate D-NEW-01 as REVISED and D-NEW-06 as RESOLVED)
3. Write `.planning/phases/04-compositor/04-DISCUSSION-LOG.md` (advisor A/B/C summaries + collision + user picks)
4. Commit both, update STATE.md to `context_captured_ready_for_plan`
5. Before `/gsd:plan-phase 04.2`: get operator confirmation on D-NEW-01 walk-back and chrome PNG authoring contract (4 human-authored PNGs required)
6. Still-open carry-over gaps: ROADMAP renumbering, Phase 5 split (5a/5b), deprecated module deletion timing, Phase 2 v2 migration scope

## Retired by Product Reset 2 (2026-04-15 evening)
All Variant Asset Delivery / Scene Asset Delivery / Phase 5 Preconditions / Path 2 / Path C items retired. See `.planning/PRODUCT-RESET-2.md`.

## flux-krea Technical Debt (external audit, 2026-04-15)
Audit flagged 6 findings — none block Phase 4.2 Path 2 (Mirror Post runtime doesn't call flux-krea). Track for future flux-krea maintenance phase.
- [ ] HIGH: Config validation runs before command overrides — --width/--steps reach pipeline unvalidated (config.py:278, main.py:227, pipeline.py:287)
- [ ] HIGH: Optimization fallback chain doesn't unload failed model — OOM spiral risk on large model (pipeline.py:103, pipeline.py:134). WORKAROUND during variant gen: small batches, --optimization standard, restart on failure
- [ ] MED-HIGH: Web UI signal-based timeout silently no-ops off-main-thread (app.py:113-115)
- [ ] MED-HIGH: Neural Engine path is dead — compile_vae_decoder saves .mlpackage but pipeline never calls it, optimize_pipeline returns pipeline unchanged, MAXIMUM mode only invokes Metal (neural_engine.py:162, :366, pipeline.py:213). Tests lock in broken behavior (test_neural_engine.py:297)
- [ ] MEDIUM: Profiler reports fake stage timings — callback never wired, falls back to synthetic breakdowns (profiler.py:79, :124)
- [ ] MEDIUM: ThermalManager produces profile fields that pipeline never applies (only inference_steps_scale used; max_cpu_threads, max_gpu_utilization, memory_fraction ignored — thermal.py:44, pipeline.py:229, :299)
- [ ] BLOCKER for re-running audit: torch not installed in audit workspace (Python 3.14 vs required 3.10-3.12). Re-run audit after venv setup (already tracked under Phase 5 Preconditions)
