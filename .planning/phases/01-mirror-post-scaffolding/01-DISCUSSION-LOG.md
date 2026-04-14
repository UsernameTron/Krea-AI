# Phase 1: Mirror Post Scaffolding - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-14
**Phase:** 01-mirror-post-scaffolding
**Areas discussed:** Post Brief schema shape, mirror-post git strategy, fixture depth, voice slider encoding

---

## Post Brief Schema Shape

### Props Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Normalize to uniform array | All props as { type, text, placement } objects in a single array | ✓ |
| Keep PLAN_01 mixed structure | mug/whiteboard as named objects, desk_items as array | |

**User's choice:** Normalize to uniform array
**Notes:** Eliminates branching in all downstream consumers. Cheapest time to fix is at scaffolding.

### Field Optionality

| Option | Description | Selected |
|--------|-------------|----------|
| Always present | tweet_embed and nav_easter_eggs both required | |
| Optional per post | Both optional | |
| Tweet always, nav_easter_eggs optional | Tweet is core format, nav easter eggs are bonus detail | ✓ |

**User's choice:** Tweet always required, nav_easter_eggs optional
**Notes:** Not every post needs nav misspellings but every post has a tweet embed.

---

## mirror-post Git Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Separate repo (like flux-krea) | git init inside mirror-post/, gitignored by Krea-AI | ✓ |
| Subdirectory of Krea-AI | No separate git, lives in parent repo | |
| Git submodule | Formal submodule reference | |

**User's choice:** Separate repo
**Notes:** Matches established flux-krea workspace pattern. Independent commit history and CI.

---

## Fixture Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid: 1 gold + 3 skeletons | Brent Vellum full depth, Trevor x2 and Pete skeletons | ✓ |
| Full depth for all 4 | Every field populated for all fixtures | |
| Skeletons only for all 4 | Key fields only for all fixtures | |

**User's choice:** Hybrid — Brent Vellum gold, 3 skeletons
**Notes:** Gold fixture validates schema completeness, skeletons validate structural shape without brittle field-level assertions.

---

## Voice Slider Encoding

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror persona spec (mixed) | Floats for continuous (sarcasm, cynicism, warmth) + ordinal 1-5 for intensity | ✓ |
| All floats 0.0-1.0 | Single encoding, simpler but loses intensity semantics | |
| All ordinal 1-5 | 5-level scale for everything, coarser but cross-slider constraints possible | |

**User's choice:** Mirror persona spec encoding
**Notes:** Preserves intentional heterogeneity from the persona spec. Both validation types are single-line deterministic assertions per Pattern 2.

---

## Claude's Discretion

- Schema validation library choice (ajv, Zod, or hand-written)
- Barrel export structure
- Directory naming convention
- README.md content depth

## Deferred Ideas

- Asset source locations — not discussed, planner handles at execution
- nav_easter_eggs rename to chrome_details — not worth churn for v1
