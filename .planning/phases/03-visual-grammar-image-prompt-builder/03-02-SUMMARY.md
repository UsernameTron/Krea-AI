---
phase: 03-visual-grammar-image-prompt-builder
plan: "02"
subsystem: mirror-post/grammar
tags: [visual-grammar, engagement-generator, zone-spec, prop-taxonomy, tdd]
dependency_graph:
  requires: [01-03, 02-07]
  provides: [grammar-module, zone-spec, engagement-metrics, prop-taxonomy]
  affects: [03-01, phase-04-compositor, phase-05-artifact-ui]
tech_stack:
  added: []
  patterns: [mulberry32-seeded-prng, barrel-exports, tdd-red-green, zero-dependencies]
key_files:
  created:
    - mirror-post/src/grammar/zone-spec.json
    - mirror-post/src/grammar/engagement.js
    - mirror-post/src/grammar/prop-taxonomy.js
    - mirror-post/test/grammar.js
  modified:
    - mirror-post/src/grammar/index.js
    - mirror-post/package.json
decisions:
  - "Mulberry32 PRNG for deterministic engagement: fast, portable, no dependencies, correct statistical properties"
  - "Funny reaction hard-capped at 4% (not 5%) across all tiers: satire lands harder when characters don't see the joke"
  - "readFileSync for JSON import in index.js: avoids JSON import assertion compatibility concerns with Node 20/22"
  - "tier-detection inferred from scene_template, not archetype.category: Post Brief doesn't carry category directly"
metrics:
  duration: "~3 minutes"
  completed: "2026-04-14"
  tasks_completed: 2
  files_created: 4
  files_modified: 2
---

# Phase 03 Plan 02: Visual Grammar Module Summary

**One-liner:** Mulberry32-seeded engagement generator with 4-tier satirical calibration, 1920x1080 zone spec, and 10-type prop taxonomy — all zero-dependency, zero-LLM-calls.

## Objective

Build the Visual Grammar module: compositor zone specification (pixel coordinates for LinkedIn chrome layout on 1920x1080), engagement metric generator (satirically calibrated by character tier and post tone), and prop taxonomy (rendering metadata for each prop type).

## Tasks Completed

### Task 1 — Zone spec, prop taxonomy, and engagement generator

**TDD RED:** Wrote 22 failing tests in `test/grammar.js` covering all zone bounds, engagement tiers, Funny suppression, determinism, and prop taxonomy. Tests failed with `SyntaxError: The requested module does not provide an export named 'PROP_TAXONOMY'` as expected.

**TDD GREEN:** Implemented all four source files:

- **`zone-spec.json`** — 1920x1080 canvas with 6 top-level zones. text_overlay_zone width = 730px (38% of 1920, within 35-40% constraint). All zone coordinates verified within canvas bounds. engagement_bar has sub_zones for reaction_icons, reaction_count, comment_count, action_buttons.

- **`engagement.js`** — Mulberry32 PRNG seeded from `options.seed`. Tier detection from `image_seed.scene_template`: office-executive→c-suite, office-middle-mgmt→middle-management, airport-hustle→hustle-culture, call-center-floor→support-ops (default: middle-management). Each tier has calibrated min/max reaction counts, dominant reaction type, dominant_pct range, and comment_ratio range. Funny hard-capped at 4% via `FUNNY_CAP_PCT` constant. High satirical intensity (≥4) biases generation toward upper 40% of tier range. Returns `{ reactions: { count, types, distribution }, comments, dominant_reaction }`.

- **`prop-taxonomy.js`** — 10 prop types (mug, whiteboard, nameplate, business_card, folder, sticky_note, monitor, headset, laptop, plant) with category, has_text, typical_placement array, render_hint, and typical_dimensions. `getPropMeta()` returns a "unknown" fallback for unrecognized prop types.

- **`index.js`** — Barrel export using `readFileSync` for zone-spec.json (avoids JSON import assertion compatibility), re-exporting `generateEngagement`, `PROP_TAXONOMY`, `getPropMeta`.

All 22 tests passed on first GREEN run.

### Task 2 — Test harness and package.json scripts

Added `test:grammar` and appended `&& node test/grammar.js` to `test:all` in `package.json`.

## Commits

| Hash | Message |
|------|---------|
| 296ef22 | test(03-02): add failing tests for grammar module (zone-spec, engagement, prop-taxonomy) |
| 16a5dc0 | feat(03-02): implement grammar module — zone-spec, engagement generator, prop taxonomy |
| bb1dd1b | chore(03-02): add test:grammar and update test:all scripts in package.json |

## Verification Results

```
22 passed, 0 failed, 22 total
```

- `ZONE_SPEC.canvas`: 1920x1080 ✓
- `text_overlay_zone.width`: 730px (38%, within 35-40%) ✓
- All zone bounds within 1920x1080 ✓
- C-suite (seed=42): reactions.count=2025, Funny=26 (1.3%) ✓
- Support/ops (seed=42): reactions.count=84 ✓
- Middle-mgmt (seed=42): reactions.count=238 ✓
- Hustle-culture (seed=42): reactions.count=1162 ✓
- Deterministic with seed: same count across calls ✓
- Variable without seed: 3 unique values on 3 calls ✓
- Funny < 5% across all tiers ✓
- Module exports: PROP_TAXONOMY, ZONE_SPEC, generateEngagement, getPropMeta ✓

## Deviations from Plan

None — plan executed exactly as written, with one minor implementation note:

**Implementation note:** The plan suggested `import zoneSpecData from './zone-spec.json' with { type: 'json' }` with a fallback to `readFileSync`. The `readFileSync` fallback was used directly in `index.js` to avoid any JSON import assertion edge cases across Node 20/22 ESM environments. Behavior is identical.

## Design Decisions

**Mulberry32 PRNG over LCG:** Mulberry32 is a well-tested 32-bit PRNG with good avalanche properties and no external dependencies. It produces statistically random output for seed values in the normal integer range and is faster and more portable than crypto-based alternatives for this non-security use case.

**Tier detection from scene_template:** The Post Brief's `image_seed.scene_template` field drives tier selection rather than reading archetype `category` from `library.json`. This keeps the grammar module stateless and brief-only — it doesn't need to know the archetype library schema or perform lookups. The Phase 2 LLM already assigns `scene_template` correctly.

**4% Funny cap vs 5% spec:** The spec says "< 5%", so 4% is the correct hard cap. The implementation uses `FUNNY_CAP_PCT = 0.04` (floor at 4%), ensuring the constraint is always met even at floating-point boundaries.

**distribution field as extension:** The plan specified `distribution` as an additional field beyond the Post Brief schema (which only has `types` as string array). The implementation returns both: `types` (array of reaction type names with count > 0) for schema compatibility and `distribution` (object with exact counts) for compositor use.

## Requirements Coverage

| REQ-ID | Description | Status |
|--------|-------------|--------|
| REQ-M-020 | Prop taxonomy | COMPLETE |
| REQ-M-021 | Composition zones | COMPLETE |
| REQ-M-022 | Engagement metric generator | COMPLETE |
| REQ-M-023 | Engagement satirically calibrated | COMPLETE |

## Self-Check: PASSED

Files verified:
- `mirror-post/src/grammar/zone-spec.json` — FOUND
- `mirror-post/src/grammar/engagement.js` — FOUND
- `mirror-post/src/grammar/prop-taxonomy.js` — FOUND
- `mirror-post/src/grammar/index.js` — FOUND (populated)
- `mirror-post/test/grammar.js` — FOUND

Commits verified:
- 296ef22 — FOUND (test RED phase)
- 16a5dc0 — FOUND (feat GREEN phase)
- bb1dd1b — FOUND (chore package.json)
