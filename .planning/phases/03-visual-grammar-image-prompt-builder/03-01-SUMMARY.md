---
phase: 03-visual-grammar-image-prompt-builder
plan: "01"
subsystem: image-prompt-engine
tags: [js-yaml, deterministic, flux, image-prompts, scene-templates, foundation-prompt]

# Dependency graph
requires:
  - phase: 01-mirror-post-scaffolding
    provides: "foundation-prompt.yaml, ultra-fidelity.json, 12k-modifiers.json, Post Brief schema"
  - phase: 02-post-brief-generator
    provides: "Post Brief JSON contract with image_seed fields (scene_template, environment, subject_pose, mood)"

provides:
  - "buildImagePrompt(brief) — deterministic Post Brief to Flux prompt transformer"
  - "loadFoundation() / getFoundation() — singleton loader for foundation-prompt.yaml"
  - "4 scene templates (office-executive, office-middle-mgmt, airport-hustle, call-center-floor)"
  - "Structured prompt object: { positive_prompt, negative_prompt, parameters, composition_notes }"
  - "21-test harness covering singleton, template routing, freeform pass-through, determinism, modifier count"

affects:
  - "Phase 5 — Compositor (consumes composition_notes.text_clear_zone and zone positions)"
  - "Phase 6 — Artifact UI (consumes full 4-key prompt object for display)"
  - "flux-krea integration (positive_prompt, negative_prompt, parameters sent to --prompt-file)"

# Tech tracking
tech-stack:
  added:
    - "js-yaml ^4.1.1 — YAML parsing for foundation-prompt.yaml"
  patterns:
    - "Singleton module-level caching: load once, return cached ref on subsequent calls"
    - "Deterministic index selection: FNV-1a hash of brief content → stable modifier picks without Math.random()"
    - "Fixed D-05 concatenation order: foundation → character → environment → props → composition → fidelity modifiers"
    - "Template routing with freeform fallback: known IDs load JSON, unknown/absent uses image_seed.environment directly"
    - "Negative prompt assembled once at module init (no brief dependency) for efficiency"

key-files:
  created:
    - "mirror-post/src/image/foundation-loader.js — singleton YAML loader, exports loadFoundation/getFoundation"
    - "mirror-post/src/image/prompt-builder.js — buildImagePrompt(brief), 180+ lines, full D-05 pipeline"
    - "mirror-post/src/image/templates/office-executive.json — corner office, floor-to-ceiling windows, mahogany desk"
    - "mirror-post/src/image/templates/office-middle-mgmt.json — fluorescent cubicle grid, institutional lighting"
    - "mirror-post/src/image/templates/airport-hustle.json — business-class lounge, tarmac window, mixed lighting"
    - "mirror-post/src/image/templates/call-center-floor.json — vast workstation rows, 6000K fluorescent, monitor glow"
    - "mirror-post/test/image-prompt.js — 21-test harness for image prompt builder"
  modified:
    - "mirror-post/src/image/index.js — populated barrel export (was empty placeholder)"
    - "mirror-post/package.json — added test:image script, appended to test:all, added js-yaml dependency"

key-decisions:
  - "Modifier count fixed at 16 (2 per 8 categories) — consistently within 15-20 range per REQ-M-032"
  - "Negative prompt built once at module init (not per-call) since it has no brief dependency — avoids repeated string construction"
  - "Template cache Map in prompt-builder.js to avoid re-reading JSON on repeated calls for same template"
  - "FNV-1a hash (not MD5/SHA) for modifier seed — lightweight, zero deps, stable across Node.js versions"
  - "Props-with-text described in positive_prompt (not compositor layer) per D-14 — Flux renders hero image text, compositor handles overlay text"

patterns-established:
  - "D-05 concatenation order is canonical for all future positive_prompt construction"
  - "All new image module files follow ESM with import.meta.url path resolution (no __dirname)"
  - "Test harness pattern: load fixtures via readFile, inline minimal briefs as helpers, pass/fail counters, process.exit"

requirements-completed:
  - REQ-M-030
  - REQ-M-031
  - REQ-M-032
  - REQ-M-033
  - REQ-M-034
  - REQ-M-035
  - REQ-X-060
  - REQ-X-061
  - REQ-X-062
  - REQ-X-063
  - REQ-X-064
  - REQ-X-065
  - REQ-X-066
  - REQ-X-070
  - REQ-X-071

# Metrics
duration: 4min
completed: 2026-04-14
---

# Phase 3 Plan 01: Foundation Loader, Scene Templates, and Prompt Builder Summary

**Deterministic buildImagePrompt(brief) function using FNV-1a hash-seeded modifier selection, 4 scene templates with freeform fallback, and foundation-prompt.yaml singleton — all 21 tests green, zero LLM calls**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-04-14T23:50:51Z
- **Completed:** 2026-04-14T23:55:03Z
- **Tasks:** 2 (TDD RED + GREEN for Task 1; Task 2 script update)
- **Files modified:** 9 (7 created, 2 updated)

## Accomplishments

- `buildImagePrompt(brief)` transforms any valid Post Brief into a 4-key Flux-compatible prompt object following D-05 concatenation order — foundation prepended verbatim, character delta, environment (template or freeform), props with text, composition directives, 16 fidelity modifiers
- 4 scene templates created with full environment/lighting/camera specs; known IDs route to templates, unknown/absent values pass image_seed.environment directly
- Foundation singleton (`loadFoundation` / `getFoundation`) loads YAML once per session; `getFoundation()` returns same object reference on repeated calls
- 21-test harness covers all must-have truths: singleton identity, template routing, freeform pass-through, absent template, determinism, no MidJourney syntax, modifier count, second fixture (Trevor airport-hustle)

## Task Commits

1. **TDD RED — failing test harness** - `804b52d` (test)
2. **TDD GREEN — foundation loader, 4 templates, prompt builder** - `48720ec` (feat)
3. **Task 2 — test:image script and test:all update** - `ab6f92e` (chore)

## Files Created/Modified

- `mirror-post/src/image/foundation-loader.js` — singleton YAML loader; loadFoundation() caches on first call using FNV-1a-like module-level variable
- `mirror-post/src/image/prompt-builder.js` — buildImagePrompt(brief) implementing full D-05 pipeline with FNV-1a hash for deterministic modifier selection
- `mirror-post/src/image/templates/office-executive.json` — corner office, cityscape windows, Bloomberg terminal, low-angle executive framing
- `mirror-post/src/image/templates/office-middle-mgmt.json` — fluorescent cubicle grid, 5000-5500K institutional lighting, desk-level medium shot
- `mirror-post/src/image/templates/airport-hustle.json` — business-class lounge, tarmac windows, warm/cool mixed lighting, three-quarter chair framing
- `mirror-post/src/image/templates/call-center-floor.json` — vast workstation rows, 6000K blue-white fluorescent, monitor glow, elevated medium shot
- `mirror-post/test/image-prompt.js` — 21-test harness following harness.js pattern
- `mirror-post/src/image/index.js` — barrel export now populated (was empty comment)
- `mirror-post/package.json` — test:image added, test:all extended, js-yaml in dependencies

## Decisions Made

- Modifier count fixed at 16 (2 picks per 8 ultra-fidelity categories) — consistently within the 15-20 range without edge cases
- FNV-1a hash over character.name + scene_template used as modifier seed — lightweight, zero external dependencies, deterministic across Node.js versions
- Negative prompt built once at module initialization since it has no brief dependency — avoids string construction overhead on every call
- Template JSON file cache (`Map`) in prompt-builder.js to avoid re-reading disk on repeated calls for same template ID
- Props with text described in positive_prompt per D-14 — Flux renders prop text in hero image; compositor owns headline/body text overlays separately

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Modifier count was 14 instead of >=15**

- **Found during:** Task 1 (first test run)
- **Issue:** Initial modifier selection picked 2 per 6 categories + 1 each from post_processing + background_layout = 14. Test requires >=15.
- **Fix:** Changed post_processing_output and background_layout_constraints picks from 1 to 2 each — total now 16.
- **Files modified:** `mirror-post/src/image/prompt-builder.js` (selectFidelityModifiers function)
- **Verification:** `Fidelity modifier count >= 15 (found: 16)` PASS
- **Committed in:** `48720ec` (feat commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — off-by-two in modifier count)
**Impact on plan:** Minimal — single function change, no architectural impact.

## Issues Encountered

The parallel 03-02 agent had already added `js-yaml` to package.json and a `test:grammar` script before Task 2 ran. The `package.json` read-before-write rule caught a stale file error, which was resolved by re-reading before editing. The `test:image` and `test:all` additions were applied cleanly to the updated file.

## Known Stubs

None — all 4 template files contain real structured scene data. All output paths produce real prompt strings from real inputs. No placeholder values in created files.

## Next Phase Readiness

- `buildImagePrompt(brief)` is ready for consumption by Phase 5 (Compositor) and Phase 6 (Artifact UI)
- The 4-key output object matches the planned flux-krea `--prompt-file` contract (composition_notes stripped for flux-krea use)
- 03-02 (Visual Grammar — zone spec and engagement generator) runs in parallel and is the only remaining plan in Phase 3

---
*Phase: 03-visual-grammar-image-prompt-builder*
*Completed: 2026-04-14*
