---
phase: 04-compositor
plan: "01"
subsystem: compositor
tags: [sharp, node-canvas, canvas, png, linkedin-chrome, inter-font, svg, rendering]

requires:
  - phase: 03-visual-grammar-image-prompt-builder
    provides: zone-spec.json pixel coordinates, engagement.js metric shape, Post Brief contract
  - phase: 01-mirror-post-scaffolding
    provides: repo structure, barrel export pattern, test harness pattern

provides:
  - compositePost(brief, heroImagePath) → 1920x1080 PNG Buffer
  - renderTopNav, renderProfileBar, renderEngagementBar canvas 2D renderers
  - COLORS, FONTS, ZONES, REACTION_EMOJI, CHROME_BLUE constants
  - Inter font TTF files (Regular, Bold, SemiBold) committed to repo
  - 6 SVG assets (5 nav icons + verify badge)

affects:
  - 04-02 (text overlay + tweet embed will add compositing layers on top of this chrome foundation)

tech-stack:
  added: [sharp@0.34.5, canvas@3.2.3]
  patterns:
    - canvas 2D render → raw RGBA buffer → sharp composite pipeline (D-03)
    - exact-version npm pins for deterministic pixel output (REQ-X-052)
    - fonts bundled in repo via registerFont at module load (D-08)
    - SVG assets pre-loaded as base64 data URIs at import time (performance + determinism)
    - graceful null/undefined handling for optional brief fields

key-files:
  created:
    - mirror-post/src/compositor/constants.js
    - mirror-post/src/compositor/render-chrome.js
    - mirror-post/src/compositor/composite.js
    - mirror-post/src/compositor/index.js
    - mirror-post/src/compositor/assets/linkedin-home.svg
    - mirror-post/src/compositor/assets/linkedin-network.svg
    - mirror-post/src/compositor/assets/linkedin-jobs.svg
    - mirror-post/src/compositor/assets/linkedin-messaging.svg
    - mirror-post/src/compositor/assets/linkedin-notifications.svg
    - mirror-post/src/compositor/assets/verify-badge.svg
    - mirror-post/src/compositor/fonts/Inter-Regular.ttf
    - mirror-post/src/compositor/fonts/Inter-Bold.ttf
    - mirror-post/src/compositor/fonts/Inter-SemiBold.ttf
    - mirror-post/test/compositor-chrome.js
  modified:
    - mirror-post/package.json (added sharp, canvas with exact pins; test:compositor + test:all)

key-decisions:
  - "REACTION_EMOJI map includes both lowercase (fixture values) and Title-case (engagement.js REACTION_TYPES) keys — handles both conventions without normalization overhead"
  - "SVG files pre-loaded as base64 data URIs at module import time — avoids async I/O inside render calls, ensures determinism"
  - "Chrome renderers are async to allow loadImage() for SVG icons; compositePost calls them in parallel via Promise.all"
  - "sharp exact-version pin 0.34.5, canvas exact-version pin 3.2.3 — REQ-X-052 byte-identical output precondition"

patterns-established:
  - "Canvas 2D render pattern: createCanvas → fill background → draw elements → return toBuffer('raw') with {x, y} position"
  - "Sharp composite pattern: create white base → overlay hero → overlay raw RGBA chrome layers in fixed order → .png({compressionLevel:9, adaptiveFiltering:false})"
  - "Font registration pattern: registerFont at module load with family name matching ctx.font strings"

requirements-completed:
  - REQ-M-040
  - REQ-M-043
  - REQ-M-044
  - REQ-X-050
  - REQ-X-051
  - REQ-X-052
  - REQ-X-060
  - REQ-X-061
  - REQ-X-064
  - REQ-X-065
  - REQ-X-066

duration: 15min
completed: 2026-04-15
---

# Phase 04 Plan 01: Compositor Chrome Layer Summary

**sharp + node-canvas compositor foundation: 1920x1080 PNG with LinkedIn chrome (blue top nav, profile bar with initials avatar, engagement bar with emoji reactions), byte-identical deterministic output verified via Buffer.compare**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-15T13:00:00Z
- **Completed:** 2026-04-15T13:15:00Z
- **Tasks:** 1 (all steps executed)
- **Files modified:** 15

## Accomplishments

- compositePost(brief, heroImagePath) produces 1920x1080 PNG Buffer with full LinkedIn chrome in a single sharp composite pass
- Byte-identical output confirmed: Buffer.compare(buf1, buf2) === 0 on two calls with identical inputs (REQ-X-052)
- Inter fonts (Regular, Bold, SemiBold, ~400KB each) committed to repo — no runtime font fetching, deterministic across platforms
- 10/10 chrome rendering tests pass; 153/153 prior-phase tests remain green (zero regression)

## Task Commits

1. **Task 1: Chrome layer — constants, fonts, SVGs, renderers, compositePost** - `267bd30` (feat)

## Files Created/Modified

- `mirror-post/src/compositor/constants.js` - CHROME_BLUE, COLORS, FONTS, ZONES, REACTION_EMOJI, ACTION_BUTTONS
- `mirror-post/src/compositor/render-chrome.js` - renderTopNav, renderProfileBar, renderEngagementBar
- `mirror-post/src/compositor/composite.js` - compositePost entry point with sharp compositing pipeline
- `mirror-post/src/compositor/index.js` - barrel export
- `mirror-post/src/compositor/assets/*.svg` - 5 nav icons + verify badge (6 SVG files)
- `mirror-post/src/compositor/fonts/Inter-{Regular,Bold,SemiBold}.ttf` - ~407KB each
- `mirror-post/test/compositor-chrome.js` - 10 tests (Buffer, PNG magic bytes, 1920x1080, determinism, avatar, engagement counts, nav dimensions, emoji circles, null guards)
- `mirror-post/package.json` - added sharp@0.34.5, canvas@3.2.3 with exact pins; test:compositor and test:all scripts

## Decisions Made

- REACTION_EMOJI map uses both lowercase and Title-case keys to handle Post Brief fixture values (`"like"`) and engagement.js REACTION_TYPES (`"Like"`) without normalization.
- SVG nav icons pre-loaded as base64 data URIs at module import time rather than per render call — avoids async I/O in the hot path and removes a source of timing variance.
- Chrome renderers declared async to support loadImage() for SVG icons; compositePost runs all three in parallel via Promise.all for performance.

## Deviations from Plan

None — plan executed exactly as written. All files, tests, and verification criteria met on first pass.

## Issues Encountered

macOS ARM runtime warning: `Class GNotificationCenterDelegate is implemented in both libvips-cpp.dylib (sharp) and libgio-2.0.0.dylib (canvas)`. This is a known library conflict between sharp and node-canvas on Apple Silicon — a spurious objc warning, not an error. Output is correct and deterministic. No fix required; tracked as known non-issue.

## User Setup Required

None — no external service configuration required. sharp and canvas install prebuilt binaries on macOS ARM.

## Known Stubs

None — all rendered content is data-driven from the Post Brief. Text overlay and tweet embed card are intentionally absent (deferred to Plan 04-02 per plan scope).

## Next Phase Readiness

Plan 04-02 (text overlay + tweet embed) can proceed immediately. The compositePost pipeline accepts additional sharp composite layers — the text overlay gradient buffer and tweet embed card buffer slot in at positions already defined in zone-spec.json. The chrome layer is complete and tested.

## Self-Check: PASSED

- `mirror-post/src/compositor/composite.js` — FOUND
- `mirror-post/src/compositor/render-chrome.js` — FOUND
- `mirror-post/src/compositor/constants.js` — FOUND
- `mirror-post/src/compositor/index.js` — FOUND
- `mirror-post/src/compositor/fonts/Inter-Regular.ttf` — FOUND (407KB)
- `mirror-post/test/compositor-chrome.js` — FOUND
- Commit `267bd30` — FOUND (feat(04-01): compositor chrome layer)
- 10 tests pass, 0 failures
- Buffer.compare determinism verified

---
*Phase: 04-compositor*
*Completed: 2026-04-15*
