# Phase 4: Compositor - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Assemble a flux-krea hero image (1920x1080 PNG) plus a Post Brief JSON into a final 1920x1080 PNG with LinkedIn UI chrome (top nav, profile bar, engagement bar), left-side gradient text overlay (headline with gold highlight words, body with inline bold/italic), tweet embed card (lower-right white rounded card with verification badge), and engagement metrics. Output is deterministic — byte-identical given identical inputs. The compositor is the assembly layer; it does NOT render prop text (mugs, whiteboards) — those are in the hero image from Flux.

</domain>

<decisions>
## Implementation Decisions

### Rendering Engine
- **D-01:** Sharp + node-canvas hybrid. Canvas 2D API handles text overlay rendering (gradients, per-word highlight color, inline bold/italic, word-wrap reflow) and gradient mask generation. Sharp handles image compositing — loading the hero PNG, layering the gradient buffer, chrome elements, and final PNG export. Two npm dependencies: `sharp` (prebuilt arm64 binary, zero compile) and `canvas` (node-canvas, requires Cairo + Pango via Homebrew on macOS).
- **D-02:** Puppeteer eliminated — documented subpixel font variation across headless Chrome runs is incompatible with REQ-X-052 pixel-comparison tests.
- **D-03:** Rendering pipeline: (1) Load hero image via Sharp, (2) Render gradient + text overlay to Canvas 2D buffer, (3) Render chrome elements (nav, profile bar, engagement bar) to Canvas 2D buffer, (4) Render tweet embed card to Canvas 2D buffer, (5) Composite all layers via Sharp `composite()`, (6) Export final 1920x1080 PNG.

### Chrome Visual Style
- **D-04:** LinkedIn-inspired simplified SVG outline icons for top nav (home, network, briefcase, speech bubble, bell). Recognizable shapes without being pixel-copies of LinkedIn brand assets. REQ-M-044 excludes the wordmark and logo — functional nav glyphs are standard across platforms.
- **D-05:** Custom "corporate blue" `#0b5ca8` for chrome elements — perceptually identical to LinkedIn but legally distinct per REQ-M-044 ("no exact color match"). Single style constant, documented as Mirror Post chrome blue.
- **D-06:** Profile bar avatar is a circle with styled initials derived from `character.name` in the Post Brief (first + last initial, chrome-blue on white or white on chrome-blue). Fully deterministic (REQ-X-052), data-driven, zero external assets.
- **D-07:** Engagement reaction icons are simplified colored circles with emoji-like Unicode characters (👍❤👏💡😂🤗) rendered via Canvas `fillText` at 24-32px in the 40px reaction zone. Zero custom icon assets. Upgradeable to custom SVGs in future iteration if visual review flags quality.

### Text Overlay Styling
- **D-08:** Font family is **Inter** (Google Fonts, OFL license), bundled as TTF files in `src/compositor/fonts/`. Two weights minimum: Regular (400) for body, Bold (700) for headline. Consistent cross-platform rendering via Canvas/Cairo font registration. System fonts rejected — Canvas uses Cairo/Pango fallbacks that violate determinism on different OS configurations.
- **D-09:** Headline highlight treatment is **gold color change (`#FFD700`)** — already committed in Post Brief schema's `highlight_color` field (canonical value in Brent Vellum fixture). Highlighted words render in gold, non-highlighted in white. No background rectangles or geometry needed.
- **D-10:** Body text uses **pre-tokenized inline formatting**. Body string is split around `bold_phrases` and `italic_phrases` matches. Canvas 2D `ctx.font` toggles weight/style per segment. Word-wrap reflow with per-word `measureText` is required anyway for the 650px-wide body zone, so tokenizer cost is incremental.
- **D-11:** Subtle drop shadow via `ctx.shadowBlur = 8` in `rgba(0,0,0,0.6)`. Invisible on dark gradient (left edge), adds legibility on mid-tone hero backgrounds where gradient thins. Two canvas state assignments per draw call.
- **D-12:** Text overflow handling: hard truncation with ellipsis at zone boundary as safety net. Primary prevention is upstream — add lightweight character-count check to Brief Validator (headline max ~60 chars, body max ~400 chars). Font-size reduction within ±4pt for body only as secondary fallback.

### Tweet Embed Card
- **D-13:** Realistic X/Twitter embed style — white rounded card (border-radius ~12px), card drop shadow. The satirical mechanism depends on verisimilitude: absurd characters presented with full corporate legitimacy.
- **D-14:** Card metadata: avatar circle (solid fill placeholder, no external fetch), blue verification badge SVG (satirical element — absurd characters get a blue check), handle in blue, body text, hashtags inline. Timestamp and like/retweet counts deferred — optional gilding for future iteration.
- **D-15:** Defensive null handling — even though `tweet_embed` is required per Phase 1 D-03, compositor gracefully omits the card on null/empty. ROADMAP success criterion 4 explicitly requires "omitted cleanly when absent."

### Claude's Discretion
- Sharp composite layer ordering and blend modes
- Canvas 2D rendering optimization (buffer reuse, state save/restore patterns)
- SVG icon drawing approach (inline Canvas path commands vs SVG file assets)
- Inter font weight selection beyond Regular/Bold (if Semi-Bold or Medium improves hierarchy)
- Exact emoji Unicode codepoints for each LinkedIn reaction type
- Tweet embed card internal padding, font sizes, and spacing
- Gradient mask implementation (Canvas gradient to buffer vs Sharp SVG overlay)
- File organization within `src/compositor/` (renderers/, assets/, fonts/, etc.)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Plans
- `Plans for Krea AI/PLAN_02_MODULES.md` — Module 4 (Compositor) spec: chrome template, text overlay renderer, tweet embed renderer, acceptance criteria
- `Plans for Krea AI/KNOWLEDGE_BASE.md` — Integration strategy, prompt-file contract, visual identity analysis

### Requirements & Constraints
- `.planning/REQUIREMENTS.md` — Phase 4 requirements: REQ-M-040 (chrome template), REQ-M-041 (text overlay with highlights), REQ-M-042 (tweet embed card), REQ-M-043 (1920x1080 PNG output), REQ-M-044 (stylistically similar, not pixel-perfect); cross-cutting: REQ-X-050 (fixed input dimensions), REQ-X-051 (compositor template is fixed asset), REQ-X-052 (byte-identical deterministic output), REQ-X-060 (3:2 horizontal), REQ-X-061-066 (visual identity spec), REQ-X-027 (compositor owns headline/body/tweet/engagement text rendering)
- `.planning/ROADMAP.md` §Phase 4 — Success criteria, plan breakdown (04-01, 04-02)

### Prior Phase Context
- `.planning/phases/01-mirror-post-scaffolding/01-CONTEXT.md` — Props normalized array (D-02), tweet_embed required (D-03)
- `.planning/phases/02-post-brief-generator/02-CONTEXT.md` — Structured Outputs (D-01), Pattern 5 prompt layout (D-05/D-06)
- `.planning/phases/03-visual-grammar-image-prompt-builder/03-CONTEXT.md` — Zone spec pixel coords (D-08/D-09), prop text in hero image (D-14), engagement tiers (D-10/D-11)

### Static Assets (already committed)
- `mirror-post/src/grammar/zone-spec.json` — Pixel-level zone coordinates for all compositor zones (top nav, profile bar, hero image, text overlay, tweet embed slot, engagement bar with sub-zones)
- `mirror-post/src/grammar/engagement.js` — Engagement metric generator (tier-bounded, seeded PRNG)
- `mirror-post/src/grammar/prop-taxonomy.js` — Prop taxonomy (10 types)
- `mirror-post/src/brief/schema.js` — Post Brief v1 schema with `highlight_color`, `highlight_words`, `bold_phrases`, `italic_phrases` fields

### Codebase Context
- `mirror-post/src/compositor/` — Directory exists with empty barrel `index.js`, empty `templates/` and `typography/` subdirectories (scaffolded in Phase 1)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `zone-spec.json` — Complete pixel-level layout for all compositor zones. Compositor loads this once and uses coordinates directly — no computation needed.
- `engagement.js` — Generates reaction counts, dominant reaction type, comment count by character tier. Compositor reads these values for the engagement bar.
- `schema.js` — Post Brief validator confirms all fields the compositor consumes: `post.headline`, `post.body`, `post.hashtags`, `tweet_embed`, `engagement`, `character.name`, `character.title`, `post.headline.highlight_words`, `post.headline.highlight_color`.
- `src/compositor/templates/` and `src/compositor/typography/` — Empty directories ready for chrome template data and font files.

### Established Patterns
- ESM modules with barrel exports (`index.js` in each directory)
- Static JSON/YAML assets loaded once (foundation-prompt.yaml, zone-spec.json precedents)
- Hand-written validators (no external schema library)
- Test harness at `test/harness.js` with fixture-based validation

### Integration Points
- `compositor/index.js` barrel will export `compositePost()` as the primary entry point
- Input contract: hero image path (PNG) + Post Brief JSON object
- Output: single 1920x1080 PNG buffer or file path
- Zone positions consumed from `grammar/zone-spec.json`
- Engagement values consumed from Post Brief's `engagement` object (already generated by `engagement.js` in Phase 3)
- New npm dependencies: `sharp`, `canvas` (node-canvas)

</code_context>

<specifics>
## Specific Ideas

- Verification badge on tweet embed characters is itself a satirical element — Brent Vellum with a blue check is funnier than Brent Vellum without one
- Inter font bundled in repo (not fetched at runtime) ensures deterministic rendering regardless of system font state
- The gradient overlay from zone-spec.json (82% black at left → 45% at 70% → transparent at right edge) provides sufficient contrast for white text with gold highlights; the subtle shadow is insurance for edge cases with high-luminance hero images
- Emoji reaction circles at 24-32px in a 40px zone are visually acceptable — the engagement bar is chrome, not the show (REQ-X-065)
- Brief Validator character-count gate (headline ~60, body ~400) catches overflow upstream before the compositor ever sees it

</specifics>

<deferred>
## Deferred Ideas

- Custom SVG reaction icons — emoji circles are MVP; upgrade to hand-drawn SVGs if visual review flags quality
- Tweet embed timestamp and like/retweet counts — optional gilding, add if render complexity budget allows
- AI-generated profile avatars — violates REQ-X-052 determinism; defer to post-M1 stretch
- Font-size reduction for headline overflow — headline size is fixed; only body gets ±4pt fallback
- Source Sans Pro as alternative font — revisit if Inter doesn't pass visual review for LinkedIn fidelity
- Dynamic zone computation — zones are fixed for v1 (REQ-X-051); revisit only if output dimensions change

</deferred>

---

*Phase: 04-compositor*
*Context gathered: 2026-04-14*
