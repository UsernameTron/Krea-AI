# Phase 4: Compositor - Context

**Gathered:** 2026-04-14
**Revised:** 2026-04-15 (architecture reset) · **Finalized:** 2026-04-15 (all six blockers resolved inline)
**Status:** Ready for planning (plan 04.2)

**Supersedes:** `04-CONTEXT.md` (original, 2026-04-14) and `04-CONTEXT-RESET.md` (interim, 2026-04-15). Both preserved in git history; this file is the single authoritative context for Phase 04.2 planning.

---

<domain>
## Phase Boundary

Composite a **static per-variant LinkedIn chrome PNG** over a **Phase-5-generated scene PNG** (hero figure + environment), then overlay **headline + body text** into fixed per-variant slot coordinates. Output is a deterministic 1920×1080 PNG.

**Input contract:**
- Scene PNG path from `post_brief.image_output.scene_png` (Phase 5 output, 1920×1080, full-bleed diffusion).
- Post Brief v2 JSON (includes `image_seed.scene_template` enum `A|B|C|D` driving chrome + slot selection).
- Chrome PNG asset `mirror-post/src/compositor/chrome-assets/variant-{a,b,c,d}.png` (1920×1080 RGBA, scene zone transparent).
- Variant slot coordinates from `mirror-post/src/compositor/constants.js` (`VARIANT_SLOTS`).

**Output:** 1920×1080 PNG buffer (or file path) — byte-identical given identical inputs per REQ-X-052b.

**Out of scope:** Scene pixels (hero figure, environment, office props, prop text, mini post card, reaction row, comment group). All of those are baked into the Phase 5 scene render. Phase 4 does not render any chrome procedurally — chrome is a pre-authored transparent PNG asset, not a canvas draw.

</domain>

<decisions>
## Implementation Decisions

### D-NEW-01 (SECOND REVISION 2026-04-15 PM/Path 2) — Variant PNGs are pre-authored composite assets
Variant PNGs are pre-authored assets containing both scene and chrome, committed to `mirror-post/src/compositor/variant-assets/{a,b,c,d}/`. No per-post scene generation. Phase 5 per-post rendering eliminated. Phase 4 selects a sub-variant asset, overlays text into `VARIANT_SLOTS` coordinates, and exports. There is no runtime scene layer, no runtime chrome layer — the full background (scene + chrome) is a single committed PNG.

### D-NEW-01 (REVISED 2026-04-15 PM — SUPERSEDED by Path 2 above, retained for audit) — A+B compositing boundary: Phase 4 owns chrome PNG
Phase 5 generates **scene-only** PNG (hero figure + office environment; NO chrome, NO nav, NO mini post card, NO reaction row, NO comment group). Phase 4 composites a **static per-variant chrome PNG** (A/B/C/D, transparent where scene shows through) over the Phase 5 scene, then overlays text into `VARIANT_SLOTS` coordinates.
- **Revision history:** Original D-NEW-01 (2026-04-15 AM, in `04-CONTEXT-RESET.md`) said "scene generation belongs to Phase 5" including chrome. Afternoon session walked back chrome ownership from Phase 5 to Phase 4 as static PNG asset (FROZEN-01). Rationale: diffusion cannot bake LinkedIn chrome cleanly enough to not require iterative fixups; a one-time human-authored chrome asset per variant is deterministic, auditable, and decouples chrome from scene prompt churn.
- **Why (net):** Programmatic Cairo/node-canvas chrome produced fragile output (Twemoji COLR swap, BGRA channel mismatch, golden-PNG churn). Diffusion-baked chrome couples chrome fidelity to prompt tuning. Static transparent-PNG chrome is the minimum-surface, maximum-determinism option.
- **How to apply:** Any proposal to draw chrome procedurally, to regenerate chrome per-post from diffusion, or to parameterize chrome beyond the four variants is out of scope.

### D-NEW-02 — Compositor overlays text only (headline + body)
Phase 4 renders the satirical post's headline and body text into fixed per-variant slot coordinates on the composited (scene + chrome) base.
- **Why:** FLUX text fidelity is good for short labels but unreliable for multi-sentence satirical body copy with punctuation and brand names. Text overlay is the one job the compositor must own.
- **How to apply:** `render-text-overlay.js` and `buildStyledSegments` survive. All chrome rendering modules (`render-chrome.js`, `render-engagement-bar.js`, `render-tweet-card.js`, nav-icon module, Twemoji COLR pipeline, BGRA channel-swap patch, `NAV_LABELS`, `REACTION_EMOJI`) are deprecated — their scope is fully absorbed by the Phase 5 scene render + the chrome PNG asset.

### D-NEW-03 — VARIANT_SLOTS coordinates per variant, in code
Each variant (A/B/C/D) gets a fixed slot block for headline + body stored as a `VARIANT_SLOTS` constant in `mirror-post/src/compositor/constants.js`. Selection driven by `post_brief.image_seed.scene_template`.
- **Slot schema:** `{ headline: {x, y, width, height, fontSize, lineHeight, maxLines, fontWeight, color}, body: {x, y, width, height, fontSize, lineHeight, maxLines, fontWeight, color} }`.
- **Why:** Scene-aware CV positioning (option b) and flux-krea sidecar JSON (option c) were both rejected during the AM reset session — too unreliable, too coupled. Static constants match the fact that each variant has a fixed UI shell by template + chrome PNG design. Versioning slots alongside the chrome PNG makes them a coherent unit.
- **How to apply:** When a new variant is added, the contributor measures the slot on the rendered chrome PNG and commits the new entry. Changing coordinates invalidates golden tests.

### D-NEW-04 — REQ-X-052 split: REQ-X-052a (scene, Phase 5) + REQ-X-052b (overlay, Phase 4)
REQ-X-052 ("byte-identical PNG output") splits into two sub-contracts at the pipeline seam:
- **REQ-X-052a (scene determinism, Phase 5):** same flux-krea seed + same variant + same model weights → same scene pixels. Verified by Phase 5 golden test.
- **REQ-X-052b (overlay determinism, Phase 4):** same Post Brief v2 + same scene PNG + same chrome PNG + same `VARIANT_SLOTS` version → byte-identical composited output. Verified by Phase 4 pixel-compare golden test against a committed scene PNG fixture.
- **Why:** Diffusion and canvas pipelines have entirely different determinism guarantees; the end-to-end byte-identical contract was unenforceable. Splitting at the engine seam with a versioned fixture handoff makes each side testable.
- **How to apply:** Phase 4 golden test consumes `mirror-post/test/fixtures/scenes/variant-{a,b,c,d}.png` (at least variant-a required from Phase 5a before 04.2 can complete).

### D-NEW-05 — Archetype → variant static table in Phase 2; fallback B
Post Brief Generator (Phase 2) gains a static `archetype → variant` lookup. `image_seed.scene_template` becomes enum `"A" | "B" | "C" | "D"`. Unmapped archetypes fall back to **B (executive strategy office)** — the most generic professional baseline. Mapping table preserved in `04-CONTEXT-RESET.md` (not duplicated here to avoid drift); rationale per row is in that document's `Archetype → Variant Mapping` section.
- **Why:** B reads as a generic "LinkedIn post on a workday" without coding to a specific stereotype.
- **How to apply:** v2 schema validator enforces the enum; mapping table lives in a single source file in `mirror-post/src/brief/`.

### D-NEW-06 (RESOLVED 2026-04-15 AM — DECISION 2) — Post Brief schema v2 bump
Three sub-decisions confirmed as a bundle (advisor C recommendations accepted in full):
- **C1 — Optional nullable overrides in v2:** `image_seed.environment`, `image_seed.subject_pose`, `image_seed.mood`, `image_seed.color_temperature` survive as optional nullable fields in v2. Default source = variant template; `null` = use template default.
- **C2 — Placeholder substitution at Phase 5 render time:** `[Profile Name]`, `[Board Title]`, and all other placeholder tokens are substituted at Phase 5 render time (not Phase 2, not Phase 4). Phase 4 receives finished scene pixels.
- **C3 — `image_output.scene_png` field:** Post Brief v2 gains an `image_output` object with a `scene_png` string (path to Phase 5 output). Phase 4 reads this path directly; no cache lookup.
- **Migration:** 4 existing v1 fixtures migrated to v2; `schema_version` bumps `"1.0"` → `"2.0"`. Compatibility note added to CLAUDE.md.
- **Why:** Breaking change to the Post Brief contract honors the lessons.md rule about explicit version bumps for contract changes (same principle as the persona spec version gate).
- **How to apply:** v2 migration runs as a sub-plan **inside** Phase 04.2 execution (see D-NEW-14) — not as a separate Phase 02-revisit.

### D-NEW-07 (TIMING CLARIFIED — FROZEN-05) — Legacy compositor modules deleted from main
`render-chrome.js`, `render-engagement-bar.js`, `render-tweet-card.js`, the nav-icon module, the emoji pipeline, and the BGRA channel-swap patch are removed from `main`. They stay preserved on `archive/phase-4-programmatic-chrome` (mirror-post SHA `c25d88a`) for audit.
- **Timing:** Deletion happens **after Phase 04.2 plan is approved** — not before. Rationale: the plan references "what's being deleted" when scoping survivors; deleting pre-plan loses that reference point.
- **Why:** Honors the global "No Orphaned Code" rule. No dead code on main.
- **How to apply:** Deletion is a task inside the 04.2 plan, not a pre-plan cleanup.

### D-NEW-08 — `nav_easter_eggs` removal retroactively justified
Cross-phase removal of `nav_easter_eggs` from the Post Brief schema (logged in STATE during original Phase 04 work) is justified under the new architecture — the entire programmatic chrome path including nav rendering is gone. The field stays removed; no further action needed.

### D-NEW-09 (NEW — FROZEN-02) — Chrome PNG authoring contract
Four chrome PNG assets (one per variant) are the new static foundation for compositing.
- **Dimensions:** 1920×1080 RGBA (matches REQ-X-060 horizontal 16:9).
- **Transparency:** scene zone transparent (alpha = 0); chrome elements opaque. Masking via `rembg` or `sharp`-based background removal; operator QA before commit.
- **Authoring pipeline:** flux-krea generates 4 reference scenes (one per variant). Scene region masked out. Chrome frozen as transparent-layer PNG.
- **Repo location:** `mirror-post/src/compositor/chrome-assets/variant-{a,b,c,d}.png`. Version-controlled in mirror-post repo alongside `constants.js` (VARIANT_SLOTS).
- **Authoring sequence:** authoring is a **task inside the Phase 04.2 plan** — operator generates the 4 PNGs via flux-krea and hands them off during plan execution, between `/gsd:plan-phase 04.2` and `/gsd:execute-phase 04.2`.
- **Regeneration trigger:** only if a variant prompt changes substantively. Otherwise frozen.

### D-NEW-10 (NEW — DECISION 2 C2) — Placeholder substitution is Phase 5 responsibility
`[Profile Name]`, `[Board Title]`, and all other placeholder tokens in variant prompts are substituted at Phase 5 render time. Phase 4 receives finished scene pixels; Phase 4 never sees a placeholder token.

### D-NEW-11 (NEW — FROZEN-03) — ROADMAP renumbering
Phases renumber to reflect the architecture reset:
- **Phase 4: Compositor** (current, rescoped to chrome PNG overlay + text overlay).
- **Phase 5: Image Prompt Engine** (was "Phase 5: Artifact UI"). Split into 5a/5b per D-NEW-12.
- **Phase 6: Artifact UI** (shifted from Phase 5).
- **Phase 7: Integration** (shifted from Phase 6).
- **Parallel: flux-krea Optimization** unchanged.

### D-NEW-12 (NEW — FROZEN-04) — Phase 5 split (5a + 5b)
- **Phase 5a:** variant A end-to-end (prompt engine foundation + variant A scene generator + first scene PNG fixture committed to `mirror-post/test/fixtures/scenes/variant-a.png`).
- **Phase 5b:** variants B/C/D + remaining Image Prompt Engine scope.
- **Rationale:** one scene fixture unblocks Phase 04.2's golden test. Waiting for all four serializes unnecessarily.

### D-NEW-13 (NEW — FROZEN-05) — Deprecated module deletion timing
Modules listed in D-NEW-07 are deleted from `main` **after** Phase 04.2 plan approval, not before. Preserved on archive branch in the meantime. See D-NEW-07.

### D-NEW-15 (NEW 2026-04-15 Path 2) — Sub-variant pool
3-5 scenes per letter. Selection via `SHA-256(character.name + post.headline.text) mod pool_size`. Same brief always picks same sub-variant. `VARIANT_SLOTS` coords are per-letter, not per-sub-variant — operator generates sub-variants with seed-only variation to keep layout stable. Pool size per letter is stored alongside assets (e.g., directory listing of `variant-assets/{letter}/`).

### D-NEW-14 (NEW — FROZEN-06) — Phase 2 v2 migration lives inside Phase 04.2 execution
Post Brief v2 schema bump (D-NEW-06) runs as a **sub-plan inside Phase 04.2 execution**, not as a separate Phase 02-revisit.
- **Scope of sub-plan:** `mirror-post/src/brief/schema.js` v2 update, 4 fixture migrations (v1 → v2 with `image_output.scene_png` placeholder path), validator update to accept v2 + enforce enum, compatibility note in root `CLAUDE.md`.
- **Why:** the schema change is motivated by Phase 04.2 decisions (specifically D-NEW-06 sub-items C1/C2/C3), not by general Phase 2 evolution. Keeping the migration coupled to the phase that needs it reduces coordination overhead.

### Claude's Discretion
- Masking tool choice (`rembg` vs `sharp`-based approach) for chrome PNG authoring — operator QA is the gate.
- Exact `VARIANT_SLOTS` coordinates per variant — operator measures each chrome PNG before plan execution.
- File layout within `src/compositor/` beyond the two prescribed subpaths (`chrome-assets/` and `constants.js`).
- Composite implementation detail (Sharp `composite()` layer ordering, PNG encoder parameters).
- Text rendering internals (Canvas state save/restore patterns, gradient+shadow application).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project governance
- `.planning/PROJECT.md` — Project context, constraints, key decisions log
- `.planning/REQUIREMENTS.md` — 71 requirements including REQ-M-040..044 (compositor), REQ-X-052a/052b (determinism split), REQ-X-060 (16:9 dimensions), REQ-X-027 (text-rendering ownership — must be rewritten to match D-NEW-01/02)
- `.planning/ROADMAP.md` — Milestone 1 phase breakdown (post-renumbering)
- `.planning/STATE.md` — Current execution state and session continuity

### Prior phase context (Phase 4 audit trail)
- `.planning/phases/04-compositor/04-CONTEXT-RESET.md` — Interim reset context (2026-04-15 AM). Preserved for audit. Contains full archetype→variant mapping with per-row rationale (source of truth for D-NEW-05 table).
- `.planning/phases/04-compositor/04-CONTEXT.md` — THIS FILE (final, 2026-04-15 PM).
- `.planning/phases/04-compositor/04-DISCUSSION-LOG.md` — Two-session discuss-phase audit trail.

### Phase 5 source material (Image Prompt Engine templates)
- `.planning/phases/05-image-prompt-engine/templates/linkedin-template-variants-master-wrapper.md` — Cross-variant invariants (UI shell, camera, lighting, finish)
- `.planning/phases/05-image-prompt-engine/templates/linkedin-template-variant-a-startup-glass-desk.md` — Variant A prompt
- `.planning/phases/05-image-prompt-engine/templates/linkedin-template-variant-b-executive-strategy-office.md` — Variant B prompt (default fallback per D-NEW-05)
- `.planning/phases/05-image-prompt-engine/templates/linkedin-template-variant-c-collaborative-creative-office.md` — Variant C prompt
- `.planning/phases/05-image-prompt-engine/templates/linkedin-template-variant-d-sterile-analyst-war-room.md` — Variant D prompt

### mirror-post codebase
- `mirror-post/src/persona/spec/mirror_pete.v2.json` — Persona spec (REQ-X-001, **DO NOT MODIFY without version bump**)
- `mirror-post/src/brief/schema.js` — Post Brief schema (target of v2 bump per D-NEW-06 / D-NEW-14)
- `mirror-post/src/grammar/zone-spec.json` — Zone coordinates (most zones now superseded by VARIANT_SLOTS; retained for Phase 5 prompt consistency)
- `mirror-post/src/grammar/engagement.js` — Engagement metric generator (seeded; values baked into Phase 5 scene render, no longer read by compositor)
- `mirror-post/src/grammar/prop-taxonomy.js` — Prop taxonomy (feeds Phase 5 prompt construction)

### Prior CONTEXT documents (for continuity)
- `.planning/phases/01-mirror-post-scaffolding/01-CONTEXT.md` — Scaffolding decisions
- `.planning/phases/02-post-brief-generator/02-CONTEXT.md` — Post Brief Generator decisions
- `.planning/phases/03-visual-grammar-image-prompt-builder/03-CONTEXT.md` — Grammar decisions

### Cross-cutting
- `tasks/lessons.md` — Cross-session rules (architecture-review trigger, determinism boundary split — both born from this Phase 4 reset)

</canonical_refs>

<code_context>
## Existing Code Insights

### Survives (imported into 04.2 scope)
- `mirror-post/src/compositor/render-text-overlay.js` — Text overlay renderer. Becomes the entire compositor body.
- `mirror-post/src/compositor/compose.js` — Top-level composite. Shrinks dramatically; drops chrome/tweet/engagement composite layers.
- `buildStyledSegments` tokenizer (inline bold/italic for headline/body) — reused as-is.

### Dies (after 04.2 plan approval, per D-NEW-07 / D-NEW-13)
- `mirror-post/src/compositor/render-chrome.js`
- `mirror-post/src/compositor/render-engagement-bar.js`
- `mirror-post/src/compositor/render-tweet-card.js`
- Nav-icon module
- Emoji pipeline (Twemoji COLR `NotoColorEmoji.ttf`)
- BGRA channel-swap patch
- `NAV_LABELS` constant
- `REACTION_EMOJI` map
- Current golden PNG baseline (regenerates against Phase-5 scene fixture)

### New (created in 04.2 execution)
- `mirror-post/src/compositor/chrome-assets/variant-{a,b,c,d}.png` — 4 static chrome PNGs, 1920×1080 RGBA (authoring task inside plan, per D-NEW-09)
- `mirror-post/src/compositor/constants.js` — `VARIANT_SLOTS` map per D-NEW-03
- Scene fixture loader — reads `mirror-post/test/fixtures/scenes/variant-{a,b,c,d}.png`
- Updated `compositor/index.js` barrel — exports `compositePost(postBriefV2)` accepting Post Brief v2 with `image_output.scene_png` path; returns 1920×1080 PNG buffer

### Integration
- Primary entry: `compositor/index.js` → `compositePost(postBrief)`
- Dependencies retained: `sharp`. **Removed:** `canvas` (node-canvas) — only text rendering remains, and that uses Sharp's built-in text support OR a minimal Canvas surface; plan will pick the simpler path.

### Established patterns (preserved)
- ESM modules with barrel exports
- Static assets loaded once (chrome PNGs and `VARIANT_SLOTS` follow this precedent)
- Hand-written validators (no external schema library)
- Test harness at `test/harness.js` with fixture-based validation

</code_context>

<specifics>
## Specific Details

### Chrome PNG asset
- **Dimensions:** 1920×1080 RGBA
- **Transparency:** scene zone alpha = 0; chrome opaque
- **Authoring:** flux-krea generates reference → mask scene → freeze chrome (operator-driven, task inside plan)
- **Commit location:** `mirror-post/src/compositor/chrome-assets/variant-{a,b,c,d}.png`
- **Version control:** Git-tracked in mirror-post; regeneration only on substantive prompt change

### Composite layering order (bottom → top)
1. **Scene PNG** (from `post_brief.image_output.scene_png`) — 1920×1080, full-bleed diffusion from Phase 5
2. **Chrome PNG** (`variant-{a|b|c|d}.png` selected by `image_seed.scene_template`) — 1920×1080 RGBA overlay
3. **Text overlay layer** (headline + body) — drawn into `VARIANT_SLOTS[variant]` coordinates

### VARIANT_SLOTS coordinate format
```js
{
  headline: { x, y, width, height, fontSize, lineHeight, maxLines, fontWeight, color },
  body:     { x, y, width, height, fontSize, lineHeight, maxLines, fontWeight, color }
}
```
Four keyed entries: `VARIANT_SLOTS.A`, `VARIANT_SLOTS.B`, `VARIANT_SLOTS.C`, `VARIANT_SLOTS.D`.

### Scene PNG dimensions
- 1920×1080 (matches chrome). Diffusion owns all pixels except where chrome PNG is opaque.

### Determinism (REQ-X-052b)
Same Post Brief v2 + same scene PNG + same chrome PNG + same `VARIANT_SLOTS` version → **byte-identical composited output**. Golden test consumes a committed scene PNG fixture (variant-a required from Phase 5a as a blocker for 04.2 golden test; variants B/C/D fixtures land with Phase 5b).

### Phase 5a prerequisite
Phase 04.2 execution can start once Phase 5a ships `test/fixtures/scenes/variant-a.png`. The 04.2 plan can be written and approved before Phase 5a ships — only the golden-test task inside 04.2 depends on the fixture.

</specifics>

---

*Phase: 04-compositor*
*Plan target: 04.2*
*Context finalized: 2026-04-15*
