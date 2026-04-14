# Phase 3: Visual Grammar + Image Prompt Builder - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Deterministically transform any valid Post Brief into a Flux-compatible image prompt and define the compositor zone spec. The image prompt builder reads the Post Brief's `image_seed` fields, loads the static foundation prompt, selects a scene template (or passes through freeform environment), appends character/props/modifier deltas, and returns a structured object ready for flux-krea consumption. The compositor zone spec defines pixel-level layout positions for LinkedIn chrome, text overlay zones, tweet embed slot, and engagement bar on a fixed 1920x1080 canvas. The engagement generator produces satirically calibrated metrics by character tier and post tone.

Zero LLM calls in this phase. Everything is deterministic data transformation.

</domain>

<decisions>
## Implementation Decisions

### Scene-to-Archetype Routing
- **D-01:** Template lookup + freeform pass-through. The prompt builder reads `image_seed.scene_template` from the Post Brief. If it matches one of the 4 known templates (office-executive, office-middle-mgmt, airport-hustle, call-center-floor), load the matching template file from `src/image/templates/`. If unrecognized or absent (freeform input), use `image_seed.environment` directly as the environment delta.
- **D-02:** Composition zone constraints are injected universally regardless of whether the scene came from a template or freeform pass-through. The `text_clear_zone` (left 35-40%) and tweet embed slot constraints apply to every prompt.
- **D-03:** No per-archetype scene override in library.json for v1. The LLM in Phase 2 already assigns `scene_template` in the Post Brief — Phase 3 trusts that assignment.

### Image Prompt Builder Output
- **D-04:** Builder returns a 4-key structured object: `{ positive_prompt, negative_prompt, parameters, composition_notes }`. This is a superset of the flux-krea `--prompt-file` contract (REQ-F-016). Callers extract what they need — flux-krea gets 3 keys (strip `composition_notes`), Artifact UI gets all 4.
- **D-05:** Concatenation order within `positive_prompt` is fixed and semantic:
  1. `foundation.reusable_master_prompt` (verbatim, always first — REQ-X-070)
  2. Character delta (appearance, expression, pose from Post Brief)
  3. Environment (scene template or freeform environment string)
  4. Props with surfaces + text for Flux to render (mug labels, whiteboard content)
  5. Composition directives (text_clear_zone left 35-40%, tweet zone, subject position)
  6. Fidelity modifiers (15-20 from ultra-fidelity library, always last)
- **D-06:** Negative prompt assembled from surface-text guards (`text, words, letters, signage, watermark` — preventing stray text outside of prop surfaces) plus negative safety modifiers from ultra-fidelity library. Flat comma-separated string.
- **D-07:** `parameters` includes width (1920), height (1080), steps, guidance_scale, and seed from Post Brief's `image_seed` fields. Matches the planned prompt-file JSON contract.

### Compositor Zone Spec
- **D-08:** Format is `zone-spec.json` — pixel coordinates for all zones on the 1920x1080 canvas. Static file, loaded once at compositor initialization. No semantic names or percentage-based values — pixel precision avoids any resolution/multiplication layer.
- **D-09:** Zones to define: top nav bar, profile bar (avatar + name + title), hero image area (full-bleed), left gradient text zone (~40% width for headline + body), tweet embed card slot (lower-right), engagement bar (bottom), reaction icons area, comment count area.

### Engagement Metric Generator
- **D-10:** Tier-bounded ranges with tone-weighted reaction distribution. Character tier (derived from archetype category) sets floor/ceiling for total reaction count. Post tone drives dominant reaction type distribution.
- **D-11:** Calibration bands:
  - C-suite / viral: 47-2,400 reactions, ~60-75% Insightful, 0.8-2% comment ratio
  - Middle-management: 85-340 reactions, flat distribution across types, 3-8% comment ratio
  - Hustle-culture / sales: 200-1,800 reactions, ~50% Celebrate, 1-3% comment ratio
  - Support / ops: 30-120 reactions, ~40% Like, 5-12% comment ratio
- **D-12:** LinkedIn reaction types: Like, Celebrate, Insightful, Love, Funny, Support. Funny is rare across all tiers — LinkedIn culture suppresses Funny on "serious" posts. The satire is funnier when characters don't know they're being laughed at.
- **D-13:** Controlled randomness with seed parameter. Default `null` = random in production, integer = deterministic in test harness. Enables both satirical freshness and snapshot testability.

### Prop Text Rendering Boundary
- **D-14:** Props with text (mug labels, whiteboard content, business card text, nameplate text) ARE described in the Flux image prompt for Flux to render in the hero image. The compositor does NOT overlay prop text — it only renders headline, body, tweet card, and engagement text as overlays. This refines REQ-X-011 and REQ-X-027: "surfaces" in those constraints refers to the compositor overlay layer, not to props within the hero image.

### Claude's Discretion
- Scene template file format and internal structure (lighting, camera, prop positions)
- Modifier selection algorithm (match by scene type, archetype domain, or weighted random from categories)
- Exact fidelity modifier count per scene (15-20 range, Claude picks the strategy)
- Negative prompt composition beyond the surface-text guards
- `composition_notes` content depth and structure
- Internal file organization within `src/image/` and `src/grammar/`

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Plans
- `Plans for Krea AI/PLAN_02_MODULES.md` — Module 2 (Visual Grammar) and Module 3 (Image Prompt Engine) specs
- `Plans for Krea AI/KNOWLEDGE_BASE.md` — Integration strategy, prompt-file JSON contract, performance targets

### Requirements & Constraints
- `.planning/REQUIREMENTS.md` — Phase 3 requirements: REQ-M-020 through REQ-M-023 (Visual Grammar), REQ-M-030 through REQ-M-035 (Image Prompt Builder); cross-cutting: REQ-X-010 (no MidJourney), REQ-X-011/X-027 (text rendering boundary — refined by D-14), REQ-X-060 through X-066 (visual identity), REQ-X-070 (foundation prompt static), REQ-X-071 (deterministic construction)
- `.planning/ROADMAP.md` section Phase 3 — Success criteria, plan breakdown (03-01, 03-02)

### Prior Phase Context
- `.planning/phases/01-mirror-post-scaffolding/01-CONTEXT.md` — Props normalized array (D-02), voice sliders encoding (D-05)
- `.planning/phases/02-post-brief-generator/02-CONTEXT.md` — Structured Outputs (D-01), Pattern 5 prompt layout (D-05/D-06), model choice claude-opus-4-6 (D-22)

### Static Assets (already committed)
- `mirror-post/src/config/foundation-prompt.yaml` — Foundation prompt v2.1 with `reusable_master_prompt` (prepended verbatim to every image prompt)
- `mirror-post/src/image/modifiers/ultra-fidelity.json` — Ultra-fidelity modifier library
- `mirror-post/src/image/modifiers/12k-modifiers.json` — 12K modifier library (9 categories)
- `mirror-post/src/brief/schema.js` — Post Brief v1 schema with `image_seed` field definitions

### Codebase Context
- `mirror-post/src/image/templates/` — Empty directory, to be populated with scene template files
- `mirror-post/src/image/index.js` — Empty barrel export, to be populated
- `mirror-post/src/grammar/index.js` — Empty barrel export, to be populated

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `foundation-prompt.yaml` — Already committed with v2.1 `reusable_master_prompt`. Builder loads this once and prepends verbatim.
- `ultra-fidelity.json` and `12k-modifiers.json` — Modifier libraries ready for selection algorithm.
- `schema.js` — Post Brief validator confirms `image_seed` structure including `scene_template`, `environment`, `lighting`, `camera_angle` fields.
- `src/persona/archetypes/library.json` — 17 archetypes in 5 categories. Category field drives engagement tier mapping.

### Established Patterns
- ESM modules with barrel exports (`index.js` in each directory)
- Hand-written validators (no external schema library)
- Static YAML assets loaded via `js-yaml` (foundation-prompt.yaml precedent)
- Test harness at `test/harness.js` with fixture-based validation

### Integration Points
- `image/index.js` barrel will export `buildImagePrompt()` as the primary entry point
- `grammar/index.js` barrel will export `generateEngagement()` and the zone spec
- Post Brief's `image_seed` object is the input contract — all fields defined in `schema.js`
- Output `{ positive_prompt, negative_prompt, parameters, composition_notes }` becomes input to flux-krea (Phase 6 integration) and Artifact UI (Phase 5)

</code_context>

<specifics>
## Specific Ideas

- Funny reactions suppressed across all tiers — the satire lands harder when characters don't realize they're being laughed at, which means engagement shows earnest Insightful/Celebrate, not Funny
- Support/ops tier added (30-120 reactions, ~40% Like, highest comment ratio at 5-12%) for call-center and support archetypes — small audience, high relative engagement
- Composition directives in the positive prompt explicitly reserve the left 35-40% as a text-clear zone for compositor overlays — this constraint is part of the Flux prompt, not just the zone spec
- Negative prompt includes surface-text guards to prevent Flux from generating stray text outside of prop surfaces (mug labels, whiteboards are intentional; random signage is not)

</specifics>

<deferred>
## Deferred Ideas

- Per-archetype scene overrides in library.json — not needed for v1 since Phase 2 LLM assigns scene_template. Can add explicit overrides later if LLM choices are inconsistent.
- Dynamic zone computation per-brief — zones are fixed for v1 (REQ-X-051). Revisit only if output dimensions change.
- Modifier weighting by archetype domain — Claude's discretion for v1. Could be tuned later based on visual quality review.

</deferred>

---

*Phase: 03-visual-grammar-image-prompt-builder*
*Context gathered: 2026-04-14*
