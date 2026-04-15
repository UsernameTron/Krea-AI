# Phase 04 — Architecture Reset Context

> **Supersedes** `04-CONTEXT.md` as of 2026-04-15. Old CONTEXT preserved for audit.
> **Trigger:** Phase 4 abandoned programmatic LinkedIn chrome rendering. Compositor now
> overlays text only onto AI-generated full-scene LinkedIn screenshots produced by Phase 5
> (Image Prompt Engine, `.planning/phases/05-image-prompt-engine/`).

## Status

- **Prior work:** preserved on `archive/phase-4-programmatic-chrome` (mirror-post repo) at
  SHA `c25d88a430f080c58854c16a248584c7ae245587`. No code reverted on `main`.
- **In-flight bug fix (BGRA channel-swap):** STOPPED. Will not merge.
- **Workspace template seeds:** committed at `45dfdfa` —
  `.planning/phases/05-image-prompt-engine/templates/` (4 variants + master wrapper).

## Architectural Pivot

Old contract (deprecated):
> Compositor renders LinkedIn chrome programmatically (top nav, profile bar, engagement
> bar, tweet card, nav icons, emoji) and overlays headline/body text on a flux-krea hero
> image of the character + scene.

New contract:
> Phase 5 produces a complete LinkedIn screenshot — chrome, hero, mini post card,
> reaction row, comment group, all baked into the diffusion output — for one of four
> scene variants. Phase 4 receives that screenshot and overlays only the satirical
> headline and body text into pre-defined slot coordinates for the variant used.

## Decisions

### D-NEW-01 — Scene generation belongs to Phase 5
The hero image *is* the LinkedIn screenshot. Phase 5 (Image Prompt Engine) is responsible
for emitting a flux-krea prompt that produces a full-scene render: LinkedIn chrome, top
nav, profile bar, mini post card, reaction row, comment group, and the in-scene office
environment. Phase 4 no longer renders any of that.
- **Why:** Programmatic chrome rendering produced fragile output — Cairo/FreeType emoji
  font drift, BGRA channel mismatches, golden-PNG churn on every nav-icon tweak. Diffusion
  bakes the chrome into pixels once, and the satirical look is more cohesive end-to-end.
- **How to apply:** Any new compositor work that proposes drawing chrome, icons, avatars,
  reaction emoji, or tweet-card geometry is out of scope for Phase 4. Push it to Phase 5
  prompt design or reject it.

### D-NEW-02 — Compositor (Phase 4) overlays text only
Phase 4's surviving responsibility is rendering the satirical post's headline and body
text into a pre-positioned slot on a Phase-5-generated screenshot. Nothing else.
- **Why:** Diffusion cannot reliably render the *exact* satirical copy from the Post
  Brief (FLUX text fidelity is good for short labels, not multi-sentence body copy with
  punctuation, em-dashes, and brand names). Text overlay is the one job the compositor
  must own.
- **How to apply:** `render-text-overlay.js` survives. `render-chrome.js`,
  `render-engagement-bar.js`, `render-tweet-card.js`, nav-icon rendering, the emoji
  pipeline, and the BGRA channel-swap fix are all deprecated. Move them to a `legacy/`
  folder on the reset branch or delete after the reset is approved.

### D-NEW-03 — Text slot coordinates per variant, defined in code
Each of the four variants (A/B/C/D) gets its own fixed text-slot coordinate block
(`{x, y, width, height, fontSize, lineHeight, maxLines}`) for headline and body, stored
as a `VARIANT_SLOTS` constant in the compositor. Selection driven by
`post_brief.image_seed.scene_template`.
- **Why:** Considered three options — (a) fixed per-variant constants, (b) scene-aware
  positioning via image analysis, (c) flux-krea sidecar JSON with slot coordinates.
  - (b) is too unreliable: detecting where the LinkedIn post-body region lives on a
    diffusion output requires an OCR/CV pipeline we don't want to build.
  - (c) couples Phase 5 too tightly to compositor internals and forces a sidecar contract
    that doesn't yet exist in flux-krea.
  - (a) is simple, deterministic, debuggable, and matches the fact that each variant has
    a *fixed* LinkedIn UI shell by template design (the master wrapper enforces "same UI
    shell across all variants"). The text slot is pinned by the prompt, not floating.
- **How to apply:** When a new variant is added, the contributor must measure the slot
  in the rendered template and add an entry to `VARIANT_SLOTS`. Treat slot coordinates
  as a versioned asset (touching them invalidates golden tests).

### D-NEW-04 — Determinism contract: seed-deterministic scene + byte-identical overlay
REQ-X-052 ("byte-identical PNG output") split into two sub-contracts:
- **REQ-X-052a (scene determinism):** Same flux-krea seed + same variant + same model
  weights → same scene pixels. Owned by Phase 5; verified by Phase 5 golden test.
- **REQ-X-052b (overlay determinism):** Same Post Brief + same scene PNG + same
  `VARIANT_SLOTS` version → byte-identical composited PNG. Owned by Phase 4; verified by
  the existing pixel-compare golden test, but the golden baseline is regenerated against
  a Phase-5-produced scene rather than a programmatic chrome render.
- **Why:** Asking flux-krea output to be byte-identical against a chrome-on-canvas render
  was the wrong contract — they're produced by entirely different pipelines. Splitting
  the determinism boundary at the Phase 4 / Phase 5 seam isolates the testable surface
  on each side.
- **How to apply:** Phase 4 golden test now consumes a checked-in scene PNG fixture from
  `mirror-post/test/fixtures/scenes/variant-{a,b,c,d}.png`. The test asserts the overlay
  output. Phase 5 owns regenerating those fixtures when the scene prompts change.

### D-NEW-05 — Archetype → variant mapping with documented fallback
The Post Brief Generator (Phase 2) gains a static `archetype → variant` table. The
`image_seed.scene_template` field becomes an enum: `"A" | "B" | "C" | "D"`.
Fallback for unmapped archetypes: **`B` (executive strategy office)** — the most generic
professional baseline that reads as "LinkedIn post on a workday" without coding to a
specific stereotype.

Proposed mapping (rationale per row in the *Archetype → Variant Mapping* section below):

| # | Archetype                                | Variant |
|---|------------------------------------------|---------|
| 1 | The Resume Padder                        | B       |
| 2 | The Governance Vampire                   | D       |
| 3 | The Metrics Illusionist                  | D       |
| 4 | The Strategy Deck Artist                 | B       |
| 5 | The Demo Magician                        | A       |
| 6 | The Vaporware Evangelist                 | A       |
| 7 | The Implementation Partner Grifter       | B       |
| 8 | The AI Hype Surfer                       | A       |
| 9 | The Security Theater Director            | D       |
|10 | The Center of Excellence Bureaucrat      | D       |
|11 | The Steering Committee Seat Warmer       | B       |
|12 | The Passive-Aggressive Stakeholder       | C       |
|13 | The WFM Wizard (Self-Proclaimed)         | A       |
|14 | The Quality Assurance Zealot             | D       |
|15 | The Digital Transformation Cheerleader   | C       |
|16 | The Denial Management Denier             | D       |
|17 | The EHR Apologist                        | C       |

Distribution: A=4, B=4, C=3, D=6. D is heavy because metrics / process / control
archetypes dominate the persona library — acceptable but worth a re-read at UAT.

## Decisions Surfaced During the Reset Discussion

### D-NEW-06 — Post Brief schema migration
`post_brief.image_seed.scene_template` changes from free-form string to enum
`"A" | "B" | "C" | "D"`. This is a breaking change to the Post Brief v1 schema. Two
options:
- **Bump to v2** (preferred) — explicit version bump, migration script for the 4 fixture
  briefs, schema validator updated, lessons.md rule honored ("never modify the persona
  spec without version bump" — same principle applies to the Post Brief contract).
- **Patch v1 in place** — faster, but breaks the contract the rest of the pipeline
  depends on without a version marker.

Defer the choice to Phase 04.2 planning, but flag it now so the planner sees it.

### D-NEW-07 — Legacy code disposition
`render-chrome.js`, `render-engagement-bar.js`, `render-tweet-card.js`, the nav-icon
module, the emoji pipeline, and any tests bound to them must be removed (not
silently retained as dead code — see global rule "No Orphaned Code"). They are preserved
on `archive/phase-4-programmatic-chrome` for audit; `main` should not carry deprecated
modules through Phase 04.2 execution.

### D-NEW-08 — `nav_easter_eggs` already removed cross-phase
Cross-phase removal of `nav_easter_eggs` from the Post Brief schema (logged in STATE
under Phase 04 decisions) is now retroactively justified — the field was removed because
chrome-baked nav labels were misspelling, but the entire programmatic chrome path is now
gone. The removal stands; no further action needed.

## Archetype → Variant Mapping (rationale)

| Archetype                              | Variant | Reasoning |
|----------------------------------------|---------|-----------|
| Resume Padder                          | B       | Executive optics — corner-office vibe matches the "AI Transformation Leader" LinkedIn theater |
| Governance Vampire                     | D       | Sterile war room mirrors process-gatekeeping cold control |
| Metrics Illusionist                    | D       | Dashboards and dual monitors literally match the "redefined metrics" gag |
| Strategy Deck Artist                   | B       | Executive office — strategy boards, framed art, "let me share my vision" |
| Demo Magician                          | A       | Bright startup desk — vendor pitch energy, glass-and-laptop aesthetic |
| Vaporware Evangelist                   | A       | Startup hype, "we're shipping next quarter" optimism |
| Implementation Partner Grifter         | B       | Consultant gravitas — exec office signals "trust me, I've done this before" |
| AI Hype Surfer                         | A       | Startup buzz, bright collaborative startup energy |
| Security Theater Director              | D       | Sterile, controlled, frosted glass — mirrors compliance-as-performance |
| Center of Excellence Bureaucrat        | D       | Analyst war room — process and dashboards over delivery |
| Steering Committee Seat Warmer         | B       | Executive optics — committee theater happens in corner offices |
| Passive-Aggressive Stakeholder         | C       | Collaborative space — passive aggression lives in shared rooms |
| WFM Wizard (Self-Proclaimed)           | A       | Startup pitch energy — workforce-mgmt vendor demo aesthetic |
| Quality Assurance Zealot               | D       | Sterile scoring environment — QA scorecards on dual monitors |
| Digital Transformation Cheerleader     | C       | Collaborative ra-ra room — sticky notes, "alignment workshops" |
| Denial Management Denier               | D       | Analyst dashboards — denial rates on dashboard screens |
| EHR Apologist                          | C       | Collaborative ops grind — shared table, marker board, frustrated team |

**Fallback:** `B` — picked because the executive strategy office reads as the most
generic "professional LinkedIn post" baseline that does not code to a specific
psychological archetype.

## Risks Surfaced (warrant ROADMAP / dependency updates)

1. **Phase 5 is now blocking for Phase 4** — previously sequential but loosely coupled,
   now strictly blocking. Phase 5 must produce at least one variant scene PNG before
   Phase 04.2 can build the overlay golden test. Suggest splitting Phase 5 into Phase 5a
   (variant A end-to-end) and Phase 5b (B/C/D + the rest of Phase 5's prompt-engine
   work) so Phase 04.2 can start as soon as 5a lands.
2. **Phase 5 naming collision in the ROADMAP** — current `ROADMAP.md` still lists
   "Phase 5: Artifact UI" while the new directory is `05-image-prompt-engine/`. The
   roadmap needs renumbering (Image Prompt Engine becomes 5, Artifact UI shifts to 6,
   Integration to 7) or a clean rename. Flagged for the operator — not auto-fixed in
   this session per the "do not progress without approval" rule.
3. **Post Brief schema bump (D-NEW-06)** — touches Phase 2 code. Phase 2 is "complete"
   in STATE.md but a v2 schema migration is a non-trivial change and should be tracked
   as either Phase 02-revisit or a sub-plan inside Phase 04.2.
4. **Variant distribution skew (D-NEW-05)** — six archetypes mapped to D. If UAT finds
   the war-room aesthetic boring across that many posts, several archetypes can re-route
   to B. Worth a dedicated UAT pass once at least one D scene exists.
5. **Golden-test fixture lifecycle** — the new compositor golden test depends on a
   Phase-5-produced scene PNG fixture. When Phase 5 regenerates scenes (prompt tweak,
   model bump), Phase 4 goldens invalidate. Need an explicit fixture-version contract
   between the two phases.
6. **REQ-X-027 reread** — current REQ-X-027 says "compositor renders headline, body,
   tweet card, engagement text — props text remains IN the hero image." Under D-NEW-01
   this becomes "compositor renders headline and body only; everything else lives in
   the Phase 5 scene render." REQ-X-027 needs a rewrite or sunset.

## What Survives, What Dies

| Module                          | Disposition | Notes |
|---------------------------------|-------------|-------|
| `render-text-overlay.js`        | SURVIVES    | Becomes the entire compositor body |
| `compose.js` (top-level)        | SURVIVES (shrinks) | Drops chrome/tweet/engagement composite layers |
| `buildStyledSegments` tokenizer | SURVIVES    | Bold/italic styling for headline/body still needed |
| `render-chrome.js`              | DIES        | Diffusion bakes chrome |
| `render-engagement-bar.js`      | DIES        | Diffusion bakes the reaction row |
| `render-tweet-card.js`          | DIES        | Diffusion bakes the floating mini post card |
| Nav icon module                 | DIES        | Diffusion bakes nav |
| Emoji pipeline (Twemoji COLR)   | DIES        | Reaction emoji are in the scene render |
| BGRA channel-swap bug fix       | DIES        | Module it patched is deprecated |
| `NAV_LABELS` constant           | DIES        | Nav text is inside the scene |
| `REACTION_EMOJI` map            | DIES        | Emoji are in the scene |
| Golden PNG (current baseline)   | REGENERATES | Against Phase-5 scene fixture |
| Variant-slot constants          | NEW         | Add `VARIANT_SLOTS` per D-NEW-03 |
| Scene fixture loader            | NEW         | Read variant scene PNG from `test/fixtures/scenes/` |

## Next Step

**STOP.** Awaiting operator approval of this CONTEXT before any planning or execution
of Phase 04.2. Do not auto-invoke `/gsd:plan-phase 04.2`.
