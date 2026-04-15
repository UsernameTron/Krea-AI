# Product Reset 2 — Scope Collapse to Prompt Artifact

**Date:** 2026-04-15 evening
**Authority:** Operator decision, atomic commit
**Supersedes:** Phase 4 (Compositor), Phase 5 (Image/Scene Asset Library), Phase 6 (Artifact UI), Phase 7 (Integration) — as previously scoped.

## Reason

The compositor architecture spiraled through Path 1 → Path 2 → Path 2b → Path C over a single 24-hour window without producing a shippable artifact:

- **Path 1 (programmatic chrome):** Cairo/FreeType CBDT emoji broke; Twemoji COLR swap; nav_easter_eggs misspellings; BGRA channel-order bug. Abandoned 2026-04-15 AM.
- **Path 2 (full variant PNGs):** Operator authors 12–20 per-variant PNGs via FLUX; compositor overlays text into measured VARIANT_SLOTS. Committed, then amended (Path 2b = Path 2 with sub-variant pools).
- **Path C (HTML template + FLUX scene-only):** Shared LinkedIn HTML template rendered via Puppeteer; FLUX generates scene content only into HERO_ZONE; sharp composites. Committed today.

None of these paths produced a working artifact. The operator pulled scope back to the core product value: **a user wants filled-in image prompts, not a compositor.** The time spent arguing pixel layering could have built the actual product three times over.

## New Scope

A **single Claude Desktop React artifact** that:

1. Accepts an idea (textarea), four voice sliders (sarcasm, cynicism, warmth, satirical_intensity), and an archetype selection (17 archetypes + auto-assign).
2. Calls `window.claude.complete` with persona distillate + archetype + slider values + idea → returns a Post Brief JSON.
3. Maps archetype to variant (A/B/C/D) via the D-NEW-05 mapping table.
4. Loads the corresponding variant template from `.planning/phases/05-image-prompt-engine/templates/` and substitutes `{{placeholder}}` tokens with Post Brief fields.
5. Displays: filled prompt text (with a Copy button), a Post Brief JSON preview, and archetype match confidence.

**Success:** User enters an idea, adjusts sliders, and gets a ready-to-paste image prompt in under 5 seconds. The app ends at prompt text. The user pastes the prompt into whatever image tool they choose (Midjourney, Flux, DALL·E, Krea, etc.).

**Out of scope (forever, not deferred):**
- Compositor, chrome rendering, text overlay, byte-identical output
- Scene/variant asset library
- FLUX / flux-krea integration for Mirror Post (flux-krea remains its own general-purpose project)
- HTML templates, Puppeteer, sharp, node-canvas
- Deployed web UI (optional much later; the artifact IS the product for now)

## What Survives

| Asset | Location | Status |
|---|---|---|
| Phase 1 scaffolding | `mirror-post/` repo structure | Kept |
| Phase 2 Post Brief Generator | `mirror-post/src/brief/`, `mirror-post/src/persona/` | Shipped — logic ports into artifact or is called externally |
| Persona spec | `mirror-post/src/persona/spec/mirror_pete.v2.json` | Canonical, immutable without version bump |
| Archetype library (17) | `mirror-post/src/brief/archetypes/` | Kept |
| Comedy structures | `mirror-post/src/brief/comedy/` | Kept |
| Variant template files (4) | `.planning/phases/05-image-prompt-engine/templates/` | **Kept — directly consumed by artifact** |

## What Dies

| Artifact | Final Resting Place |
|---|---|
| Phase 4 programmatic chrome (sharp + node-canvas + Twemoji + BGRA fix) | `archive/phase-4-programmatic-chrome` branch in `mirror-post` (SHA c25d88a) |
| Phase 4.2 Path 2 variant assets + Path C HTML template + FLUX scene-only | `archive/phase-4.2-path-c-abandoned` branch in `mirror-post` |
| Phase 4.2 plans 04.2-01..07 | Historical record in `.planning/phases/04-compositor/` — not executed |
| Phase 5 Variant Asset Library / Scene Asset Library | Collapsed; operator asset authoring task retired |
| Phase 6 Artifact UI (as scoped) | Collapsed into single Claude Desktop artifact |
| Phase 7 Integration | Absorbed — the artifact IS the integration |
| 04-CONTEXT.md decisions D-NEW-01..17 | Historical record; not load-bearing for new scope |

**Nothing is deleted.** Branches exist for audit. Documentation exists for history.

## Path Forward

1. **This session (Claude Code):** Reset the planning docs (this file, ROADMAP, STATE, todo, lessons), archive the mirror-post Path C branch, atomic commit. **Stop.**
2. **Next session (Claude Desktop):** Build the React artifact directly in a Claude Desktop chat with artifact support. Iterate in-chat until satisfaction.
3. **Finishing (Claude Code or Desktop):** Commit the finished artifact to `mirror-post/artifact/MirrorPoster.jsx`.
4. **Optional later:** Promote to a deployed web app (Vercel) only if the artifact proves its value first.

## Hard Rules

- Do NOT delete any prior work. Branches and docs are permanent.
- Do NOT run `/gsd:execute-phase 04.2` or any successor.
- Do NOT attempt to build the React artifact in Claude Code — that work happens in Claude Desktop with direct artifact support.
- Do NOT touch `mirror_pete.v2.json` (still canonical, still immutable without version bump).

## Signal That Triggered The Reset

From the operator: *"I could have built this app 3 times over in the time we've spent arguing about pixel layering."*

That sentence is the lesson. The architecture was not the product. The product was always the prompt generator.
