# Phase 4: Compositor - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

This log spans two phases of work:
- **Section A (2026-04-14):** Original Phase 4 discuss-phase. Programmatic chrome architecture. Superseded by the 2026-04-15 reset but preserved as audit trail.
- **Section B (2026-04-15 AM):** Phase 04.2 discuss-phase — three advisor-researcher subagents, two operator decisions (DECISION 1, DECISION 2).
- **Section C (2026-04-15 PM):** Phase 04.2 resume session — six carry-over blockers (FROZEN-01..06) resolved inline, birthing D-NEW-09 through D-NEW-14.

---

# SECTION A — Original Phase 4 Discussion (2026-04-14, SUPERSEDED)

**Date:** 2026-04-14
**Phase:** 04-compositor (original architecture, abandoned 2026-04-15)
**Areas discussed:** Rendering engine, Chrome visual style, Text overlay styling, Tweet embed card design
**Status:** All decisions below were VALID at the time and drove 04-01..04-03 plan execution. Abandoned when Phase 4 was reset — chrome rendering moves out of compositor entirely. Retained here as audit trail.

## Rendering Engine

| Option | Description | Selected |
|--------|-------------|----------|
| Sharp + node-canvas hybrid | Canvas 2D for text/gradient, Sharp for image compositing. 2 npm deps, requires Homebrew Cairo/Pango. | ✓ |
| Sharp + SVG only | Single dep, zero system installs. Text as SVG strings. More verbose but deterministic. | |
| node-canvas only | Full Canvas 2D for everything. Requires Homebrew system deps. | |
| Puppeteer | Full CSS rendering. Eliminated — subpixel variation violates REQ-X-052. | |

**User's choice:** Sharp + node-canvas hybrid (Recommended)
**Notes:** Puppeteer eliminated pre-presentation due to determinism requirement. Hybrid gives best text API (Canvas 2D) with best image compositing (Sharp).
**Phase 04.2 status:** Sharp + node-canvas hybrid SURVIVES for text-overlay-only scope. Chrome rendering no longer in compositor.

## Chrome Visual Style

| Option | Description | Selected |
|--------|-------------|----------|
| Accept all 4 recommendations | Simplified SVG nav icons, custom blue #0b5ca8, initials avatar, emoji reaction circles | ✓ |
| Custom SVG reaction icons instead | Same but 6 hand-drawn SVG reaction icons instead of emoji | |
| Silhouette avatar instead | Generic gray silhouette placeholder instead of initials | |

**User's choice:** Accept all 4 recommendations
**Notes:** Package deal: LinkedIn-inspired simplified SVG icons, custom corporate blue #0b5ca8, circle-with-initials avatar from character.name, emoji Unicode reaction circles at 24-32px.
**Phase 04.2 status:** ENTIRE SECTION OBSOLETE. Chrome becomes a static PNG per variant (D-NEW-09). No programmatic icons, no avatar, no emoji pipeline.

## Text Overlay Styling

| Option | Description | Selected |
|--------|-------------|----------|
| Accept all 5 recommendations | Inter font, gold #FFD700 highlights, pre-tokenized bold/italic, subtle shadow, truncation + validator | ✓ |
| Source Sans Pro instead of Inter | Closer to LinkedIn's actual typeface. Same approach otherwise. | |
| Skip body bold/italic for MVP | Single-style body, defer formatting. Simpler but loses comedic timing. | |

**User's choice:** Accept all 5 recommendations
**Notes:** Inter bundled as TTF, gold highlight from schema's highlight_color field, inline bold/italic with pre-tokenized segments, ctx.shadowBlur=8 for readability, hard truncation + upstream Brief Validator char limits.
**Phase 04.2 status:** ALL DECISIONS SURVIVE. Text overlay is the surviving scope. render-text-overlay.js and buildStyledSegments carry forward unchanged.

## Tweet Embed Card Design

| Option | Description | Selected |
|--------|-------------|----------|
| Accept realistic embed | White rounded card, avatar circle, blue verification badge, handle, text, hashtags. Defensive null handling. | ✓ |
| Include timestamp + like/RT counts | Same plus fake timestamp and engagement numbers. More render work. | |
| Skip verification badge | Same but no blue checkmark. Reduces satirical punch. | |

**User's choice:** Accept realistic embed (Recommended)
**Notes:** Verisimilitude drives the satirical mechanism. Verified badge on absurd characters is itself a punchline. Timestamp and like/RT counts deferred as optional gilding.
**Phase 04.2 status:** OBSOLETE. Tweet embed card becomes part of the chrome PNG if it appears at all. render-tweet-card.js deleted per D-NEW-07.

## Claude's Discretion (original)

- Sharp composite layer ordering and blend modes
- Canvas 2D rendering optimization (buffer reuse, state save/restore)
- SVG icon drawing approach (inline Canvas paths vs SVG file assets)
- Inter font weight selection beyond Regular/Bold
- Emoji Unicode codepoints for each reaction type
- Tweet embed card internal layout (padding, font sizes, spacing)
- Gradient mask implementation approach
- File organization within src/compositor/

## Deferred Ideas (original)

- Custom SVG reaction icons (upgrade from emoji if visual review flags quality)
- Tweet embed timestamp and like/retweet counts
- AI-generated profile avatars (violates REQ-X-052)
- Source Sans Pro as alternative font
- Dynamic zone computation (zones fixed for v1)

---

# SECTION B — Phase 04.2 Discuss-Phase (2026-04-15 AM)

**Date:** 2026-04-15 (morning)
**Phase:** 04.2 (architecture reset follow-through)
**Format:** Three gsd-advisor-researcher subagents spawned in parallel, each assigned one gray-area decision from `04-CONTEXT-RESET.md`.
**Outcome:** Two operator decisions captured — DECISION 1 (A+B compositing boundary) and DECISION 2 (Schema v2 bump bundle).

## Advisors Spawned

### Advisor A — "Compositor owns nothing but text"

**Recommendation:** Phase 5 generates the complete LinkedIn screenshot including chrome. Phase 4 compositor only overlays text.
**Pros cited:**
- Single source of truth for pixel layout (diffusion output)
- Compositor becomes trivial — draw text into fixed rectangles
- No mismatch risk between chrome PNG and scene PNG
**Cons cited:**
- Chrome consistency across the 4 variants depends on diffusion prompt fidelity
- Any chrome detail change requires re-prompting all 4 scenes
- LinkedIn's UI has crisp typographic edges — diffusion introduces softness/artifacts on small text
- No way to version-control the chrome separately from the scene

### Advisor B — "Compositor owns the chrome, scene owns the scene"

**Recommendation:** Phase 5 generates scene-only (hero figure + environment, no chrome pixels). Phase 4 composites a static chrome PNG + text overlay onto that scene.
**Pros cited:**
- Chrome pixels are pin-sharp (rasterized from a known-good source, no diffusion jitter)
- Chrome versioned separately — swap one of the four PNGs without regenerating scenes
- Text slots (VARIANT_SLOTS) bind to the chrome PNG, not the diffusion output, so determinism is firm
- Scene diffusion is simpler (no "render a believable LinkedIn nav bar" pressure)
**Cons cited:**
- Phase 4 picks up a new asset authoring step (build the 4 chrome PNGs)
- Extra composite layer adds one more source of bugs at the mask boundary
- Requires an authoring pipeline for the chrome PNGs (flux-krea + masking tool)

### Advisor C — "Schema v2 bump with optional override fields"

**Recommendation:** Bump Post Brief schema to v2. Add optional nullable override fields for diffusion parameters (environment, subject_pose, mood, color_temperature). Move placeholder substitution to Phase 5 render time. Add `image_output.scene_png` field carrying the path to Phase 5's output.
**Pros cited:**
- Versioning is explicit — consumers can switch cleanly
- Overrides unblock per-post creative deviation without forcing a template edit
- Placeholder substitution at Phase 5 keeps Phase 2 free of Phase-5-specific concerns
- `image_output.scene_png` gives Phase 4 a direct file path, no cache lookup
**Cons cited:**
- Four existing v1 fixtures must be migrated
- Validator must learn v2 schema
- CLAUDE.md rules (persona spec frozen) mean v1 validator must live alongside v2 for the migration window

## Collisions and Tensions

- **A vs B is the core collision.** Both are internally coherent; the disagreement is who owns the chrome pixels.
- **C is independent but coupled to whichever A/B outcome is chosen.** If A wins, `scene_png` is the final screenshot and `image_output` can be renamed. If B wins, `scene_png` is the scene-only intermediate.

## Operator Decisions

### DECISION 1 — A+B boundary (Advisor B wins outright)

**Operator call:** "B. Chrome stays crisp. I want the four chrome PNGs versioned separately. Regenerating scenes when I change one icon on the nav bar is a bad trade."
**Consequences captured:**
- REVISES D-NEW-01 from 04-CONTEXT-RESET.md. D-NEW-01 had chrome ownership walking Phase 5→Phase 4 as "full scene with chrome." Now chrome ownership is explicitly Phase 4, scene is explicitly Phase 5.
- Adds new responsibility to Phase 4: author 4 chrome PNGs.
- REQ-X-052b (overlay determinism) now guards the chrome+text composite specifically. REQ-X-052a stays pinned to Phase 5 scene generation.

### DECISION 2 — Schema v2 bump (Advisor C accepted as bundle)

**Operator call:** "Take all three of C's recommendations. v2 bump, override fields optional nullable, placeholder substitution at Phase 5, `image_output.scene_png` on the brief. Migrate the four v1 fixtures as part of this."
**Consequences captured:**
- RESOLVES D-NEW-06 (schema bump question from 04-CONTEXT-RESET.md). v2 bundle confirmed — not an in-place v1 patch.
- Migration becomes a sub-plan inside Phase 04.2 execution (not a separate Phase 02-revisit). This coupling is deliberate: schema change is motivated by Phase 04.2 decisions, keep them coupled.

## Left Open After AM Session

Six questions remained because the operator paused to context-check before ROADMAP and code edits. These became the FROZEN-01..06 carry-over blockers resolved in Section C.

---

# SECTION C — Phase 04.2 Resume Session (2026-04-15 PM)

**Date:** 2026-04-15 (afternoon)
**Context:** Operator ran `/gsd:resume-work`, read the checkpoint, chose Option 1 (resume + write), and resolved all six carry-over blockers inline in one message before any artifacts were touched.
**Outcome:** Six frozen operator decisions (FROZEN-01..06) giving rise to six new Phase 04.2 decisions (D-NEW-09..14).

## FROZEN-01 — D-NEW-01 reversal confirmed

**Question:** Re-confirm that chrome ownership really does move Phase 5 → Phase 4 (vs. hybrid or reversion to Phase 5-owns-all).
**Operator rationale:** "Chrome stays Phase 4. Scene stays Phase 5. The A+B boundary from this morning is the right boundary. Anything else is re-litigating a closed decision."
**Birthed:** No new decision — this is a reaffirmation of DECISION 1. It is surfaced as a first-class frozen item because the ripple effects (chrome authoring task, REQ-X-052 split, deprecated module disposition) all depend on this being definitive rather than tentative.

## FROZEN-02 — Chrome PNG authoring contract

**Question:** How are the 4 chrome PNGs actually produced, versioned, and regenerated?
**Operator rationale:** "Author with flux-krea using the same variant prompts the scene pipeline will use. Mask out the scene zone. Keep the chrome opaque. 1920×1080 RGBA, committed to mirror-post at `src/compositor/chrome-assets/variant-{a,b,c,d}.png`. VARIANT_SLOTS versioned alongside as `constants.js`. Chrome authoring is a task inside the Phase 04.2 plan — I generate, plan execution consumes."
**Birthed D-NEW-09.**

## FROZEN-03 — ROADMAP renumbering

**Question:** Resolve the Phase 5 naming collision (current ROADMAP lists Phase 5 = Artifact UI, but new directory is `05-image-prompt-engine/`).
**Operator rationale:** "Renumber. Image Prompt Engine becomes Phase 5 because it's now strictly blocking Phase 4. Artifact UI shifts to Phase 6. Integration shifts to Phase 7. Apply to ROADMAP.md and STATE.md atomically with the CONTEXT writeup."
**Birthed D-NEW-11.**

## FROZEN-04 — Phase 5 split

**Question:** Does Phase 5 run as a single phase covering all four variants + prompt engine, or split into 5a (variant A end-to-end) and 5b (remaining variants + rest of engine)?
**Operator rationale:** "Split it. Phase 5a delivers variant A end-to-end — one scene PNG fixture is enough to build the Phase 04.2 golden test. Waiting on all four serializes for no reason. 5b finishes B/C/D and any engine work 5a didn't need."
**Birthed D-NEW-12.**

## FROZEN-05 — Deprecated module deletion timing

**Question:** Delete the programmatic chrome modules from mirror-post main now, or after Phase 04.2 plan approval?
**Operator rationale:** "After plan approval. The plan-phase language will reference 'what's being deleted' when scoping survivors; easier to write that while the modules are still readable on main. Archive branch already holds them safely."
**Birthed D-NEW-13.**

## FROZEN-06 — Phase 2 v2 migration scoping

**Question:** Is the Post Brief v1→v2 migration its own Phase 02-revisit, or a sub-plan inside Phase 04.2?
**Operator rationale:** "Sub-plan inside 04.2. The schema change is motivated by 04.2 decisions specifically, not a generic Phase 2 evolution. Keep them coupled or I'll have two plans drifting apart."
**Birthed D-NEW-14.**

## New Decisions Born in Section C

- **D-NEW-09** — Chrome PNG authoring contract (from FROZEN-02).
- **D-NEW-10** — Placeholder substitution at Phase 5 render time (from DECISION 2 C2, formally surfaced as its own decision here because Phase 4's contract depends on receiving fully-substituted scene pixels).
- **D-NEW-11** — ROADMAP renumbering (from FROZEN-03).
- **D-NEW-12** — Phase 5 split 5a/5b (from FROZEN-04).
- **D-NEW-13** — Deprecated module deletion after 04.2 plan approval (from FROZEN-05).
- **D-NEW-14** — Post Brief v2 migration as sub-plan inside Phase 04.2 execution (from FROZEN-06).

## Session Transition

Operator ran `/gsd:pause-work` after the six resolutions landed but before any artifact writes — context had reached 87%. The checkpoint (`.continue-here.md`) carries the frozen answers as authoritative input to the next session's writeup, converting open questions into closed decisions. Next session (this one) transcribes them into canonical files and commits atomically.

---

## Claude's Discretion (Phase 04.2)

- Masking tool choice for chrome authoring (rembg vs sharp-based background removal — operator QA decides at authoring time)
- Exact VARIANT_SLOTS coordinate values (operator measures per chrome PNG before plan execution)
- File layout within `src/compositor/` beyond the two prescribed subpaths (`chrome-assets/` and `constants.js`)
- Composite implementation detail (sharp composite vs canvas 2D for layering — whichever gives byte-identical output under REQ-X-052b)

## Deferred (Phase 04.2)

No open blockers. Classic scope cuts (most inherited from Section A):

- AI-generated profile avatars — violates REQ-X-052; also moot since chrome is static
- Custom SVG reaction icons — moot; chrome is rasterized
- Tweet timestamp and engagement counts — moot if not in chrome PNGs
- Source Sans Pro alternative — Inter stays; moot for text overlay scope
- Dynamic zone computation — replaced by static VARIANT_SLOTS per variant

## Appendix: Path 2 Amendment (2026-04-15 evening)

Operator chose Path 2 (fixed-per-variant) after validating variant renders from flux-krea looked production-quality. Per-post Phase 5 scene generation eliminated. Compositor simplified to variant-asset lookup + text overlay. Sub-variant pool (D-NEW-15) added for visual diversity within each letter. See D-NEW-01 second revision and D-NEW-15 in CONTEXT.md for architectural implications. REQUIREMENTS and ROADMAP amended in follow-up commit.
