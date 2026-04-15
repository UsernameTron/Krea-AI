# Lessons

## Active Rules

### Seed Rules
- [2026-04-13] [Config]: Never modify shared config files without checking downstream consumers.
- [2026-04-13] [Scope]: If a "quick fix" requires 3+ files, it is not quick. Re-plan.
- [2026-04-13] [Testing]: Run the full test suite, not just tests for the changed module.
- [2026-04-13] [Dependencies]: Never add dependencies without explicit user approval.
- [2026-04-13] [Data]: Never delete production data, migrations, or seed data without approval.

### Learned Rules
- [2026-04-15] [Architecture]: When a module accumulates repeated symptom fixes (golden-PNG regenerations, font-format swaps, channel-order bugs, emoji pipeline rewrites), STOP patching and escalate to an architecture review. Repeated fragility in one subsystem is a signal the contract is wrong, not that the next patch will be the last one.
  Why: Phase 4 compositor went through Twemoji COLR swap, nav_easter_eggs removal, golden PNG regeneration, and a BGRA channel-swap fix in sequence — all symptoms of "render LinkedIn chrome programmatically" being the wrong architecture. The right move was to pivot to AI-generated full-scene screenshots, which the operator did via the Phase 4 reset on 2026-04-15.
  How to apply: Track patch-count-per-module across a phase. Three or more symptom fixes in the same module within one phase = mandatory architecture-review pause before the next patch. Surface this in /gsd:verify-work and /gsd:debug.
- [2026-04-15] [Determinism]: Do not enforce byte-identical pixel contracts across pipeline stages produced by different rendering engines (e.g., diffusion output vs. canvas chrome). Split the determinism boundary at the seam — each engine owns determinism on its own side, with a fixture handoff in between.
  Why: REQ-X-052 originally asserted byte-identical compositor output against a flux-krea hero. Diffusion and canvas pipelines have entirely different determinism guarantees; the contract was unenforceable end-to-end. Phase 4 reset split it into REQ-X-052a (scene determinism, Phase 5) + REQ-X-052b (overlay determinism, Phase 4) with a checked-in scene PNG fixture as the handoff.
  How to apply: When writing a determinism requirement that spans two pipeline stages, ask "are these stages produced by the same engine?" If no, split the contract and define a versioned fixture at the seam.
- [2026-04-14] [Architecture]: Mirror Post is a three-layer visual system, not a multi-template engine.
  Layer 1: Static assets (foundation prompt YAML for Flux style lock, LinkedIn chrome HTML/Canvas template, compositor zone spec).
  Layer 2: Per-post generation (Post Brief supplies text content; image prompt = foundation prefix + character/scene/props delta built via mirror-vision-prompt-crafter logic adapted for Flux; flux-krea generates full-bleed hero image locally).
  Layer 3: Compositor assembly (hero image + headline/body text overlay with left-side gradient + LinkedIn chrome + tweet embed card + engagement bar = final PNG).
  The hero image is full-bleed — character sits IN the scene with props that have text rendered by Flux. Compositor overlays text and chrome ON TOP of the hero image. Props with text (mug labels, nameplates, whiteboard content) are IN the diffusion output, not compositor overlays.

## Archived
<!-- Rules that no longer apply -->
