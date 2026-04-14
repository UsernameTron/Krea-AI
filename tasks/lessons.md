# Lessons

## Active Rules

### Seed Rules
- [2026-04-13] [Config]: Never modify shared config files without checking downstream consumers.
- [2026-04-13] [Scope]: If a "quick fix" requires 3+ files, it is not quick. Re-plan.
- [2026-04-13] [Testing]: Run the full test suite, not just tests for the changed module.
- [2026-04-13] [Dependencies]: Never add dependencies without explicit user approval.
- [2026-04-13] [Data]: Never delete production data, migrations, or seed data without approval.

### Learned Rules
- [2026-04-14] [Architecture]: Mirror Post is a three-layer visual system, not a multi-template engine.
  Layer 1: Static assets (foundation prompt YAML for Flux style lock, LinkedIn chrome HTML/Canvas template, compositor zone spec).
  Layer 2: Per-post generation (Post Brief supplies text content; image prompt = foundation prefix + character/scene/props delta built via mirror-vision-prompt-crafter logic adapted for Flux; flux-krea generates full-bleed hero image locally).
  Layer 3: Compositor assembly (hero image + headline/body text overlay with left-side gradient + LinkedIn chrome + tweet embed card + engagement bar = final PNG).
  The hero image is full-bleed — character sits IN the scene with props that have text rendered by Flux. Compositor overlays text and chrome ON TOP of the hero image. Props with text (mug labels, nameplates, whiteboard content) are IN the diffusion output, not compositor overlays.

## Archived
<!-- Rules that no longer apply -->
