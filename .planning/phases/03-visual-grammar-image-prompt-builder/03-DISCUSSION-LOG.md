# Phase 3: Visual Grammar + Image Prompt Builder - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-14
**Phase:** 03-visual-grammar-image-prompt-builder
**Areas discussed:** Scene-to-archetype mapping, Prompt assembly output format, Zone spec precision, Engagement satirical calibration

---

## Scene-to-Archetype Mapping

| Option | Description | Selected |
|--------|-------------|----------|
| Template lookup + freeform pass-through | Read image_seed.scene_template, load matching template file. Freeform inputs use image_seed.environment directly. Composition zone constraints injected universally. | ✓ |
| Category-to-scene map only | 5-entry category map. No per-archetype overrides. Freeform defaults to office-middle-mgmt. | |
| Template lookup only, fail on unknown | Strict lookup — throw error on unrecognized scene_template. Freeform must produce a recognized template. | |

**User's choice:** Template lookup + freeform pass-through (Recommended)
**Notes:** Research revealed that Post Brief's `image_seed.scene_template` is already LLM-assigned in Phase 2. Phase 3 consumes it directly — routing is upstream. Composition zone constraints injected universally regardless of path.

---

## Prompt Assembly Output Format

| Option | Description | Selected |
|--------|-------------|----------|
| 4-key object | { positive_prompt, negative_prompt, parameters, composition_notes }. Superset of prompt-file contract. Callers extract what they need. | ✓ |
| 3-key object (prompt-file match) | { positive_prompt, negative_prompt, parameters }. Exact match to flux-krea contract. Composition notes returned separately. | |
| You decide | Claude picks during planning. | |

**User's choice:** 4-key object (Recommended)
**Notes:** Concatenation order locked: foundation → character → environment → props → composition directives → fidelity modifiers. Negative prompt includes surface-text guards. `composition_notes` stripped before serialization to prompt-file.

---

## Zone Spec Precision

| Option | Description | Selected |
|--------|-------------|----------|
| JSON with pixel coords | zone-spec.json — pixel coordinates for all zones on 1920x1080 canvas. Static file, loaded once. No new deps. | ✓ |
| YAML with pixel coords | zone-spec.yaml — same data with inline comments. Matches foundation-prompt.yaml pattern. | |
| JS constants file | zone-spec.js — exported named constants. Tree-shakeable, IDE autocomplete. JS-only consumers. | |

**User's choice:** JSON with pixel coords (Recommended)
**Notes:** REQ-X-051 (fixed template) and REQ-X-052 (byte-identical output) favor simplest static format. No semantic names or percentages — pixel precision avoids resolution layers.

---

## Engagement Satirical Calibration

| Option | Description | Selected |
|--------|-------------|----------|
| Tier-bounded + tone-weighted | Character tier sets count ranges, post tone drives dominant reaction type. Controlled randomness with seed for test determinism. | ✓ |
| Fully deterministic | Hash character_type + tone to produce fixed metrics. Same Brief = same numbers. | |
| You decide | Claude picks during planning. | |

**User's choice:** Tier-bounded + tone-weighted (Recommended)
**Notes:** 4 tiers defined (C-suite, middle-mgmt, hustle/sales, support/ops). 6 LinkedIn reaction types with Funny suppressed. Seed parameter: null = random, integer = deterministic. Calibration bands specified in CONTEXT.md D-11.

---

## Claude's Discretion

- Scene template file format and internal structure
- Modifier selection algorithm
- Negative prompt composition beyond surface-text guards
- `composition_notes` content depth
- File organization within `src/image/` and `src/grammar/`

## Deferred Ideas

- Per-archetype scene overrides in library.json — not needed for v1
- Dynamic zone computation per-brief — zones fixed for v1
- Modifier weighting by archetype domain — Claude's discretion for v1
