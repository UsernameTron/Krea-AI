# Phase 1: Mirror Post Scaffolding - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the complete mirror-post project skeleton: directory structure, persona asset imports, Post Brief v1 JSON schema, and 4 reference test fixtures. Zero functional code. After this phase, a Claude Code session can `cd mirror-post/` and know exactly what to build next — all structure, data, schema, and fixtures in place.

</domain>

<decisions>
## Implementation Decisions

### Git Strategy
- **D-01:** mirror-post is a separate git repo inside the Krea-AI workspace, matching the flux-krea pattern. `git init` inside `mirror-post/`, add `mirror-post/` to Krea-AI `.gitignore`. Independent commit history, CI, and release cadence.

### Post Brief Schema Shape
- **D-02:** Props normalized to a uniform array of `{ type, text, placement }` objects. No mixed structure (no named object keys for mug/whiteboard alongside a desk_items array). All props are array entries — eliminates branching in compositor, prompt-builder, and validator.
- **D-03:** `tweet_embed` is required (always present in every Post Brief).
- **D-04:** `nav_easter_eggs` is optional (not every post needs nav misspellings or chrome details).
- **D-05:** Voice sliders mirror the persona spec's own encoding: float 0.0–1.0 for continuous dimensions (sarcasm, cynicism, warmth) and ordinal integer 1–5 for satirical intensity (matching roast_doctrine's 5-level scale). Both are single-line deterministic validations per Pattern 2 (REQ-X-029). Note: the PLAN_01 schema draft has no voice slider field — this must be added as a new top-level `voice` object.

### Test Fixtures
- **D-06:** Hybrid fixture depth. Brent Vellum is the gold standard — full field-by-field depth covering every schema field. Trevor B. (hustle), Trevor B. (closer), and Pete C. (titles) are structural skeletons with key fields only: meta.input_type, character.name, character.title, post.headline.text, props[0].type, tweet_embed.handle, engagement.dominant_reaction, image_seed.scene_template.

### Claude's Discretion
- Schema validation library choice (ajv, Zod, hand-written) — planner decides based on dependency philosophy
- Barrel export structure for index.js files
- Exact directory naming (kebab-case vs snake_case for subdirectories)
- README.md content depth

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Plans
- `Plans for Krea AI/PLAN_01_SCAFFOLDING.md` — Complete scaffolding spec: directory structure (Section 1), architectural decisions (Section 2), foundational files (Section 3), asset import mapping (Section 4), Post Brief schema draft (Section 5), fixture descriptions (Section 6), execution checklist (Section 7)
- `Plans for Krea AI/PLAN_02_MODULES.md` — Module specs for downstream consumers of the Post Brief schema
- `Plans for Krea AI/KNOWLEDGE_BASE.md` — Comprehensive analysis and integration strategy

### Requirements & Constraints
- `.planning/REQUIREMENTS.md` — 78 requirements; Phase 1 requirements are REQ-M-001 through REQ-M-006; cross-cutting constraints REQ-X-001 (persona immutable), REQ-X-002 (Post Brief is the contract), REQ-X-042 (Node.js 20+)
- `.planning/ROADMAP.md` §Phase 1 — Success criteria, plan breakdown (01-01, 01-02, 01-03)

### Codebase Context
- `.planning/codebase/STRUCTURE.md` — flux-krea directory layout and conventions (reference for workspace pattern)
- `.planning/codebase/CONVENTIONS.md` — flux-krea coding conventions (Python; mirror-post is JS but workspace consistency matters)

### Persona Spec
- The canonical persona spec file (mirror_pete.v2.json) must be located on Pete's machine for import — source path TBD at execution time

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- No existing mirror-post code (greenfield project)
- flux-krea exists as a shipped Python project in `flux-krea/` — demonstrates the workspace pattern (separate repo, own CI, gitignored by parent)

### Established Patterns
- Workspace pattern: sub-projects are separate git repos, gitignored by Krea-AI parent repo
- flux-krea uses: snake_case files, PascalCase classes, Google-style docstrings, ruff + mypy, pytest
- mirror-post will use: Node.js 20+, ESM (`"type": "module"`), .js files

### Integration Points
- mirror-post/ will be added to Krea-AI `.gitignore` (matching flux-krea/ entry)
- Post Brief schema is the contract consumed by all downstream phases (2–7)
- Voice slider encoding in schema must match persona spec's own encoding for prompt construction consistency

</code_context>

<specifics>
## Specific Ideas

- Props array normalization: every prop is `{ type, text, placement }` — mug becomes `{ type: "mug", text: "ALIGNMENT", placement: "right_hand" }`, whiteboard becomes `{ type: "whiteboard", text: ["Role Title A", ...], placement: "background_right" }` (text field can be string or array depending on prop type)
- Voice object added to schema: `{ sarcasm: 0.8, cynicism: 0.85, warmth: 0.25, satirical_intensity: 4 }` — floats for continuous, integer for intensity
- Gold fixture (Brent Vellum) should include nav_easter_eggs to validate the optional field path; skeleton fixtures should omit it to validate the absence path

</specifics>

<deferred>
## Deferred Ideas

- Asset source locations — not discussed; planner should check for existing persona assets in the workspace or prompt for paths at execution time
- Schema validation library choice — left to Claude's discretion at planning time
- `nav_easter_eggs` rename to `chrome_details` — not worth the churn for v1; revisit if scope broadens beyond nav misspellings

</deferred>

---

*Phase: 01-mirror-post-scaffolding*
*Context gathered: 2026-04-14*
