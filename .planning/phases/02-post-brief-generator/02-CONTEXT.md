# Phase 2: Post Brief Generator - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the Post Brief Generator module: input classifier, system prompt builder, LLM generation via Anthropic API, and Brief Validator. Users provide a corporate archetype or freeform scenario and receive a complete, validated Post Brief JSON containing every element of a satirical LinkedIn post. This is the module that produces the canonical JSON contract every downstream module consumes.

</domain>

<decisions>
## Implementation Decisions

### JSON Extraction Strategy
- **D-01:** Primary path is Anthropic Structured Outputs (`output_config.format`) with the Post Brief schema. Constrained decoding guarantees valid JSON shape at token-generation level. Response arrives in `content[0].text` as a JSON string — no tool-invocation framing.
- **D-02:** Brief Validator (Pattern 2, REQ-X-029) still runs on every response regardless of extraction method. Structured Outputs guarantees shape, not semantic validity (safety, voice, completeness).
- **D-03:** Fallback path is plain text generation + `JSON.parse` + 1 retry with correction hint, then throw to user. Activated only if Structured Outputs cannot accommodate the Post Brief schema (see D-04).
- **D-04:** Early planning task required: enumerate all optional leaves in the Post Brief schema against Structured Outputs' 24-optional-parameter ceiling. If count exceeds 24, apply "required + sentinel value" mitigation (empty string, 0, empty array). If sentinel pattern is ugly or count far exceeds 24, fall back to plain text + parse as primary path. Decision deferred to planning, not locked here.

### System Prompt Architecture (Pattern 5)
- **D-05:** System prompt uses hybrid distillation. A `persona-voice-distillate.md` file (~900 words of imperative prose) is derived from mirror_pete.v2.json voice sections. The full persona spec is never sent to the API.
- **D-06:** Concrete block assignment for system prompt builder:
  - TOP: Role framing (2-3 sentences, ~100 tokens)
  - MIDDLE: PERSONA_VOICE — distilled prose from spec (~900 tokens)
  - MIDDLE: COMEDY_TOOLKIT — 2-3 structures selected per archetype (~400 tokens)
  - MIDDLE: OUTPUT_SCHEMA — verbatim Post Brief v1 schema (~600 tokens)
  - MIDDLE: PROP_INTELLIGENCE — static prop rules (~200 tokens)
  - END: SAFETY_RAILS — verbatim `safety.hard_boundaries` from persona spec (~400 tokens) — Pattern 5 highest-attention position
- **D-07:** First user message (Pattern 5 conversational frame):
  - Selected archetype full definition (~100-200 tokens)
  - User's raw scenario or archetype name (~50-100 tokens)
  - "Generate a complete Post Brief."
- **D-08:** Comedy structures: load all 13, select 2-3 per archetype at generation time (Pattern 12 lazy loading), inject into COMEDY_TOOLKIT slot.
- **D-09:** Budget: ~2,600 of 8,192 tokens used. Generous headroom for iteration.

### Distillate Synchronization
- **D-10:** `persona-voice-distillate.md` must declare `derived_from_spec_version: "2.0.0"` in a frontmatter block.
- **D-11:** Test harness must assert the distillate's declared version matches `mirror_pete.v2.json`'s `version` field. Mismatch = test failure.
- **D-12:** Distillate regeneration is an explicit step on any persona spec version bump, documented in the spec's version-bump procedure.

### Prompt Caching
- **D-13:** Dual-position caching aligned to Pattern 5 layout:
  - `cache_control: {type: "ephemeral"}` on last block of `system` array (covers persona distillate + safety rails + schema, ~2,600 tokens)
  - `cache_control: {type: "ephemeral"}` on last content block of first user message (covers selected archetype definition)
- **D-14:** System prompt cache hits on every call after the first (90% cost reduction). Archetype cache hits when user iterates on same archetype, busts (1.25x write cost) on archetype switch.
- **D-15:** Tool-level schema caching deferred until schema stabilizes post-development.
- **D-16:** 2,048-token minimum threshold for Sonnet 4.6 is cleared by the system prompt block.

### LLM Failure Handling
- **D-17:** Primary path (Structured Outputs): invalid JSON is effectively impossible. No retry logic needed for structural parse failures.
- **D-18:** Fallback path (plain text): 1 retry with correction hint ("Return only a JSON object matching this schema: ..."), then throw to user. Keeps worst-case latency under 25 seconds.
- **D-19:** Safety violation (REQ-X-020 through X-024): 0 retries. Reject immediately with named error code (`SAFETY_INDIVIDUAL_TARGET`, `SAFETY_MARGINALIZED_GROUP`, etc.). Surface to user with plain-language description. The input is the problem — retrying the same scenario rarely produces different safety outcomes.
- **D-20:** Voice drift (structurally valid, safety-compliant, but voice sliders off-target): soft warning, no retry. Brief is usable — user can inspect and edit in-place (Phase 6, REQ-M-051). Voice drift is a quality signal, not a safety gate. Applies regardless of extraction path.

### SDK and Model
- **D-21:** `@anthropic-ai/sdk@latest` for v1 scaffold. Pin to specific version once flake is observed.
- **D-22:** Model: `claude-opus-4-6` for Brief Generator. Persona complexity, voice fidelity, and satirical coherence across all Brief fields justify the cost premium for the core creative generation step.

### Streaming
- **D-23:** Non-streaming for Phase 2. Structured Outputs requires the full response for JSON parsing. Streaming deferred to Phase 6 Artifact UI if needed.

### Claude's Discretion
- Prompt builder file organization inside `src/brief/prompts/`
- Exact wording of role framing preamble
- Comedy structure selection algorithm (match by archetype domain, mood, or random subset)
- Error code naming convention and message formatting
- Distillate prose authoring style (as long as it covers the five axioms, banned phrases, deterministic transforms, anti-patterns, and linguistic DNA from the spec)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Plans
- `Plans for Krea AI/PLAN_02_MODULES.md` — Module 1 spec: Post Brief Generator (input classifier, system prompt builder, LLM generation, Brief Validator)
- `Plans for Krea AI/KNOWLEDGE_BASE.md` — Comprehensive analysis, integration strategy, and performance targets

### Requirements & Constraints
- `.planning/REQUIREMENTS.md` — Phase 2 requirements: REQ-M-010 through REQ-M-018; cross-cutting: REQ-X-028 (Anthropic API), REQ-X-029 (Pattern 2 Zero-Trust), REQ-X-031 (Pattern 5 dual-position), REQ-X-032 (Pattern 12 lazy loading), REQ-X-020 through X-024 (safety rails)
- `.planning/ROADMAP.md` §Phase 2 — Success criteria, plan breakdown (02-01, 02-02, 02-03)

### Prior Phase Context
- `.planning/phases/01-mirror-post-scaffolding/01-CONTEXT.md` — Phase 1 decisions: props normalized array (D-02), tweet_embed required (D-03), voice sliders float/int (D-05), hybrid fixtures (D-06)

### Persona & Assets
- `mirror-post/src/persona/spec/mirror_pete.v2.json` — Canonical persona spec (source of truth for distillate). Voice sections, safety.hard_boundaries, banned_patterns, deterministic_transforms, anti_patterns, consistency_rules
- `mirror-post/src/persona/archetypes/library.json` — 17 archetypes in 5 categories (metadata for upfront loading, full defs for lazy loading)
- `mirror-post/src/persona/comedy/structures.json` — 10 joke + 3 roast structures
- `mirror-post/src/brief/schema.js` — Post Brief v1 schema + hand-written validator (existing from Phase 1)

### Codebase Context
- `.planning/codebase/CONVENTIONS.md` — Coding conventions (flux-krea is Python; mirror-post is JS/ESM)
- `state/pattern-context.md` — KB v2.1 patterns: Pattern 2 (Zero-Trust), Pattern 5 (Dual-Position), Pattern 12 (Lazy Loading)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/brief/schema.js` — Post Brief v1 schema with hand-written `validateBrief()` function (130 lines). Phase 2 Brief Validator extends this with safety, voice, and completeness checks.
- `src/persona/spec/mirror_pete.v2.json` — Full persona spec. Source for distillate extraction and safety rail verbatim copy.
- `src/persona/archetypes/library.json` — 17 archetypes. Input classifier uses this for exact/fuzzy/domain matching.
- `src/persona/comedy/structures.json` — 13 structures. Comedy toolkit selection draws from this.
- `src/image/modifiers/` — Ultra-fidelity and 12K modifier libraries. NOT used in Phase 2 (consumed by Phase 4 Image Prompt Engine).

### Established Patterns
- ESM modules (`"type": "module"` in package.json)
- Barrel exports via `index.js` in each directory
- Hand-written validators (no external schema library) — Phase 1 chose this over ajv/Zod
- Test harness: `test/harness.js` with fixture-based validation

### Integration Points
- Brief Validator (Phase 2) extends existing `validateBrief()` from `schema.js` — adds safety, voice, and completeness checks on top of structural validation
- Generated Post Briefs must pass existing fixture tests AND new Phase 2 semantic tests
- `src/brief/prompts/` directory exists (empty) — ready for system prompt templates and distillate

</code_context>

<specifics>
## Specific Ideas

- Distillate file includes a frontmatter version declaration linking it to the canonical spec version — enables automated drift detection via test assertion
- Structured Outputs schema should be derived programmatically from the existing `schema.js` field definitions where possible, not hand-duplicated
- Error codes for safety violations should be human-readable strings that Pete can act on without looking up a table
- Comedy structure selection: match by archetype category/domain first, then by tone/mood. If no domain match, fall back to general-purpose structures.

</specifics>

<deferred>
## Deferred Ideas

- Streaming generation — deferred to Phase 6 (Artifact UI). Non-streaming is correct for Phase 2's JSON extraction.
- Tool-level prompt caching for Post Brief schema — deferred until schema stabilizes post-development
- Voice drift hard-reject threshold — not implemented. Soft warning is the policy. Revisit only if downstream modules (Image Prompt Engine) prove hypersensitive to voice slider values.
- Multi-provider abstraction layer — not built. Anthropic API is locked (REQ-X-028). If provider-agnosticism becomes a future requirement, Structured Outputs would be replaced with plain text + parse.

</deferred>

---

*Phase: 02-post-brief-generator*
*Context gathered: 2026-04-14*
