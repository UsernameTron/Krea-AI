# Phase 2: Post Brief Generator - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-14
**Phase:** 02-post-brief-generator
**Areas discussed:** JSON Extraction Strategy, System Prompt Architecture, Prompt Caching Strategy, LLM Failure Handling, SDK Version, Model Selection, Streaming
**Mode:** Advisor research (4 parallel agents) + direct user input

---

## JSON Extraction Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Structured Outputs (`output_config.format`) | Constrained decoding guarantees valid JSON shape. Response in `content[0].text`. No tool-invocation framing. | ✓ (primary) |
| tool_use with `strict: true` | Same constrained decoding, but requires tool-invocation semantics. Extra framing for a data-extraction use case. | |
| Plain text + JSON.parse + retry | Provider-agnostic, no schema complexity limits. Parse failures possible, retry adds latency. | ✓ (fallback) |

**User's choice:** Structured Outputs as primary, plain text + parse as fallback. Added constraint B: early planning task to enumerate optional leaves against the 24-parameter ceiling before committing.
**Notes:** Advisor surfaced Structured Outputs as a third option not in the original framing. User accepted with the ceiling-check constraint.

---

## System Prompt Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Deep Distillation | Extract ~1,500 tokens of voice rules from 46KB spec. Prose distillate. Stable, lossy. | |
| Selective Section Load | Parse and include only voice/safety/anti_patterns sections. No separate file. ~2,775 tokens. | |
| Hybrid (Distilled Core + Verbatim Safety at END) | ~900-word prose distillate for voice + verbatim safety.hard_boundaries last per Pattern 5. ~2,600 total. | ✓ |
| Full Spec Reference | Cannot fit. 11,600 tokens vs 8K budget. Structurally impossible. | |

**User's choice:** Hybrid approach (Option C). Added constraint A: distillate must declare `derived_from_spec_version` in frontmatter, test harness must assert version match, regeneration is explicit step on version bump.
**Notes:** User identified distillate synchronization as a silent drift risk and added three specific enforcement mechanisms.

---

## Prompt Caching Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Cache system prompt only | Single breakpoint on last system block. Simplest. | |
| Cache system + archetype (dual-position) | Two breakpoints: system end + first user message end. 90% reduction on system, archetype hits on same-archetype iteration. | ✓ |
| Cache tools + system prompt | Tool definitions cached at highest hierarchy level. Busts on any schema change. | |
| Top-level automatic caching | Request-root ephemeral. Not suited for single-call pattern. | |

**User's choice:** Dual-position caching. Deferred tool-level caching until schema stabilizes.
**Notes:** Maps cleanly to Pattern 5 layout already decided for system prompt architecture.

---

## LLM Failure Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Invalid JSON: 1 retry + correction hint | Recovers transient parse failures. Keeps worst-case under 25s. | ✓ (fallback path only) |
| Invalid JSON: single retry then throw | Stricter latency guarantee. | |
| Safety violation: 0 retries, escalate with error code | Hard constraints. Input is the problem. Named codes for user action. | ✓ |
| Safety violation: 1 retry with stricter prompt | Masks the signal. Adds latency. | |
| Voice drift: soft warning, no retry | Quality signal, not safety gate. Brief is usable, editable in Phase 6. | ✓ |
| Voice drift: hard reject + retry | Latency cost for recoverable quality issue. Arbitrary threshold maintenance. | |

**User's choice:** Accepted all three recommended policies. Added constraint C: clarified that retry logic only applies to the fallback (plain text) path — Structured Outputs makes invalid JSON effectively impossible, so retry logic is skipped on the primary path.
**Notes:** User identified the internal inconsistency between "guaranteed JSON" (Structured Outputs) and "retry on invalid JSON" and required explicit path-conditional logic.

---

## SDK Version (Medium Priority)

**User's choice:** `@anthropic-ai/sdk@latest` for v1 scaffold. Pin once flake observed.
**Notes:** User pre-stated this preference in the discussion prompt. Confirmed.

## Model Selection (Medium Priority)

**User's choice:** `claude-opus-4-6` for Brief Generator.
**Notes:** Persona complexity and satirical coherence justify the cost premium for the core creative generation step.

## Streaming (Medium Priority)

**User's choice:** Non-streaming for Phase 2. Deferred to Phase 6 Artifact UI.
**Notes:** Structured Outputs requires full response for JSON parsing.

---

## Claude's Discretion

- Prompt builder file organization inside `src/brief/prompts/`
- Role framing preamble wording
- Comedy structure selection algorithm
- Error code naming convention
- Distillate prose authoring style

## Deferred Ideas

- Streaming generation → Phase 6
- Tool-level prompt caching → post schema stabilization
- Voice drift hard-reject threshold → not implemented, revisit if downstream proves sensitive
- Multi-provider abstraction → not built (REQ-X-028 locks Anthropic)
