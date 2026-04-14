ROLE
You are implementing the FIRST NON-NEGOTIABLE FOUNDATION for the Pete Content Factory monorepo:
Persona must become an EXECUTABLE SPEC (rules + renderer + validation), not “just a prompt.”

STOP CONDITION
Do not work on dashboards, observability, cross-service glue, new features, or refactors until this foundation exists, is wired into at least one pipeline path end-to-end, and is verifiable.

TARGET REPO
/Users/cpconnor/projects/pete-content-factory

KNOWN CONTEXT (VERIFY IN CODE, DO NOT GUESS)
- Root App defines platform-specific persona prompts in `src/App.jsx` for LinkedIn/Substack/Medium (tone compositions and constraints).
- central-smart-ass already has a persona transformer agent `MirrorUniversePeteAgent.js` and a 6-stage pipeline (`ContentPipeline.js`) where “Style Transformation” is stage 2.
- satirical-pipeline contains persona prompts and comedy structures; treat as reference library, not the canonical spec.

PRIMARY OBJECTIVE
Create a single, canonical Persona Engine that consists of:
1) Persona Spec (data file, versioned)
2) Persona Renderer (deterministic + LLM rewrite is fine)
3) Persona Validator (hard fails + scored checks)
4) Gate: pipelines must route outputs through Renderer → Validator before final output is considered valid

DELIVERABLES (IN THIS EXACT ORDER)

A) INVENTORY & EXTRACTION (read-only, evidence-based)
1. Locate the persona definitions currently embedded in:
   - `src/App.jsx` (platform prompts and constraints)
   - `central-smart-ass/src/agents/MirrorUniversePeteAgent.js`
   - `satirical-pipeline/prompts/` and `satirical-pipeline/mcp-lmstudio/server.py`
2. Summarize differences and overlaps.
3. Identify the minimum set of persona rules that can be made machine-enforceable.

B) CREATE CANONICAL PERSONA SPEC (data, not prose)
Create a new file (pick one format, JSON or YAML):
- `persona/spec/mirror_pete.v1.json` (preferred)
This spec MUST include:
- version, name, description
- voice sliders (sarcasm/cynicism/warmth, verbosity)
- stance rules (what we roast; what we avoid)
- signature motifs (recurring bits, stylistic tics)
- banned phrases/patterns (generic assistant voice, corporate fluff)
- platform adapters: linkedin/substack/medium with explicit constraints (length, structure, CTA style)
- safety boundaries (avoid punching down; disallowed targets)

RULE: This spec becomes the only source of truth going forward.
Do NOT leave persona logic only in `src/App.jsx`.

C) IMPLEMENT PERSONA RENDERER (the compiler)
Implement a renderer that transforms any draft into persona-consistent output.
Target location (choose ONE as canonical and justify):
Option 1: central-smart-ass as the platform “engine”
Option 2: a shared package/module at repo root (preferred long-term)
Minimum renderer function signature:
render(input_text, platform, intent, spec, context_meta) -> { rendered_text, metadata }

Renderer MUST:
- apply deterministic transforms (remove banned phrases, enforce structure skeleton, normalize tone)
- then apply LLM rewrite using the spec + platform adapter (if configured)
- return metadata (platform, spec_version, render_passes, warnings)

D) IMPLEMENT PERSONA VALIDATOR (the bouncer)
Validator returns PASS/FAIL + score breakdown + repair suggestions.
validate(candidate_text, platform, spec) -> {
  pass: bool,
  hard_fail_reasons: [],
  scores: {voice_fit, motif_presence, structure_compliance, safety, genericness},
  suggestions: []
}

Hard-fail checks MUST include:
- banned phrases/patterns
- missing required structural elements for the platform adapter (e.g. LinkedIn hook/beat structure if defined)
- obvious tone drift into generic assistant voice

Scored checks can be LLM-judged, but:
- include deterministic checks where possible (length, headings, bullets, CTA presence)
- keep it reproducible (store scoring prompt in code)

E) WIRE THE GATE (MANDATORY)
Choose ONE path to wire end-to-end first:
Path 1 (recommended): Root App generation → Persona Engine → output displayed
Path 2: central-smart-ass pipeline stage 2 “Style Transformation” → Persona Engine → downstream stages
The gate rule:
No output is “final” unless it passes Validator.
If it fails:
- auto-retry rendering up to N=2 with validator feedback injected
- after retries, return failure with reasons and the best candidate draft

F) PROOF IT WORKS (tests + demo)
1. Add a tiny test harness that:
- feeds 3 intentionally generic drafts (one per platform)
- runs render + validate
- prints: pass/fail, score breakdown, and final text length
2. Add 5 fixture examples:
- 2 passing (persona-strong)
- 3 failing (generic, too mean, structure missing)
3. Provide a “how to run” command for the harness.

G) OUTPUT A SHORT IMPLEMENTATION REPORT
In markdown, include:
- Files added/modified (paths)
- How Persona Spec is versioned
- Where the renderer/validator live and why
- How the gate is enforced
- What’s next (ONLY after this foundation exists)

CONSTRAINTS
- No secrets in output. Redact values. Only list required env var names.
- No broad refactors. Keep changes surgical.
- Keep one foot in satire: the persona is the baseplate, not a skin. But don’t turn logs and errors into a stand-up routine unless they’re user-facing.

SUCCESS CRITERIA (OBJECTIVE)
- A canonical persona spec exists as a file.
- Renderer + validator exist in code and can run locally.
- At least one pipeline path is gated by validator pass/fail.
- A harness or tests prove the gate works with fixtures.
- Persona logic is not trapped only inside `src/App.jsx`.