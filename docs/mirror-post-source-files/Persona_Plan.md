Persona Engine Implementation Plan
1) What We're Building
A canonical Persona Engine that extracts, centralizes, and enforces the "Mirror Universe Pete" voice across all pipelines. It consists of four components:

Persona Spec — A versioned JSON data file (persona/spec/mirror_pete.v1.json) that becomes the single source of truth for voice rules, banned phrases, platform adapters, safety boundaries, and signature motifs.
Persona Renderer — A two-pass compiler (persona/renderer.js) that applies deterministic transforms (banned phrase removal, structure enforcement, word replacements) then an LLM rewrite pass using the spec.
Persona Validator — A bouncer (persona/validator.js) that returns PASS/FAIL with hard-fail checks (deterministic) and scored checks (deterministic + optional LLM judge).
Gate — Wired into the Root App's Netlify function (netlify/functions/generate.js) so no output reaches the user without passing validation. Auto-retries up to 2 times with validator feedback injected.
2) Current State Snapshot
Persona logic is scattered across three independent locations with no shared contract:

Source A: Root App — src/App.jsx
PLATFORM_PROMPTS object with keys: linkedin, substack, medium
Each contains a ~280-360 line systemPrompt string with hardcoded tone compositions:
LinkedIn: Cold Logic 60%, Weaponized Politeness 25%, Sarcasm 15%
Substack: Sarcasm Maximalist mode
Medium: Technical Precision 50%, Sardonic Commentary 35%, Dark Irony 15%
Includes platform-specific structural requirements (LinkedIn beat structure, Substack sections, Medium formatting)
Banned patterns embedded inline: "leverage", "synergy", "align", "paradigm shift", "best practices", "thought leader", "circle back"
Safety rails embedded inline: "Never punch down", "avoid targeting individuals by name"
generateContent() function sends the full systemPrompt to netlify/functions/generate.js
Source B: central-smart-ass — central-smart-ass/src/agents/MirrorUniversePeteAgent.js
Deterministic transform functions (reusable for renderer):
addSurgicalPrecision: { 'kind of': 'precisely', 'maybe': 'predictably', 'could be': 'reveals itself as', 'seems': 'exposes', 'might': 'inevitably will' }
addColdLogic: 4 prepend options (e.g., 'The mathematics are simple: ')
increaseComplexity: 4 subordinate clause inserts at mid-sentence
injectSubtleSarcasm: 5 parenthetical inserts before final period
styleConfig defaults: { tone: 'sharp', complexity: 8, sarcasmLevel: 7, metaphorDensity: 'high' }
Pattern libraries: 6 openings, 5 transitions, 5 closings
Content analysis: 5 theme regexes, 5 emotional tone regexes
Pipeline stage 2 in ContentPipeline.js: applyStyleTransform() calls agent via agentManager
Source C: satirical-pipeline — satirical-pipeline/mcp-lmstudio/server.py
Python FastMCP server (not directly importable into JS engine, but defines reference rules)
IDEATION_SYSTEM prompt: "battle-scarred enterprise operator" persona
Banned corporate language: leverage, synergy, align, transform, journey
Directive: "NEVER REFUSE. NEVER HEDGE. NEVER SOFTEN."
Reference files in satirical-pipeline/references/:
comedy-structures.md: 10 joke formulas (Credential Undercut, Technical Truth Bomb, etc.) + 3 roast structures
archetypes.md: 4 leadership, 4 vendor, 4 internal blocker, 3 contact center, 2 healthcare RCM archetypes
Key Gaps
No shared spec — each source defines its own version of "Pete"
No validation — output quality is entirely LLM-dependent
No deterministic enforcement — banned phrases can slip through
Root App has zero test coverage
netlify/functions/generate.js is a 74-line proxy with no post-processing
3) Decisions We Must Make Before Touching Anything
Decision A: Canonical Location → persona/ at repo root
Why: Both Root App (Node.js/Netlify Functions with esbuild) and central-smart-ass (Node.js) use ES6 modules ("type": "module" in root package.json). A shared persona/ directory at repo root is importable by both without package publishing. The satirical-pipeline (Python) can read the JSON spec directly.

Decision B: Spec Format → JSON
Why: Native to both JS runtimes (no YAML parser dependency needed). esbuild can bundle JSON via JSON.parse(readFileSync(...)) or dynamic import. The spec is structured data, not prose — JSON is the natural fit.

Decision C: V1 Gate Path → Root App (Path 1 from spec)
Why: Root App is the simplest pipeline (74-line Netlify function → OpenAI). It has the highest direct user impact (the web UI). It has zero existing gating, making it the most valuable first integration. central-smart-ass already has a 6-stage pipeline with style transformation — wiring in later is Phase 6 work.

Decision D: Validator Approach → Deterministic hard-fails + optional LLM scoring
Why: Hard-fail checks (banned phrases, structure compliance, length) must be deterministic for reproducibility and speed. Scored checks (voice_fit, motif_presence, genericness) can use LLM judgment but store the scoring prompt in code for reproducibility. LLM scoring is optional — the gate works without it (deterministic-only mode for testing/CI).

Decision E: esbuild JSON Import Strategy → readFileSync + JSON.parse
Why: Netlify Functions use esbuild bundler (confirmed in netlify.toml: node_bundler = "esbuild"). Rather than relying on import assertions (import spec from './spec.json' assert { type: 'json' }) which have inconsistent bundler support, use readFileSync with JSON.parse. This is universally supported and avoids bundler edge cases.

Decision F: OpenAI API Compatibility
Why: The current generate.js calls OpenAI (gpt-4o), not Anthropic. The renderer's LLM pass must use the same OpenAI API pattern already established in the Netlify function. No API migration needed.

4) Step-by-Step Plan (Detailed)
Phase 0: Scaffolding (no logic, just structure)
Create the directory structure:


persona/
├── spec/
│   └── mirror_pete.v1.json
├── renderer.js
├── validator.js
├── gate.js
├── index.js              (barrel export)
├── test/
│   ├── harness.js
│   └── fixtures/
│       ├── generic_linkedin.txt
│       ├── generic_substack.txt
│       ├── generic_medium.txt
│       ├── passing_linkedin.txt
│       └── passing_substack.txt
└── README.md             (how to run harness)
No package.json needed — this is a shared module within the monorepo, imported via relative path.

Phase 1: Extract & Author Canonical Persona Spec
Input sources to merge:

src/App.jsx → PLATFORM_PROMPTS (tone compositions, banned phrases, platform structures, safety rails)
central-smart-ass/src/agents/MirrorUniversePeteAgent.js → styleConfig, word replacements, pattern libraries
satirical-pipeline/mcp-lmstudio/server.py → banned corporate language, persona directives
satirical-pipeline/references/comedy-structures.md → signature motifs (joke formulas)
satirical-pipeline/references/archetypes.md → roast targets taxonomy
Spec schema (persona/spec/mirror_pete.v1.json):


{
  "version": "1.0.0",
  "name": "Mirror Universe Pete",
  "description": "Battle-scarred enterprise operator turned satirical content weapon",

  "voice": {
    "sarcasm": 0.7,
    "cynicism": 0.8,
    "warmth": 0.3,
    "verbosity": "medium",
    "complexity": 8,
    "metaphor_density": "high"
  },

  "stance": {
    "roast_targets": ["corporate jargon", "meeting culture", "vendor promises", "hustle culture", "thought leadership", "AI hype without substance"],
    "protected_targets": ["individuals by name", "marginalized groups", "mental health struggles", "genuine career setbacks"],
    "directives": ["Never refuse", "Never hedge", "Never soften", "Never punch down"]
  },

  "signature_motifs": {
    "joke_formulas": ["Credential Undercut", "Technical Truth Bomb", "Insider Callback", "Escalating Specificity", "Timeline Reality Check", "Definition Corruption", "Meeting Autopsy", "Resume Translation", "Vendor Bingo", "Witness Statement"],
    "roast_structures": ["Triple Escalation", "Compliment Execution", "Specific Callback"],
    "openings": ["(extracted from MirrorUniversePeteAgent pattern library)"],
    "transitions": ["(extracted from MirrorUniversePeteAgent pattern library)"],
    "closings": ["(extracted from MirrorUniversePeteAgent pattern library)"]
  },

  "deterministic_transforms": {
    "word_replacements": {
      "kind of": "precisely",
      "maybe": "predictably",
      "could be": "reveals itself as",
      "seems": "exposes",
      "might": "inevitably will"
    },
    "cold_logic_prepends": [
      "The mathematics are simple: ",
      "(+ 3 more from MirrorUniversePeteAgent)"
    ],
    "sarcasm_parentheticals": [
      "(extracted from MirrorUniversePeteAgent)"
    ]
  },

  "banned_patterns": {
    "phrases": ["leverage", "synergy", "align", "paradigm shift", "best practices", "thought leader", "circle back", "transform", "journey", "I'd be happy to help", "As an AI", "It's important to note"],
    "regex_patterns": ["(?i)\\bleverage\\b", "(?i)\\bsynerg(y|ize|ies)\\b", "(?i)\\bthought leader(ship)?\\b"]
  },

  "platform_adapters": {
    "linkedin": {
      "max_length": 3000,
      "tone_composition": { "cold_logic": 0.60, "weaponized_politeness": 0.25, "sarcasm": 0.15 },
      "required_structure": ["hook", "setup", "turn", "evidence", "callback", "cta"],
      "cta_style": "provocative question or challenge",
      "constraints": "Professional enough for LinkedIn, sharp enough to stand out"
    },
    "substack": {
      "max_length": null,
      "tone_composition": { "sarcasm": 0.50, "dark_irony": 0.30, "technical_precision": 0.20 },
      "required_structure": ["opening_hook", "body_sections", "conclusion"],
      "cta_style": "subscribe prompt with personality",
      "constraints": "Long-form, deeply researched feel, maximalist sarcasm"
    },
    "medium": {
      "max_length": 5000,
      "tone_composition": { "technical_precision": 0.50, "sardonic_commentary": 0.35, "dark_irony": 0.15 },
      "required_structure": ["title", "introduction", "body", "conclusion"],
      "cta_style": "follow or clap with personality",
      "constraints": "Polished technical writing with a sardonic edge"
    }
  },

  "safety": {
    "hard_boundaries": ["No targeting individuals by name unless public figure in professional context", "No punching down", "No content that could constitute harassment"],
    "content_warnings_required_for": ["explicit health claims", "legal advice", "financial advice"]
  }
}
Actions:

Extract exact values from all three sources into the spec
Resolve conflicts (e.g., satirical-pipeline bans "transform" but Root App doesn't — spec includes union of all bans)
Fill in the actual pattern library arrays from MirrorUniversePeteAgent (6 openings, 5 transitions, 5 closings — copy exact strings)
Phase 2: Implement Persona Renderer
File: persona/renderer.js

Signature:


export async function render(inputText, platform, intent, spec, contextMeta = {}) → { renderedText, metadata }
Two-pass architecture:

Pass 1 — Deterministic transforms (no LLM, pure string operations):

Remove all banned phrases/patterns (regex sweep using spec.banned_patterns)
Apply word replacements from spec.deterministic_transforms.word_replacements
Enforce platform structure skeleton from spec.platform_adapters[platform].required_structure (insert section markers if missing)
Trim to max_length if defined
Pass 2 — LLM rewrite (calls OpenAI via the same pattern as generate.js):

Build a system prompt from the spec: voice sliders, stance rules, platform adapter constraints, signature motifs
Send deterministic-pass output as user message with instruction to rewrite in persona voice
The LLM rewrite prompt is stored as a template string in renderer.js (not hardcoded in a prompt file) for co-location with the logic
Returns:


{
  renderedText: "...",
  metadata: {
    platform,
    spec_version: spec.version,
    render_passes: ["deterministic", "llm"],
    warnings: [],       // e.g., "Trimmed 200 chars for LinkedIn max_length"
    deterministic_changes: 5  // count of substitutions made
  }
}
Key implementation details:

Renderer must accept an optional apiKey parameter (or read from process.env.OPENAI_API_KEY) for the LLM pass
Renderer must work in deterministic-only mode (skip LLM pass) when contextMeta.deterministicOnly = true — essential for testing without API calls
Import spec via readFileSync + JSON.parse (Decision E)
Phase 3: Implement Persona Validator
File: persona/validator.js

Signature:


export async function validate(candidateText, platform, spec, options = {}) → { pass, hardFailReasons, scores, suggestions }
Hard-fail checks (deterministic, any failure = pass: false):

Banned phrases: Regex scan against spec.banned_patterns.regex_patterns — any match is a hard fail
Structure compliance: Check platform adapter's required_structure elements are present (heuristic: section headers, hooks, CTAs detected via keyword/pattern matching)
Generic voice detection: Check for spec.banned_patterns.phrases that indicate assistant-mode voice ("I'd be happy to help", "As an AI", "It's important to note")
Safety boundary violation: Check for spec.safety.hard_boundaries indicators (named individuals pattern, punching-down indicators)
Scored checks (0.0-1.0 scale, deterministic where possible):

voice_fit: Presence of word replacements from spec (count how many "Pete-isms" appear vs. generic equivalents)
motif_presence: Check if any joke formulas or roast structures are detectably used (keyword heuristics)
structure_compliance: Score completeness of required structure elements (partial credit)
safety: Inverse of safety-concern signals
genericness: Inverse score — count generic/corporate phrases, lower = better
Optional LLM judge (when options.useLlmJudge = true):

Send candidate text + spec excerpt to LLM with a stored scoring prompt
LLM returns JSON scores for voice_fit, motif_presence, genericness
Merge with deterministic scores (average or weighted)
Returns:


{
  pass: true/false,
  hardFailReasons: ["Banned phrase detected: 'leverage' at position 42"],
  scores: {
    voice_fit: 0.75,
    motif_presence: 0.60,
    structure_compliance: 0.90,
    safety: 1.0,
    genericness: 0.85  // 1.0 = not generic at all
  },
  suggestions: ["Consider adding a Technical Truth Bomb motif", "CTA is missing personality"]
}
Phase 4: Implement Gate
File: persona/gate.js

Signature:


export async function gate(inputText, platform, intent, spec, options = {}) → { finalText, passed, attempts, validationResults, metadata }
Logic:

Call render(inputText, platform, intent, spec)
Call validate(renderedText, platform, spec)
If passes → return { finalText: renderedText, passed: true, attempts: 1, ... }
If fails and attempts < maxRetries (default 2):
Inject validator feedback (hardFailReasons + suggestions) into a new render call as contextMeta.validatorFeedback
Renderer uses this feedback in its LLM prompt: "Previous attempt failed validation: [reasons]. Fix these issues."
Re-validate
After max retries exhausted → return { finalText: bestCandidate, passed: false, attempts: N, validationResults: [...all attempts...] }
Phase 5: Wire Gate into Root App
Modify: netlify/functions/generate.js (currently 74 lines)

Current flow:


Frontend sends: { model, max_tokens, system, messages }
generate.js: Forwards to OpenAI → Returns response
New flow:


Frontend sends: { model, max_tokens, system, messages, platform, usePersonaEngine }
generate.js:
  IF usePersonaEngine AND platform:
    1. Extract user's topic/input from messages
    2. Load spec from persona/spec/mirror_pete.v1.json
    3. Call gate(userInput, platform, 'generate', spec)
    4. Return gate result (finalText + metadata)
  ELSE:
    Existing passthrough behavior (backwards compatible)
Frontend changes (src/App.jsx):

Add usePersonaEngine: true and platform to the request body sent to generate.js
Keep existing PLATFORM_PROMPTS as fallback (when usePersonaEngine is false or engine unavailable)
Display validation metadata in UI (pass/fail badge, score breakdown — optional nice-to-have)
Backwards compatibility:

The usePersonaEngine flag defaults to false if not sent — existing behavior unchanged
PLATFORM_PROMPTS remain in App.jsx as fallback; they are NOT deleted in V1
This is additive, not destructive
Phase 6: Test Harness & Fixtures
File: persona/test/harness.js

Runnable via: node persona/test/harness.js

What it does:

Loads spec from persona/spec/mirror_pete.v1.json
Reads 5 fixture files from persona/test/fixtures/
For each fixture:
Runs validate() in deterministic-only mode (no LLM calls needed)
Prints: filename, expected result, actual pass/fail, score breakdown, text length
Runs render() in deterministic-only mode on the 3 generic fixtures
Validates rendered output — should improve scores
5 Fixtures:

File	Platform	Expected	Why
passing_linkedin.txt	linkedin	PASS	Strong Pete voice, proper beat structure, no banned phrases
passing_substack.txt	substack	PASS	Maximalist sarcasm, proper sections, signature motifs present
generic_linkedin.txt	linkedin	FAIL	Corporate language, "I'd be happy to help", no hook structure
generic_substack.txt	substack	FAIL	Generic assistant voice, "It's important to note", no personality
too_mean_medium.txt	medium	FAIL	Targets individual by name, punches down, violates safety boundaries
Run command: node persona/test/harness.js (no dependencies beyond Node.js built-ins + the persona module itself)

5) Concrete Artifacts List (Planned)
New Files
Path	Purpose	~Lines
persona/spec/mirror_pete.v1.json	Canonical persona spec	~150
persona/renderer.js	Two-pass renderer (deterministic + LLM)	~200
persona/validator.js	Hard-fail + scored checks	~250
persona/gate.js	Render→Validate→Retry loop	~80
persona/index.js	Barrel export	~10
persona/test/harness.js	Test runner	~100
persona/test/fixtures/passing_linkedin.txt	Passing fixture	~30
persona/test/fixtures/passing_substack.txt	Passing fixture	~50
persona/test/fixtures/generic_linkedin.txt	Failing fixture (generic)	~30
persona/test/fixtures/generic_substack.txt	Failing fixture (generic)	~50
persona/test/fixtures/too_mean_medium.txt	Failing fixture (safety)	~30
Modified Files
Path	Change	Scope
netlify/functions/generate.js	Add persona engine gate path (additive, ~40 lines added)	Moderate
src/App.jsx	Add usePersonaEngine + platform to request body (~5 lines changed)	Minimal
NOT Modified (Intentionally)
Path	Why
central-smart-ass/	V2 integration — after V1 gate is proven
satirical-pipeline/	Reference only — Python service, separate integration path
PLATFORM_PROMPTS in src/App.jsx	Kept as fallback; extracted INTO spec but not deleted
6) Execution Checklist for the Coding Agent

Phase 0: Scaffolding
[ ] Create persona/ directory structure (spec/, test/, test/fixtures/)
[ ] Create persona/index.js barrel export (empty, will populate)

Phase 1: Persona Spec
[ ] Read src/App.jsx — extract PLATFORM_PROMPTS tone compositions, banned phrases, structural requirements, safety rails
[ ] Read central-smart-ass/src/agents/MirrorUniversePeteAgent.js — extract word_replacements, cold_logic_prepends, sarcasm_parentheticals, styleConfig, pattern libraries (openings/transitions/closings exact strings)
[ ] Read satirical-pipeline/mcp-lmstudio/server.py — extract IDEATION_SYSTEM banned language, directives
[ ] Read satirical-pipeline/references/comedy-structures.md — extract joke formula names and roast structures
[ ] Read satirical-pipeline/references/archetypes.md — extract archetype categories for roast targets
[ ] Author persona/spec/mirror_pete.v1.json merging all sources
[ ] Verify: JSON is valid, all fields from schema are populated with real values (not placeholders)

Phase 2: Renderer
[ ] Implement persona/renderer.js with render() function
[ ] Implement deterministic pass: banned phrase removal, word replacements, structure skeleton
[ ] Implement LLM pass: build system prompt from spec, call OpenAI (same pattern as generate.js)
[ ] Support deterministicOnly mode (skip LLM pass)
[ ] Return metadata object with spec_version, render_passes, warnings
[ ] Export from persona/index.js

Phase 3: Validator
[ ] Implement persona/validator.js with validate() function
[ ] Implement hard-fail: banned phrases regex scan
[ ] Implement hard-fail: required structure check per platform adapter
[ ] Implement hard-fail: generic voice detection
[ ] Implement hard-fail: safety boundary check
[ ] Implement scored checks: voice_fit, motif_presence, structure_compliance, safety, genericness
[ ] Return structured result with pass/hardFailReasons/scores/suggestions
[ ] Export from persona/index.js

Phase 4: Gate
[ ] Implement persona/gate.js with gate() function
[ ] Implement render→validate→retry loop (max 2 retries)
[ ] Inject validator feedback into retry render calls
[ ] Return best candidate on failure after retries exhausted
[ ] Export from persona/index.js

Phase 5: Wire into Root App
[ ] Modify netlify/functions/generate.js — add persona engine import and gate path
[ ] Add usePersonaEngine flag check (backwards compatible)
[ ] Load spec via readFileSync + JSON.parse
[ ] Call gate() when usePersonaEngine=true
[ ] Return gate result with metadata
[ ] Modify src/App.jsx — add usePersonaEngine:true and platform to request body
[ ] Test: existing non-persona-engine requests still work (regression check)

Phase 6: Test Harness & Fixtures
[ ] Write 2 passing fixtures (passing_linkedin.txt, passing_substack.txt) — must contain Pete-isms, proper structure, no banned phrases
[ ] Write 3 failing fixtures (generic_linkedin.txt, generic_substack.txt, too_mean_medium.txt) — must trigger specific hard-fail reasons
[ ] Implement persona/test/harness.js — loads spec, runs validate on all 5 fixtures, prints results
[ ] Run harness: node persona/test/harness.js — verify 2 pass, 3 fail
[ ] Run deterministic render on 3 generic fixtures, re-validate — verify improvement

Final Verification
[ ] All new files use ES6 module syntax (import/export)
[ ] No secrets in any file (only env var names referenced: OPENAI_API_KEY)
[ ] persona/spec/mirror_pete.v1.json contains real extracted values, not placeholders
[ ] netlify/functions/generate.js still works without usePersonaEngine flag
[ ] node persona/test/harness.js runs and produces expected pass/fail results
Required Environment Variables
Variable	Where Used	Already Exists?
OPENAI_API_KEY	persona/renderer.js (LLM pass), netlify/functions/generate.js (existing)	Yes — already used by generate.js
No new secrets required.