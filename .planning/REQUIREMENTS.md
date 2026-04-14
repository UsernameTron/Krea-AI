# Requirements — Krea-AI Workspace

Derived from: KNOWLEDGE_BASE.md, PLAN_01_SCAFFOLDING.md, PLAN_02_MODULES.md, PROJECT.md, state/pattern-context.md, CLAUDE.md.

---

## Notation

- **REQ-F-xxx**: flux-krea (validated/shipped)
- **REQ-M-xxx**: mirror-post (active, to be built)
- **REQ-X-xxx**: cross-cutting constraints (architectural, safety, patterns)
- **Status**: VALIDATED = shipped and tested | ACTIVE = to be built | DEFERRED = out of scope for M1

---

## 1. flux-krea — Validated / Shipped

These requirements are satisfied by the existing flux-krea codebase (299 tests, 96% coverage, CI green).

| REQ-ID | Requirement | Status | Evidence |
|--------|-------------|--------|----------|
| REQ-F-001 | Local FLUX.1 image generation on Apple Silicon via MPS | VALIDATED | pipeline.py, 299 tests |
| REQ-F-002 | CLI entry point with generate, profile, and benchmark subcommands | VALIDATED | main.py |
| REQ-F-003 | Gradio web UI for interactive image generation | VALIDATED | app.py |
| REQ-F-004 | Config-driven pipeline with layered YAML loading (yaml > env > CLI) | VALIDATED | config.py, config.yaml |
| REQ-F-005 | MPS optimization with automatic CPU fallback | VALIDATED | optimizers/metal.py |
| REQ-F-006 | Thermal management and memory monitoring | VALIDATED | optimizers/thermal.py |
| REQ-F-007 | Profiling and benchmarking tools for generation performance | VALIDATED | utils/profiler.py |
| REQ-F-008 | 3-tier optimization pipeline (MAXIMUM > STANDARD > NONE) | VALIDATED | optimizers/ |

### flux-krea — Active Optimization (Parallel Work Stream)

| REQ-ID | Requirement | Status | Source |
|--------|-------------|--------|--------|
| REQ-F-010 | Baseline benchmarks on M4 Pro hardware before any optimization | ACTIVE | KNOWLEDGE_BASE Part 3, Milestone 1 |
| REQ-F-011 | Scheduler optimization — support Euler/DPM++ schedulers via config field | ACTIVE | KNOWLEDGE_BASE Part 3, Milestone 2 |
| REQ-F-012 | Step count reduction from 28 to 20 with quality verification (same-seed comparison) | ACTIVE | KNOWLEDGE_BASE Part 3, Milestone 2 |
| REQ-F-013 | MPS memory tuning — watermark ratios, cleanup frequency, allocator policy | ACTIVE | KNOWLEDGE_BASE Part 3, Milestone 3 |
| REQ-F-014 | torch.compile experiment — feature-flagged, default off, compile-time gated per REQ-X-030 | ACTIVE | KNOWLEDGE_BASE Part 3, Milestone 4 |
| REQ-F-015 | Neural Engine VAE decoder — feature-flagged, default off, compile-time gated per REQ-X-030 | ACTIVE | KNOWLEDGE_BASE Part 3, Milestone 5 |
| REQ-F-016 | --prompt-file flag accepting JSON contract for structured input from Mirror Post | ACTIVE | KNOWLEDGE_BASE Part 3, Milestone 6 |
| REQ-F-017 | Generation latency target: 30-45 seconds on M4 Pro (from current 60-90s) | ACTIVE | PROJECT.md |

---

## 2. mirror-post — Active

### 2.1 Scaffolding

| REQ-ID | Requirement | Status | Source |
|--------|-------------|--------|--------|
| REQ-M-001 | mirror-post/ directory structure per PLAN_01 Section 1 | ACTIVE | PLAN_01 |
| REQ-M-002 | Import mirror_pete.v2.json as canonical persona spec (no modifications) | ACTIVE | PLAN_01 Section 4 |
| REQ-M-003 | Convert and import archetype library, comedy structures, ultra-fidelity modifiers | ACTIVE | PLAN_01 Section 4 |
| REQ-M-004 | Post Brief v1 JSON schema definition (src/brief/schema.js) | ACTIVE | PLAN_01 Section 5 |
| REQ-M-005 | 4 reference test fixtures (Brent Vellum, Trevor B. x2, Pete C.) with expected briefs | ACTIVE | PLAN_01 Section 6 |
| REQ-M-006 | package.json with minimal dependencies, Node.js 20+ | ACTIVE | PLAN_01 Section 3e |

### 2.2 Post Brief Generator (Module 1)

| REQ-ID | Requirement | Status | Source |
|--------|-------------|--------|--------|
| REQ-M-010 | Input classifier — archetype lookup (exact, fuzzy, domain), scenario detection, freeform | ACTIVE | PLAN_02 Module 1 |
| REQ-M-011 | System prompt builder assembling persona spec + archetype + comedy structures + schema | ACTIVE | PLAN_02 Module 1, Phase 1 |
| REQ-M-012 | Prompt assembly per Pattern 5: persona + safety at system prompt END, archetype + scenario as first user message (see REQ-X-031) | ACTIVE | PLAN_02 Module 1 + pattern-context.md |
| REQ-M-013 | LLM generation via Anthropic API (claude-opus-4-6) | ACTIVE | PLAN_02 Module 1, Phase 2 |
| REQ-M-014 | Brief Validator — schema, safety, voice, completeness checks independent of LLM reasoning (see REQ-X-029) | ACTIVE | PLAN_02 Module 1, Phase 3 |
| REQ-M-015 | Archetype metadata loaded upfront; full definitions loaded only on selection (see REQ-X-032) | ACTIVE | pattern-context.md Pattern 12 |
| REQ-M-016 | Assembled system prompt fits within 8K tokens | ACTIVE | PLAN_02 Module 1, Phase 1 |
| REQ-M-017 | Generation completes in < 15 seconds per brief | ACTIVE | PLAN_02 Module 1, Phase 2 |
| REQ-M-018 | Freeform scenarios produce original characters, not archetype copies | ACTIVE | PLAN_02 Module 1, Phase 2 |

### 2.3 LinkedIn Visual Grammar (Module 2)

| REQ-ID | Requirement | Status | Source |
|--------|-------------|--------|--------|
| REQ-M-020 | Prop taxonomy — mugs, whiteboards, desk items, business cards, nameplates | ACTIVE | PLAN_02 Module 2, Phase 1 |
| REQ-M-021 | Composition zones — LinkedIn header, profile bar, hero image sub-zones, engagement bar | ACTIVE | PLAN_02 Module 2, Phase 2 |
| REQ-M-022 | Engagement metric generator — satirically calibrated per character type and post tone | ACTIVE | PLAN_02 Module 2, Phase 3 |
| REQ-M-023 | Pure data/logic module — no LLM calls, no external dependencies | ACTIVE | PLAN_02 Module 2 |

### 2.4 Image Prompt Engine (Module 3)

| REQ-ID | Requirement | Status | Source |
|--------|-------------|--------|--------|
| REQ-M-030 | Scene templates for 4 archetypes: office-executive, office-middle-mgmt, airport-hustle, call-center-floor | ACTIVE | PLAN_02 Module 3, Phase 1 |
| REQ-M-031 | Prompt builder: Post Brief to local diffusion prompt with positive/negative/parameters | ACTIVE | PLAN_02 Module 3, Phase 2 |
| REQ-M-032 | Modifier selection: 15-20 modifiers per scene from ultra-fidelity library | ACTIVE | PLAN_02 Module 3, Phase 3 |
| REQ-M-033 | Composition-aware prompting: text_clear_zone (left 35-40%) kept empty for overlays | ACTIVE | PLAN_02 Module 3 |
| REQ-M-034 | Prompts describe prop SURFACES without text content — compositor handles all text (see REQ-X-027) | ACTIVE | PLAN_02 Module 3 |
| REQ-M-035 | Output includes composition_notes for manual adjustment | ACTIVE | PLAN_02 Module 3, Phase 2 |

### 2.5 Compositor (Module 4)

| REQ-ID | Requirement | Status | Source |
|--------|-------------|--------|--------|
| REQ-M-040 | LinkedIn UI chrome template — header, profile bar, engagement footer | ACTIVE | PLAN_02 Module 4, Phase 1 |
| REQ-M-041 | Text overlay renderer — headline with highlight words, body with bold/italic formatting | ACTIVE | PLAN_02 Module 4, Phase 2 |
| REQ-M-042 | Tweet embed card renderer — author, handle, text, hashtags | ACTIVE | PLAN_02 Module 4, Phase 3 |
| REQ-M-043 | Output: single PNG at 1920x1080 | ACTIVE | PLAN_02 Module 4 |
| REQ-M-044 | LinkedIn chrome is stylistically similar, not pixel-perfect — no LinkedIn logo, no exact color match | ACTIVE | PLAN_02 Module 4, Phase 1 |

### 2.6 Artifact UI (Module 5)

| REQ-ID | Requirement | Status | Source |
|--------|-------------|--------|--------|
| REQ-M-050 | Input form with freeform text and archetype browser | ACTIVE | PLAN_02 Module 5, Phase 1 |
| REQ-M-051 | Post Brief display with all fields editable in-place | ACTIVE | PLAN_02 Module 5, Phase 2 |
| REQ-M-052 | Image prompt output with copy-to-clipboard and composition guide | ACTIVE | PLAN_02 Module 5, Phase 3 |
| REQ-M-053 | Compositor preview in artifact via HTML Canvas (STRETCH) | DEFERRED | PLAN_02 Module 5, Phase 4 |
| REQ-M-054 | Obsidian dark-mode aesthetic (see REQ-X-033) | ACTIVE | PLAN_02 Module 5, CLAUDE.md |
| REQ-M-055 | Full input-to-prompt flow completes in < 20 seconds | ACTIVE | PLAN_02 Module 5 |

### 2.7 Integration

| REQ-ID | Requirement | Status | Source |
|--------|-------------|--------|--------|
| REQ-M-060 | End-to-end: input → brief → prompt → flux-krea → hero → composite | ACTIVE | PLAN_02 Cross-Module |
| REQ-M-061 | 4 reference post roundtrips produce structurally matching outputs | ACTIVE | PLAN_02 Cross-Module |
| REQ-M-062 | Total pipeline completes in < 2 minutes end-to-end | ACTIVE | PLAN_02 Cross-Module |

---

## 3. Cross-Cutting Constraints

These are hard constraints that apply across the entire workspace. Violations are blocking.

### 3.1 Data Contracts

| REQ-ID | Constraint | Enforcement | Source |
|--------|-----------|-------------|--------|
| REQ-X-001 | mirror_pete.v2.json is the canonical persona spec — never modify without version bump | Code review gate | CLAUDE.md Rules |
| REQ-X-002 | Post Brief JSON is THE contract between all Mirror Post modules — never bypass it | Schema validation at every module boundary | CLAUDE.md Rules, PLAN_01 Decision 3 |
| REQ-X-003 | prompt-file.json is the contract between Mirror Post and flux-krea | JSON schema validation | KNOWLEDGE_BASE Part 4 |

### 3.2 Image Generation Constraints

| REQ-ID | Constraint | Enforcement | Source |
|--------|-----------|-------------|--------|
| REQ-X-010 | Image prompts target LOCAL diffusion only — no --ar, --v, or MidJourney syntax | Prompt validator rejects MidJourney flags | CLAUDE.md Rules |
| REQ-X-011 | Diffusion model does NOT render text — all text on surfaces handled by compositor | Prompt builder must not include text content for surfaces | PLAN_02 Module 3, KNOWLEDGE_BASE Part 4 |

### 3.3 Safety

| REQ-ID | Constraint | Enforcement | Source |
|--------|-----------|-------------|--------|
| REQ-X-020 | Punch systems, patterns, and archetypes — never individuals | Brief Validator safety check | mirror_pete.v2.json safety rails |
| REQ-X-021 | No content targeting marginalized groups, mental health, genuine setbacks | Brief Validator safety check | mirror_pete.v2.json safety rails |
| REQ-X-022 | Characters are FICTIONAL composites, not real people | Brief Validator name check | mirror_pete.v2.json safety rails |
| REQ-X-023 | Satire must contain insight — if removing humor leaves no point, rewrite | Brief Validator scored check | mirror_pete.v2.json safety rails |
| REQ-X-024 | Safety rails from persona spec are HARD constraints, not suggestions | All rails enforced in validator, not advisory | CLAUDE.md Rules |

### 3.4 Architecture

| REQ-ID | Constraint | Enforcement | Source |
|--------|-----------|-------------|--------|
| REQ-X-025 | flux-krea stays general-purpose — Mirror Post generates standard prompts, no flux-krea source modifications | Separate repos, gitignore boundary | KNOWLEDGE_BASE Part 2 |
| REQ-X-026 | Two-stage image pipeline: diffusion generates hero, compositor overlays text/chrome | Module separation (Modules 3 + 4) | PLAN_01 Decision 4 |
| REQ-X-027 | Compositor owns ALL text rendering on the final image | Diffusion prompts describe surfaces only | REQ-X-011, PLAN_02 Module 3 |
| REQ-X-028 | Anthropic API throughout — no OpenAI, no local LLM | Code review, dependency check | PLAN_01 Decision 6, PROJECT.md |

### 3.5 Pattern-Derived (from KB v2.1)

| REQ-ID | Pattern | Constraint | Enforcement | Source |
|--------|---------|-----------|-------------|--------|
| REQ-X-029 | Pattern 2: Zero-Trust on Model Output | Brief Validator must enforce schema, safety, and voice constraints independently of LLM reasoning — treat LLM output as untrusted input | Validator checks are deterministic, not prompt-based; no validator logic references model reasoning or "the model said X" | pattern-context.md |
| REQ-X-030 | Pattern 11: Feature Flags as Security Perimeters | flux-krea torch.compile (REQ-F-014) and Neural Engine VAE (REQ-F-015) are compile-time gated — when disabled, code paths do not execute, not just skipped by runtime if-check | Config flag evaluated at pipeline load time, gated code not imported when disabled | pattern-context.md |
| REQ-X-031 | Pattern 5: Dual-Position Context Injection | Persona identity + safety rails placed at system prompt END (high attention). Archetype context + scenario placed as first user message (conversational frame) | System prompt builder enforces position — no mixing positions | pattern-context.md |
| REQ-X-032 | Pattern 12: Lazy Prompt Loading | Archetype metadata (name, description, domain) loaded upfront. Full archetype definition, comedy structures, and modifiers loaded only when that archetype is selected for a Post Brief | Prompt builder reads full definitions at generation time, not at module import | pattern-context.md |
| REQ-X-033 | N/A (aesthetic) | Obsidian dark-mode aesthetic for Artifact UI — deep navy, gold accents, cream backgrounds | UI component styles | CLAUDE.md, PROJECT.md |

### 3.6 Platform

| REQ-ID | Constraint | Source |
|--------|-----------|--------|
| REQ-X-040 | macOS Apple Silicon (M-series) — MPS compute, not CUDA | PROJECT.md |
| REQ-X-041 | Python 3.10 for flux-krea | PROJECT.md |
| REQ-X-042 | Node.js 20+ for mirror-post | PROJECT.md |
| REQ-X-043 | React for Artifact UI (Claude Desktop artifact) | PLAN_01 Decision 1 |

### 3.7 Compositor Consistency

| REQ-ID | Constraint | Enforcement | Source |
|--------|-----------|-------------|--------|
| REQ-X-050 | flux-krea output dimensions MUST be exactly 1920x1080 or a defined hero zone dimension. Compositor depends on fixed input dimensions. | Pipeline config validation, compositor input check | PROJECT.md, PLAN_02 Module 4 |
| REQ-X-051 | Compositor template (LinkedIn chrome) is a FIXED asset — PNG/SVG overlay with text injection points. It is never generated, only applied. | Template stored as static asset, not generated per-run | PLAN_02 Module 4 |
| REQ-X-052 | Compositor must produce byte-identical output given identical Post Brief + identical hero image. Pixel-comparison tests required. | Deterministic rendering tests in integration suite | Integration testing |

### 3.8 Visual Identity Spec

Hard constraints, not suggestions.

| REQ-ID | Constraint | Enforcement | Source |
|--------|-----------|-------------|--------|
| REQ-X-060 | Output aspect ratio is 3:2 horizontal. No square crop, no portrait framing of the final composite. | Compositor output dimension check | Visual identity spec |
| REQ-X-061 | Visual language is hyperreal polished corporate-social design — premium professional-networking-platform aesthetic with editorial-commercial finish and serious executive polish. | Visual review at verification | Visual identity spec |
| REQ-X-062 | Desktop-first composition with clean modular layout. No clutter, no meme chaos, no cartoon parody. | Visual review at verification | Visual identity spec |
| REQ-X-063 | Typography is crisp sans-serif with sparse high-impact text behavior. Text is restrained — never dense, never decorative. | Compositor font/layout config | Visual identity spec |
| REQ-X-064 | Color palette for the compositor chrome is corporate blue, white, and cool gray. Hero image palette is unconstrained — it follows the Post Brief context. | Compositor style constants | Visual identity spec |
| REQ-X-065 | Tone is deadpan satirical business aesthetic with restrained corporate absurdity. The humor lives in the content and props, not the visual chrome. | Brief Validator tone check, visual review | Visual identity spec |
| REQ-X-066 | Hero image content (characters, props, environment, scene) is fully dynamic — driven by the Post Brief context each generation. What is FIXED across every output is the compositor template: LinkedIn chrome, zone layout (text left / hero right), typography style, engagement bar, and overall composition. The template never changes. Only the hero image and injected text content change. | Compositor architecture (static template + dynamic injection) | Visual identity spec |

---

## Summary

| Category | Count | Status Breakdown |
|----------|-------|-----------------|
| flux-krea validated (REQ-F-001 to F-008) | 8 | 8 VALIDATED |
| flux-krea active optimization (REQ-F-010 to F-017) | 8 | 8 ACTIVE |
| mirror-post (REQ-M-001 to M-062) | 33 | 32 ACTIVE, 1 DEFERRED |
| cross-cutting (REQ-X-001 to X-066) | 32 | 32 constraints (always enforced) |
| **Total** | **81** | |

### Ambiguity Flags

| REQ-ID | Issue | Resolution Needed |
|--------|-------|-------------------|
| REQ-M-053 | Compositor preview marked STRETCH/DEFERRED — scope unclear | Confirm whether this is in M1 scope or deferred to M2 |
| REQ-F-014 / REQ-F-015 | "Compile-time gated" in Python context — Pattern 11 is from TypeScript. Need to define Python equivalent (conditional import vs if-check) | Define what "compile-time gated" means in Python: conditional import at module level, not runtime if-check in hot path |
| REQ-M-044 | "Stylistically similar, not pixel-perfect" — degree of LinkedIn fidelity is subjective | Accept as-is; Pete will judge visually during verification |

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REQ-F-001 | N/A | VALIDATED |
| REQ-F-002 | N/A | VALIDATED |
| REQ-F-003 | N/A | VALIDATED |
| REQ-F-004 | N/A | VALIDATED |
| REQ-F-005 | N/A | VALIDATED |
| REQ-F-006 | N/A | VALIDATED |
| REQ-F-007 | N/A | VALIDATED |
| REQ-F-008 | N/A | VALIDATED |
| REQ-F-010 | Parallel | Pending |
| REQ-F-011 | Parallel | Pending |
| REQ-F-012 | Parallel | Pending |
| REQ-F-013 | Parallel | Pending |
| REQ-F-014 | Parallel | Pending |
| REQ-F-015 | Parallel | Pending |
| REQ-F-016 | Parallel | Pending |
| REQ-F-017 | Parallel | Pending |
| REQ-M-001 | Phase 1 | Pending |
| REQ-M-002 | Phase 1 | Pending |
| REQ-M-003 | Phase 1 | Pending |
| REQ-M-004 | Phase 1 | Pending |
| REQ-M-005 | Phase 1 | Pending |
| REQ-M-006 | Phase 1 | Pending |
| REQ-M-010 | Phase 2 | Pending |
| REQ-M-011 | Phase 2 | Pending |
| REQ-M-012 | Phase 2 | Pending |
| REQ-M-013 | Phase 2 | Pending |
| REQ-M-014 | Phase 2 | Pending |
| REQ-M-015 | Phase 2 | Pending |
| REQ-M-016 | Phase 2 | Pending |
| REQ-M-017 | Phase 2 | Pending |
| REQ-M-018 | Phase 2 | Pending |
| REQ-M-020 | Phase 3 | Pending |
| REQ-M-021 | Phase 3 | Pending |
| REQ-M-022 | Phase 3 | Pending |
| REQ-M-023 | Phase 3 | Pending |
| REQ-M-030 | Phase 4 | Pending |
| REQ-M-031 | Phase 4 | Pending |
| REQ-M-032 | Phase 4 | Pending |
| REQ-M-033 | Phase 4 | Pending |
| REQ-M-034 | Phase 4 | Pending |
| REQ-M-035 | Phase 4 | Pending |
| REQ-M-040 | Phase 5 | Pending |
| REQ-M-041 | Phase 5 | Pending |
| REQ-M-042 | Phase 5 | Pending |
| REQ-M-043 | Phase 5 | Pending |
| REQ-M-044 | Phase 5 | Pending |
| REQ-M-050 | Phase 6 | Pending |
| REQ-M-051 | Phase 6 | Pending |
| REQ-M-052 | Phase 6 | Pending |
| REQ-M-053 | DEFERRED | Deferred |
| REQ-M-054 | Phase 6 | Pending |
| REQ-M-055 | Phase 6 | Pending |
| REQ-M-060 | Phase 7 | Pending |
| REQ-M-061 | Phase 7 | Pending |
| REQ-M-062 | Phase 7 | Pending |

---

*81 requirements. Every pattern has a home. Every constraint has enforcement.*
