# Roadmap: Krea-AI Workspace

## Milestone 1: Mirror Post v1

Build the complete Mirror Post satirical LinkedIn compositor pipeline: from user input through Post Brief generation, image prompting, compositing, and artifact UI. flux-krea optimization runs as a parallel work stream.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3...7): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)
- Parallel work stream: flux-krea optimization (independent, non-blocking)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Mirror Post Scaffolding** - Directory structure, persona assets, Post Brief schema, test fixtures
- [x] **Phase 2: Post Brief Generator** - Input classifier, system prompt builder, LLM generation, Brief Validator
- [ ] **Phase 3: LinkedIn Visual Grammar** - Prop taxonomy, composition zones, engagement generator
- [ ] **Phase 4: Image Prompt Engine** - Scene templates, prompt builder, modifier selection
- [ ] **Phase 5: Compositor** - LinkedIn chrome, text overlay, tweet embed rendering
- [ ] **Phase 6: Artifact UI** - Input form, brief display/edit, image prompt output
- [ ] **Phase 7: Integration** - End-to-end pipeline testing and validation
- [ ] **Parallel: flux-krea Optimization** - Scheduler, MPS tuning, torch.compile, --prompt-file

## Phase Details

### Phase 1: Mirror Post Scaffolding
**Goal**: A Claude Code session can cd into mirror-post/, read the project files, and know exactly what to build next -- all structure, data, schema, and fixtures in place with zero functional code
**Depends on**: Nothing (first phase)
**Requirements**: REQ-M-001, REQ-M-002, REQ-M-003, REQ-M-004, REQ-M-005, REQ-M-006
**Patterns**: None (structural phase)
**Enforces**: REQ-X-001 (persona spec imported as-is), REQ-X-002 (Post Brief schema defined as canonical contract)
**Success Criteria** (what must be TRUE):
  1. mirror-post/ directory exists with full structure per PLAN_01 Section 1 and all barrel exports resolve
  2. mirror_pete.v2.json is importable from src/persona/spec/ and archetype library, comedy structures, and modifier data are converted and importable as JSON
  3. Post Brief v1 JSON schema exists in src/brief/schema.js and validates all 4 reference test fixtures without error
  4. `node test/harness.js` runs without errors (schema validation only, no generation)
  5. package.json exists with Node.js 20+ target and minimal dependencies
**Plans**: TBD

Plans:
- [ ] 01-01: Directory structure, config, foundational files
- [ ] 01-02: Asset import and conversion (persona spec, archetypes, comedy structures, modifiers)
- [ ] 01-03: Post Brief schema definition and 4 reference test fixtures

### Phase 2: Post Brief Generator
**Goal**: Users can provide a corporate archetype or freeform scenario and receive a complete, validated Post Brief JSON containing every element of a satirical LinkedIn post
**Depends on**: Phase 1
**Requirements**: REQ-M-010, REQ-M-011, REQ-M-012, REQ-M-013, REQ-M-014, REQ-M-015, REQ-M-016, REQ-M-017, REQ-M-018
**Patterns**:
  - Pattern 5 (REQ-X-031) -- Persona identity + safety rails at system prompt END; archetype context + scenario as first user message
  - Pattern 2 (REQ-X-029) -- Brief Validator treats LLM JSON output as untrusted input; all checks are deterministic, never reference model reasoning
  - Pattern 12 (REQ-X-032) -- Archetype metadata (name, description, domain) loaded upfront; full definitions loaded only on selection
**Enforces**: REQ-X-020, REQ-X-021, REQ-X-022, REQ-X-023, REQ-X-024 (safety rails enforced by Brief Validator), REQ-X-028 (Anthropic API only)
**Success Criteria** (what must be TRUE):
  1. Input classifier correctly identifies archetypes by exact match, fuzzy match, and domain match from the library
  2. Freeform scenarios (e.g., "Cloud architect who only deploys to PowerPoint") produce original characters with coherent props, not archetype copies
  3. Generated Post Briefs pass all Brief Validator hard-fail checks: schema compliance, safety, banned phrases, prop completeness, engagement sanity
  4. All 4 reference fixture inputs produce structurally similar Post Briefs to their expected outputs
  5. Generation completes in under 60 seconds per brief with assembled system prompt under 8K tokens (revised from <15s per D-22: Opus 4 selected for quality over latency; observed p50 ~32s, p95 ~50s acceptable for satirical-quality requirements)
**Plans**: 8 plans in 4 waves

Plans:
- [ ] 02-01-PLAN.md — Schema optional-parameter audit + Structured Outputs decision gate (Wave 1)
- [ ] 02-02-PLAN.md — Persona voice distillate generation + version-assertion test (Wave 2)
- [ ] 02-03-PLAN.md — Input classifier with archetype/domain/freeform routing (Wave 2)
- [ ] 02-04-PLAN.md — Comedy structure selector with Pattern 12 lazy loading (Wave 2)
- [ ] 02-05-PLAN.md — System prompt builder with Pattern 5 positioning + cache markers (Wave 3)
- [ ] 02-06-PLAN.md — LLM generation client with Structured Outputs + fallback (Wave 3)
- [ ] 02-07-PLAN.md — Brief Validator extension with safety/voice/completeness checks (Wave 4)
- [ ] 02-08-PLAN.md — End-to-end integration test + human verification (Wave 4)

### Phase 3: LinkedIn Visual Grammar
**Goal**: The visual rules for satirical LinkedIn post construction are codified as importable data -- prop taxonomy, composition zones, and engagement generation -- ready for consumption by Image Prompt Engine and Compositor
**Depends on**: Phase 1
**Requirements**: REQ-M-020, REQ-M-021, REQ-M-022, REQ-M-023
**Patterns**: None (pure data/logic module)
**Enforces**: REQ-X-002 (grammar data consumed via Post Brief contract)
**Success Criteria** (what must be TRUE):
  1. Props JSON covers all prop types observed in the 4 reference outputs (mugs, whiteboards, desk items, business cards, nameplates) with placement options and text constraints
  2. Zones JSON accurately describes the 1920x1080 spatial layout: LinkedIn header, profile bar, hero image with sub-zones (headline overlay, body overlay, subject, tweet embed), and engagement bar
  3. Engagement generator produces satirically calibrated metrics (reaction counts, dominant reaction type, comment count) that vary by character type and post tone
  4. Module has zero external dependencies and makes zero LLM calls
**Plans**: TBD

Plans:
- [ ] 03-01: Prop taxonomy, composition zones, and engagement generator

### Phase 4: Image Prompt Engine
**Goal**: A Post Brief can be transformed into a complete local diffusion prompt with intelligent modifier selection, composition-aware framing, and explicit text-free surface descriptions
**Depends on**: Phase 2, Phase 3
**Requirements**: REQ-M-030, REQ-M-031, REQ-M-032, REQ-M-033, REQ-M-034, REQ-M-035, REQ-X-060, REQ-X-061, REQ-X-062, REQ-X-063, REQ-X-064, REQ-X-065, REQ-X-066
**Patterns**: None specific (consumes pattern-driven data from Phase 2)
**Enforces**: REQ-X-010 (no MidJourney flags -- prompt validator rejects --ar, --v syntax), REQ-X-011 (diffusion prompts describe surfaces only, no text content), REQ-X-027 (compositor owns all text rendering), REQ-X-025 (standard prompts, no flux-krea source modifications), REQ-X-060 (3:2 horizontal output), REQ-X-061 (hyperreal polished corporate-social aesthetic), REQ-X-062 (clean modular layout), REQ-X-063 (crisp sans-serif typography), REQ-X-064 (hero palette follows Post Brief), REQ-X-065 (deadpan satirical tone), REQ-X-066 (dynamic hero, fixed template)
**Success Criteria** (what must be TRUE):
  1. Scene templates exist for all 4 archetype environments (office-executive, office-middle-mgmt, airport-hustle, call-center-floor) with environment, subject, prop positions, and composition specs
  2. Prompt builder produces positive prompt, negative prompt, parameters, and composition_notes from any valid Post Brief
  3. Generated prompts leave the left 35-40% of frame clear for text overlays (text_clear_zone) and describe prop surfaces without text content
  4. Modifier selection pulls 15-20 modifiers per scene from the ultra-fidelity library, matched to scene type and mood
  5. Brent Vellum Post Brief produces a coherent office-middle-mgmt prompt with correct prop surface descriptions
**Plans**: TBD

Plans:
- [ ] 04-01: Scene templates for 4 archetype environments
- [ ] 04-02: Prompt builder and modifier selection logic

### Phase 5: Compositor
**Goal**: A diffusion-generated hero image plus a Post Brief can be assembled into a final 1920x1080 PNG with LinkedIn UI chrome, text overlays, tweet embed card, and engagement metrics
**Depends on**: Phase 3, Phase 4
**Requirements**: REQ-M-040, REQ-M-041, REQ-M-042, REQ-M-043, REQ-M-044, REQ-X-050, REQ-X-051, REQ-X-052, REQ-X-060, REQ-X-061, REQ-X-062, REQ-X-063, REQ-X-064, REQ-X-065, REQ-X-066
**Patterns**: None specific
**Enforces**: REQ-X-027 (ALL text rendering happens here -- headline, body, prop text, tweet, engagement), REQ-X-026 (two-stage pipeline: diffusion hero + compositor overlay), REQ-X-050 (fixed input dimensions from flux-krea), REQ-X-051 (compositor template is fixed asset), REQ-X-052 (deterministic output -- byte-identical given same inputs), REQ-X-060 (3:2 horizontal output), REQ-X-061 (hyperreal polished corporate-social aesthetic), REQ-X-062 (clean modular layout), REQ-X-063 (crisp sans-serif typography), REQ-X-064 (corporate blue/white/gray chrome), REQ-X-065 (deadpan satirical tone), REQ-X-066 (fixed template, dynamic hero + text)
**Success Criteria** (what must be TRUE):
  1. LinkedIn UI chrome renders recognizably (dark header bar, profile section with avatar/name/title, engagement footer) -- stylistically similar, not pixel-perfect, no LinkedIn logo
  2. Headline text renders with highlight words in accent color, body text renders with bold/italic formatting, hashtags render below body
  3. Tweet embed card renders as a white rounded card with author, handle, text, and hashtags in correct layout
  4. Output is a single PNG at 1920x1080 with all zones populated per the composition spec
  5. Engagement bar displays correct reaction icons, count, and comment count from the Post Brief
**Plans**: TBD
**UI hint**: yes

Plans:
- [ ] 05-01: LinkedIn UI chrome template and canvas setup
- [ ] 05-02: Text overlay renderer and tweet embed renderer

### Phase 6: Artifact UI
**Goal**: Users can input a scenario or browse archetypes, generate and edit a Post Brief, and produce a copy-ready image prompt -- all within a Claude Desktop artifact with Obsidian dark-mode aesthetic
**Depends on**: Phase 2, Phase 4
**Requirements**: REQ-M-050, REQ-M-051, REQ-M-052, REQ-M-054, REQ-M-055
**Patterns**: None specific
**Enforces**: REQ-X-033 (Obsidian dark-mode aesthetic: deep navy, gold accents, cream backgrounds), REQ-X-043 (React artifact)
**Deferred**: REQ-M-053 (Compositor preview in artifact -- marked DEFERRED, out of M1 scope)
**Success Criteria** (what must be TRUE):
  1. Input form accepts freeform text and provides an archetype browser displaying all archetypes grouped by domain with name, signature move, and tell
  2. Generated Post Brief displays with all fields editable in-place (character, post content, props, tweet embed, engagement)
  3. "Generate Image Prompt" produces a diffusion-ready prompt with composition guide showing zone layout, and "Copy" buttons work for prompt, negative prompt, and Brief JSON
  4. Full input-to-prompt flow completes in under 20 seconds
  5. UI follows Obsidian dark-mode aesthetic (deep navy background, gold accents, cream text)
**Plans**: TBD
**UI hint**: yes

Plans:
- [ ] 06-01: Input form with archetype browser
- [ ] 06-02: Post Brief display with inline editing
- [ ] 06-03: Image prompt output with composition guide

### Phase 7: Integration
**Goal**: The complete pipeline -- input to brief to prompt to hero image to composite -- works end-to-end and produces structurally matching outputs for all 4 reference posts
**Depends on**: Phase 5, Phase 6
**Requirements**: REQ-M-060, REQ-M-061, REQ-M-062
**Patterns**: All patterns validated end-to-end
**Enforces**: REQ-X-002 (Post Brief contract never bypassed across full pipeline), REQ-X-003 (prompt-file.json contract between Mirror Post and flux-krea)
**Success Criteria** (what must be TRUE):
  1. Input "HR manager who strips job descriptions" produces a complete composite image with Brent Vellum character, correct props, and LinkedIn chrome
  2. All 4 reference post roundtrips (Brent Vellum, Trevor B. hustle, Trevor B. closer, Pete C. titles) produce structurally matching outputs
  3. A freeform original scenario produces a coherent end-to-end output with original character
  4. Edit flow works: generate brief, edit mug text, regenerate prompt -- updated prompt reflects the edit
  5. Total pipeline completes in under 2 minutes end-to-end
  6. Compositor produces byte-identical output given identical Post Brief + identical hero image (REQ-X-052 pixel-comparison test)
**Plans**: TBD

Plans:
- [ ] 07-01: End-to-end pipeline wiring and 4-post roundtrip validation

### Parallel: flux-krea Optimization
**Goal**: Generation latency drops from 60-90 seconds to 30-45 seconds on M4 Pro, with --prompt-file flag enabling structured input from Mirror Post
**Depends on**: Nothing (independent work stream, runs in flux-krea/ repo)
**Does NOT block**: Phases 1-7 (Mirror Post proceeds regardless of optimization progress)
**Requirements**: REQ-F-010, REQ-F-011, REQ-F-012, REQ-F-013, REQ-F-014, REQ-F-015, REQ-F-016, REQ-F-017
**Patterns**:
  - Pattern 11 (REQ-X-030) -- torch.compile (REQ-F-014) and Neural Engine VAE (REQ-F-015) are compile-time gated via conditional import; when disabled, code paths are not imported, not just skipped by runtime if-check
**Enforces**: REQ-X-025 (flux-krea stays general-purpose), REQ-X-040 (Apple Silicon MPS), REQ-X-041 (Python 3.10)
**Success Criteria** (what must be TRUE):
  1. Baseline benchmarks captured on M4 Pro hardware with current 28-step configuration
  2. Scheduler optimization (Euler/DPM++) reduces steps from 28 to 20 with quality verification via same-seed comparison
  3. MPS memory tuning (watermark ratios, cleanup frequency, allocator policy) shows measurable improvement
  4. torch.compile and Neural Engine VAE are feature-flagged with compile-time gating (conditional import, not runtime if-check) and default to OFF
  5. --prompt-file flag accepts the JSON contract defined in Phase 1 for structured input from Mirror Post
  6. Generation latency achieves 30-45 second target on M4 Pro
**Plans**: TBD

Plans:
- [ ] P-01: Baseline benchmarks on M4 Pro
- [ ] P-02: Scheduler optimization and step reduction
- [ ] P-03: MPS memory tuning
- [ ] P-04: torch.compile experiment (feature-flagged)
- [ ] P-05: Neural Engine VAE decoder (feature-flagged)
- [ ] P-06: --prompt-file flag for Mirror Post integration

## Constraints (Hard -- Reject if Violated)

These constraints apply across ALL phases. Any phase output that violates these is rejected.

| Constraint | REQ-ID | Enforcement |
|------------|--------|-------------|
| Post Brief JSON is THE contract between all modules -- never bypass | REQ-X-002 | Schema validation at every module boundary |
| Diffusion prompts describe surfaces only -- compositor owns ALL text | REQ-X-011, REQ-X-027 | Prompt builder never includes text content for surfaces |
| Image prompts target local diffusion only -- no MidJourney syntax | REQ-X-010 | Prompt validator rejects --ar, --v flags |
| Safety rails are hard constraints -- punch systems, never individuals | REQ-X-020 to REQ-X-024 | Brief Validator deterministic checks |
| Persona spec is immutable without version bump | REQ-X-001 | Code review gate |
| Anthropic API throughout -- no OpenAI, no local LLM | REQ-X-028 | Dependency check |
| flux-krea stays general-purpose -- no source modifications for Mirror Post | REQ-X-025 | Separate repos, gitignore boundary |
| flux-krea optimization is parallel -- does NOT block mirror-post phases | Architecture | Independent work stream |

## Coverage

### Requirement Traceability

| Requirement | Phase | Category |
|-------------|-------|----------|
| REQ-M-001 | Phase 1 | Scaffolding |
| REQ-M-002 | Phase 1 | Scaffolding |
| REQ-M-003 | Phase 1 | Scaffolding |
| REQ-M-004 | Phase 1 | Scaffolding |
| REQ-M-005 | Phase 1 | Scaffolding |
| REQ-M-006 | Phase 1 | Scaffolding |
| REQ-M-010 | Phase 2 | Post Brief Generator |
| REQ-M-011 | Phase 2 | Post Brief Generator |
| REQ-M-012 | Phase 2 | Post Brief Generator |
| REQ-M-013 | Phase 2 | Post Brief Generator |
| REQ-M-014 | Phase 2 | Post Brief Generator |
| REQ-M-015 | Phase 2 | Post Brief Generator |
| REQ-M-016 | Phase 2 | Post Brief Generator |
| REQ-M-017 | Phase 2 | Post Brief Generator |
| REQ-M-018 | Phase 2 | Post Brief Generator |
| REQ-M-020 | Phase 3 | Visual Grammar |
| REQ-M-021 | Phase 3 | Visual Grammar |
| REQ-M-022 | Phase 3 | Visual Grammar |
| REQ-M-023 | Phase 3 | Visual Grammar |
| REQ-M-030 | Phase 4 | Image Prompt Engine |
| REQ-M-031 | Phase 4 | Image Prompt Engine |
| REQ-M-032 | Phase 4 | Image Prompt Engine |
| REQ-M-033 | Phase 4 | Image Prompt Engine |
| REQ-M-034 | Phase 4 | Image Prompt Engine |
| REQ-M-035 | Phase 4 | Image Prompt Engine |
| REQ-M-040 | Phase 5 | Compositor |
| REQ-M-041 | Phase 5 | Compositor |
| REQ-M-042 | Phase 5 | Compositor |
| REQ-M-043 | Phase 5 | Compositor |
| REQ-M-044 | Phase 5 | Compositor |
| REQ-M-050 | Phase 6 | Artifact UI |
| REQ-M-051 | Phase 6 | Artifact UI |
| REQ-M-052 | Phase 6 | Artifact UI |
| REQ-M-053 | DEFERRED | Artifact UI (stretch) |
| REQ-M-054 | Phase 6 | Artifact UI |
| REQ-M-055 | Phase 6 | Artifact UI |
| REQ-M-060 | Phase 7 | Integration |
| REQ-M-061 | Phase 7 | Integration |
| REQ-M-062 | Phase 7 | Integration |
| REQ-F-010 | Parallel | flux-krea Optimization |
| REQ-F-011 | Parallel | flux-krea Optimization |
| REQ-F-012 | Parallel | flux-krea Optimization |
| REQ-F-013 | Parallel | flux-krea Optimization |
| REQ-F-014 | Parallel | flux-krea Optimization |
| REQ-F-015 | Parallel | flux-krea Optimization |
| REQ-F-016 | Parallel | flux-krea Optimization |
| REQ-F-017 | Parallel | flux-krea Optimization |

**ACTIVE requirements mapped:** 40/40 (32 mirror-post ACTIVE + 8 flux-krea ACTIVE)
**DEFERRED:** 1 (REQ-M-053 -- compositor preview in artifact)
**VALIDATED (shipped):** 8 (REQ-F-001 to F-008 -- already satisfied by flux-krea)
**Cross-cutting constraints:** 22 (REQ-X-001 to X-043 -- enforced across phases, not assigned to individual phases)

## Progress

**Execution Order:**
Phases 1-7 execute sequentially (with Phase 3 potentially parallel to Phase 2).
Parallel work stream executes independently in flux-krea/ repo.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Scaffolding | 3/3 | COMPLETE | 2026-04-14 |
| 2. Post Brief Generator | 0/3 | Not started | - |
| 3. Visual Grammar | 0/1 | Not started | - |
| 4. Image Prompt Engine | 0/2 | Not started | - |
| 5. Compositor | 0/2 | Not started | - |
| 6. Artifact UI | 0/3 | Not started | - |
| 7. Integration | 0/1 | Not started | - |
| P. flux-krea Optimization | 0/6 | Not started | - |
