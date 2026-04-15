# Roadmap: Krea-AI Workspace

## Milestone 1: Mirror Post v1

Build the complete Mirror Post satirical LinkedIn compositor pipeline: from user input through Post Brief generation, image prompting, compositing, and artifact UI. flux-krea optimization runs as a parallel work stream.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3...7): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)
- Parallel work stream: flux-krea optimization (independent, non-blocking)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Mirror Post Scaffolding** - Directory structure, persona assets, Post Brief schema, test fixtures
- [x] **Phase 2: Post Brief Generator** - Input classifier, system prompt builder, LLM generation, Brief Validator
- [x] **Phase 3: Visual Grammar + Image Prompt Builder** - Foundation prompt asset, deterministic prompt builder, zone spec, engagement generator (completed 2026-04-14)
- [ ] **Phase 4: Compositor** - LinkedIn chrome, text overlay, tweet embed, engagement bar over hero image
- [ ] **Phase 5: Artifact UI** - Input form, brief display/edit, image prompt output
- [ ] **Phase 6: Integration** - End-to-end pipeline testing and 4-post roundtrip validation
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

### Phase 3: Visual Grammar + Image Prompt Builder
**Goal**: A Post Brief can be deterministically transformed into a Flux-compatible image prompt by prepending the static foundation prompt and appending character/scene/props delta — with zero LLM calls — and the compositor zone spec accurately describes the reference layout
**Depends on**: Phase 2
**Requirements**: REQ-M-020, REQ-M-021, REQ-M-022, REQ-M-023, REQ-M-030, REQ-M-031, REQ-M-032, REQ-M-033, REQ-M-034, REQ-M-035, REQ-X-060, REQ-X-061, REQ-X-062, REQ-X-063, REQ-X-064, REQ-X-065, REQ-X-066, REQ-X-070, REQ-X-071
**Patterns**: None (pure data/logic module, deterministic construction)
**Enforces**: REQ-X-002 (Post Brief contract), REQ-X-010 (no MidJourney flags), REQ-X-011 (diffusion prompts describe surfaces — but props with text are rendered IN the hero image by Flux), REQ-X-025 (no flux-krea source modifications), REQ-X-060 (3:2 horizontal), REQ-X-061 to REQ-X-066 (visual identity), REQ-X-070 (foundation prompt is a static asset, loaded once), REQ-X-071 (deterministic prompt construction, no LLM)
**Success Criteria** (what must be TRUE):
  1. Foundation prompt YAML at `mirror-post/src/config/foundation-prompt.yaml` is committed as a static asset and loaded exactly once per session (not per call)
  2. `image-prompt-builder.js` takes any valid Post Brief and outputs a Flux-compatible prompt that begins with `foundation.reusable_master_prompt` followed by character delta, props (with text rendered IN diffusion output), environment, and fidelity modifiers — fully deterministic, zero LLM calls
  3. Compositor zone spec describes the full-bleed 3:2 hero layout with: left ~40% gradient text zone (headline + body), LinkedIn chrome positions (top nav, profile bar, engagement bar), tweet embed card slot (lower-right), engagement bar at bottom — matching the 5 reference images
  4. Engagement generator produces satirically calibrated metrics (reaction counts, dominant reaction type, comment count) that vary by character type and post tone
  5. Brent Vellum Post Brief produces a coherent middle-management office prompt where every prompt starts with the foundation prefix and prop text (mug label, whiteboard content) is described as part of the diffusion output
**Plans**: 2 plans in 1 wave

Plans:
- [x] 03-01-PLAN.md — Foundation prompt loader, scene templates, and image prompt builder (Wave 1)
- [x] 03-02-PLAN.md — Compositor zone spec, prop taxonomy, and engagement generator (Wave 1)

### Phase 4: Compositor
**Goal**: A flux-krea hero image plus a Post Brief can be assembled into a final 1920x1080 PNG with LinkedIn UI chrome, left-side gradient text overlay, tweet embed card, and engagement metrics — chrome layout consistent across any hero image
**Depends on**: Phase 3
**Requirements**: REQ-M-040, REQ-M-041, REQ-M-042, REQ-M-043, REQ-M-044, REQ-X-050, REQ-X-051, REQ-X-052, REQ-X-060, REQ-X-061, REQ-X-062, REQ-X-063, REQ-X-064, REQ-X-065, REQ-X-066
**Patterns**: None specific
**Enforces**: REQ-X-027 (compositor renders headline, body, tweet card, engagement text — props text remains IN the hero image), REQ-X-026 (two-stage pipeline), REQ-X-050 (fixed input dimensions), REQ-X-051 (compositor template is fixed asset), REQ-X-052 (deterministic output)
**Success Criteria** (what must be TRUE):
  1. Sharp + node-canvas hybrid compositor takes a hero image + Post Brief and outputs a single 1920x1080 PNG
  2. LinkedIn chrome (top nav, profile bar with avatar/name/title, engagement bar with reactions and comment count) renders consistently regardless of the underlying hero image
  3. Headline + body text overlay on the left ~40% with semi-transparent dark gradient renders readably against any hero background
  4. Tweet embed card renders correctly as a white rounded card (lower-right) when present in the Post Brief, and is omitted cleanly when absent
  5. Output is byte-identical given identical Post Brief + identical hero image (REQ-X-052 pixel-comparison test)
**Plans**: 2 plans in 2 waves
**UI hint**: yes

Plans:
- [x] 04-01-PLAN.md — LinkedIn chrome template + sharp/node-canvas setup (Wave 1)
- [ ] 04-02-PLAN.md — Text overlay gradient zone + tweet embed card renderer (Wave 2)

### Phase 5: Artifact UI
**Goal**: Users can input a scenario or browse archetypes, generate and edit a Post Brief, and produce a copy-ready image prompt — all within a Claude Desktop React artifact with Obsidian dark-mode aesthetic
**Depends on**: Phase 3
**Requirements**: REQ-M-050, REQ-M-051, REQ-M-052, REQ-M-054, REQ-M-055
**Patterns**: None specific
**Enforces**: REQ-X-033 (Obsidian dark-mode aesthetic), REQ-X-043 (React artifact)
**Deferred**: REQ-M-053 (compositor preview in artifact — out of M1 scope)
**Success Criteria** (what must be TRUE):
  1. Input form accepts freeform text and provides an archetype browser grouped by domain
  2. Sliders adjust satirical intensity (1-5), character expression, environment type, and prop density
  3. Generated Post Brief displays with all fields editable in-place
  4. One-click "Generate Image Prompt" runs brief → image prompt and provides Copy buttons for the Flux-ready prompt
  5. Full input-to-prompt flow completes in under 20 seconds and the UI follows Obsidian dark-mode aesthetic (deep navy, gold accents, cream text)
**Plans**: TBD
**UI hint**: yes

Plans:
- [ ] 05-01: Input form + archetype browser + sliders
- [ ] 05-02: Brief display with inline editing + image prompt output

### Phase 6: Integration + End-to-End
**Goal**: The complete pipeline — input → Post Brief → image prompt → flux-krea → compositor → final PNG — works end-to-end and produces structurally matching outputs for all 4 reference posts plus a freeform original scenario
**Depends on**: Phase 4, Phase 5
**Requirements**: REQ-M-060, REQ-M-061, REQ-M-062
**Patterns**: All patterns validated end-to-end
**Enforces**: REQ-X-002 (Post Brief contract never bypassed), REQ-X-003 (prompt-file contract between Mirror Post and flux-krea)
**Success Criteria** (what must be TRUE):
  1. Pipeline test runs input → Post Brief → image prompt → flux-krea → compositor → final PNG without manual intervention
  2. All 4 reference post roundtrips (Brent Vellum, Trevor B. hustle, Trevor B. closer, Pete C. titles) produce structurally matching outputs
  3. A freeform original scenario produces a coherent end-to-end output with original character and props
  4. Edit flow works: generate brief → edit a field → regenerate prompt → updated prompt reflects the edit
  5. Compositor produces byte-identical output given identical Post Brief + identical hero image
**Plans**: TBD

Plans:
- [ ] 06-01: End-to-end pipeline wiring and 4-post roundtrip validation

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
| REQ-M-020 | Phase 3 | Visual Grammar + Image Prompt Builder |
| REQ-M-021 | Phase 3 | Visual Grammar + Image Prompt Builder |
| REQ-M-022 | Phase 3 | Visual Grammar + Image Prompt Builder |
| REQ-M-023 | Phase 3 | Visual Grammar + Image Prompt Builder |
| REQ-M-030 | Phase 3 | Visual Grammar + Image Prompt Builder |
| REQ-M-031 | Phase 3 | Visual Grammar + Image Prompt Builder |
| REQ-M-032 | Phase 3 | Visual Grammar + Image Prompt Builder |
| REQ-M-033 | Phase 3 | Visual Grammar + Image Prompt Builder |
| REQ-M-034 | Phase 3 | Visual Grammar + Image Prompt Builder |
| REQ-M-035 | Phase 3 | Visual Grammar + Image Prompt Builder |
| REQ-M-040 | Phase 4 | Compositor |
| REQ-M-041 | Phase 4 | Compositor |
| REQ-M-042 | Phase 4 | Compositor |
| REQ-M-043 | Phase 4 | Compositor |
| REQ-M-044 | Phase 4 | Compositor |
| REQ-M-050 | Phase 5 | Artifact UI |
| REQ-M-051 | Phase 5 | Artifact UI |
| REQ-M-052 | Phase 5 | Artifact UI |
| REQ-M-053 | DEFERRED | Artifact UI (stretch) |
| REQ-M-054 | Phase 5 | Artifact UI |
| REQ-M-055 | Phase 5 | Artifact UI |
| REQ-M-060 | Phase 6 | Integration |
| REQ-M-061 | Phase 6 | Integration |
| REQ-M-062 | Phase 6 | Integration |
| REQ-X-070 | Phase 3 | Foundation prompt is static asset, loaded once |
| REQ-X-071 | Phase 3 | Image prompt construction is deterministic, no LLM |
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
Phases 1-6 execute sequentially. Phase 5 (Artifact UI) may run in parallel with Phase 4 (Compositor) since both depend on Phase 3.
Parallel work stream executes independently in flux-krea/ repo.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Scaffolding | 3/3 | COMPLETE | 2026-04-14 |
| 2. Post Brief Generator | 8/8 | COMPLETE | 2026-04-14 |
| 3. Visual Grammar + Image Prompt Builder | 2/2 | Complete   | 2026-04-14 |
| 4. Compositor | 1/2 | In Progress|  |
| 5. Artifact UI | 0/2 | Not started | - |
| 6. Integration | 0/1 | Not started | - |
| P. flux-krea Optimization | 0/6 | Not started | - |
