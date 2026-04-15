# Roadmap: Krea-AI Workspace

## Milestone 1: Mirror Post v1 (Prompt Artifact Scope — Reset 2)

**Product Reset 2 (2026-04-15 evening):** Scope collapsed from full compositor pipeline to a single Claude Desktop React artifact that generates filled-in image prompts. See `.planning/PRODUCT-RESET-2.md` for the full rationale. Prior Phase 4–7 work archived on branches in `mirror-post` (`archive/phase-4-programmatic-chrome`, `archive/phase-4.2-path-c-abandoned`).

## Active Phases

- [x] **Phase 1: Mirror Post Scaffolding** — Directory structure, persona assets, Post Brief schema, test fixtures (completed 2026-04-14)
- [x] **Phase 2: Post Brief Generator** — Input classifier, system prompt builder, LLM generation, Brief Validator (completed 2026-04-14)
- [ ] **Phase 3: Prompt Artifact** — Single-file Claude Desktop React artifact. Supersedes all prior Phase 3–7 scope.
- [ ] **Parallel: flux-krea Optimization** — Independent work stream, non-blocking

### Phase 3: Prompt Artifact

**Goal:** A Claude Desktop React artifact lets a user enter an idea, adjust voice sliders, select an archetype (or auto-assign), and receive a ready-to-paste image prompt in under 5 seconds. The app ends at prompt text — the user pastes into whatever image tool they want.

**Depends on:** Phase 2 Post Brief Generator (logic may be ported into the artifact or called externally), persona spec, archetype library, comedy structures, variant templates at `.planning/phases/05-image-prompt-engine/templates/`.

**Build location:** Claude Desktop chat with artifact support. **NOT** Claude Code `/gsd:execute-phase`.

**Scope:**
1. Input form: idea textarea; four voice sliders (sarcasm, cynicism, warmth, satirical_intensity); archetype dropdown (17 archetypes + "auto-assign").
2. Generate button calls `window.claude.complete` with persona distillate + archetype + slider values + idea → returns Post Brief JSON.
3. Archetype-to-variant mapping (A/B/C/D) via the D-NEW-05 table.
4. Load the corresponding variant template from `05-image-prompt-engine/templates/`, substitute `{{placeholders}}` with Post Brief fields.
5. Display: filled prompt text with Copy button, Post Brief JSON preview, archetype match confidence.

**Success Criteria:**
1. User can enter an idea, adjust sliders, and see a filled image prompt in under 5 seconds.
2. All 17 archetypes map to one of 4 variant templates without gaps.
3. Copy-to-clipboard works for the filled prompt text.
4. Archetype auto-assign produces a sensible match (visual confidence indicator).
5. Obsidian dark-mode aesthetic (deep navy, gold accents, cream text).

**Finished state:** React artifact committed to `mirror-post/artifact/MirrorPoster.jsx`.

**Plans:** None in Claude Code. Build happens interactively in Claude Desktop.

### Parallel: flux-krea Optimization

Unchanged from prior roadmap. Independent work stream in the `flux-krea/` repo. Does not block Phase 3.

**Plans:**
- [ ] P-01: Baseline benchmarks on M4 Pro
- [ ] P-02: Scheduler optimization and step reduction
- [ ] P-03: MPS memory tuning
- [ ] P-04: torch.compile experiment (feature-flagged)
- [ ] P-05: Neural Engine VAE decoder (feature-flagged)
- [ ] P-06: --prompt-file flag for Mirror Post integration

## Deferred / Abandoned (Historical Record)

Preserved for audit. Do not execute. Branches exist for code recovery.

| Phase | Prior Scope | Disposition | Audit Trail |
|---|---|---|---|
| Phase 3 (prior) | Visual Grammar + Image Prompt Builder (deterministic builder, foundation prompt YAML, zone spec, engagement generator) | Shipped but superseded — artifact uses variant templates directly, deterministic builder logic not called | `mirror-post` main |
| Phase 4 (prior) | Compositor (programmatic LinkedIn chrome via sharp + canvas) | Abandoned — architecture spiral | `archive/phase-4-programmatic-chrome` (mirror-post, SHA c25d88a) |
| Phase 4.2 | Compositor reset — Path 2 variant PNGs / Path C HTML template + FLUX scene-only | Abandoned — scope reset to artifact-only | `archive/phase-4.2-path-c-abandoned` (mirror-post) + plans in `.planning/phases/04-compositor/` |
| Phase 5 (prior) | Scene Asset Library (operator-authored FLUX scene PNGs) | Abandoned — compositor eliminated | `.planning/phases/05-image-prompt-engine/` (templates survive and are consumed by Phase 3) |
| Phase 6 (prior) | Artifact UI (full React app with brief editor, prompt output, etc.) | Collapsed into new Phase 3 | — |
| Phase 7 (prior) | Integration + End-to-End (pipeline wiring, 4-post roundtrip) | Absorbed — the artifact IS the integration | — |

## Constraints (Hard — Reject if Violated)

| Constraint | REQ-ID | Enforcement |
|---|---|---|
| Post Brief JSON is THE contract between all modules — never bypass | REQ-X-002 | Schema validation at module boundary |
| Persona spec is immutable without version bump | REQ-X-001 | Code review gate |
| Anthropic API throughout — no OpenAI, no local LLM | REQ-X-028 | Dependency check |
| Safety rails are hard constraints — punch systems, never individuals | REQ-X-020 to REQ-X-024 | Brief Validator deterministic checks |
| flux-krea stays general-purpose — no Mirror-Post-specific source modifications | REQ-X-025 | Separate repos, gitignore boundary |

## Progress

| Phase | Status | Completed |
|---|---|---|
| 1. Scaffolding | COMPLETE | 2026-04-14 |
| 2. Post Brief Generator | COMPLETE | 2026-04-14 |
| 3. Prompt Artifact | NOT STARTED (Claude Desktop build) | — |
| P. flux-krea Optimization | NOT STARTED | — |

**Active requirements mapped:** TBD — requirement trace will be reconciled against the artifact scope once it ships. Compositor/chrome/scene requirements (REQ-M-040..044, REQ-X-052a, REQ-X-052b) are retired alongside the compositor work.
