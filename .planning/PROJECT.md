# Mirror Post (Krea-AI Workspace)

## What This Is

A satirical LinkedIn post compositor that takes a corporate archetype or scenario as input and outputs a complete visual post — AI-generated hero image via local FLUX diffusion + contextually intelligent text overlays + LinkedIn UI chrome — all driven by the Mirror Universe Pete voice. Built on top of flux-krea, a shipped and production-hardened local image generation pipeline for Apple Silicon.

## Core Value

The Post Brief is the product. A single structured JSON document that contains every element of a satirical LinkedIn post — character, headline, body, props with text, tweet embed, engagement metrics, and image seed — all satirically coherent because they're generated from one voice-aware pass. Everything downstream consumes the Post Brief.

## Requirements

### Validated

- ✓ Local FLUX image generation on Apple Silicon (MPS) with 3-tier optimization — existing (flux-krea)
- ✓ CLI + Gradio web UI for image generation — existing (flux-krea)
- ✓ Profiling and benchmarking tools for generation performance — existing (flux-krea)
- ✓ Config-driven pipeline with layered YAML loading — existing (flux-krea)
- ✓ Metal Performance Shaders optimization with automatic fallback — existing (flux-krea)
- ✓ Thermal management and memory monitoring — existing (flux-krea)
- ✓ 299 tests, 96% coverage, CI green (pytest + ruff + mypy) — existing (flux-krea)

### Active

#### Performance Stabilization (flux-krea)
- [ ] Baseline benchmarks on M4 Pro hardware
- [ ] Scheduler optimization (Euler/DPM++) to reduce steps from 28 to 20
- [ ] MPS memory tuning (watermark ratios, cleanup frequency)
- [ ] torch.compile experiment (feature-flagged)
- [ ] Neural Engine VAE decoder activation (feature-flagged)
- [ ] Structured prompt-file input (`--prompt-file` JSON contract)

#### Mirror Post — Scaffolding
- [ ] Project directory structure with GSD planning docs
- [ ] Import existing persona assets (mirror_pete.v2.json, archetypes, comedy structures, modifiers)
- [ ] Post Brief v1 JSON schema + 4 reference test fixtures

#### Mirror Post — Post Brief Generator
- [ ] Input classifier (archetype lookup, scenario detection, freeform)
- [ ] System prompt builder from persona spec + archetype + comedy structures
- [ ] LLM generation via Anthropic API (claude-sonnet-4)
- [ ] Brief validator (schema, safety, voice, completeness)
- [ ] Test harness against 4 reference fixtures

#### Mirror Post — LinkedIn Visual Grammar
- [ ] Prop taxonomy (mugs, whiteboards, desk items, cards, nameplates)
- [ ] Composition zones (LinkedIn header, profile bar, hero image sub-zones, engagement bar)
- [ ] Engagement metric generator (satirically calibrated)

#### Mirror Post — Image Prompt Engine
- [ ] Scene templates (office-executive, office-middle-mgmt, airport-hustle, call-center-floor)
- [ ] Prompt builder (Post Brief to local diffusion prompt with modifier selection)
- [ ] Composition-aware prompting (text-clear zones, no text in diffusion output)

#### Mirror Post — Compositor
- [ ] LinkedIn UI chrome template (header, profile bar, engagement footer)
- [ ] Text overlay renderer (headline with highlights, body with formatting)
- [ ] Tweet embed card renderer

#### Mirror Post — Artifact UI
- [ ] Input form with archetype browser
- [ ] Post Brief display with inline editing
- [ ] Image prompt output with composition guide
- [ ] Compositor preview (stretch)

### Out of Scope

- SaaS deployment / Netlify hosting — local creative tool first, deployment is future concern
- OAuth / user authentication — single-user tool on Pete's machine
- Content scheduling / posting to LinkedIn — compositor produces images, not API posts
- MidJourney integration — local FLUX diffusion is the render engine, no --ar or --v flags
- Mobile app — desktop-first (Claude Desktop artifact + local Node.js pipeline)
- Real-time collaboration — single-operator tool
- OpenAI/GPT backend — Anthropic API throughout, persona spec designed for Claude
- Pixel-perfect LinkedIn UI cloning — stylistic similarity for satirical context, not reproduction

## Context

**Two-project workspace:** flux-krea (Python, shipped) lives in `flux-krea/` as its own git repo. Mirror Post will be built in `mirror-post/` as a separate Node.js project. The outer `Krea-AI/` workspace orchestrates both.

**Existing assets to import:**
- `mirror_pete_v2.json` — Comprehensive persona spec (identity, voice sliders, 17+ archetypes, safety rails)
- Archetype library, comedy structures, ultra-fidelity modifiers — all from prior Persona Engine work
- 4 reference posts (Brent Vellum, Trevor B. x2, Pete C.) reverse-engineered as test fixtures

**Architecture:** Mirror Post generates a Post Brief (structured JSON) → Image Prompt Engine translates it to a diffusion prompt → flux-krea generates the hero image → Compositor overlays LinkedIn UI chrome + text + tweet embed. Systems are fully decoupled — flux-krea requires zero source code changes for Mirror Post (only a `--prompt-file` flag addition).

**Key design insight:** The diffusion model does NOT render text. All text on mugs, whiteboards, sticky notes, and overlays is handled by the compositor layer. The image prompt describes surfaces and objects that will receive text — not the text itself.

**Performance target:** Generation latency on M4 Pro needs to drop from 60-90 seconds to 30-45 seconds. Primary lever: scheduler optimization + step reduction (28 → 20 steps).

## Constraints

- **Platform**: macOS Apple Silicon (M-series) — MPS compute, not CUDA
- **Stack**: Python 3.10 for flux-krea, Node.js 20+ / JavaScript for Mirror Post
- **LLM**: Anthropic API (claude-sonnet-4) — no OpenAI, no local LLM
- **Image Generation**: Local FLUX diffusion via flux-krea — no cloud API, no MidJourney
- **Persona**: mirror_pete.v2.json is canonical and immutable without version bump
- **Safety**: Punch systems and archetypes, never individuals. No marginalized groups, mental health, genuine setbacks. Characters are fictional composites.
- **UI Aesthetic**: Obsidian dark-mode (deep navy, gold accents, cream backgrounds)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| flux-krea stays general-purpose; Mirror Post generates standard prompts | Keeps systems decoupled, flux-krea usable for non-Mirror-Post purposes | -- Pending |
| Two-stage image (diffusion + compositing) | Plays to diffusion strengths (photorealistic scenes), avoids weakness (text rendering) | -- Pending |
| Anthropic API throughout, no OpenAI | Persona spec designed for Claude, existing artifacts use Anthropic | -- Pending |
| Performance optimization before Mirror Post build | 60-90s generation is too slow for creative iteration; 30-45s target needed first | -- Pending |
| Post Brief as single canonical contract | Every downstream module consumes one structured JSON; prevents module coupling | -- Pending |
| No deployment in v1 | Local creative tool first; Pete runs it on his machine + Claude Desktop | -- Pending |
| Compositor handles all text rendering | Diffusion can't reliably render text on surfaces; programmatic overlay is reliable | -- Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-13 after initialization*
