# Krea-AI Workspace

## What This Is

Workspace for the Mirror Post project and its dependency, flux-krea. Mirror Post is a satirical LinkedIn post compositor. flux-krea is a local FLUX image generation pipeline for Apple Silicon. They connect via a structured JSON prompt-file contract.

## Architecture

```
Krea-AI/                    # This workspace (GSD-managed)
  .planning/                # GSD execution state
  Plans for Krea AI/        # Source plans (KNOWLEDGE_BASE, PLAN_01, PLAN_02)
  flux-krea/                # Image generation engine (own git repo, shipped)
  mirror-post/              # Satirical compositor (to be built)
```

**Pipeline:** User Input -> Post Brief (JSON) -> Image Prompt -> flux-krea -> hero.png -> Compositor -> final.png

## Sub-Projects

### flux-krea (Python, shipped)
- Status: Production-hardened, 299 tests, 96% coverage
- Location: `flux-krea/` (separate git repo)
- Stack: Python 3.10, PyTorch, diffusers, Apple MPS
- Next: Performance optimization (scheduler, MPS tuning, --prompt-file flag)

### mirror-post (JavaScript, to be built)
- Status: Planning complete, awaiting scaffolding
- Location: `mirror-post/` (to be created)
- Stack: Node.js 20+, React (Claude Desktop artifact), Anthropic API
- Modules: Post Brief Generator -> Visual Grammar -> Image Prompt Engine -> Compositor -> Artifact UI

## Key Files

| File | Purpose |
|------|---------|
| `.planning/PROJECT.md` | Project context and requirements |
| `.planning/ROADMAP.md` | Phase structure (when created) |
| `.planning/STATE.md` | Current execution state (when created) |
| `Plans for Krea AI/KNOWLEDGE_BASE.md` | Comprehensive analysis and integration strategy |
| `Plans for Krea AI/PLAN_01_SCAFFOLDING.md` | Mirror Post framework scaffolding spec |
| `Plans for Krea AI/PLAN_02_MODULES.md` | Mirror Post module specs (5 modules) |
| `tasks/lessons.md` | Session correction rules |

## Rules

- flux-krea source lives in its own repo — do not modify flux-krea code from this workspace without switching into that directory
- mirror_pete.v2.json is the canonical persona spec — never modify without version bump
- Post Brief JSON is the contract between all Mirror Post modules — never bypass it
- Image prompts target LOCAL diffusion (not MidJourney) — no --ar or --v flags
- Compositor handles all text rendering — diffusion prompts should NOT attempt text on surfaces
- Safety rails from persona spec are hard constraints, not suggestions
- Obsidian dark-mode aesthetic for all UI (deep navy, gold accents, cream backgrounds)

## Commands

See `~/.claude/CLAUDE.md` for global session commands and GSD workflow.

## GSD State

Check `.planning/STATE.md` for current phase. Check `tasks/todo.md` for work queue.

## Test & Coverage

| Sub-Project | Tests | Coverage | Status |
|-------------|-------|----------|--------|
| flux-krea | 299 | 96% | Green |
| mirror-post | 0 | 0% | Not yet built |

## Dependencies

- Python 3.10.13 (flux-krea)
- Node.js 20+ (mirror-post)
- Anthropic API key (mirror-post Post Brief generation)
- FLUX.1-Krea-dev model weights (flux-krea, ~12GB)
