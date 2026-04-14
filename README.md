# Krea-AI

Workspace for **Mirror Post** — a satirical LinkedIn post compositor that generates complete visual posts from corporate archetypes or scenarios. Powered by local FLUX image generation on Apple Silicon.

## How It Works

```
User describes a scenario     "HR manager who strips job descriptions"
         |
    Post Brief Generator      Anthropic API produces structured JSON with
         |                    character, headline, props, tweet, engagement
    Image Prompt Engine       Translates Post Brief to diffusion prompt
         |
    flux-krea                 Local FLUX model generates hero image on MPS
         |
    Compositor                Overlays LinkedIn chrome, text, tweet embed
         |
    Final Image               1920x1080 satirical LinkedIn post
```

## Projects

### flux-krea

Local FLUX image generation pipeline optimized for Apple Silicon (M-series). Production-hardened with 299 tests and 96% coverage.

- **Stack:** Python 3.10, PyTorch, diffusers, Metal Performance Shaders
- **Status:** Shipped. Performance optimization pending.
- **Location:** `flux-krea/`

### mirror-post

Satirical LinkedIn post compositor. Takes a corporate archetype or scenario and produces a complete visual post.

- **Stack:** Node.js 20+, React (Claude Desktop artifact), Anthropic API
- **Status:** Planning complete. Scaffolding phase next.
- **Location:** `mirror-post/` (to be created)

## Quick Start

### flux-krea

```bash
cd flux-krea
pip install -r requirements.txt
python main.py generate -p "a professional at a corporate desk, photorealistic"
python main.py benchmark --quick
```

### mirror-post (after build)

```bash
cd mirror-post
npm install
npm run brief -- "LinkedIn hustle bro who posts airport selfies"
```

## File Structure

```
Krea-AI/
  .planning/                  GSD execution state
    PROJECT.md                Project context and requirements
    codebase/                 Codebase mapping (7 analysis documents)
  Plans for Krea AI/          Source planning documents
    KNOWLEDGE_BASE.md         Integration strategy and analysis
    PLAN_01_SCAFFOLDING.md    Mirror Post framework spec
    PLAN_02_MODULES.md        Mirror Post module specs
  flux-krea/                  Image generation engine (separate repo)
  mirror-post/                Satirical compositor (to be built)
  tasks/
    lessons.md                Session correction rules
```

## Author

Pete Connor — AI transformation leader. Built by directing AI.

## Status

Phase 0 bootstrap in progress. GSD project initialized. Next: workflow configuration and roadmap creation.
