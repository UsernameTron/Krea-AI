# PLAN 01 — SCAFFOLDING: Mirror Post Framework

## Project Identity

**Product Name:** Mirror Post
**Tagline:** Corporate satire, rendered.
**What it is:** A satirical LinkedIn post compositor that takes a corporate archetype or scenario as input and outputs a complete visual post — AI-generated hero image + contextually intelligent text overlays + LinkedIn UI chrome — all driven by the Mirror Universe Pete voice.

**What it is NOT:**
- Not the Persona Engine monorepo (that was a content factory for written-only output)
- Not a MidJourney prompt generator (local diffusion is the render engine)
- Not a generic meme maker (the satirical intelligence IS the product)

---

## GSD Phase: DISCUSS → PLAN

This document covers the scaffolding — directory structure, dependency decisions, configuration, and the foundational files that every module will import from. No functional code ships in this phase. The goal is: after scaffolding, a Claude Code session can `cd` into the project, read `.planning/`, and know exactly what to build next.

---

## 1. Directory Structure

```
mirror-post/
├── .planning/                          # GSD state management
│   ├── PROJECT.md                      # Project brief (this plan, condensed)
│   ├── ROADMAP.md                      # Milestones → Plan 02
│   ├── STATE.md                        # Current phase/milestone
│   └── phases/                         # Phase-specific specs
│       ├── 01-scaffolding/
│       │   └── spec.md
│       ├── 02-post-brief-generator/
│       │   └── spec.md
│       ├── 03-image-prompt-engine/
│       │   └── spec.md
│       ├── 04-compositor/
│       │   └── spec.md
│       └── 05-artifact-ui/
│           └── spec.md
│
├── CLAUDE.md                           # Claude Code instructions
├── tasks/
│   ├── todo.md                         # Current work queue
│   └── lessons.md                      # Corrections and learnings
│
├── src/
│   ├── index.js                        # Barrel export
│   │
│   ├── persona/                        # Layer 1: Voice engine
│   │   ├── spec/
│   │   │   └── mirror_pete.v2.json     # Canonical persona spec (from existing)
│   │   ├── archetypes/
│   │   │   └── library.json            # Enterprise archetype library (from existing)
│   │   ├── comedy/
│   │   │   └── structures.json         # Comedy formulas + roast structures (from existing)
│   │   └── index.js                    # Persona module exports
│   │
│   ├── brief/                          # Layer 2: Post Brief Generator
│   │   ├── generator.js                # Input → Post Brief (structured JSON)
│   │   ├── schema.js                   # Post Brief JSON schema + validation
│   │   ├── prompts/                    # System prompts for brief generation
│   │   │   └── brief-system.md         # LLM system prompt built from persona spec
│   │   └── index.js
│   │
│   ├── image/                          # Layer 3: Image Prompt Engine
│   │   ├── prompt-builder.js           # Post Brief → local diffusion prompt
│   │   ├── modifiers/
│   │   │   ├── ultra-fidelity.json     # From existing modifier library
│   │   │   └── composition-zones.json  # LinkedIn-specific spatial rules
│   │   ├── templates/                  # Scene templates per archetype category
│   │   │   ├── office-executive.json
│   │   │   ├── office-middle-mgmt.json
│   │   │   ├── airport-hustle.json
│   │   │   └── call-center-floor.json
│   │   └── index.js
│   │
│   ├── compositor/                     # Layer 4: LinkedIn UI Compositor
│   │   ├── renderer.js                 # Assembles final image (canvas-based)
│   │   ├── templates/
│   │   │   ├── linkedin-post.json      # UI chrome layout spec
│   │   │   └── tweet-embed.json        # Inset tweet card layout
│   │   ├── typography/
│   │   │   └── styles.json             # Font specs for each text zone
│   │   └── index.js
│   │
│   ├── grammar/                        # LinkedIn Visual Grammar (NEW)
│   │   ├── props.json                  # Prop taxonomy (mugs, whiteboards, folders, etc.)
│   │   ├── zones.json                  # Composition zones (hero, overlay, tweet, metrics)
│   │   ├── engagement.json             # Reaction/comment/share number generators
│   │   └── index.js
│   │
│   └── utils/
│       ├── llm.js                      # Anthropic API wrapper
│       └── validators.js               # Shared validation utilities
│
├── artifact/                           # Claude Desktop artifact (React)
│   └── mirror-post-generator.jsx       # The UI artifact
│
├── test/
│   ├── fixtures/
│   │   ├── input-brent-vellum.json     # Reference: Brent Vellum post
│   │   ├── input-trevor-hustle.json    # Reference: Trevor B. hustle post
│   │   ├── input-trevor-closer.json    # Reference: Trevor B. closer post
│   │   ├── input-pete-titles.json      # Reference: Pete C. titles post
│   │   └── expected-briefs/            # Expected Post Brief outputs
│   │       ├── brent-vellum.json
│   │       ├── trevor-hustle.json
│   │       ├── trevor-closer.json
│   │       └── pete-titles.json
│   └── harness.js                      # Test runner
│
├── package.json
└── README.md
```

---

## 2. Key Architectural Decisions

### Decision 1: Runtime Environment → Claude Artifact (React) + Local Node.js

**Why:** The primary UI is a Claude Desktop artifact (React component using Anthropic API for generation). The image prompt engine and compositor run locally via Node.js (Pete's machine). The artifact handles Layers 1-2 (persona + brief generation). Local scripts handle Layers 3-4 (image prompting + compositing).

**Implication:** The artifact is self-contained — it bundles the persona spec, archetype library, and comedy structures as inline data. The Node.js modules are for local pipeline execution.

### Decision 2: Persona Spec → Import existing `mirror_pete_v2.json` as-is

**Why:** The v2 spec is comprehensive (identity, philosophy, voice sliders, intellectual range, 17+ archetypes, platform adapters, safety rails, anti-patterns, consistency rules, evolution rules, pattern libraries). No reason to rewrite. Import and reference.

**Change from existing plans:** The Persona Engine Plan wanted to extract and merge from 3 scattered sources. That work is ALREADY DONE — `mirror_pete_v2.json` IS the merged spec. Skip the extraction phase entirely.

### Decision 3: Post Brief Schema → The new canonical artifact

**Why:** This is the thing that doesn't exist yet. The Post Brief is a structured JSON document that contains EVERY element of a satirical LinkedIn post — persona metadata, headline, body copy, prop list with text content, tweet embed text, engagement metrics, AND the image prompt seed. Everything downstream consumes the Post Brief. This is the product's beating heart.

### Decision 4: Image Generation → Two-stage (diffusion + compositing)

**Why:** Local diffusion generates the base hero image (person at desk with physical props). A separate compositor overlays LinkedIn UI chrome, text, tweet embed, and engagement metrics. This matches Pete's current manual workflow and plays to diffusion's strengths (photorealistic scenes) while avoiding its weaknesses (reliable text rendering on arbitrary surfaces).

**Implication:** The image prompt engine focuses on scene composition and prop placement. Text on mugs/whiteboards/sticky notes is handled by the compositor layer, not the diffusion model. This is the key insight from Pete's existing outputs — the text elements are overlaid, not generated.

### Decision 5: No Netlify/deployment in v1

**Why:** This is a local creative tool first. Pete runs it on his machine. The artifact runs in Claude Desktop. There's no server, no database, no auth. Deployment is a Phase 2 concern if/when this becomes a product others use.

### Decision 6: LLM Backend → Anthropic API (claude-sonnet-4-20250514)

**Why:** The existing `pete-content-generator.jsx` already calls Anthropic. The persona spec was designed for Claude. Stay consistent.

**Change from existing plans:** The Persona Engine Plan referenced OpenAI/gpt-4o because the Netlify function used it. We're not using that Netlify function. Anthropic throughout.

---

## 3. Foundational Files to Create

### 3a. `.planning/PROJECT.md`

Condensed project brief. Contains:
- Product name, tagline, one-paragraph description
- 5-layer architecture summary (one sentence each)
- Success criteria: "Can reproduce the Brent Vellum, Trevor B., and Pete C. posts from a single input each"
- Non-goals: "Not a SaaS platform. Not a content scheduler. Not a brand management tool."

### 3b. `.planning/ROADMAP.md`

```
Milestone 1: Scaffolding (this plan)
  Phase 1: Directory structure + config + foundational files
  Phase 2: Import existing assets (persona spec, archetypes, comedy structures, modifiers)
  Phase 3: Post Brief schema definition + 4 reference fixtures

Milestone 2: Post Brief Generator (Plan 02, Module 1)
  Phase 1: Brief generation system prompt
  Phase 2: Generator implementation
  Phase 3: Validation + test harness

Milestone 3: LinkedIn Visual Grammar (Plan 02, Module 2)
  Phase 1: Prop taxonomy
  Phase 2: Composition zones
  Phase 3: Engagement generator

Milestone 4: Image Prompt Engine (Plan 02, Module 3)
  Phase 1: Scene templates
  Phase 2: Prompt builder (Post Brief → local diffusion prompt)
  Phase 3: Modifier selection logic

Milestone 5: Compositor (Plan 02, Module 4)
  Phase 1: LinkedIn UI chrome template
  Phase 2: Text overlay renderer
  Phase 3: Tweet embed renderer

Milestone 6: Artifact UI (Plan 02, Module 5)
  Phase 1: Input form + archetype selector
  Phase 2: Post Brief display + edit
  Phase 3: Image prompt output + copy-to-clipboard
  Phase 4: Compositor preview (stretch)
```

### 3c. `.planning/STATE.md`

```
Project: Mirror Post
Current Milestone: 1 (Scaffolding)
Current Phase: 1 (Directory structure + config)
Status: IN_PROGRESS
Last Updated: 2026-04-13
```

### 3d. `CLAUDE.md`

```markdown
# Mirror Post — Claude Code Instructions

## What This Project Is
Satirical LinkedIn post compositor. Input: corporate archetype or scenario.
Output: complete visual LinkedIn post with AI-generated hero image + text overlays.

## Architecture
5 layers: Persona (voice) → Brief (structured post plan) → Image (diffusion prompt)
→ Compositor (LinkedIn UI overlay) → Artifact (React UI)

## Key Files
- Persona spec: src/persona/spec/mirror_pete.v2.json (DO NOT MODIFY without version bump)
- Post Brief schema: src/brief/schema.js (the canonical output format)
- Test fixtures: test/fixtures/ (4 reference posts from actual outputs)

## Rules
- All persona-driven generation must reference mirror_pete.v2.json
- Post Brief is the contract between all layers — never bypass it
- Archetype library and comedy structures are reference data, not generated
- Image prompts target LOCAL diffusion (not MidJourney) — no --ar or --v flags
- Compositor handles text rendering — diffusion prompt should NOT attempt text on surfaces
- Safety rails from persona spec are hard constraints, not suggestions

## Dependencies
- Node.js 20+
- React (artifact only — runs in Claude Desktop)
- Anthropic API (via artifact's built-in API access)
- Canvas/Sharp (compositor — local only)

## GSD State
Check .planning/STATE.md for current phase. Check tasks/todo.md for work queue.
```

### 3e. `package.json`

```json
{
  "name": "mirror-post",
  "version": "0.1.0",
  "type": "module",
  "description": "Satirical LinkedIn post compositor — corporate satire, rendered.",
  "scripts": {
    "test": "node test/harness.js",
    "brief": "node src/brief/cli.js"
  },
  "dependencies": {},
  "devDependencies": {}
}
```

Minimal. Dependencies added per-module as needed. No upfront bloat.

---

## 4. Asset Import Mapping

These existing files get imported into the scaffolding. No modifications — just placement.

| Source File | Destination | Notes |
|---|---|---|
| `mirror_pete_v2.json` | `src/persona/spec/mirror_pete.v2.json` | As-is. This is the canonical spec. |
| `archetypes.md` | `src/persona/archetypes/library.json` | Convert MD tables → structured JSON. Add `id` fields for referencing. |
| `comedy-structures.md` | `src/persona/comedy/structures.json` | Convert formulas + templates → structured JSON. |
| `ultra_fidelity_modifiers.json` | `src/image/modifiers/ultra-fidelity.json` | As-is. Already structured. |
| `12k_modifier_library.txt` | `src/image/modifiers/12k-modifiers.json` | Convert TXT → structured JSON by category. |

---

## 5. Post Brief Schema (the new canonical format)

This is the single most important artifact of scaffolding. Every module produces or consumes this.

```json
{
  "$schema": "post-brief-v1",
  "meta": {
    "generated_at": "ISO timestamp",
    "persona_spec_version": "2.0.0",
    "input_type": "archetype | scenario | theme",
    "input_raw": "the original user input"
  },
  "character": {
    "name": "Brent Vellum",
    "title": "VP, Workforce Optimization Theater",
    "tagline": "Process Purist | Defender of Standard Procedure",
    "avatar_prompt": "silver-haired man, 50s, smug expression, navy sweater over dress shirt",
    "recurring": false
  },
  "post": {
    "headline": {
      "text": "Moving the Needle: How Different Data Sources Help You ALWAYS Look Productive",
      "highlight_words": ["ALWAYS"],
      "highlight_color": "gold"
    },
    "body": {
      "text": "Our selective approach to reporting allows us to alternate which metric...",
      "bold_phrases": ["need improvement"],
      "italic_phrases": ["(again)"]
    },
    "hashtags": ["#OperationalDiscipline", "#ProcessIntegrity"]
  },
  "props": {
    "mug": {
      "text": "ALIGNMENT",
      "placement": "right_hand"
    },
    "whiteboard": {
      "content": ["Role Title A", "Role Title B", "Role Title C"],
      "annotation": "All interchangeable. Please stop asking...",
      "placement": "background_right"
    },
    "desk_items": [
      { "item": "folder", "label": "ACCOMMODATION WORKFLOW", "placement": "desk_left_buried" },
      { "item": "spreadsheet", "label": "WCS", "placement": "desk_center" },
      { "item": "sticky_note", "label": "NO NEED TO REISSUE", "placement": "desk_right" },
      { "item": "document", "label": "TITLE FLEXIBILITY", "placement": "desk_right_under" }
    ]
  },
  "tweet_embed": {
    "author_name": "Brent Vellum.",
    "handle": "@BrentBehindTheDocs",
    "text": "Hard conversations build strong organizations. Sometimes excellence requires reducing noise...",
    "hashtags": ["#OperationalDiscipline", "#ProcessIntegrity"]
  },
  "engagement": {
    "reactions": { "count": 695, "types": ["like", "heart", "laugh"] },
    "comments": null,
    "dominant_reaction": "laugh"
  },
  "image_seed": {
    "scene_template": "office-middle-mgmt",
    "environment": "corporate office, multiple monitors, fluorescent lighting",
    "subject_pose": "seated at desk, holding mug, smug smile",
    "mood": "self-satisfied, oblivious",
    "color_temperature": "warm office tones, slightly desaturated"
  },
  "nav_easter_eggs": {
    "misspelling": { "original": "Evaluations", "replacement": "Evanrations" }
  }
}
```

---

## 6. Test Fixtures (4 Reference Posts)

Each fixture is a pair: `input-*.json` (what the user would type) and `expected-briefs/*.json` (what the Post Brief Generator should produce). These are reverse-engineered from Pete's actual outputs.

| Fixture | Input | Key Validation Points |
|---|---|---|
| Brent Vellum | "HR manager who strips job descriptions and calls it standard procedure" | Mug: ALIGNMENT. Whiteboard: 3 titles. Props: accommodation folder buried. Tweet: weaponized politeness. Nav: Evanrations. |
| Trevor B. (Hustle) | "LinkedIn hustle culture bro who posts airport selfies" | Selfie pose with coffee. Laptop open to LinkedIn. #BlessedAndBusy. Tweet: knowledge bombs. |
| Trevor B. (Closer) | "Aggressive sales bro who thinks empathy is expensive" | Headset + cash in hand. SMILE.LIE.CLOSE sticky. Garbled tweet text. EXCUSES: 0 whiteboard. |
| Pete C. (Titles) | "Director whose job description is a quaintly retro concept" | Deadpan expression. Vague Duty Elixir mug. JOB DESCRIPTIONS (CLOSED ARCHIVE) box. Spirograph org chart. |

---

## 7. Scaffolding Execution Checklist

```
Phase 1: Structure
[ ] Create mirror-post/ directory
[ ] Create all subdirectories per structure above
[ ] Create .planning/PROJECT.md
[ ] Create .planning/ROADMAP.md
[ ] Create .planning/STATE.md
[ ] Create CLAUDE.md
[ ] Create tasks/todo.md (populated with Milestone 1 tasks)
[ ] Create tasks/lessons.md (empty)
[ ] Create package.json
[ ] Create README.md (product description + architecture diagram)
[ ] git init + initial commit

Phase 2: Asset Import
[ ] Copy mirror_pete_v2.json → src/persona/spec/
[ ] Convert archetypes.md → src/persona/archetypes/library.json
[ ] Convert comedy-structures.md → src/persona/comedy/structures.json
[ ] Copy ultra_fidelity_modifiers.json → src/image/modifiers/
[ ] Convert 12k_modifier_library.txt → src/image/modifiers/12k-modifiers.json
[ ] Create barrel exports (src/index.js, src/persona/index.js, etc.)
[ ] Commit: "feat(persona): import canonical spec and reference libraries"

Phase 3: Schema + Fixtures
[ ] Create src/brief/schema.js (Post Brief JSON schema with JSDoc types)
[ ] Create test/fixtures/input-brent-vellum.json
[ ] Create test/fixtures/input-trevor-hustle.json
[ ] Create test/fixtures/input-trevor-closer.json
[ ] Create test/fixtures/input-pete-titles.json
[ ] Create test/fixtures/expected-briefs/ (4 files, reverse-engineered from actual outputs)
[ ] Create test/harness.js (skeleton — validates fixtures against schema)
[ ] Commit: "feat(schema): Post Brief v1 schema + 4 reference fixtures"

Final Verification
[ ] `node test/harness.js` runs without errors (schema validation only)
[ ] All barrel exports resolve
[ ] .planning/STATE.md updated to Phase 1 complete
[ ] tasks/todo.md refreshed for Milestone 2
```

---

## 8. Success Criteria

Scaffolding is DONE when:

1. A Claude Code session can `cd mirror-post/`, read `CLAUDE.md`, and understand the full architecture
2. The Post Brief schema exists and can validate the 4 reference fixtures
3. All existing assets (persona spec, archetypes, comedy structures, modifiers) are imported and importable
4. `.planning/` accurately reflects current state
5. No functional code exists yet — only structure, data, schema, and fixtures
6. `git log` shows clean, atomic commits with conventional commit messages

---

*Scaffolding is the foundation. Nothing ships without it. Nothing built on it should surprise you.*
