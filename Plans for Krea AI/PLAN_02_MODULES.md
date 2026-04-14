# PLAN 02 — MODULES: Building Mirror Post Functionality

## Overview

This plan covers Milestones 2–6 from the roadmap. Each milestone maps to one functional module. Each module follows the GSD lifecycle: discuss → plan → execute → verify → ship. Modules are built sequentially because each one's output feeds the next.

**Dependency chain:**
```
Persona (existing) → Post Brief Generator → LinkedIn Visual Grammar → Image Prompt Engine → Compositor → Artifact UI
         ↓                    ↓                      ↓                      ↓                ↓
   voice rules          structured JSON          prop/zone specs       diffusion prompt    final image
```

---

## MODULE 1: Post Brief Generator
**Milestone 2 | Priority: CRITICAL | This is the product.**

### What It Does
Takes a user input (archetype name, scenario description, or theme) and produces a complete Post Brief JSON — the structured document that contains every element of a satirical LinkedIn post. Character name, title, tagline, headline, body copy, prop list with text, tweet embed, engagement metrics, image seed, and nav easter eggs. All elements are satirically coherent because they're generated from a single voice-aware pass.

### Why It's First
Everything downstream consumes the Post Brief. The image prompt engine reads it. The compositor reads it. The artifact displays it. Without this, there is no product — just disconnected tools.

### Architecture

```
User Input (string)
       ↓
┌─────────────────────┐
│  Input Classifier    │  Determines: archetype | scenario | theme
│  (deterministic)     │  Looks up archetype library if match found
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  Brief System Prompt │  Built dynamically from:
│  Builder             │  - mirror_pete.v2.json (voice, stance, safety)
│  (deterministic)     │  - Matched archetype (if any)
│                      │  - Comedy structures (available formulas)
│                      │  - Post Brief schema (required output format)
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  LLM Generation      │  Anthropic API call
│  (claude-sonnet-4)   │  System: built prompt
│                      │  User: input + "Generate a Post Brief"
│                      │  Response: JSON matching Post Brief schema
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  Brief Validator     │  Schema validation (required fields, types)
│  (deterministic)     │  Safety check (no punching down)
│                      │  Voice check (no banned phrases in output)
│                      │  Completeness check (all prop texts populated)
└─────────┬───────────┘
          ↓
     Post Brief JSON
```

### Phases

#### Phase 1: Brief Generation System Prompt
**GSD: discuss → plan**

Build the system prompt template that gets assembled at generation time. This is the most important prompt in the entire system.

**Deliverables:**
- `src/brief/prompts/brief-system.md` — the master system prompt template
- Template uses `{{PERSONA_VOICE}}`, `{{ARCHETYPE_CONTEXT}}`, `{{COMEDY_STRUCTURES}}`, `{{OUTPUT_SCHEMA}}` interpolation slots

**System prompt structure:**
```
ROLE: You are the Post Brief Generator for Mirror Post.
Your job is to create a complete satirical LinkedIn post plan.

VOICE RULES:
{{PERSONA_VOICE}}
— Extracted from mirror_pete.v2.json: voice sliders, stance, safety rails,
  consistency rules, anti-patterns to avoid

ARCHETYPE CONTEXT (if matched):
{{ARCHETYPE_CONTEXT}}
— The matched archetype's name, signature move, tell, and domain
— If no match: "Generate an original character from the scenario"

COMEDY TOOLKIT:
{{COMEDY_STRUCTURES}}
— Available joke formulas (Credential Undercut, Technical Truth Bomb, etc.)
— Available roast structures (Triple Escalation, Compliment Execution, etc.)
— Instruction: "Select 2-3 formulas that fit the input. Do not force all of them."

OUTPUT FORMAT:
{{OUTPUT_SCHEMA}}
— The complete Post Brief JSON schema with field descriptions
— Every field is required unless marked optional
— Instruction: "Return ONLY valid JSON. No preamble. No markdown fences."

PROP INTELLIGENCE:
— Mugs carry slogans that reveal character (1-2 words max)
— Whiteboards carry lists, org charts, or mantras (3-5 items)
— Desk items are evidence — folders, documents, sticky notes that tell a story
— The tweet embed is the character's unfiltered inner voice
— Engagement metrics should be satirically calibrated:
  laughing emoji dominant = audience sees through it
  high likes low comments = performative engagement
  high comments = controversial take
— Nav easter eggs are optional misspellings or wrong labels in the LinkedIn chrome

SAFETY RAILS (HARD CONSTRAINTS):
— Punch systems, patterns, and archetypes — never individuals
— No content targeting marginalized groups, mental health, genuine setbacks
— Characters are FICTIONAL composites, not real people
— Satire must contain insight — if removing the humor leaves no point, rewrite
```

**Acceptance criteria:**
- [ ] System prompt template exists with all interpolation slots
- [ ] Prompt can be assembled with real data from persona spec + archetype library
- [ ] Assembled prompt fits within 8K tokens (leaving room for generation)

#### Phase 2: Generator Implementation
**GSD: execute**

**Deliverables:**
- `src/brief/generator.js` — the main generator function
- `src/brief/classifier.js` — input classification (archetype lookup, scenario detection)
- `src/brief/prompt-builder.js` — assembles system prompt from template + data

**Generator function signature:**
```javascript
async function generateBrief(input, options = {}) {
  // input: string (user's topic/archetype/scenario)
  // options: { temperature, maxRetries, archetypeOverride }
  // returns: { brief: PostBrief, metadata: GenerationMetadata }
}
```

**Classifier logic:**
```javascript
function classifyInput(input, archetypeLibrary) {
  // 1. Exact match: input matches an archetype name
  // 2. Fuzzy match: input contains archetype keywords (signature move, tell)
  // 3. Domain match: input matches a domain (contact center, vendor, etc.)
  // 4. No match: treat as freeform scenario
  // Returns: { type, matchedArchetype?, confidence }
}
```

**Prompt builder logic:**
```javascript
function buildSystemPrompt(personaSpec, archetype, comedyStructures, briefSchema) {
  // 1. Read template from brief-system.md
  // 2. Extract voice rules from personaSpec.voice + personaSpec.stance + personaSpec.consistency_rules
  // 3. Format archetype context (or "generate original" instruction)
  // 4. Format comedy structures as available toolkit
  // 5. Stringify Post Brief schema as output format
  // 6. Interpolate all slots
  // Returns: assembled system prompt string
}
```

**LLM call pattern:**
```javascript
const response = await fetch('https://api.anthropic.com/v1/messages', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 4000,
    system: assembledSystemPrompt,
    messages: [{ role: 'user', content: `INPUT: ${input}\n\nGenerate a complete Post Brief.` }]
  })
});
```

**Acceptance criteria:**
- [ ] `generateBrief("HR manager who strips job descriptions")` returns valid Post Brief JSON
- [ ] Classifier correctly identifies archetypes from the library
- [ ] Freeform scenarios produce original characters (not just archetype copies)
- [ ] All Post Brief fields are populated (no nulls in required fields)
- [ ] Generation completes in < 15 seconds

#### Phase 3: Validation + Test Harness
**GSD: verify → ship**

**Deliverables:**
- `src/brief/validator.js` — Post Brief validation
- Updated `test/harness.js` — runs generation against 4 reference fixtures

**Validator checks:**

Hard-fail (deterministic):
```
[ ] Schema compliance — all required fields present and correct types
[ ] Safety — no real names, no punching-down patterns
[ ] Banned phrases — persona spec banned_phrases not in any text field
[ ] Prop completeness — mug.text, whiteboard.content, and tweet_embed.text all populated
[ ] Engagement sanity — reaction count > 0, dominant_reaction is valid enum
[ ] Headline length — ≤ 80 characters
[ ] Body length — ≤ 500 characters (LinkedIn visual constraint)
[ ] Character name — not a real public figure
```

Scored (heuristic):
```
[ ] Voice fit — does the headline sound like Pete? (keyword matching against power_words + pattern libraries)
[ ] Comedy structure usage — at least one identifiable formula in the body or headline
[ ] Prop coherence — do desk items relate to the scenario? (semantic check)
[ ] Tweet voice distinction — tweet text sounds different from body text (more unfiltered)
[ ] Easter egg presence — at least one nav misspelling or visual gag
```

**Test harness runs:**
1. Load 4 input fixtures
2. Generate Post Brief for each
3. Validate each against schema + hard-fail checks
4. Compare against expected briefs (structural similarity, not exact match)
5. Print: pass/fail, score breakdown, generation time

**Acceptance criteria:**
- [ ] `node test/harness.js` produces 4 passing Post Briefs
- [ ] Each brief is structurally similar to its expected reference
- [ ] No hard-fail violations
- [ ] Average generation time < 15 seconds per brief
- [ ] Harness prints clear, readable output

---

## MODULE 2: LinkedIn Visual Grammar
**Milestone 3 | Priority: HIGH | This codifies Pete's creative instincts.**

### What It Does
Defines the rules for how satirical LinkedIn posts are visually constructed. Prop taxonomy, composition zones, and engagement metric generation. This module is pure data and logic — no LLM calls, no images. It's the reference library that the image prompt engine and compositor consume.

### Why It's Second
The Post Brief Generator references props, zones, and engagement patterns. This module formalizes what's currently in Pete's head so the generator can make better choices and the downstream modules have a contract to build against.

### Phases

#### Phase 1: Prop Taxonomy
**GSD: discuss → plan → execute**

**Deliverable:** `src/grammar/props.json`

```json
{
  "props": {
    "mug": {
      "description": "Coffee mug with 1-2 word slogan revealing character",
      "placement_options": ["right_hand", "left_hand", "desk_surface"],
      "text_constraints": { "max_words": 3, "style": "all_caps_or_title_case" },
      "examples": ["ALIGNMENT", "Vague Duty Elixir", "CLOSERS CLOSE"],
      "satirical_function": "Reveals what the character worships or believes without irony"
    },
    "whiteboard": {
      "description": "Office whiteboard or flip chart with lists, charts, or annotations",
      "placement_options": ["background_right", "background_left", "background_center"],
      "content_types": ["list", "org_chart", "metrics_board", "mantra"],
      "annotation_slot": true,
      "examples": {
        "list": ["Role Title A", "Role Title B", "Role Title C"],
        "metrics_board": { "DIALS": 1247, "CONV": 89, "CLOSES": 12, "EXCUSES": 0 },
        "mantra": "All interchangeable. Please stop asking..."
      },
      "satirical_function": "Background evidence that contradicts or amplifies the headline"
    },
    "desk_items": {
      "types": {
        "folder": {
          "description": "Labeled folder or binder",
          "text_constraints": { "max_words": 4, "style": "all_caps" },
          "placement_options": ["desk_left", "desk_right", "desk_buried"],
          "satirical_function": "Evidence hidden in plain sight"
        },
        "sticky_note": {
          "description": "Post-it note with short phrase",
          "text_constraints": { "max_words": 5, "style": "handwritten_caps" },
          "placement_options": ["monitor", "desk_surface", "laptop"],
          "satirical_function": "The quiet part said loud"
        },
        "document": {
          "description": "Visible document or report",
          "text_constraints": { "max_words": 4, "style": "printed_title" },
          "satirical_function": "Bureaucratic artifact that tells the real story"
        },
        "box": {
          "description": "File box or archive box",
          "text_constraints": { "max_words": 5, "style": "marker_on_cardboard" },
          "satirical_function": "Where things go to die"
        },
        "laptop": {
          "description": "Open laptop showing screen content",
          "screen_content_options": ["linkedin_feed", "email", "spreadsheet", "slack"],
          "satirical_function": "Meta-recursion (LinkedIn open while posting on LinkedIn)"
        },
        "phone": {
          "description": "Smartphone in hand",
          "pose_options": ["selfie", "texting", "showing_screen"],
          "satirical_function": "Performative documentation of the moment"
        }
      }
    },
    "business_card": {
      "description": "Optional visible business card",
      "satirical_function": "Credential display that undercuts itself"
    },
    "nameplate": {
      "description": "Desk nameplate with title",
      "satirical_function": "Official title that contradicts the actual role"
    }
  }
}
```

#### Phase 2: Composition Zones
**GSD: execute**

**Deliverable:** `src/grammar/zones.json`

Defines the spatial layout of a satirical LinkedIn post image. Based on analysis of Pete's 4 example outputs.

```json
{
  "layout": {
    "aspect_ratio": "16:9",
    "resolution": "1920x1080",
    "zones": {
      "linkedin_header": {
        "position": { "x": 0, "y": 0, "width": "100%", "height": "60px" },
        "contains": ["nav_bar", "search", "nav_items"],
        "style": "linkedin_dark_blue_chrome",
        "easter_egg_slot": "nav_items"
      },
      "profile_bar": {
        "position": { "x": 0, "y": 60, "width": "100%", "height": "50px" },
        "contains": ["avatar_circle", "name", "title_tagline", "update_button"],
        "style": "linkedin_profile_header"
      },
      "hero_image": {
        "position": { "x": 0, "y": 110, "width": "100%", "height": "calc(100% - 170px)" },
        "contains": ["diffusion_generated_image"],
        "sub_zones": {
          "headline_overlay": {
            "position": "left_40pct",
            "vertical": "top_third",
            "text_style": "bold_headline_with_highlight"
          },
          "body_overlay": {
            "position": "left_40pct",
            "vertical": "middle",
            "text_style": "body_with_formatting"
          },
          "subject": {
            "position": "right_60pct",
            "vertical": "full_height",
            "note": "Person at desk — generated by diffusion"
          },
          "tweet_embed": {
            "position": "bottom_right",
            "size": "300x200px",
            "style": "twitter_card_with_shadow",
            "optional": true
          }
        }
      },
      "engagement_bar": {
        "position": { "x": 0, "y": "bottom", "width": "100%", "height": "60px" },
        "contains": ["reaction_icons", "reaction_count", "comment_count", "like_comment_share_buttons"],
        "style": "linkedin_engagement_footer"
      }
    }
  }
}
```

#### Phase 3: Engagement Generator
**GSD: execute → verify**

**Deliverable:** `src/grammar/engagement.js`

```javascript
function generateEngagement(characterType, postTone) {
  // Rules:
  // - Satirical/self-aware posts: laugh emoji dominant, moderate count (500-2000)
  // - Hustle/motivational posts: like emoji dominant, high count (2000-5000), low comments
  // - Aggressive/controversial posts: like+heart, moderate count, visible comment count
  // - Self-deprecating posts: laugh+heart, high count, high comments
  //
  // Comment count rules:
  // - null/hidden if the joke is that nobody engages meaningfully
  // - visible and high (300+) if the joke is controversy
  // - visible and specific (482) for realism
  //
  // Returns: { reactions: { count, types }, comments, dominant_reaction }
}
```

**Acceptance criteria:**
- [ ] Props JSON covers all prop types seen in 4 reference outputs
- [ ] Zones JSON accurately describes the spatial layout of reference outputs
- [ ] Engagement generator produces plausible, satirically calibrated numbers
- [ ] All files are pure data/logic — no LLM calls, no external dependencies

---

## MODULE 3: Image Prompt Engine
**Milestone 4 | Priority: HIGH | Bridges the brief to the diffusion model.**

### What It Does
Takes a Post Brief and produces a prompt formatted for Pete's local diffusion pipeline. Selects appropriate modifiers from the ultra-fidelity library. Accounts for composition zones (leaves space for text overlays). Generates scene descriptions that match the character archetype and satirical tone.

### Key Design Constraint
**The diffusion model does NOT render text.** All text on mugs, whiteboards, sticky notes, and overlays is handled by the compositor (Module 4). The image prompt should describe *surfaces and objects that will receive text* — but not the text itself. Example: "coffee mug with blank white label area" not "coffee mug that says ALIGNMENT."

### Phases

#### Phase 1: Scene Templates
**GSD: discuss → plan → execute**

**Deliverables:** `src/image/templates/*.json` (4 scene templates)

Each template defines a scene archetype with: environment description, subject pose, camera angle, lighting setup, key props (positions only — content comes from Post Brief), and mood keywords.

Templates to create:
1. **office-executive** — Corner office, natural light, expensive furniture, power pose
2. **office-middle-mgmt** — Open plan or glass-walled office, multiple monitors, fluorescent + warm mix
3. **airport-hustle** — Airport lounge or business class, glass walls, passing travelers
4. **call-center-floor** — Headset, phone, cramped desk, bullpen background, harsh overhead light

**Template structure:**
```json
{
  "id": "office-middle-mgmt",
  "description": "Mid-level manager in standard corporate office environment",
  "environment": {
    "base": "corporate office interior, glass partitions, multiple monitors",
    "lighting": "overhead fluorescent mixed with warm desk lamp, 5600K key with 3200K practicals",
    "depth": "shallow depth of field, background slightly soft",
    "atmosphere": "clean but lived-in, papers on desk, pen holder"
  },
  "subject": {
    "framing": "medium shot, waist up, seated at desk",
    "eye_line": "direct to camera or slight smirk angle",
    "hands": "one hand on mug, other on desk or phone",
    "expression_map": {
      "smug": "self-satisfied half-smile, slight head tilt",
      "deadpan": "flat expression, direct stare, slight frown",
      "performative": "exaggerated smile, duck lips, selfie angle"
    }
  },
  "prop_positions": {
    "mug": "right_hand or desk_right — must have visible blank label area",
    "whiteboard": "background, partially visible — leave 30% of right frame",
    "desk_surface": "foreground — space for 2-3 items with readable surfaces",
    "laptop": "desk_left, angled open — screen visible but not primary focus"
  },
  "composition": {
    "subject_position": "right_of_center (rule of thirds, right intersection)",
    "text_clear_zone": "left 35-40% of frame — no props, no subject, for headline overlay",
    "tweet_zone": "bottom_right 25% — leave clean or simple background"
  }
}
```

#### Phase 2: Prompt Builder
**GSD: execute**

**Deliverable:** `src/image/prompt-builder.js`

```javascript
function buildImagePrompt(postBrief, options = {}) {
  // 1. Select scene template from brief.image_seed.scene_template
  // 2. Build subject description from brief.character.avatar_prompt + template.subject
  // 3. Map expression from brief.image_seed.mood → template.subject.expression_map
  // 4. Build environment from template.environment + brief.image_seed overrides
  // 5. Position props per template.prop_positions (surfaces only, no text)
  // 6. Select modifiers from ultra-fidelity library (3-5 per category, matched to scene type)
  // 7. Apply composition zone constraints (text_clear_zone, tweet_zone)
  // 8. Format for local diffusion (positive prompt + negative prompt + parameters)
  //
  // Returns: {
  //   positive_prompt: string,
  //   negative_prompt: string,
  //   parameters: { width, height, steps, cfg_scale, sampler },
  //   composition_notes: string (human-readable placement guide for manual adjustment)
  // }
}
```

**Modifier selection logic:**
```javascript
function selectModifiers(sceneType, ultraFidelityLib) {
  // Use the Mirror Vision modifier selection strategy:
  // - Corporate interiors: Resolution + Lighting (fluorescent vs warm) + Material (leather, glass)
  // - Airport/travel: Lighting (natural + practicals) + Material (glass, chrome) + Color (warm gold)
  // - Call center: Lighting (harsh overhead) + Material (plastic, fabric) + Color (cool/desaturated)
  //
  // Always include from each category:
  // - resolution_render_engine: 3 items
  // - sensor_optics: 2 items
  // - lighting_atmospherics: 3 items
  // - material_surface_fidelity: 3 items
  // - color_science_grade: 2 items
  // - post_processing_output: 2 items
  // - negative_safety_prompts: all relevant
}
```

#### Phase 3: Modifier Selection Logic
**GSD: execute → verify**

**Deliverable:** Updated `src/image/prompt-builder.js` with intelligent modifier selection

The modifier library has 100+ options. The prompt builder must select the RIGHT 15-20 for each scene without dumping everything. Selection is based on scene template type + mood + environment.

**Acceptance criteria:**
- [ ] `buildImagePrompt(brentVellumBrief)` produces a coherent prompt for office-middle-mgmt scene
- [ ] Prompt leaves text_clear_zone empty (no props in left 35-40% of frame)
- [ ] Prompt describes prop SURFACES without text content
- [ ] Modifier count is 15-20 (not 5, not 80)
- [ ] Negative prompt includes all safety items
- [ ] Output includes composition_notes for manual adjustment

---

## MODULE 4: Compositor
**Milestone 5 | Priority: MEDIUM | The assembly layer.**

### What It Does
Takes the diffusion-generated hero image + the Post Brief and composites the final LinkedIn post image. Overlays: LinkedIn UI chrome (header, profile bar, engagement footer), headline text with highlights, body text with formatting, tweet embed card, prop text (if not generated by diffusion), and nav easter eggs.

### Technical Approach
Node.js canvas rendering (node-canvas or Sharp). Template-driven — the zones.json from Module 2 defines positions. Typography from styles.json. All text rendering is programmatic.

### Phases

#### Phase 1: LinkedIn UI Chrome Template
**GSD: discuss → plan → execute**

**Deliverables:**
- `src/compositor/templates/linkedin-post.json` — layout spec for all chrome elements
- `src/compositor/assets/` — LinkedIn-style icons (home, network, jobs, messaging, notifications), reaction icons (like, heart, laugh), profile photo placeholder circle

**Chrome elements:**
```
Top bar: Dark blue (#0A66C2) with white icons and search bar
Profile section: Avatar circle (60px), Name (bold, white), Title|Tagline (regular, white/70%), Update button, "..." menu
Engagement bar: Reaction icons + count (left), Comment count (right, if present), Like/Comment/Share buttons
```

**Note:** These are styled to LOOK like LinkedIn but are not pixel-perfect copies. Satirical context makes this transformative use. Still — no LinkedIn logo, no exact color match, stylistic similarity only.

#### Phase 2: Text Overlay Renderer
**GSD: execute**

**Deliverable:** `src/compositor/renderers/text-overlay.js`

```javascript
function renderTextOverlay(canvas, postBrief, zones) {
  // 1. Render headline in headline_overlay zone
  //    - Split by highlight_words → render those in highlight_color
  //    - Font: bold condensed, ~48px, white with slight drop shadow
  // 2. Render body in body_overlay zone
  //    - Apply bold_phrases and italic_phrases formatting
  //    - Font: regular, ~20px, white with slight transparency
  // 3. Render hashtags below body (if present)
  //    - Font: regular, ~16px, accent color
}
```

#### Phase 3: Tweet Embed Renderer
**GSD: execute → verify**

**Deliverable:** `src/compositor/renderers/tweet-embed.js`

```javascript
function renderTweetEmbed(canvas, tweetData, position) {
  // 1. Draw card background (white, rounded corners, drop shadow)
  // 2. Render avatar circle (small, top-left of card)
  // 3. Render author_name (bold) + handle (gray)
  // 4. Render tweet text (regular, dark, with line wrapping)
  // 5. Render hashtags (blue, linked style)
  // 6. Optional: render "..." menu icon (top-right)
  // 7. Optional: render like/reply counts below text
}
```

**Acceptance criteria:**
- [ ] Compositor takes a hero image + Post Brief and produces a final composite
- [ ] LinkedIn chrome is visually convincing (not pixel-perfect, but recognizable)
- [ ] Headline text renders with highlight words in accent color
- [ ] Tweet embed card renders with proper layout
- [ ] Engagement bar shows correct reaction icons and counts
- [ ] Output is a single PNG at 1920x1080
- [ ] Comparison against reference outputs shows structural match

---

## MODULE 5: Artifact UI
**Milestone 6 | Priority: MEDIUM | The user-facing interface.**

### What It Does
A React artifact for Claude Desktop that ties everything together. Input form → archetype browser → Post Brief generation → brief display/edit → image prompt output → compositor preview (stretch goal).

### Design Direction
Obsidian dark-mode aesthetic. Minimal. The content is the spectacle — the UI is the stage, not the performer.

### Phases

#### Phase 1: Input Form + Archetype Selector
**GSD: discuss → plan → execute**

**Deliverable:** `artifact/mirror-post-generator.jsx` (initial version)

**UI structure:**
```
┌─────────────────────────────────────────────┐
│  ⚔️ Mirror Post                              │
│  Corporate satire, rendered.                 │
├─────────────────────────────────────────────┤
│                                             │
│  [Text input: Describe a scenario...]       │
│                                             │
│  — OR —                                     │
│                                             │
│  Browse Archetypes:                         │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │
│  │Leader│ │Vendor│ │Block-│ │Contact│      │
│  │ ship │ │      │ │  er  │ │Center │      │
│  └──────┘ └──────┘ └──────┘ └──────┘      │
│                                             │
│  [Selected: The Metrics Illusionist]        │
│  Signature Move: Redefines success...       │
│  The Tell: Slides have more footnotes...    │
│                                             │
│  [⚔️ Generate Post Brief]                   │
│                                             │
├─────────────────────────────────────────────┤
│  Advanced Options (collapsed by default)     │
│  Temperature: [0.8] Scene: [auto-detect]    │
└─────────────────────────────────────────────┘
```

**Archetype browser:**
- Cards grouped by domain (Leadership, Vendor, Blocker, Contact Center, Healthcare)
- Each card shows: name, signature move (truncated), the tell (truncated)
- Click to select → populates input with archetype context
- Can still type freeform — archetype selection is a shortcut, not a requirement

#### Phase 2: Post Brief Display + Edit
**GSD: execute**

**UI after generation:**
```
┌─────────────────────────────────────────────┐
│  POST BRIEF: Brent Vellum                    │
│  VP, Workforce Optimization Theater          │
├──────────────────┬──────────────────────────┤
│  CHARACTER        │  POST CONTENT            │
│  Name: [editable] │  Headline: [editable]    │
│  Title: [editable]│  Body: [editable]        │
│  Tagline: [edit]  │  Hashtags: [editable]    │
├──────────────────┼──────────────────────────┤
│  PROPS            │  TWEET EMBED             │
│  Mug: [editable]  │  Author: [editable]      │
│  Whiteboard:      │  Handle: [editable]      │
│    - [editable]   │  Text: [editable]        │
│    - [editable]   │  Tags: [editable]        │
│  Desk items:      │                          │
│    - [editable]   │  ENGAGEMENT              │
│    - [editable]   │  Reactions: [editable]   │
│                   │  Type: [select]          │
├──────────────────┴──────────────────────────┤
│  [📋 Copy Brief JSON] [🎨 Generate Image Prompt] [🔄 Regenerate] │
└─────────────────────────────────────────────┘
```

Every field is editable. Pete can tweak any element before generating the image prompt. The brief is the creative control surface.

#### Phase 3: Image Prompt Output
**GSD: execute → verify**

When "Generate Image Prompt" is clicked:
1. Take current (possibly edited) Post Brief
2. Run image prompt builder (Module 3)
3. Display in a code block with copy button
4. Show composition notes as a visual guide

```
┌─────────────────────────────────────────────┐
│  IMAGE PROMPT                                │
│  ┌─────────────────────────────────────────┐│
│  │ medium shot of a silver-haired man in   ││
│  │ his 50s, navy sweater over dress shirt, ││
│  │ seated at corporate office desk,        ││
│  │ smug half-smile, holding white coffee   ││
│  │ mug with blank label area in right      ││
│  │ hand...                                 ││
│  └─────────────────────────────────────────┘│
│  [📋 Copy Prompt] [📋 Copy Negative]        │
│                                             │
│  COMPOSITION GUIDE                           │
│  ┌─────────────────────────────────────────┐│
│  │ [TEXT ZONE]  │  [SUBJECT]  │[WHITEBOARD]││
│  │ headline +   │  person at  │ background ││
│  │ body text    │  desk       │ right      ││
│  │ goes here    │             │            ││
│  │              │      [TWEET EMBED]       ││
│  └─────────────────────────────────────────┘│
│  Leave left 35-40% clear for text overlay.   │
│  Whiteboard in background right.             │
│  Tweet embed zone: bottom right.             │
└─────────────────────────────────────────────┘
```

#### Phase 4: Compositor Preview (STRETCH)
**GSD: execute → verify → ship**

If time permits: render a rough preview of the final composite directly in the artifact using HTML Canvas. Not production quality — just a layout preview showing where all elements will land. This lets Pete see the composition before running local diffusion.

**Acceptance criteria for full Milestone 6:**
- [ ] Archetype browser displays all archetypes from library
- [ ] Freeform input generates a Post Brief
- [ ] Archetype selection generates a Post Brief
- [ ] All Post Brief fields are editable in-place
- [ ] "Copy Brief JSON" produces valid JSON
- [ ] "Generate Image Prompt" produces a diffusion-ready prompt
- [ ] Composition guide accurately shows zone layout
- [ ] Regenerate button produces a different (but valid) Post Brief
- [ ] UI follows Obsidian dark-mode aesthetic
- [ ] Full flow completes in < 20 seconds

---

## Cross-Module Integration Tests

After all modules are built, run these end-to-end tests:

| Test | Input | Expected Flow | Pass Criteria |
|---|---|---|---|
| **Brent Vellum roundtrip** | "HR manager who strips job descriptions" | Brief → Image Prompt → Composite | Final image structurally matches reference |
| **Trevor B. roundtrip** | "LinkedIn hustle bro who posts airport selfies" | Brief → Image Prompt → Composite | Airport scene, selfie pose, garbled tweet |
| **Freeform original** | "Cloud architect who only deploys to PowerPoint" | Brief → Image Prompt → Composite | Original character, coherent props, valid engagement |
| **Archetype direct** | Select "The Metrics Illusionist" from browser | Brief → Image Prompt → Composite | Uses archetype's signature move and tell |
| **Edit flow** | Generate brief → edit mug text → regenerate prompt | Updated prompt reflects edited mug | Changed mug described in prompt |

---

## Execution Order Summary

```
Week 1:  Module 1 (Post Brief Generator) — Phases 1-3
         This is the product. Ship it first. Everything else builds on it.

Week 2:  Module 2 (Visual Grammar) — Phases 1-3
         Pure data work. Can be done quickly. Unblocks Modules 3-4.

Week 3:  Module 3 (Image Prompt Engine) — Phases 1-3
         Bridges brief to diffusion. Pete can start generating images after this.

Week 4:  Module 4 (Compositor) — Phases 1-3
         Text overlay and UI chrome. After this, full pipeline works.

Week 5:  Module 5 (Artifact UI) — Phases 1-4
         Wraps everything in a usable interface. Polish and ship.
```

**Critical path:** Module 1 → Module 2 → Module 3. After Module 3 ships, Pete has a working pipeline (input → brief → image prompt → manual diffusion → manual compositing). Modules 4 and 5 automate the manual steps but aren't blockers to producing output.

---

*Every module has one job. Every module's output feeds the next. No module is precious — if it doesn't serve the final image, cut it.*
