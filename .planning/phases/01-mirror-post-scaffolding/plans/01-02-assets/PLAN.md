# PLAN-01-02: Asset Import and Conversion

## Metadata

- **Phase:** 1 (Mirror Post Scaffolding)
- **Plan:** 01-02
- **Wave:** 2a (parallel with 01-03)
- **Requirements:** REQ-M-002, REQ-M-003, REQ-X-001
- **Depends on:** 01-01
- **Commit message:** `feat(persona): import canonical spec and reference libraries`

## Dependencies

- PLAN-01-01 must complete first (directory structure must exist).

## Source Asset Paths

All source assets at: `/Users/cpconnor/projects/Krea-AI/docs/mirror-post-source-files/`

## Tasks

### T1: Copy mirror_pete.v2.json (REQ-M-002, REQ-X-001)

- Source: `docs/mirror-post-source-files/mirror_pete_v2.json` (45 KB)
- Dest: `mirror-post/src/persona/spec/mirror_pete.v2.json`
- Byte-identical copy. No modifications.
- **Acceptance:** `diff docs/mirror-post-source-files/mirror_pete_v2.json mirror-post/src/persona/spec/mirror_pete.v2.json` is empty

### T2: Convert archetypes.md -> library.json

- Source: `docs/mirror-post-source-files/archetypes.md` (5.1 KB, 17 archetypes in 5 categories)
- Dest: `mirror-post/src/persona/archetypes/library.json`
- Structure: `{ "$schema": "archetype-library-v1", "categories": [...], "total_archetypes": 17 }`
- Each archetype: `{ id, name, category, traits: [], signature_move, tell }`
- ID rule: kebab-case from name, strip "The ". e.g. "The Resume Padder" -> `resume-padder`
- Categories: leadership (4), vendor (4), internal-blocker (4), contact-center (3), healthcare-rcm (2)
- **Acceptance:** `node -e "const j=JSON.parse(require('fs').readFileSync('mirror-post/src/persona/archetypes/library.json','utf8'));console.log(j.total_archetypes)"` outputs `17`

### T3: Convert comedy-structures.md -> structures.json

- Source: `docs/mirror-post-source-files/comedy-structures.md` (6.5 KB)
- Dest: `mirror-post/src/persona/comedy/structures.json`
- Structure: `{ "$schema": "comedy-structures-v1", "core_principle": "...", "joke_structures": [10], "roast_structures": [3], "rules": [6] }`
- Each joke: `{ id, name, description, formula, examples: [] }`
- Each roast: `{ id, name, description, example }`
- 10 joke IDs: credential-undercut, technical-truth-bomb, insider-callback, escalating-specificity, timeline-reality-check, definition-corruption, meeting-autopsy, resume-translation, vendor-bingo, witness-statement
- 3 roast IDs: triple-escalation, compliment-execution, specific-callback
- **Acceptance:** `node -e "const j=JSON.parse(require('fs').readFileSync('mirror-post/src/persona/comedy/structures.json','utf8'));console.log(j.joke_structures.length,j.roast_structures.length)"` outputs `10 3`

### T4: Copy ultra_fidelity_modifiers.json

- Source: `docs/mirror-post-source-files/ultra_fidelity_modifiers.json`
- Dest: `mirror-post/src/image/modifiers/ultra-fidelity.json` (underscore to hyphen)
- **Acceptance:** valid JSON with top-level keys for resolution, sensor, lighting, etc.

### T5: Convert 12k_modifier_library.txt -> 12k-modifiers.json

- Source: `docs/mirror-post-source-files/12k_modifier_library.txt` (2.1 KB, 9 categories)
- Dest: `mirror-post/src/image/modifiers/12k-modifiers.json`
- Structure: `{ "$schema": "12k-modifiers-v1", "categories": { "resolution_render_fidelity": [...], ... } }`
- 9 categories: resolution_render_fidelity, camera_lens, lighting, surface_texture, post_processing_film_look, depth_focus, color_mood, environment_detail, negatives
- **Acceptance:** `node -e "const j=JSON.parse(require('fs').readFileSync('mirror-post/src/image/modifiers/12k-modifiers.json','utf8'));console.log(Object.keys(j.categories).length)"` outputs `9`
