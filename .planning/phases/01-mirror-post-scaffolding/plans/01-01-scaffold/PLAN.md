# PLAN-01-01: Directory Structure, Config, Foundational Files

## Metadata

- **Phase:** 1 (Mirror Post Scaffolding)
- **Plan:** 01-01
- **Wave:** 1 (no dependencies)
- **Requirements:** REQ-M-001, REQ-M-006, REQ-X-042
- **Commit message:** `feat(scaffold): initialize mirror-post directory structure and config`

## Dependencies

None (first plan in phase).

## Tasks

### T1: Initialize mirror-post/ as separate git repo (D-01)

- `mkdir -p mirror-post && cd mirror-post && git init`
- Add `mirror-post/` to Krea-AI `.gitignore` (below existing `flux-krea/` entry)
- **Acceptance:** `test -d mirror-post/.git` passes; `grep 'mirror-post/' .gitignore` matches

### T2: Create full directory tree

```
mirror-post/
  src/
    index.js
    persona/spec/ archetypes/ comedy/
    brief/ (+ prompts/)
    image/modifiers/ templates/
    compositor/templates/ typography/
    grammar/
    utils/
  artifact/
  test/fixtures/inputs/ expected-briefs/
  .planning/phases/
  tasks/
```

- Each leaf dir gets `.gitkeep`; each module dir gets placeholder `index.js` with `// Barrel export — populated as modules are built`
- **Acceptance:** `find mirror-post/src -type d | wc -l` >= 12

### T3: Create package.json

```json
{
  "name": "mirror-post",
  "version": "0.1.0",
  "type": "module",
  "engines": { "node": ">=20.0.0" },
  "description": "Satirical LinkedIn post compositor",
  "main": "src/index.js",
  "scripts": { "test": "node test/harness.js" },
  "dependencies": {},
  "devDependencies": {},
  "private": true
}
```

- **Acceptance:** `node -e "const p=JSON.parse(require('fs').readFileSync('mirror-post/package.json','utf8'));console.log(p.type,p.engines.node)"` outputs `module >=20.0.0`

### T4: Create .gitignore

- `node_modules/`, `.DS_Store`, `.env`, `.env.*`, `context/`, `state/`, `.claude/worktrees/`, `.claude/hooks/*.log`

### T5: Create CLAUDE.md

- Architecture: 5-layer pipeline (Post Brief Generator -> Visual Grammar -> Image Prompt Engine -> Compositor -> Artifact UI)
- Key Files table pointing to persona spec, schema, fixtures, modifiers
- Rules: 6 hard rules from workspace CLAUDE.md (immutable persona, Post Brief contract, local diffusion only, compositor owns text, safety rails hard, Anthropic API only)
- Dependencies: Node.js 20+, React (artifact), Anthropic API, Canvas/Sharp (compositor)
- **Acceptance:** `grep 'mirror_pete.v2.json' mirror-post/CLAUDE.md` matches

### T6: Create README.md

- What it is, architecture diagram, directory structure, prerequisites (Node.js 20+), quick start (`npm test`), status
- **Acceptance:** `grep 'Node.js 20' mirror-post/README.md` matches

### T7: Create GSD planning files

- `.planning/PROJECT.md` — condensed project brief with D-01 through D-06 decisions
- `.planning/STATE.md` — initial state (Phase 1 in progress)
- `tasks/todo.md` — milestone checklist
- `tasks/lessons.md` — seed rules template
- **Acceptance:** `test -f mirror-post/.planning/PROJECT.md` passes
