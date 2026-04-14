# PLAN-01-03: Post Brief Schema + Test Fixtures

## Metadata

- **Phase:** 1 (Mirror Post Scaffolding)
- **Plan:** 01-03
- **Wave:** 2b (parallel with 01-02)
- **Requirements:** REQ-M-004, REQ-M-005, REQ-X-002
- **Depends on:** 01-01
- **Commit message:** `feat(schema): Post Brief v1 schema + 4 reference fixtures`

## Dependencies

- PLAN-01-01 must complete first (directory structure must exist).

## Tasks

### T1: Define Post Brief v1 schema in schema.js (REQ-M-004)

- File: `mirror-post/src/brief/schema.js`
- Exports: `POST_BRIEF_VERSION` ("post-brief-v1"), `SCHEMA` (shape reference), `validateBrief(brief)` -> `{ valid, errors }`
- Schema shape applies all decisions:
  - **D-02**: `props` is uniform array of `{ type, text, placement }` (text can be string or string[])
  - **D-03**: `tweet_embed` required (always present)
  - **D-04**: `nav_easter_eggs` optional (absence is valid)
  - **D-05**: `voice` added as top-level: `{ sarcasm: float 0-1, cynicism: float 0-1, warmth: float 0-1, satirical_intensity: int 1-5 }`
- Full field list: `meta` (generated_at, persona_spec_version, input_type, input_raw), `character` (name, title, tagline, avatar_prompt, recurring), `voice`, `post` (headline.text/highlight_words/highlight_color, body.text/bold_phrases/italic_phrases, hashtags), `props[]`, `tweet_embed` (author_name, handle, text, hashtags), `engagement` (reactions.count/types, comments, dominant_reaction), `image_seed` (scene_template, environment, subject_pose, mood, color_temperature), `nav_easter_eggs?`
- Hand-written validator (no external deps). ~100 lines of field/type checks.
- **Acceptance:** `node -e "import('./mirror-post/src/brief/schema.js').then(m=>console.log(m.POST_BRIEF_VERSION))"` outputs `post-brief-v1`; empty object returns `{ valid: false }`

### T2: Create Brent Vellum gold fixture (D-06: full depth)

- Input: `test/fixtures/inputs/input-brent-vellum.json` — scenario text + notes
- Expected: `test/fixtures/expected-briefs/brent-vellum.json` — EVERY field populated
- Key values: name "Brent Vellum", mug "ALIGNMENT", whiteboard ["Role Title A","B","C"], folder "ACCOMMODATION WORKFLOW", tweet handle "@BrentBehindTheDocs", nav_easter_eggs present (Evanrations), voice { sarcasm: 0.8, cynicism: 0.85, warmth: 0.25, satirical_intensity: 4 }, scene "office-middle-mgmt", 6 props in normalized array format
- **Acceptance:** passes `validateBrief()` with 0 errors

### T3: Create 3 skeleton fixtures (D-06: structural only)

- Inputs: `input-trevor-hustle.json`, `input-trevor-closer.json`, `input-pete-titles.json`
- Expected briefs: `trevor-hustle.json`, `trevor-closer.json`, `pete-titles.json`
- All required fields present with valid types; text content is "PLACEHOLDER" where not fixture-specific
- Fixture-specific values:
  - Trevor Hustle: "Trevor B.", scene "airport-hustle", mug "HUSTLE", hashtag "#BlessedAndBusy"
  - Trevor Closer: "Trevor B.", scene "office-middle-mgmt", sticky "SMILE.LIE.CLOSE", whiteboard "EXCUSES: 0"
  - Pete Titles: "Pete C.", scene "office-middle-mgmt", mug "Vague Duty Elixir"
- `nav_easter_eggs` intentionally ABSENT from skeletons (validates D-04 optional path)
- **Acceptance:** all 3 pass `validateBrief()` with 0 errors

### T4: Create test/harness.js

- ESM, zero dependencies
- Reads all `.json` from `test/fixtures/expected-briefs/`, validates each via `validateBrief()`
- Reports per-file PASS/FAIL, exits 0 if all pass, 1 if any fail
- **Acceptance:** `cd mirror-post && node test/harness.js` outputs `4 passed, 0 failed, 4 total`

### T5: Wire barrel exports

- `src/brief/index.js`: re-exports from `./schema.js`
- `src/index.js`: re-exports from `./brief/index.js`
- **Acceptance:** `node -e "import('./mirror-post/src/index.js').then(m=>console.log(m.POST_BRIEF_VERSION))"` outputs `post-brief-v1`
