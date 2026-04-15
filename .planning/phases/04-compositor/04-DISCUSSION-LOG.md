# Phase 4: Compositor - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-14
**Phase:** 04-compositor
**Areas discussed:** Rendering engine, Chrome visual style, Text overlay styling, Tweet embed card design

---

## Rendering Engine

| Option | Description | Selected |
|--------|-------------|----------|
| Sharp + node-canvas hybrid | Canvas 2D for text/gradient, Sharp for image compositing. 2 npm deps, requires Homebrew Cairo/Pango. | ✓ |
| Sharp + SVG only | Single dep, zero system installs. Text as SVG strings. More verbose but deterministic. | |
| node-canvas only | Full Canvas 2D for everything. Requires Homebrew system deps. | |
| Puppeteer | Full CSS rendering. Eliminated — subpixel variation violates REQ-X-052. | |

**User's choice:** Sharp + node-canvas hybrid (Recommended)
**Notes:** Puppeteer eliminated pre-presentation due to determinism requirement. Hybrid gives best text API (Canvas 2D) with best image compositing (Sharp).

---

## Chrome Visual Style

| Option | Description | Selected |
|--------|-------------|----------|
| Accept all 4 recommendations | Simplified SVG nav icons, custom blue #0b5ca8, initials avatar, emoji reaction circles | ✓ |
| Custom SVG reaction icons instead | Same but 6 hand-drawn SVG reaction icons instead of emoji | |
| Silhouette avatar instead | Generic gray silhouette placeholder instead of initials | |

**User's choice:** Accept all 4 recommendations
**Notes:** Package deal: LinkedIn-inspired simplified SVG icons, custom corporate blue #0b5ca8, circle-with-initials avatar from character.name, emoji Unicode reaction circles at 24-32px.

---

## Text Overlay Styling

| Option | Description | Selected |
|--------|-------------|----------|
| Accept all 5 recommendations | Inter font, gold #FFD700 highlights, pre-tokenized bold/italic, subtle shadow, truncation + validator | ✓ |
| Source Sans Pro instead of Inter | Closer to LinkedIn's actual typeface. Same approach otherwise. | |
| Skip body bold/italic for MVP | Single-style body, defer formatting. Simpler but loses comedic timing. | |

**User's choice:** Accept all 5 recommendations
**Notes:** Inter bundled as TTF, gold highlight from schema's highlight_color field, inline bold/italic with pre-tokenized segments, ctx.shadowBlur=8 for readability, hard truncation + upstream Brief Validator char limits.

---

## Tweet Embed Card Design

| Option | Description | Selected |
|--------|-------------|----------|
| Accept realistic embed | White rounded card, avatar circle, blue verification badge, handle, text, hashtags. Defensive null handling. | ✓ |
| Include timestamp + like/RT counts | Same plus fake timestamp and engagement numbers. More render work. | |
| Skip verification badge | Same but no blue checkmark. Reduces satirical punch. | |

**User's choice:** Accept realistic embed (Recommended)
**Notes:** Verisimilitude drives the satirical mechanism. Verified badge on absurd characters is itself a punchline. Timestamp and like/RT counts deferred as optional gilding.

---

## Claude's Discretion

- Sharp composite layer ordering and blend modes
- Canvas 2D rendering optimization (buffer reuse, state save/restore)
- SVG icon drawing approach (inline Canvas paths vs SVG file assets)
- Inter font weight selection beyond Regular/Bold
- Emoji Unicode codepoints for each reaction type
- Tweet embed card internal layout (padding, font sizes, spacing)
- Gradient mask implementation approach
- File organization within src/compositor/

## Deferred Ideas

- Custom SVG reaction icons (upgrade from emoji if visual review flags quality)
- Tweet embed timestamp and like/retweet counts
- AI-generated profile avatars (violates REQ-X-052)
- Source Sans Pro as alternative font
- Dynamic zone computation (zones fixed for v1)
