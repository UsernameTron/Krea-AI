---
status: awaiting_human_verify
trigger: "Mirror Post integration test returns 0/5. TIMEOUT on all 5, SEMANTIC failures on 3/5."
created: 2026-04-14T00:00:00Z
updated: 2026-04-14T00:01:00Z
---

## Current Focus

hypothesis: CONFIRMED — three distinct root causes, all fixed
test: 111 unit/mock tests re-run — all pass, zero regressions
expecting: Integration tests to pass when re-run with real API key
next_action: Await human verification with real Anthropic API key

## Symptoms

expected: All 5 integration test scenarios pass when run with a real Anthropic API key
actual: 0/5 pass. All 5 exceed 15s timeout (actual 39-45s). 3/5 fail SAFETY_INDIVIDUAL_TARGET and BANNED_CORPORATE_JARGON.
errors: TIMEOUT (all 5), SAFETY_INDIVIDUAL_TARGET (3/5), BANNED_CORPORATE_JARGON (3/5)
reproduction: cd mirror-post && ANTHROPIC_API_KEY=sk-ant-... node test/integration.js
started: First run against real API. 111 unit/mock tests pass green.

## Eliminated

## Evidence

- timestamp: 2026-04-14T00:00:10Z
  checked: test/integration.js timeout constant
  found: Hardcoded 15000ms at line 59. No named constant. Opus 4.6 with ~2600-token system prompt + structured output takes 39-45s per call.
  implication: 15s is unrealistic for real API. Needs 90s to accommodate latency variance.

- timestamp: 2026-04-14T00:00:20Z
  checked: src/brief/validator.js TARGETING_PATTERNS
  found: /@\w{3,}/ regex matches any @-handle in text. tweet_embed.text and prop text are scanned. LLM generates fake email addresses (e.g., karen@synergycorp.com) on business card props, triggering the @handle regex.
  implication: Safety rails in system prompt don't prohibit fictional contact details. LLM fills in "realistic" business card details including fake emails.

- timestamp: 2026-04-14T00:00:30Z
  checked: src/brief/prompts/safety-rails.md
  found: No rule against generating fictional contact details (emails, phones, @handles). Only rules about real people, punching down, harassment.
  implication: Need explicit rule #6 prohibiting fake contact details in any text field.

- timestamp: 2026-04-14T00:00:40Z
  checked: src/brief/validator.js containsUnquoted() and persona-voice-distillate.md
  found: Validator correctly allows quoted jargon and flags unquoted. The persona-voice-distillate lists "align" and "transform" as banned but instruction density is low — appears in a dense paragraph easily lost in ~2600 tokens. Safety-rails.md (highest-attention position, Pattern 5) has no jargon rule.
  implication: LLM uses "align" and "transform" naturally because the ban instruction lacks emphasis. Fix: reinforce in both persona-voice-distillate (stronger wording) AND safety-rails (new rule #7 in final position).

- timestamp: 2026-04-14T00:00:50Z
  checked: All 111 unit/mock tests after all three fixes
  found: 111 passed, 0 failed. Zero regressions across all 8 test files.
  implication: Fixes are structurally safe. Integration test needs real API key to confirm semantic fixes.

## Resolution

root_cause: Three independent issues. (1) TIMEOUT: 15s limit was calibrated for mocks, not real Opus 4.6 API calls which take 39-45s. (2) SAFETY_INDIVIDUAL_TARGET: Safety rails prompt lacked explicit prohibition on fictional contact details (emails, phone numbers, @handles), so LLM generated realistic-looking details on business cards and in text, triggering the /@\w{3,}/ validator regex. (3) BANNED_CORPORATE_JARGON: Words "align" and "transform" are on the banned list but the system prompt instruction to avoid them was buried in a dense paragraph and not reinforced in the safety-rails block (highest-attention position). LLM used them unquoted.

fix: (1) Raised timeout from 15s to 90s in test/integration.js with calibration comment. (2) Added rule #6 to safety-rails.md explicitly prohibiting fictional contact details in all text fields. (3) Strengthened corporate jargon ban in persona-voice-distillate.md with explicit validation-failure warning and quotation-mark requirement, plus added rule #7 to safety-rails.md reinforcing the ban in the highest-attention prompt position.

verification: 111 unit/mock tests pass. Integration test awaiting human verification with real API key.

files_changed:
- mirror-post/test/integration.js
- mirror-post/src/brief/prompts/safety-rails.md
- mirror-post/src/brief/prompts/persona-voice-distillate.md
