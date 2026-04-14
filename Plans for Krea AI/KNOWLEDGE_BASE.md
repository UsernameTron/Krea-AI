# MIRROR POST × FLUX-KREA — Comprehensive Knowledge Base & Reference

## Document Purpose

This is the single reference document for building Mirror Post on top of FLUX-Krea. It combines the GSD project assessment, performance stabilization plan, architectural analysis, and the Mirror Post integration strategy into one knowledge base.

---

## PART 1: GSD STATE ASSESSMENT

### State: BETWEEN_MILESTONES with BROWNFIELD_UNCLAIMED overlay

### Evidence

The `flux-krea/` application is a completed, production-hardened codebase. The outer `Krea-AI/` workspace is partially organized but has no active development target.

**flux-krea/ (inner project) — SHIPPED:**
- Branch: `main`, all feature branches merged and deleted (session-log 2026-03-22)
- 299 tests passing, 96% coverage, CI green (pytest + ruff + mypy)
- `tasks/todo.md` says "Production Hardening — COMPLETE" with "No blockers. No pending work."
- Clean architecture: 12 Python files, 3 optimization modules, unified CLI/web/benchmark/profiler
- Last commit: `01ac85a` — mypy fixes, merged to main

**Krea-AI/ (outer workspace) — UNCLAIMED:**
- `tasks/todo.md` has a single incomplete item: "Phase 0 bootstrap"
- `memory.md` contains a ~5,000-word optimization guide (from an earlier GPT session) that was never implemented
- `Plans for Krea AI/` contains Mirror Post plans (PLAN_01, PLAN_02) but no GSD `.planning/` directory
- `state/pattern-context.md` has Claude Code KB patterns loaded but no active project applying them
- No `.planning/PROJECT.md`, no `ROADMAP.md`, no `STATE.md` at the outer level
- The `flux-krea-claude-code-prompt.md` is the original consolidation prompt — the work it describes is DONE

### Diagnosis

The inner project shipped successfully. The outer workspace was set up to continue development but never got a clear next objective. The Mirror Post concept is that objective — it transforms flux-krea from a general-purpose image generator into a purpose-built satirical content tool. But before that integration can happen, flux-krea itself needs performance optimization. The `memory.md` optimization guide identified the right areas but was never executed.

---

## PART 2: ULTRATHINK ANALYSIS

### Architect Perspective

Two projects need to converge: flux-krea (image generation engine) and Mirror Post (content intelligence layer). The architecture is a pipeline where Mirror Post generates a Post Brief (structured JSON describing every element of a satirical LinkedIn post), and flux-krea generates the base hero image from that brief. A compositor then overlays LinkedIn UI chrome, text, and props.

The key architectural question is whether flux-krea should be modified to accept Post Brief JSON as input, or whether Mirror Post should generate a standard text prompt that flux-krea processes normally. The answer is the latter — flux-krea stays a general-purpose image generator. Mirror Post's image prompt engine translates the Post Brief into a standard positive/negative prompt pair. This keeps the systems decoupled and flux-krea usable for non-Mirror-Post purposes.

The performance bottleneck is generation latency. On M4 Pro, a 1024x1024 image at 28 steps takes roughly 60-90 seconds. For Mirror Post to be a usable creative tool, this needs to drop to 30-45 seconds. The optimization targets are: scheduler selection (fewer steps), MPS memory management (fewer stalls), and optional Neural Engine offloading (parallel compute).

### Research Perspective

The `memory.md` optimization guide covers the right technical areas but was written for a pre-consolidation codebase (references `flux_metal_kernels.py`, `flux_neural_engine_accelerator.py` etc. which no longer exist). The recommendations need to be mapped to the current architecture. Specifically, the scheduler/sampler recommendation (switch from default to Euler or DPM++) is the highest-impact change — it can reduce step count from 28 to 20 while maintaining quality, cutting ~28% off generation time with zero code complexity.

The `torch.compile()` recommendation is worth attempting but risky on MPS — PyTorch's Inductor backend has inconsistent MPS support. This should be tested in isolation before integration. The CoreML/Neural Engine path is already stubbed in `optimizers/neural_engine.py` but explicitly disabled ("not altering in-place to avoid unintended regressions"). Enabling it requires careful testing.

### Coder Perspective

The codebase is clean and well-tested. Performance work should follow the existing patterns: config-driven toggles, fallback chains, and comprehensive test coverage. The scheduler change is a ~10-line modification to `pipeline.py`. The `torch.compile` experiment is a feature flag. The memory management improvements are config.yaml tweaks.

The Mirror Post integration doesn't touch flux-krea source code at all — it's a separate project that calls `python main.py generate -p "..."` or imports `FluxKreaPipeline` as a library. The only flux-krea change needed is potentially adding a `--prompt-file` flag for Mirror Post to pass generation parameters (resolution, steps, guidance) without editing config.yaml.

### Tester Perspective

The existing 299 tests cover the current architecture thoroughly. Performance work needs benchmark baselines BEFORE changes, measured on Pete's actual M4 Pro hardware. The profiler (`python main.py profile`) already provides stage-by-stage timing. Run it before any optimization, capture the numbers, then measure after each change.

The risk areas are: scheduler changes affecting image quality (test with same seed, compare outputs), `torch.compile` causing different outputs or crashes on MPS, and Neural Engine activation causing data conversion overhead that negates the compute benefit.

### Synthesis — Priority Order

1. **Baseline benchmark** — run profiler, capture numbers, establish the target
2. **Scheduler optimization** — swap to Euler/DPM++, reduce steps to 20, compare quality
3. **MPS memory tuning** — adjust watermark ratios and memory fraction based on actual usage patterns
4. **torch.compile experiment** — try compiling UNet forward, measure impact, feature-flag it
5. **Neural Engine activation** — enable the existing CoreML code path for VAE decoder, measure
6. **Mirror Post integration point** — add prompt-from-file or structured input mode

---

## PART 3: PERFORMANCE STABILIZATION PLAN

### Milestone 1: Establish Baselines

**Objective:** Measure current performance on Pete's M4 Pro hardware so every subsequent change is measured against real numbers.

**Tasks:**

```bash
cd /Users/cpconnor/projects/Krea-AI/flux-krea

# Standard generation baseline
python main.py profile --prompt "a professional man seated at a corporate desk, medium shot, photorealistic" --steps 28 --width 1024 --height 1024

# Lower step count baseline
python main.py profile --prompt "a professional man seated at a corporate desk, medium shot, photorealistic" --steps 20 --width 1024 --height 1024

# Mirror Post target resolution
python main.py profile --prompt "a professional man seated at a corporate desk, medium shot, photorealistic" --steps 28 --width 1280 --height 720

# Quick benchmark
python main.py benchmark --quick
```

**Deliverable:** `docs/BASELINE_BENCHMARKS.md` — all numbers, hardware spec, PyTorch version, date.

**Acceptance criteria:** Baseline file exists. No code changes in this milestone.

---

### Milestone 2: Scheduler Optimization

**Objective:** Reduce generation time by 25-35% through scheduler selection and step count reduction.

**Implementation:**

Add `scheduler` field to `GenerationConfig` in `config.py`:
```python
@dataclass
class GenerationConfig:
    width: int = 1024
    height: int = 1024
    steps: int = 20  # reduced default
    guidance_scale: float = 4.5
    max_sequence_length: int = 256
    scheduler: str = "default"
```

Add scheduler selection to `pipeline.py` in `_load_at_level`:
```python
scheduler_name = self.config.generation.scheduler if hasattr(self.config.generation, 'scheduler') else "default"
if scheduler_name == "euler":
    from diffusers import EulerDiscreteScheduler
    self.pipeline.scheduler = EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)
elif scheduler_name == "dpm++":
    from diffusers import DPMSolverMultistepScheduler
    self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        self.pipeline.scheduler.config, algorithm_type="dpmsolver++"
    )
```

Update `config.yaml`:
```yaml
generation:
  scheduler: "euler"
  steps: 20
```

**Verification:** Same seed comparison at 20 steps (new) vs 28 steps (old). Profiler comparison. All 299 tests still pass.

---

### Milestone 3: MPS Memory Optimization

**Objective:** Reduce MPS memory stalls and cache clearing overhead.

**Investigation areas:**
- Is `_cleanup_memory()` called too frequently? It runs after every generation AND in inference context finally block.
- Is the VAE autocast in `optimizers/metal.py` helping or adding overhead?
- Test watermark ratio values: 0.7, 0.8, 0.9 with memory fractions 0.8, 0.85, 0.9
- Test allocator policy: "expandable_segments" vs "page"

**Deliverable:** Optimal config values documented. config.yaml updated.

---

### Milestone 4: torch.compile Experiment

**Objective:** Determine if torch.compile provides meaningful speedup on MPS.

**Implementation:** Add `torch_compile` boolean and `torch_compile_mode` string to config. Apply to `self.pipeline.transformer` after load. Feature-flagged, default off. Wrap in try/except.

**Decision rule:** If speedup < 5% across generations 2-5 (excluding compile overhead), leave disabled by default.

---

### Milestone 5: Neural Engine VAE Decoder

**Objective:** Offload VAE decoding to ANE for parallel compute.

**Current state:** `optimizers/neural_engine.py` has conversion code but explicitly does NOT replace modules at runtime.

**Implementation:** Enable the replacement path for VAE decoder only. Test conversion success, measure decode time vs PyTorch MPS baseline. Feature-flagged.

**Risk:** CoreML conversion may fail for FLUX VAE architecture. Clean fallback required.

---

### Milestone 6: Mirror Post Integration Point

**Objective:** Enable flux-krea to accept structured prompts from Mirror Post.

**Implementation:** Add `--prompt-file` flag to `main.py`:
```json
{
  "positive_prompt": "medium shot of a silver-haired man...",
  "negative_prompt": "text, watermark, low quality...",
  "width": 1280,
  "height": 720,
  "steps": 20,
  "guidance_scale": 4.5,
  "seed": 42
}
```

This JSON is the contract between Mirror Post and flux-krea.

---

## PART 4: MIRROR POST INTEGRATION ARCHITECTURE

### System Connection Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    MIRROR POST                           │
│                                                          │
│  User Input → Post Brief Generator → Post Brief JSON     │
│                    (Anthropic API)        │               │
│                                          ↓               │
│                               Image Prompt Engine        │
│                               (deterministic)            │
│                                          │               │
│                                          ↓               │
│                               prompt-file.json           │
└──────────────────────────────────┬───────────────────────┘
                                   │
                                   ↓
┌──────────────────────────────────┴───────────────────────┐
│                    FLUX-KREA                              │
│                                                          │
│  prompt-file.json → FluxKreaPipeline.generate() → hero.png│
└──────────────────────────────────┬───────────────────────┘
                                   │
                                   ↓
┌──────────────────────────────────┴───────────────────────┐
│                    COMPOSITOR                            │
│                                                          │
│  hero.png + Post Brief JSON → LinkedIn overlay → final.png│
└─────────────────────────────────────────────────────────┘
```

**Key design principle:** flux-krea requires zero source code modifications for Mirror Post. Mirror Post generates a standard prompt file. flux-krea processes it like any other prompt. The compositor reads both the hero image and the Post Brief. Systems are fully decoupled.

### Optimal Generation Settings for Mirror Post

```yaml
generation:
  width: 1280
  height: 720
  steps: 20
  guidance_scale: 4.5
  scheduler: "euler"
  max_sequence_length: 256
```

Why 1280x720: Final composite is 1920x1080. Hero image fills ~60% right side. 1280x720 provides sufficient resolution for subject + props + background.

---

## PART 5: FILE INDEX & REFERENCE MAP

### flux-krea/ Core Files

| File | Purpose | Mirror Post Relevance |
|------|---------|----------------------|
| `main.py` | CLI entry point | Add `--prompt-file` flag |
| `pipeline.py` | Unified pipeline | Add scheduler selection |
| `config.py` | Configuration | Add scheduler + compile fields |
| `config.yaml` | Default settings | Tuning target |
| `app.py` | Gradio web UI | Not used by Mirror Post |
| `optimizers/metal.py` | MPS optimization | Performance tuning |
| `optimizers/neural_engine.py` | CoreML/ANE | Experimental — enable VAE |
| `optimizers/thermal.py` | Thermal management | Background concern |
| `utils/profiler.py` | Profiling | Performance measurement |

### Mirror Post Files (from PLAN_01 and PLAN_02)

| File | Purpose | Depends On |
|------|---------|------------|
| `src/persona/spec/mirror_pete.v2.json` | Voice spec | Canonical source |
| `src/persona/archetypes/library.json` | Archetype catalog | Reference data |
| `src/persona/comedy/structures.json` | Comedy formulas | Reference data |
| `src/brief/generator.js` | Post Brief generation | Persona + Anthropic API |
| `src/brief/schema.js` | Brief validation | Defines contract |
| `src/image/prompt-builder.js` | Brief → diffusion prompt | Brief + modifiers |
| `src/grammar/props.json` | Prop taxonomy | Reference data |
| `src/grammar/zones.json` | Composition zones | Reference data |
| `src/compositor/renderer.js` | Final image assembly | Canvas + Brief + zones |
| `artifact/mirror-post-generator.jsx` | React UI | All of the above |

### Existing Reference Materials

| File | Location | Status |
|------|----------|--------|
| `memory.md` | `/Krea-AI/` | Unexecuted — valid recommendations, old file names |
| `pattern-context.md` | `/Krea-AI/state/` | Active patterns applied to config architecture |
| `DEVOPS-HANDOFF.md` | `/flux-krea/docs/` | Current |
| `flux-krea-claude-code-prompt.md` | `/Krea-AI/` | Historical — work is DONE |
| `PLAN_01_SCAFFOLDING.md` | `/Plans for Krea AI/` | Active — Mirror Post framework |
| `PLAN_02_MODULES.md` | `/Plans for Krea AI/` | Active — Mirror Post modules |

---

## PART 6: EXECUTION SEQUENCE

### Phase A: Performance Stabilization (flux-krea) — Week 1-2

```
1. Run baselines (Milestone 1)                    Day 1
2. Implement scheduler selection (Milestone 2)     Day 2-3
3. MPS memory tuning (Milestone 3)                 Day 4-5
4. torch.compile experiment (Milestone 4)          Day 6-7
5. Neural Engine VAE test (Milestone 5)            Day 8-9
6. Add --prompt-file flag (Milestone 6)            Day 10
```

### Phase B: Mirror Post Construction — Week 3-5

```
7. Scaffolding (PLAN_01)                           Day 11-12
8. Post Brief Generator (PLAN_02, Module 1)        Day 13-17
9. Visual Grammar (PLAN_02, Module 2)              Day 18-19
10. Image Prompt Engine (PLAN_02, Module 3)        Day 20-22
11. Compositor (PLAN_02, Module 4)                 Day 23-26
12. Artifact UI (PLAN_02, Module 5)                Day 27-30
```

### Phase C: Integration Testing — Week 6

```
13. End-to-end: input → brief → prompt → flux-krea → hero → composite
14. 4 reference post roundtrips (Brent Vellum, Trevor x2, Pete)
15. Performance: total pipeline < 2 minutes end-to-end
16. Quality: outputs match reference post quality
```

---

## PART 7: CLAUDE CODE PROMPT — PERFORMANCE STABILIZATION

Copy this into Claude Code to begin Phase A:

```
cd /Users/cpconnor/projects/Krea-AI/flux-krea

I need to optimize FLUX-Krea generation performance on my M4 Pro.
Execute these steps in order. After EACH step, run the profiler to
measure impact.

STEP 1 — BASELINE
Run profiler:
  python main.py profile --prompt "a professional man seated at a
  corporate desk, medium shot, photorealistic, office environment,
  shallow depth of field" --steps 28

Save results to docs/BASELINE_BENCHMARKS.md.

STEP 2 — SCHEDULER SELECTION
Add scheduler field to GenerationConfig in config.py (default: "default").
In pipeline.py, add scheduler selection after FluxPipeline.from_pretrained().
Support: "default", "euler", "dpm++".
Update config.yaml: scheduler: "euler", steps: 20.
Add tests. Run full suite. Run profiler. Compare quality with seed 42.

STEP 3 — MPS TUNING
Review MPS config. Check _cleanup_memory() call frequency.
Adjust ratios if watermark errors appear. Document findings.

STEP 4 — torch.compile EXPERIMENT
Add torch_compile boolean to config (default: false).
Try torch.compile on self.pipeline.transformer.
Run 3 sequential generations to amortize. Compare gen 2-3 against
baseline. Document. If speedup < 5%, leave disabled.

STEP 5 — PROMPT FILE INPUT
Add --prompt-file flag to main.py generate subcommand.
JSON: { positive_prompt, negative_prompt, width, height, steps,
guidance_scale, seed }.
Add test. This is the Mirror Post integration contract.

After all steps: update CLAUDE.md, commit each step separately
with conventional commit messages.
```

---

*Two systems. One pipeline. Zero patience for theater.*
