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

The codebase is clean and well-tested. Performance work should follow the existing patterns: config-driven toggles, fallback chains, and comprehensive test coverage. The scheduler change is a ~10-line modification to `pipeline.py`. The `torch.compile` experiment is a feature flag. The Memory management improvements are config.yaml tweaks.

The Mirror Post integration doesn't touch flux-krea source code at all — it's a separate project that calls `python main.py generate -p "..."` or imports `FluxKreaPipeline` as a library. The only flux-krea change needed is potentially adding a `--config-override` flag for Mirror Post to pass generation parameters (resolution, steps, guidance) without editing config.yaml.

### Tester Perspective

The existing 299 tests cover the current architecture thoroughly. Performance work needs benchmark baselines BEFORE changes, measured on Pete's actual M4 Pro hardware. The profiler (`python main.py profile`) already provides stage-by-stage timing. Run it before any optimization, capture the numbers, then measure after each change.

The risk areas are: scheduler changes affecting image quality (test with same seed, compare outputs), `torch.compile` causing different outputs or crashes on MPS, and Neural Engine activation causing data conversion overhead that negates the compute benefit.

### Synthesis

**Priority order for flux-krea performance work:**

1. **Baseline benchmark** — run profiler, capture numbers, establish the target
2. **Scheduler optimization** — swap to Euler/DPM++, reduce steps to 20, compare quality
3. **MPS memory tuning** — adjust watermark ratios and memory fraction based on actual usage patterns
4. **torch.compile experiment** — try compiling UNet forward, measure impact, feature-flag it
5. **Neural Engine activation** — enable the existing CoreML code path for VAE decoder, measure
6. **Mirror Post integration point** — add prompt-from-file or structured input mode

---

## PART 3: PERFORMANCE STABILIZATION PLAN

### Milestone 1: Establish Baselines (GSD Phase: discuss → plan)

**Objective:** Measure current performance on Pete's M4 Pro hardware so every subsequent change is measured against real numbers, not guesses.

**Tasks:**

Run the full profiler suite and capture outputs:
```bash
cd /Users/cpconnor/projects/Krea-AI/flux-krea

# Standard generation baseline
python main.py profile --prompt "a professional man seated at a corporate desk, medium shot, photorealistic" --steps 28 --width 1024 --height 1024

# Lower step count baseline
python main.py profile --prompt "a professional man seated at a corporate desk, medium shot, photorealistic" --steps 20 --width 1024 --height 1024

# Higher resolution baseline (Mirror Post target)
python main.py profile --prompt "a professional man seated at a corporate desk, medium shot, photorealistic" --steps 28 --width 1280 --height 720

# Quick benchmark
python main.py benchmark --quick
```

**Capture:** total time, time per step, stage breakdown (model load, text encoding, denoising, VAE decode), MPS memory peak.

**Deliverable:** `docs/BASELINE_BENCHMARKS.md` with all numbers, hardware spec, PyTorch version, date.

**Acceptance criteria:** Baseline file exists with reproducible numbers. No code changes in this milestone.

---

### Milestone 2: Scheduler Optimization (GSD Phase: execute → verify)

**Objective:** Reduce generation time by 25-35% through scheduler selection and step count reduction.

**Current state:** Default scheduler from diffusers FluxPipeline (likely Euler or Flow Match). Steps: 28. Guidance: 4.5.

**Implementation:**

Add scheduler configuration to `config.yaml`:
```yaml
generation:
  scheduler: "euler"  # euler, dpm++, default
  width: 1024
  height: 1024
  steps: 20  # reduced from 28
  guidance_scale: 4.5
```

Add scheduler selection to `pipeline.py` in the `_load_at_level` method:
```python
# After pipeline = FluxPipeline.from_pretrained(...)
scheduler_name = self.config.generation.get("scheduler", "default")
if scheduler_name == "euler":
    from diffusers import EulerDiscreteScheduler
    self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
        self.pipeline.scheduler.config
    )
elif scheduler_name == "dpm++":
    from diffusers import DPMSolverMultistepScheduler
    self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        self.pipeline.scheduler.config, algorithm_type="dpmsolver++"
    )
```

Add `scheduler` field to `GenerationConfig` dataclass in `config.py`:
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

**Verification:** Run the same profiler commands from Milestone 1. Compare total time and time_per_step. Generate images with same seed at 20 and 28 steps with each scheduler. Visual comparison for quality regression.

**Test additions:**
- `test_pipeline.py`: Test scheduler selection for each option
- `test_config.py`: Test scheduler field validation

**Acceptance criteria:**
- 20 steps with Euler or DPM++ produces visually comparable output to 28 steps with default scheduler (same seed comparison)
- Total generation time reduced by ≥20%
- All 299 existing tests still pass
- New scheduler tests pass

---

### Milestone 3: MPS Memory Optimization (GSD Phase: execute → verify)

**Objective:** Reduce MPS memory stalls and cache clearing overhead.

**Current state:** Watermark ratio 0.8, memory fraction 0.85, allocator policy "expandable_segments". These are reasonable defaults but may not be optimal for Pete's specific hardware.

**Implementation:**

Create a tuning script that tests different MPS configurations:
```python
# utils/mps_tuner.py
"""Test different MPS memory configurations and report performance."""
```

Test matrix:
- `mps_watermark_ratio`: [0.7, 0.8, 0.9]
- `mps_memory_fraction`: [0.8, 0.85, 0.9]
- `mps_allocator_policy`: ["expandable_segments", "page"]

For each combination: run 3 generations, measure time and memory peak, record any cache clearing errors.

**Also investigate:**
- Is `torch.mps.empty_cache()` being called too frequently? The `_cleanup_memory` method runs after every generation AND in the inference context manager's finally block. If the cache clear is causing watermark errors that add overhead, reduce frequency.
- Is the VAE autocast in `optimizers/metal.py` actually helping? The `metal_vae_forward` wraps in float16 autocast — measure with and without.

**Acceptance criteria:**
- Optimal MPS config identified for M4 Pro
- config.yaml updated with optimal values
- Zero watermark ratio errors in normal operation
- Memory peak documented

---

### Milestone 4: torch.compile Experiment (GSD Phase: execute → verify)

**Objective:** Determine if torch.compile provides meaningful speedup on MPS.

**Implementation:**

Add compile option to config:
```yaml
optimization:
  torch_compile: false  # experimental
  torch_compile_mode: "reduce-overhead"  # reduce-overhead, max-autotune
```

Add to `pipeline.py` after model load:
```python
if self.config.get("optimization", {}).get("torch_compile", False):
    mode = self.config.get("optimization", {}).get("torch_compile_mode", "reduce-overhead")
    try:
        self.pipeline.transformer = torch.compile(
            self.pipeline.transformer, mode=mode
        )
        logger.info("torch.compile applied to transformer (mode=%s)", mode)
    except Exception as e:
        logger.warning("torch.compile failed, continuing without: %s", e)
```

**Key considerations:**
- First inference after compile will be slower (compilation overhead)
- Subsequent inferences should be faster
- MPS backend support for Inductor is incomplete — may fall back to eager mode
- Memory usage may increase
- Output correctness must be verified (same seed, compare pixel values)

**Verification:** Run 5 sequential generations. First one includes compile overhead. Compare generations 2-5 against baseline. If speedup < 5%, disable by default (not worth the complexity).

**Acceptance criteria:**
- Feature works behind config flag
- Measured speedup documented (even if negative — that's useful information)
- Graceful fallback if compile fails
- No test regressions

---

### Milestone 5: Neural Engine VAE Decoder (GSD Phase: execute → verify)

**Objective:** Offload VAE decoding to ANE for parallel compute.

**Current state:** `optimizers/neural_engine.py` has `NeuralEngineOptimizer` with CoreML conversion code. It can assess model compatibility and attempt VAE decoder conversion. But it explicitly does NOT replace the original modules at runtime.

**Implementation:**

Enable the VAE decoder replacement path:
```python
# In neural_engine.py, modify optimize_pipeline to actually use compiled models
def optimize_pipeline(self, pipeline):
    if not self.acceleration_enabled:
        return pipeline
    
    # Convert VAE decoder to CoreML
    vae_result = self._convert_vae_decoder(pipeline.vae.decoder)
    if vae_result and vae_result.get("success"):
        # Replace VAE decode with CoreML version
        original_decode = pipeline.vae.decode
        compiled_model = vae_result["model"]
        
        def coreml_vae_decode(latents, **kwargs):
            # Convert tensor → numpy → CoreML input
            # Run prediction
            # Convert output → tensor
            ...
        
        pipeline.vae.decode = coreml_vae_decode
        logger.info("VAE decoder replaced with CoreML/ANE version")
    
    return pipeline
```

**Risk:** CoreML conversion may fail for the FLUX VAE architecture. The existing code notes that T5-XXL exceeds ANE limits. VAE decoder is smaller but still complex.

**Verification:**
- Convert VAE decoder, run generation, compare output quality
- Measure VAE decode time: PyTorch MPS vs CoreML ANE
- If ANE is slower or produces artifacts, keep disabled

**Acceptance criteria:**
- CoreML VAE decoder conversion succeeds or fails gracefully
- If successful: measurable speedup in VAE decode stage
- If unsuccessful: documented as known limitation, clean fallback
- Feature behind config flag

---

### Milestone 6: Mirror Post Integration Point (GSD Phase: plan → execute)

**Objective:** Enable flux-krea to accept structured prompts from Mirror Post without modifying the core pipeline.

**Implementation:**

Add a `--prompt-file` flag to main.py:
```python
gen_parser.add_argument("--prompt-file", type=str, help="Read prompt from JSON file")
```

JSON format:
```json
{
  "positive_prompt": "medium shot of a silver-haired man in his 50s...",
  "negative_prompt": "text, watermark, low quality...",
  "width": 1280,
  "height": 720,
  "steps": 20,
  "guidance_scale": 4.5,
  "seed": 42
}
```

This is the contract between Mirror Post's image prompt engine and flux-krea. Mirror Post generates this JSON. flux-krea consumes it.

**Also add:** A Python library interface for direct import:
```python
from pipeline import FluxKreaPipeline
from config import get_config

config = get_config(
    optimization_level="standard",
    **{"generation.steps": 20, "generation.width": 1280, "generation.height": 720}
)
pipeline = FluxKreaPipeline(config)
pipeline.load()
image, metrics = pipeline.generate(prompt="...")
pipeline.save_image(image, "mirror_post_output")
pipeline.unload()
```

**Acceptance criteria:**
- `--prompt-file` flag works end-to-end
- Library import works without CLI
- Mirror Post can generate images by producing a JSON file and calling flux-krea

---

## PART 4: MIRROR POST INTEGRATION ARCHITECTURE

### How the Systems Connect

```
┌─────────────────────────────────────────────────────────────────┐
│                        MIRROR POST                               │
│                                                                  │
│  User Input ──→ Post Brief Generator ──→ Post Brief JSON         │
│                       (Anthropic API)         │                  │
│                                               ↓                  │
│                                    Image Prompt Engine           │
│                                    (deterministic)               │
│                                               │                  │
│                                               ↓                  │
│                                    prompt-file.json              │
└───────────────────────────────────────┬──────────────────────────┘
                                        │
                                        ↓
┌───────────────────────────────────────┴──────────────────────────┐
│                        FLUX-KREA                                 │
│                                                                  │
│  prompt-file.json ──→ FluxKreaPipeline.generate() ──→ hero.png  │
│                                                                  │
└───────────────────────────────────────┬──────────────────────────┘
                                        │
                                        ↓
┌───────────────────────────────────────┴──────────────────────────┐
│                        COMPOSITOR                                │
│                                                                  │
│  hero.png + Post Brief JSON ──→ LinkedIn UI overlay ──→ final.png│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Mirror Post and flux-krea communicate through two files: `prompt-file.json` (Mirror Post → flux-krea) and the output image `hero.png` (flux-krea → Compositor). The Post Brief JSON also goes directly to the Compositor for text overlays, engagement metrics, and UI chrome.

This architecture means:
- flux-krea requires zero source code modifications for Mirror Post integration
- Mirror Post can be developed and tested independently using manual prompts
- The compositor can be developed and tested with placeholder hero images
- Any diffusion model could replace flux-krea (Stable Diffusion, SDXL, etc.) — the interface is just a prompt file

### Optimal Generation Settings for Mirror Post

Based on the example outputs and composition zone requirements:

```yaml
# Mirror Post generation profile
generation:
  width: 1280       # 16:9 for LinkedIn post format
  height: 720       # Landscape orientation
  steps: 20         # Reduced with Euler scheduler
  guidance_scale: 4.5
  scheduler: "euler"
  max_sequence_length: 256  # Full prompt fidelity for detailed scenes
```

Why 1280x720: The final composite is 1920x1080 (LinkedIn post dimensions). The hero image occupies roughly 60% of the right side of the frame. 1280x720 gives enough resolution for the subject + props + background while the left 40% is text overlay (rendered by compositor, not diffusion).

---

## PART 5: FILE INDEX & REFERENCE MAP

### flux-krea/ Core Files

| File | Purpose | Stability | Mirror Post Relevance |
|------|---------|-----------|----------------------|
| `main.py` | CLI entry point | Stable — add `--prompt-file` only | Integration point |
| `pipeline.py` | Unified pipeline | Stable — add scheduler selection | The engine |
| `config.py` | Configuration system | Stable — add scheduler + compile fields | Settings contract |
| `config.yaml` | Default settings | Update for optimal Mirror Post profile | Tuning target |
| `app.py` | Gradio web UI | Stable — no changes needed | Not used by Mirror Post |
| `optimizers/metal.py` | MPS optimization | Stable — tune watermark behavior | Performance |
| `optimizers/neural_engine.py` | CoreML/ANE | Experimental — enable VAE path | Performance |
| `optimizers/thermal.py` | Thermal management | Stable | Background concern |
| `utils/benchmark.py` | Benchmarking | Stable | Baseline measurement |
| `utils/profiler.py` | Profiling | Stable | Performance measurement |
| `utils/monitor.py` | System info | Stable | Diagnostics |

### Mirror Post Files (to be created)

| File | Purpose | Depends On |
|------|---------|------------|
| `src/persona/spec/mirror_pete.v2.json` | Voice specification | Nothing — canonical source |
| `src/persona/archetypes/library.json` | Archetype catalog | Nothing — reference data |
| `src/persona/comedy/structures.json` | Comedy formulas | Nothing — reference data |
| `src/brief/generator.js` | Post Brief generation | Persona spec + Anthropic API |
| `src/brief/schema.js` | Brief validation | Nothing — defines contract |
| `src/image/prompt-builder.js` | Brief → diffusion prompt | Brief schema + modifier library |
| `src/image/modifiers/ultra-fidelity.json` | Image quality modifiers | Nothing — reference data |
| `src/grammar/props.json` | Prop taxonomy | Nothing — reference data |
| `src/grammar/zones.json` | Composition zones | Nothing — reference data |
| `src/compositor/renderer.js` | Final image assembly | Canvas + Brief schema + zones |
| `artifact/mirror-post-generator.jsx` | React UI | All of the above |

### Existing Reference Materials

| File | Location | Content | Status |
|------|----------|---------|--------|
| `memory.md` | `/Krea-AI/` | 5,000-word optimization guide | Unexecuted — recommendations valid but mapped to old file names |
| `pattern-context.md` | `/Krea-AI/state/` | Claude Code KB patterns (Zero-Trust, Dual-Position, Adaptive Denial) | Applied to config validation architecture |
| `DEVOPS-HANDOFF.md` | `/flux-krea/docs/` | Deployment reference | Current |
| `flux-krea-claude-code-prompt.md` | `/Krea-AI/` | Original consolidation prompt | Historical — work is DONE |
| `PLAN_01_SCAFFOLDING.md` | `/Plans for Krea AI/` | Mirror Post framework plan | Active |
| `PLAN_02_MODULES.md` | `/Plans for Krea AI/` | Mirror Post module build plan | Active |

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
10. Image Prompt Engine (PLAN_02, Module 3)         Day 20-22
11. Compositor (PLAN_02, Module 4)                  Day 23-26
12. Artifact UI (PLAN_02, Module 5)                 Day 27-30
```

### Phase C: Integration Testing — Week 6

```
13. End-to-end: input → brief → prompt → flux-krea → hero image → composite
14. 4 reference post roundtrips (Brent Vellum, Trevor x2, Pete)
15. Performance validation: total pipeline < 2 minutes end-to-end
16. Quality validation: outputs match reference post quality
```

---

## PART 7: CLAUDE CODE PROMPT — PERFORMANCE STABILIZATION

Copy this prompt into Claude Code to begin Phase A:

```
cd /Users/cpconnor/projects/Krea-AI/flux-krea

I need to optimize FLUX-Krea generation performance on my M4 Pro. Execute
these steps in order. After EACH step, run the profiler to measure impact.

STEP 1 — BASELINE
Run profiler with current settings:
  python main.py profile --prompt "a professional man seated at a corporate desk, medium shot, photorealistic, office environment, shallow depth of field" --steps 28

Capture: total time, time per step, stage breakdown.
Save results to docs/BASELINE_BENCHMARKS.md.

STEP 2 — SCHEDULER SELECTION
Add a `scheduler` field to GenerationConfig in config.py (default: "default").
In pipeline.py, add scheduler selection after FluxPipeline.from_pretrained().
Support: "default", "euler" (EulerDiscreteScheduler), "dpm++" (DPMSolverMultistepScheduler).
Add to config.yaml: scheduler: "euler".
Reduce default steps from 28 to 20.
Add tests for scheduler selection to tests/test_pipeline.py and tests/test_config.py.
Run full test suite. Run profiler with euler scheduler at 20 steps.
Compare quality: generate same image with seed 42 at 28 steps (old) and 20 steps (new).
Save both images for visual comparison.

STEP 3 — MPS TUNING
Review current MPS config values in config.yaml.
Investigate if _cleanup_memory() is being called too frequently.
If watermark errors appear in logs, adjust ratios.
Run profiler. Document findings.

STEP 4 — torch.compile EXPERIMENT  
Add torch_compile boolean to config (default: false).
Add torch_compile_mode string to config (default: "reduce-overhead").
In pipeline.py, after model load, try torch.compile on self.pipeline.transformer.
Wrap in try/except — if it fails, log warning and continue.
Run profiler with compile enabled. Run 3 sequential generations to amortize
compile cost. Compare generation 2-3 times against baseline.
Document results. If speedup < 5%, leave disabled by default.

STEP 5 — PROMPT FILE INPUT
Add --prompt-file flag to main.py generate subcommand.
JSON format: { positive_prompt, negative_prompt, width, height, steps, guidance_scale, seed }
When --prompt-file is provided, read prompt and params from file instead of CLI args.
Add test for prompt file loading.

After all steps: update CLAUDE.md, docs/BASELINE_BENCHMARKS.md, and
tasks/todo.md. Commit each step separately with conventional commit messages.
```

---

*Two systems. One pipeline. Zero patience for theater.*
