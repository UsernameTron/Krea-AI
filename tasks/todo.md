# Current Task: Production Hardening — Phases 6-8

**Branch**: `feat/production-hardening`
**Started**: 2026-03-22
**Baseline**: 52 tests passing, 53% coverage, clean architecture

## Coverage Baseline (53% → target 90%)

| Module | Current | Target | Gap |
|--------|---------|--------|-----|
| app.py | 0% | 80%+ | Full test suite needed |
| main.py | 0% | 80%+ | CLI subcommand tests |
| utils/benchmark.py | 0% | 80%+ | Mock pipeline benchmarks |
| utils/monitor.py | 0% | 80%+ | System info tests |
| neural_engine.py | 0% | 80%+ | CoreML availability tests |
| pipeline.py | 50% | 90%+ | Fallback chain, edge cases |
| metal.py | 51% | 80%+ | VAE autocast, memory stats |
| thermal.py | 50% | 80%+ | Monitor loop, temp reading |
| config.py | 85% | 90%+ | Validation edge cases |
| optimizers/__init__.py | 10% | 80%+ | Init logic |

---

## Feature 1 — Test Coverage Expansion (→ 90% overall)

### 1A: Zero-coverage modules
- [ ] tests/test_app.py — FluxWebApp: mock Gradio, test generate(), _format_error(), _background_load(), empty prompt, timeout, concurrency lock, create_ui()
- [ ] tests/test_main.py — CLI: test all 4 subcommands (generate, web, benchmark, info), --optimization flag, --verbose, no-command help, KeyboardInterrupt, fatal error
- [ ] tests/test_benchmark.py — run_benchmark(): mock pipeline, test quick mode, normal mode, step failure handling
- [ ] tests/test_monitor.py — get_system_info(): with/without psutil, with/without MPS
- [ ] tests/test_neural_engine.py — NeuralEngineOptimizer: CoreML available/unavailable, optimize_pipeline no-op, get_stats()

### 1B: Half-covered modules — close gaps
- [ ] pipeline.py gaps: fallback chain (MAX→STD→NONE), _detect_device() all paths, generate() with thermal throttling, save_image() auto-naming, unload(), _apply_cpu_offload() fallback paths
- [ ] metal.py gaps: optimize_pipeline() with VAE, metal_kernel_context() error path, get_memory_stats() error path, _safe_empty_cache() non-watermark error
- [ ] thermal.py gaps: _monitor_loop() with callback, _try_sysctl/psutil/powermetrics, stop_monitoring(), _profile_differs(), add_callback()

### 1C: Edge cases
- [ ] config.py: remaining validation errors (invalid width, negative steps, bad watermark ratios, bad thermal thresholds, invalid optimization level)
- [ ] MPS unavailable fallback path (force device=cpu)
- [ ] Invalid config values (non-multiple-of-64 resolution, negative guidance)
- [ ] Model load failure simulation

---

## Feature 2 — Neural Engine Optimization (Phase 6)

- [ ] Research: Check if coremltools can convert FLUX text encoder (T5-XXL) to CoreML
- [ ] Implement NeuralEngineOptimizer.compile_text_encoder() — try/except with clear logging
- [ ] Add benchmark: MPS-only vs MPS+ANE hybrid timing comparison
- [ ] If conversion fails (model too large/unsupported ops), document WHY in code comments
- [ ] Ensure fallback: if ANE unavailable, pipeline runs identically to before
- [ ] Tests: mock coremltools, test compile success/failure paths

---

## Feature 3 — Performance Profiler (Phase 7)

- [ ] Create utils/profiler.py with FluxProfiler class
- [ ] Track: model_load_time, encode_time, denoise_time_per_step[], decode_time, total_time
- [ ] Wrap torch.profiler for MPS kernel timing
- [ ] Output clean summary table (tabulate or manual formatting)
- [ ] Add "profile" subcommand to main.py
- [ ] Tests: test profiler with mock pipeline, test summary output format

---

## Feature 4 — CI/CD Pipeline (Phase 8)

- [ ] Create .github/workflows/ci.yml
- [ ] Job 1: pytest (skip @requires_mps tests, all tests mock the model)
- [ ] Job 2: ruff check (linting)
- [ ] Job 3: mypy (type checking)
- [ ] Ensure no test requires model download or MPS hardware
- [ ] Add ruff.toml and mypy.ini/pyproject.toml config if missing

---

## Feature 5 — Documentation Polish (depends on 1-4)

- [ ] Update CLAUDE.md: test count, coverage %, new modules (profiler), new CLI commands
- [ ] Update README.md: architecture diagram, all CLI commands including "profile", benchmark template
- [ ] Update docs/DEVOPS-HANDOFF.md: CI/CD info, profiler, test commands

---

## Verification (per feature)

- [ ] All tests pass: python3 -m pytest tests/ -v
- [ ] Coverage ≥90% overall, no module below 80%
- [ ] ruff check passes (no lint errors)
- [ ] No regressions: existing 52 tests still pass
- [ ] Diff reviewed: only intended files changed

---

## Execution Strategy

**Features 1-4 are independent → Agent Teams with parallel chains.**

| Feature | Agent | Model | Complexity |
|---------|-------|-------|------------|
| F1: Test Coverage | Feature-Tests | Sonnet | Standard (many files, mechanical) |
| F2: Neural Engine | Feature-ANE | Opus | Complex (research + experimental code) |
| F3: Profiler | Feature-Profiler | Sonnet | Standard (new module, clear spec) |
| F4: CI/CD | Feature-CICD | Sonnet | Simple (config files) |
| F5: Docs | Main context | Sonnet | Simple (depends on F1-F4 results) |

F5 runs after F1-F4 complete (needs final test count and coverage numbers).

## Completed
<!-- Completed tasks logged here -->
