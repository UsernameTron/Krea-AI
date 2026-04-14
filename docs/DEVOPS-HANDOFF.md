# DevOps Handoff — Krea-AI Workspace

## Project Summary

Multi-project workspace containing flux-krea (Python image generation) and mirror-post (JavaScript satirical compositor). Both run locally on macOS Apple Silicon. No cloud deployment in v1.

## Environment Requirements

| Requirement | Version | Purpose |
|-------------|---------|---------|
| macOS | 14+ (Sonoma) | Apple Silicon MPS compute |
| Python | 3.10.13 | flux-krea runtime |
| Node.js | 20+ | mirror-post runtime |
| Git | 2.x | Version control |
| Disk | ~15GB free | FLUX model weights + dependencies |

## Sub-Project: flux-krea

### How to Run

```bash
cd flux-krea
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate an image
python main.py generate -p "corporate office scene" --steps 20

# Run tests
pytest -v

# Run benchmarks
python main.py benchmark --quick

# Start web UI
python main.py web
```

### Configuration

- `flux-krea/config.yaml` — All runtime parameters (generation, optimization, thermal)
- `flux-krea/config.py` — FluxConfig dataclass with validation
- Environment: No `.env` file needed. Model path configured in config.yaml.

### Model Weights

FLUX.1-Krea-dev weights must be present at `flux-krea/models/FLUX.1-Krea-dev/`. Not committed to git (~12GB). Download separately.

## Sub-Project: mirror-post (planned)

### How to Run (after build)

```bash
cd mirror-post
npm install
npm run brief -- "scenario description"
npm test
```

### Configuration

- Requires `ANTHROPIC_API_KEY` environment variable for Post Brief generation
- No database, no server, no auth

## Security Notes

- No secrets in repository. API keys via environment variables only.
- flux-krea model weights are gitignored (too large for git)
- `context/` directory is gitignored (operator identity files)
- `state/` directory is gitignored (session audit trail)
- No network services exposed in v1 (Gradio web UI is localhost-only)

## Deployment Maturity

| Dimension | Status |
|-----------|--------|
| Local dev | Working (flux-krea), Planned (mirror-post) |
| CI/CD | None (local-only project) |
| Staging | N/A |
| Production | N/A — local creative tool |
| Monitoring | flux-krea has profiler + thermal monitor |
| Backup | Git repos + model weights stored separately |

## Known Tech Debt

- flux-krea generation latency (60-90s) needs optimization to 30-45s target
- memory.md contains unexecuted optimization recommendations from prior session
- No automated integration tests between flux-krea and mirror-post (will need end-to-end pipeline test)
- flux-krea `.claire/worktrees/` contains stale worktree artifacts

---
*Last updated: 2026-04-13 after GSD initialization*
