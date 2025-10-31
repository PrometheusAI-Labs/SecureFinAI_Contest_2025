# Task I (FinRL-Transformer) — Submission Checklist

Use this checklist to prepare a clean GitHub submission. The last pushed version will be evaluated.

## Must-have
- [ ] Code compiles and runs end-to-end via Makefile targets
  - `make workflow` trains and evaluates (data -> factors -> RL -> trajectories -> DT -> eval)
  - `make evaluate-dt` evaluates a provided DT weight file
- [ ] Dependencies pinned (pyproject in this folder) and reproducible install using `uv`
- [ ] README in this folder explains:
  - Data acquisition (links), preparation, and expected paths
  - Exact train/eval commands, expected outputs, runtime expectations
  - Model weights location and how to obtain them
- [ ] Model weights provided in one of the following ways:
  - [ ] Included in repo under `trained_models/` (if small / allowed)
  - [ ] Uploaded as GitHub Release asset or Git LFS with large-file policy
  - [ ] Hosted on Hugging Face Hub (preferred) with repo id documented
  - In all cases, document the exact weight filename and provide a download command
- [ ] Scripts are present for weight download if not stored in-repo
  - e.g., `python download_weights.py --repo-id <org/repo> --filename decision_transformer.pth`
- [ ] No private keys, tokens, or credentials in the repository

## Nice-to-have
- [ ] Reproducible seed settings documented
- [ ] Hardware requirements documented (GPU type, VRAM)
- [ ] Expected training time and dataset footprint documented
- [ ] `make status` shows data and model presence

## Versioning and release
- Create a Git tag (e.g., `v1.0.0`) for the version you want evaluated
- If using Release assets, attach `decision_transformer.pth` and note its SHA256
- Push your final commit to the main branch — the latest push will be used for evaluation

## Quick verification before submitting
```bash
# From repo root
cd Task_1_FinRL_DT_Crypto_Trading
uv sync
make setup-data
make create-test-split
make check-dependencies

# If weights are external
python download_weights.py --repo-id <org/repo> --filename decision_transformer.pth

# Evaluate
make evaluate-dt
```
