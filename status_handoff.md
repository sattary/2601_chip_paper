# Status Handoff

## Project Objective
Develop a Hardware-Informed Neural Network (HINN) combined with Multi-Objective Optimization (MOO) for High-Level Synthesis (HLS) design space exploration. Target: Q1 Journal publication (Expert Systems with Applications).

## Architecture (The "No Main" Rule Applies)
- **Data Recovery:** `src/loko_map.py` — Two-pass relational recovery. Maps 154,259 configuration hashes to 17 unique hardware kernels.
- **Data Pipeline:** `src/data_prep.py` — Stream-parses SQL dump and joins with kernel mapping. Result: **153,320 rows**, 22 core features (parameter OHE + synthesis hints), 4 targets.
- **Modeling:** `src/model.py` — `HINN_MultiTask` (512→256→128→64, Dropout 0.1) + `hinn_loss`.
- **Dataset:** `src/dataset.py` — `HINNDataset` for monotonicity training.
- **LOKO Framework:** `src/train.py` — Supports `--loko <kernel_name>`. Replaces random splitting with hardware-aware zero-shot evaluation (Hold out 1 kernel for testing).
- **Analysis:** `src/analysis/monotonicity_check.py` — Audits ground-truth monotonicity.
- **Visualization:** `src/visualize/` — Publication-quality figures (Nature style). Fixed relative imports for Colab compatibility.

## Critical Achievement (Session 2026-05-14)
**Zero-Shot Generalization Infra:** Successfully recovered the full hardware kernel metadata previously lost in the SQL parsing. We now have 17 kernels (e.g., `backprop`, `aes`, `stencil2d`) instead of a monolithic block. This enables **Leave-One-Kernel-Out (LOKO)** validation, providing a much stronger "Novelty" claim for the paper than simple random splits.

## Current Status
- **Data Prep:** **COMPLETE & EXPANDED**. 153k samples ready with kernel labels.
- **LOKO Framework:** **COMPLETE**. Infrastructure verified; pilot run (held out `stencil2d`) achieved **R² = 0.84** in just 1 epoch on unseen hardware.
- **Modeling/Training:** COMPLETE. Ready for GPU-scale LOKO sweep.
- **Visualization:** COMPLETE. fixed `ModuleNotFoundError` for Colab.

## Immediate Next Actions
1. **Colab Execution:** Pull the latest code and run the LOKO sweep for all 17 kernels.
2. **Comparative Analysis:** Run baselines (XGBoost/MLP) on the SAME LOKO splits to prove HINN's superior generalization.
3. **Manuscript:** Update the "Experimental Setup" section to emphasize the 17-kernel LOKO zero-shot validation strategy.

## CLI Reference
```bash
# Data Refresh (already done locally, but can be rerun in Colab)
uv run python src/loko_map.py
uv run python src/data_prep.py

# LOKO Training (Pilot example)
uv run python cli.py train train --epochs 300 --loko stencil2d --seed 42

# Plotting (Relative imports fixed)
uv run python cli.py plot training-dynamics
uv run python cli.py plot pareto
uv run python cli.py plot monotonicity
uv run python cli.py plot comparison
```
