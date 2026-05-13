# Status Handoff

## Project Objective
Develop a Hardware-Informed Neural Network (HINN) combined with Multi-Objective Optimization (MOO) for High-Level Synthesis (HLS) design space exploration. Target: Q1 Journal publication (Expert Systems with Applications).

## Architecture (The "No Main" Rule Applies)
- **Data Pipeline:** `src/data_prep.py` — streaming MySQL dump parser, 4-table join → 43,992 rows, 262 OHE features, 4 targets.
- **Modeling:** `src/model.py` — `HINN_MultiTask` (512→256→128→64, Dropout 0.1, dual-head) + `hinn_loss` (hinge penalties for area+latency monotonicity).
- **Dataset:** `src/dataset.py` — `HINNDataset` returns `(x, x_pert, y)` triplets via `param_3_` OHE perturbation.
- **Training:** `src/train.py` — `train_hinn(epochs, batch_size, seed)` with log1p on ALL targets, cosine LR, lambda annealing, test set eval. Writes `test_results.csv`.
- **Baselines:** `src/baselines.py` — XGBoost (per-target early stopping) + Vanilla MLP (no physics). Writes `baseline_results.csv`.
- **MOO:** `src/moo.py` — `run_moo()` evaluates all 43k configs through surrogate, extracts Pareto via pymoo `NonDominatedSorting`. Writes `pareto_front.csv` + `gt_pareto_front.csv`.
- **Analysis:** `src/analysis/monotonicity_check.py` — audits ground truth monotonicity (finding: 52.1% joint — reframed as soft regularizer in paper).
- **Visualization:** `src/visualize/` — training_dynamics, pareto_front (now w/ GT comparison), monotonicity_proof (inverse transform fixed), baseline_comparison.

## Critical Finding (Session 2026-05-13)
**Ground truth param_3 monotonicity holds in 52.1% of consecutive pairs — near-random.** The HINN physics penalty is therefore a soft Bayesian prior, not a hard constraint. CVR → 0% is NOT the target; the paper claims CVR is significantly lower than the vanilla MLP baseline. Paper narrative reframed accordingly.

## Current Status
- **Phase 1 (Data Prep):** Complete.
- **Phase 2 (Modeling):** Complete. Architecture upgraded: 512→256→128→64 + Dropout(0.1).
- **Phase 3 (Training infra):** Complete. log1p on area+latency, cosine LR, seed fixture, test eval.
- **Phase 4 (Baselines):** Complete. Code ready; needs Colab GPU run.
- **Phase 5 (MOO):** Complete. `moo.py` implemented; needs trained model (`hinn_best.pt`).
- **Phase 6 (Figures):** All 4 figure scripts ready; all blocked on `hinn_best.pt`.

## Immediate Next Actions
1. **Commit and push** all changes to GitHub so Colab picks them up.
2. **Run Colab notebook** (9 cells in order): `uv sync` → monotonicity audit → HINN train (300 epochs) → baselines → MOO → all 4 figures → download zip.
3. **Multi-seed runs:** After seed=42 succeeds, re-run Cell 5+6 with seeds 0, 7, 123, 999 for statistical reporting.
4. **Write LaTeX manuscript** in `docs/manuscript/` — skeleton needed.

## CLI Reference
```
uv run python cli.py train train --epochs 300 --seed 42
uv run python cli.py train baselines --epochs 300 --seed 42
uv run python cli.py analysis monotonicity
uv run python cli.py moo run
uv run python cli.py plot training-dynamics
uv run python cli.py plot pareto
uv run python cli.py plot monotonicity
uv run python cli.py plot comparison
```

