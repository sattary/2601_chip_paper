# Status Handoff

## Project Objective
Develop a Hardware-Informed Neural Network (HINN) combined with Multi-Objective Optimization (MOO) for High-Level Synthesis (HLS) design space exploration. Target: Q1 Journal publication (Expert Systems with Applications).

## Architecture (The "No Main" Rule Applies)
- **Data Recovery:** `src/loko_map.py` — Recovered 17 hardware kernels and 153k data points.
- **Data Pipeline:** `src/data_prep.py` — Fixed relational join mismatches.
- **Modeling/Training:** `src/train.py` — Dual-stage lambda schedule (0.3 -> 0.8) over 500 epochs for phased physics regularization.
- **MOO Engine:** `src/moo.py` — **UPGRADED**. Implements NSGA-II continuous search via `pymoo`. Discovers theoretical Pareto frontiers beyond discrete synthesis samples.
- **Visualization:** `src/visualize/` — Nature-style figures. Fixed relative imports. Pareto plot now supports triple-overlay (GT vs Discrete vs Continuous).

## Critical Milestone (Session 2026-05-14)
**Continuous Frontier Discovery:** We have successfully integrated **NSGA-II Genetic Search** over the learned HINN manifold. This allows the discovery of "Virtual Designs" that represent the theoretical optimality limit of the hardware design space. This is a core requirement for a Q1 paper to show how AI enables discovery, not just faster simulation.

## Current Status
- **LOKO Framework:** Ready for zero-shot validation sweep.
- **MOO Engine:** COMPLETE. Genetic search is operational.
- **Visualization:** COMPLETE. Triple-overlay Pareto plotting verified.

## Immediate Next Actions (Colab)
1. **LOKO Sweep:** Train on 16 kernels, test on 1 (e.g., `stencil2d`) to get zero-shot R².
2. **Continuous MOO:** Run `uv run python cli.py moo run --pop 100 --gen 200` to find the smooth Pareto front.
3. **Figures:** Run `uv run python cli.py plot all` to generate the final manuscript figures.

## CLI Reference
```bash
# Phased Training (500 epochs)
uv run python cli.py train train --epochs 500 --seed 42

# Continuous MOO Search
uv run python cli.py moo run --pop 100 --gen 200

# Plotting with Triple-Overlay Pareto
uv run python cli.py plot pareto
```
