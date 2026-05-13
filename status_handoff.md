# Status Handoff

## Project Objective
Develop a Hardware-Informed Neural Network (HINN) combined with Multi-Objective Optimization (MOO) for High-Level Synthesis (HLS) design space exploration. Target: Q1 Journal publication (e.g., Expert Systems with Applications).

## Architecture (The "No Main" Rule Applies)
- **Data Pipeline:** Extracted from `db4hls.sql`. Merges synthesis configuration with Area (`hls_lut`, `hls_ff`) and Timing (`average_latency`) targets.
- **Modeling Strategy (Perturbation Batching):** A PyTorch Multi-Task MLP that employs semi-supervised synthetic pair batching. The custom loss function penalizes violations of monotonic hardware scaling bounds (e.g., increasing parallelism must increase Area and decrease Latency).
- **MOO Strategy:** Utilize `pymoo` / `optuna` (NSGA-II) over the HINN surrogate to instantaneously identify the Pareto front.

## Current Status
- **Phase 1 (Data Prep):** Complete. `src/dataset.py` generated for synthetic perturbations.
- **Phase 2 (Modeling):** Complete. `src/model.py` implements the multi-task HINN constraint loss.
- **Phase 3 (Optimization):** Defining standard `pymoo` decoupling strategy.

## Immediate Next Actions
1. Implement `src/train.py` to train the HINN and export the `.pt` surrogate.
2. Implement `src/moo.py` utilizing a custom `pymoo.Problem` to extract the Pareto front.
