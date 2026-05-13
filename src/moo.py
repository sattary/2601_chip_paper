"""
Multi-Objective Optimization via Surrogate Pareto Extraction.

Strategy: Surrogate-Assisted Discrete Pareto Extraction.

Rationale for NOT using continuous NSGA-II over the 262-dim OHE space:
The design space is inherently discrete — each configuration is a row in db4hls.
Running NSGA-II over a continuous relaxation of one-hot features produces
fractional indicator vectors that are physically meaningless (a design cannot
be "0.3 block + 0.7 cyclic"). The correct approach for a discrete surrogate DSE
paper is to:
  1. Evaluate the surrogate on all known discrete configurations.
  2. Apply non-dominated sorting (pymoo) on (predicted_area, predicted_latency).
  3. Compare the surrogate Pareto front against the ground truth Pareto front.

This comparison is the core validation plot of the paper: if the HINN Pareto
front closely approximates the ground truth Pareto front, it proves the surrogate
is accurate enough for deployment in a real DSE loop.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import HINN_MultiTask
from train import log1p_transform, inverse_log1p_transform


def _load_model_and_scalers(
    model_path: str,
    scalers_dir: str,
    input_dim: int,
    device: torch.device,
) -> tuple[HINN_MultiTask, object, object, object, dict]:
    scalers_dir = Path(scalers_dir)
    scaler_X = joblib.load(scalers_dir / "scaler_X.pkl")
    scaler_y_area = joblib.load(scalers_dir / "scaler_y_area.pkl")
    scaler_y_lat = joblib.load(scalers_dir / "scaler_y_lat.pkl")

    cfg_path = scalers_dir / "transform_config.pkl"
    transform_cfg: dict = joblib.load(cfg_path) if cfg_path.exists() else {}

    model = HINN_MultiTask(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    return model, scaler_X, scaler_y_area, scaler_y_lat, transform_cfg


def _predict_physical_units(
    model: HINN_MultiTask,
    X_scaled: np.ndarray,
    scaler_y_area: object,
    scaler_y_lat: object,
    transform_cfg: dict,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Run the surrogate over X_scaled in batches and return predictions in
    physical units: [hls_lut, hls_ff, average_latency, best_latency].
    """
    all_preds: list[np.ndarray] = []

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    n = len(X_t)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X_t[i : i + batch_size].to(device)
            preds = model(batch).cpu().numpy()
            all_preds.append(preds)

    raw = np.vstack(all_preds)

    # Inverse transform: reverse StandardScaler then reverse log1p if applied.
    area_inv = scaler_y_area.inverse_transform(raw[:, :2])
    lat_inv = scaler_y_lat.inverse_transform(raw[:, 2:])

    if transform_cfg.get("area_log1p_applied", False):
        area_inv = inverse_log1p_transform(area_inv)
    if transform_cfg.get("lat_log1p_applied", False):
        lat_inv = inverse_log1p_transform(lat_inv)

    return np.hstack([area_inv, lat_inv])


def _extract_pareto(
    objectives: np.ndarray,
) -> np.ndarray:
    """
    Return boolean mask of Pareto-optimal rows.

    Uses pymoo's NonDominatedSorting (rank 0 = Pareto front).
    Minimization convention: we pass (area, latency) where both should be minimized.
    """
    nds = NonDominatedSorting()
    # front_indices is a list of arrays; front_indices[0] = rank-0 (Pareto optimal)
    front_indices = nds.do(objectives, only_non_dominated_front=True)
    mask = np.zeros(len(objectives), dtype=bool)
    mask[front_indices] = True
    return mask


def run_moo(
    model_path: str = "results/models/hinn_best.pt",
    scalers_dir: str = "results/models/",
    features_path: str = "data/processed/features.parquet",
    targets_path: str = "data/processed/targets.parquet",
    out_dir: str = "results/data/",
) -> None:
    """
    Extract Pareto front from HINN surrogate predictions over all known configs.

    Outputs:
        surrogate_predictions.csv  — all 43k predicted (lut, ff, avg_lat, best_lat)
        pareto_front.csv           — Pareto-optimal subset from surrogate
        gt_pareto_front.csv        — Ground truth Pareto front for comparison
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    features_df = pd.read_parquet(features_path)
    targets_df = pd.read_parquet(targets_path)

    model, scaler_X, scaler_y_area, scaler_y_lat, transform_cfg = _load_model_and_scalers(
        model_path, scalers_dir, input_dim=features_df.shape[1], device=device
    )
    print(f"Loaded surrogate from {model_path}")
    print(f"  Transform config: {transform_cfg}")

    # Scale features (same pipeline as training)
    X_scaled = scaler_X.transform(features_df.values)

    print(f"Running surrogate inference over {len(features_df)} configurations...")
    preds_physical = _predict_physical_units(
        model, X_scaled, scaler_y_area, scaler_y_lat, transform_cfg, device
    )

    # Build predictions DataFrame
    preds_df = pd.DataFrame(
        preds_physical,
        columns=["pred_hls_lut", "pred_hls_ff", "pred_avg_latency", "pred_best_latency"],
        index=features_df.index,
    )
    preds_df["gt_hls_lut"] = targets_df["hls_lut"].values
    preds_df["gt_hls_ff"] = targets_df["hls_ff"].values
    preds_df["gt_avg_latency"] = targets_df["average_latency"].values
    preds_df["pred_total_area"] = preds_df["pred_hls_lut"] + preds_df["pred_hls_ff"]
    preds_df["gt_total_area"] = preds_df["gt_hls_lut"] + preds_df["gt_hls_ff"]

    pred_csv = out_path / "surrogate_predictions.csv"
    preds_df.to_csv(pred_csv, index=False)
    print(f"Surrogate predictions saved to {pred_csv}")

    # --- Surrogate Pareto front ---
    # Objectives: (predicted_area, predicted_avg_latency) — both minimize
    surr_objectives = preds_df[["pred_total_area", "pred_avg_latency"]].values
    surr_pareto_mask = _extract_pareto(surr_objectives)
    surr_pareto_df = preds_df[surr_pareto_mask].copy()
    surr_pareto_df["source"] = "surrogate"

    pareto_csv = out_path / "pareto_front.csv"
    surr_pareto_df.to_csv(pareto_csv, index=False)
    print(f"Surrogate Pareto front ({surr_pareto_mask.sum()} points) saved to {pareto_csv}")

    # --- Ground truth Pareto front ---
    gt_objectives = preds_df[["gt_total_area", "gt_avg_latency"]].values
    # Filter out rows with NaN gt targets
    valid_gt = ~np.isnan(gt_objectives).any(axis=1)
    gt_pareto_mask = np.zeros(len(preds_df), dtype=bool)
    gt_pareto_subset = _extract_pareto(gt_objectives[valid_gt])
    valid_indices = np.where(valid_gt)[0][gt_pareto_subset]
    gt_pareto_mask[valid_indices] = True

    gt_pareto_df = preds_df[gt_pareto_mask].copy()
    gt_pareto_df["source"] = "ground_truth"

    gt_pareto_csv = out_path / "gt_pareto_front.csv"
    gt_pareto_df.to_csv(gt_pareto_csv, index=False)
    print(f"Ground truth Pareto front ({gt_pareto_mask.sum()} points) saved to {gt_pareto_csv}")

    # Summary stats
    print(f"\nSurrogate Pareto front range:")
    print(f"  Area:    [{surr_pareto_df['pred_total_area'].min():.0f}, {surr_pareto_df['pred_total_area'].max():.0f}] LUT+FF")
    print(f"  Latency: [{surr_pareto_df['pred_avg_latency'].min():.0f}, {surr_pareto_df['pred_avg_latency'].max():.0f}] cycles")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Pareto front from trained HINN surrogate.")
    parser.add_argument("--model", default="results/models/hinn_best.pt")
    parser.add_argument("--scalers", default="results/models/")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--targets", default="data/processed/targets.parquet")
    parser.add_argument("--out", default="results/data/")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: {args.model} not found. Run 'cli.py train train' first.")
        raise SystemExit(1)

    run_moo(args.model, args.scalers, args.features, args.targets, args.out)
