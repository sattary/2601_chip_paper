"""
Multi-Objective Optimization via Surrogate-Guided Genetic Search (NSGA-II).

This module implements a continuous search over the HINN manifold to discover
the theoretical Pareto frontier. 

Rationale for NSGA-II over HINN manifold:
1. Discovery: Finds optimal trade-offs in the 'gaps' between discrete synthesis points.
2. Continuity: Produces a smooth trade-off curve suitable for Nature-style figures.
3. Analysis: Allows for the quantification of the 'Optimality Gap' between existing 
   HLS designs and the theoretical surrogate boundary.
"""

import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import HINN_MultiTask
from train import inverse_log1p_transform

class HINNProblem(Problem):
    """
    Pymoo wrapper for the HINN surrogate.
    Search space: [0, 1] relaxation of the OHE design parameters.
    Objectives: [Total Area (LUT+FF), Average Latency].
    """
    def __init__(self, model, scaler_X, scaler_y_area, scaler_y_lat, transform_cfg, device):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y_area = scaler_y_area
        self.scaler_y_lat = scaler_y_lat
        self.transform_cfg = transform_cfg
        self.device = device
        
        # input_dim = 262 (OHE features)
        input_dim = model.input_dim
        super().__init__(n_var=input_dim, n_obj=2, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        # x shape: (pop_size, 262)
        X_t = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # HINN predicts in scaled log-space
            raw = self.model(X_t).cpu().numpy()
            
        # Inverse transform to physical units
        # Rationale: Objectives must be in physical units for meaningful Pareto comparison.
        area_inv = self.scaler_y_area.inverse_transform(raw[:, :2])
        lat_inv = self.scaler_y_lat.inverse_transform(raw[:, 2:])

        if self.transform_cfg.get("area_log1p_applied", False):
            area_inv = inverse_log1p_transform(area_inv)
        if self.transform_cfg.get("lat_log1p_applied", False):
            lat_inv = inverse_log1p_transform(lat_inv)
            
        total_area = area_inv[:, 0] + area_inv[:, 1] # LUT + FF
        avg_lat = lat_inv[:, 0]
        
        # out["F"] should be (n_samples, n_obj)
        out["F"] = np.column_stack([total_area, avg_lat])

def run_moo(
    model_path: str = "results/models/hinn_best.pt",
    scalers_dir: str = "results/models/",
    features_path: str = "data/processed/features.parquet",
    targets_path: str = "data/processed/targets.parquet",
    out_dir: str = "results/data/",
    pop_size: int = 100,
    n_gen: int = 200,
) -> None:
    """
    Executes NSGA-II search to find the continuous Pareto frontier.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load resources
    features_df = pd.read_parquet(features_path)
    targets_df = pd.read_parquet(targets_path)
    
    scalers_dir = Path(scalers_dir)
    scaler_X = joblib.load(scalers_dir / "scaler_X.pkl")
    scaler_y_area = joblib.load(scalers_dir / "scaler_y_area.pkl")
    scaler_y_lat = joblib.load(scalers_dir / "scaler_y_lat.pkl")
    transform_cfg = joblib.load(scalers_dir / "transform_config.pkl") if (scalers_dir / "transform_config.pkl").exists() else {}

    model = HINN_MultiTask(input_dim=features_df.shape[1], feature_names=list(features_df.columns))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # 1. DISCRETE BASELINE (Existing Configurations)
    print(f"Evaluating {len(features_df)} existing discrete points...")
    X_scaled = scaler_X.transform(features_df.values)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        raw_discrete = model(X_t).cpu().numpy()
    
    # Inverse transform
    area_d = scaler_y_area.inverse_transform(raw_discrete[:, :2])
    lat_d = scaler_y_lat.inverse_transform(raw_discrete[:, 2:])
    if transform_cfg.get("area_log1p_applied"): area_d = inverse_log1p_transform(area_d)
    if transform_cfg.get("lat_log1p_applied"): lat_d = inverse_log1p_transform(lat_d)
    
    discrete_objs = np.column_stack([area_d[:,0]+area_d[:,1], lat_d[:,0]])
    nds = NonDominatedSorting()
    idx = nds.do(discrete_objs, only_non_dominated_front=True)
    discrete_pareto = discrete_objs[idx]
    
    # 2. CONTINUOUS GENETIC SEARCH (NSGA-II)
    print(f"Running NSGA-II search (pop={pop_size}, gen={n_gen}) over HINN manifold...")
    problem = HINNProblem(model, scaler_X, scaler_y_area, scaler_y_lat, transform_cfg, device)
    algorithm = NSGA2(pop_size=pop_size)
    
    res = minimize(problem, algorithm, ('n_gen', n_gen), seed=1, verbose=True)
    
    # Save results
    # NSGA-II Continuous Front
    continuous_front = pd.DataFrame(res.F, columns=["total_area", "avg_latency"])
    continuous_front.to_csv(out_path / "pareto_front_continuous.csv", index=False)
    
    # Discrete Front for comparison
    discrete_front_df = pd.DataFrame(discrete_pareto, columns=["total_area", "avg_latency"])
    discrete_front_df.to_csv(out_path / "pareto_front_discrete.csv", index=False)
    
    # Ground Truth Front
    gt_objs = targets_df[['hls_lut', 'hls_ff', 'average_latency']].dropna()
    gt_vals = np.column_stack([gt_objs['hls_lut'] + gt_objs['hls_ff'], gt_objs['average_latency']])
    gt_idx = nds.do(gt_vals, only_non_dominated_front=True)
    gt_pareto = gt_vals[gt_idx]
    pd.DataFrame(gt_pareto, columns=["total_area", "avg_latency"]).to_csv(out_path / "gt_pareto_front.csv", index=False)

    print(f"Pareto analysis complete.")
    print(f"  Discrete Pareto points found: {len(discrete_pareto)}")
    print(f"  NSGA-II optimized points found: {len(res.F)}")
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    run_moo()
