import argparse
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent))
from model import HINN_MultiTask
from style import nature_style, save_figure, SINGLE_COL

def plot_monotonicity(model_path: str, scalers_dir: str, features_path: str, out_path: str):
    # Load Scalers
    scalers_dir = Path(scalers_dir)
    scaler_X = joblib.load(scalers_dir / 'scaler_X.pkl')
    scaler_y_area = joblib.load(scalers_dir / 'scaler_y_area.pkl')
    scaler_y_lat = joblib.load(scalers_dir / 'scaler_y_lat.pkl')

    # Recover whether log1p was applied to latency during training.
    # Symmetry is non-negotiable: inference must mirror the training transform pipeline.
    transform_cfg_path = scalers_dir / 'transform_config.pkl'
    lat_log1p = False
    if transform_cfg_path.exists():
        cfg = joblib.load(transform_cfg_path)
        lat_log1p = cfg.get('lat_log1p_applied', False)
    
    # Load Model
    features_df = pd.read_parquet(features_path)
    input_dim = features_df.shape[1]
    
    model = HINN_MultiTask(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Identify parallelism columns (e.g. param_3_*)
    parallelism_prefix = 'param_3_'
    parallel_cols = [c for c in features_df.columns if c.startswith(parallelism_prefix) and c.replace(parallelism_prefix, '').isdigit()]
    parallel_cols.sort(key=lambda x: int(x.replace(parallelism_prefix, '')))
    
    if not parallel_cols:
        print("No parallel columns found!")
        return

    # Base configuration: take mean of the dataset, then zero out all parallel cols
    base_x = features_df.mean().values.copy()
    col_indices = {col: i for i, col in enumerate(features_df.columns)}
    
    for col in parallel_cols:
        base_x[col_indices[col]] = 0.0

    factors = []
    areas = []
    latencies = []
    
    with torch.no_grad():
        for col in parallel_cols:
            factor = int(col.replace(parallelism_prefix, ''))
            
            # Create specific configuration
            x_test = base_x.copy()
            x_test[col_indices[col]] = 1.0  # Set this parallelism factor to active
            
            # Scale
            x_scaled = scaler_X.transform([x_test])
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
            
            # Predict
            preds_scaled = model(x_tensor).numpy()
            
            # Inverse Transform: reverse StandardScaler, then reverse log1p if applied.
            area_pred = scaler_y_area.inverse_transform(preds_scaled[:, :2])
            lat_pred_scaled = scaler_y_lat.inverse_transform(preds_scaled[:, 2:])
            lat_pred = np.expm1(lat_pred_scaled) if lat_log1p else lat_pred_scaled

            # Sum LUT and FF for total area, use average latency
            total_area = area_pred[0, 0] + area_pred[0, 1]
            avg_lat = lat_pred[0, 0]
            
            factors.append(factor)
            areas.append(total_area)
            latencies.append(avg_lat)

    # Plotting
    with nature_style():
        fig, ax1 = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))
        
        ax1.plot(factors, areas, marker='o', color='#0072B2', label='Predicted Area')
        ax1.set_xlabel('Parallelism Factor (Unroll)')
        ax1.set_ylabel('Total Area (LUT + FF)', color='#0072B2')
        ax1.tick_params(axis='y', labelcolor='#0072B2')
        
        ax2 = ax1.twinx()
        ax2.plot(factors, latencies, marker='s', color='#D55E00', label='Predicted Latency')
        ax2.set_ylabel('Average Latency (Cycles)', color='#D55E00')
        ax2.tick_params(axis='y', labelcolor='#D55E00')
        
        ax1.set_title('Physics Monotonicity Constraint Proof')
        
        # Align grid
        fig.tight_layout()
        save_figure(fig, out_path)
        print(f"Saved monotonicity proof to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='results/models/hinn_best.pt')
    parser.add_argument('--scalers', type=str, default='results/models/')
    parser.add_argument('--features', type=str, default='data/processed/features.parquet')
    parser.add_argument('--out', type=str, default='results/figs/monotonicity_proof')
    args = parser.parse_args()
    
    if Path(args.model).exists():
        plot_monotonicity(args.model, args.scalers, args.features, args.out)
    else:
        print(f"Error: {args.model} not found. Train model first.")
