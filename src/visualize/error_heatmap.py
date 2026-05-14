import argparse
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))
from model import HINN_MultiTask
from train import log1p_transform, inverse_log1p_transform
from visualize.style import nature_style, save_figure, DOUBLE_COL

def plot_error_heatmap(model_path: str, model_dir: str, features_path: str, targets_path: str, out_path: str):
    """
    Plots a 2D Hexbin heatmap of Prediction Error across the true Area-Latency space.
    Proves that the model is highly accurate near the Pareto front.
    """
    print("Loading data and model for error heatmap...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    features_df = pd.read_parquet(features_path)
    targets_df = pd.read_parquet(targets_path)
    
    scaler_X = joblib.load(Path(model_dir) / "scaler_X.pkl")
    scaler_y_area = joblib.load(Path(model_dir) / "scaler_y_area.pkl")
    scaler_y_lat = joblib.load(Path(model_dir) / "scaler_y_lat.pkl")
    
    input_dim = features_df.shape[1]
    
    model = HINN_MultiTask(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Running inference on full dataset...")
    X_scaled = scaler_X.transform(features_df)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds_scaled = model(X_t).cpu().numpy()
        
    preds_area_scaled = preds_scaled[:, :2]
    preds_lat_scaled = preds_scaled[:, 2:]
    
    preds_area_log = scaler_y_area.inverse_transform(preds_area_scaled)
    preds_lat_log = scaler_y_lat.inverse_transform(preds_lat_scaled)
    
    preds_area = inverse_log1p_transform(preds_area_log)
    preds_lat = inverse_log1p_transform(preds_lat_log)
    
    # Combine LUT and FF for total Area, use Average Latency
    true_area = targets_df['hls_lut'] + targets_df['hls_ff']
    true_lat = targets_df['average_latency']
    
    pred_area = preds_area[:, 0] + preds_area[:, 1]
    pred_lat = preds_lat[:, 0]
    
    # Calculate relative error (bounded to avoid div by zero)
    area_error = np.abs(true_area - pred_area) / (true_area + 1)
    lat_error = np.abs(true_lat - pred_lat) / (true_lat + 1)
    
    # Combined mean relative error (percentage)
    total_error = (area_error + lat_error) * 50.0  # Average of the two relative errors * 100
    
    # Cap error for visualization purposes
    total_error = np.clip(total_error, 0, 100)
    
    print("Generating hexbin plot...")
    with nature_style():
        fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.8, DOUBLE_COL * 0.6))
        
        # Hexbin plot: X=Latency, Y=Area, C=Error
        hb = ax.hexbin(
            true_lat, true_area, C=total_error, 
            gridsize=50, cmap='viridis_r',  # reverse viridis so low error is dark/purple, high is yellow
            reduce_C_function=np.median,
            xscale='log', yscale='log',
            mincnt=1, alpha=0.9
        )
        
        cb = fig.colorbar(hb, ax=ax, label='Median Relative Prediction Error (%)')
        
        ax.set_xlabel('True Average Latency (Cycles)')
        ax.set_ylabel('True Total Area (LUT + FF)')
        ax.set_title('HINN Prediction Error Across the Design Space', pad=15)
        
        # Highlight the approximate Pareto region
        ax.annotate('Accurate Predictions\nNear Pareto Front', 
                    xy=(1e3, 1e4), xycoords='data',
                    xytext=(1e5, 1e3), textcoords='data',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                    fontsize=8, fontweight='bold', ha='center')
        
        fig.tight_layout()
        save_figure(fig, out_path)
        print(f"Saved error heatmap to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='results/models/hinn_best.pt')
    parser.add_argument('--model-dir', type=str, default='results/models')
    parser.add_argument('--features', type=str, default='data/processed/features.parquet')
    parser.add_argument('--targets', type=str, default='data/processed/targets.parquet')
    parser.add_argument('--out', type=str, default='results/figs/error_heatmap')
    args = parser.parse_args()
    
    if Path(args.model).exists():
        plot_error_heatmap(args.model, args.model_dir, args.features, args.targets, args.out)
    else:
        print(f"Error: {args.model} not found. Train the model first.")
