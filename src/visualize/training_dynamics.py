import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from style import nature_style, save_figure, DOUBLE_COL

def plot_training_dynamics(metrics_csv: str, out_path: str):
    df = pd.read_csv(metrics_csv)
    
    with nature_style():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL, DOUBLE_COL * 0.7), sharex=True)
        
        # --- Top Panel: MSE Loss & Lambda Annealing ---
        ax1.plot(df['epoch'], df['train_mse'], label='Train MSE', color='gray', alpha=0.7)
        ax1.plot(df['epoch'], df['val_mse'], label='Val MSE', color='#0072B2') # Blue
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_yscale('log')
        ax1.legend(loc='upper left')
        
        # Secondary axis for Lambda
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df['epoch'], df['lambda_weight'], label='$\\lambda$ (Physics Penalty)', color='#D55E00', linestyle='--') # Vermillion
        ax1_twin.set_ylabel('$\\lambda$ Weight')
        ax1_twin.legend(loc='upper right')
        ax1.set_title('Model Convergence vs. Physics Constraint Annealing')
        
        # --- Bottom Panel: R2 Score & CVR ---
        ax2.plot(df['epoch'], df['val_r2'], label='Validation $R^2$', color='#009E73') # Teal
        ax2.set_ylabel('$R^2$ Score')
        ax2.set_ylim(0, 1.0)
        ax2.legend(loc='upper left')
        ax2.set_xlabel('Epoch')
        
        # Secondary axis for Constraint Violation Ratio (CVR)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df['epoch'], df['val_cvr'], label='Constraint Violations (%)', color='#CC79A7') # Pink
        ax2_twin.set_ylabel('Violation Ratio (%)')
        ax2_twin.set_ylim(0, 100)
        ax2_twin.legend(loc='upper right')
        
        fig.tight_layout()
        save_figure(fig, out_path)
        print(f"Saved training dynamics plot to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, default='results/models/metrics.csv')
    parser.add_argument('--out', type=str, default='results/figs/training_dynamics')
    args = parser.parse_args()
    
    plot_training_dynamics(args.metrics, args.out)
