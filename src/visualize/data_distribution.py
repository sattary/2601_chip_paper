import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import style
sys.path.append(str(Path(__file__).parent.parent))
from visualize.style import nature_style, save_figure, DOUBLE_COL

def plot_data_distribution(raw_targets_path: str, out_path: str):
    """
    Plots the distribution of Area (LUTs) and Latency (Cycles) to justify the log1p transform.
    """
    df = pd.read_parquet(raw_targets_path)
    
    with nature_style():
        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.6))
        
        # Raw Area
        axes[0, 0].hist(df['hls_lut'].dropna(), bins=50, color='#0072B2', alpha=0.7)
        axes[0, 0].set_title('Raw LUT Distribution')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_xlabel('LUT Count')
        
        # Log1p Area
        axes[0, 1].hist(np.log1p(df['hls_lut'].dropna()), bins=50, color='#0072B2', alpha=0.7)
        axes[0, 1].set_title('Log1p(LUT) Distribution')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xlabel('Log1p(LUT Count)')
        
        # Raw Latency
        axes[1, 0].hist(df['average_latency'].dropna(), bins=50, color='#D55E00', alpha=0.7)
        axes[1, 0].set_title('Raw Latency Distribution')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_xlabel('Cycles')
        
        # Log1p Latency
        axes[1, 1].hist(np.log1p(df['average_latency'].dropna()), bins=50, color='#D55E00', alpha=0.7)
        axes[1, 1].set_title('Log1p(Latency) Distribution')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xlabel('Log1p(Cycles)')
        
        fig.tight_layout()
        save_figure(fig, out_path)
        print(f"Saved data distribution plot to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=str, default='data/processed/targets.parquet')
    parser.add_argument('--out', type=str, default='results/figs/data_distribution')
    args = parser.parse_args()
    
    if Path(args.raw).exists():
        plot_data_distribution(args.raw, args.out)
    else:
        print(f"Error: {args.raw} not found.")
