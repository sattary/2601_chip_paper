import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from style import nature_style, save_figure, SINGLE_COL

def plot_pareto(pareto_csv: str, raw_data_parquet: str, out_path: str):
    pareto_df = pd.read_csv(pareto_csv)
    raw_df = pd.read_parquet(raw_data_parquet)
    
    # Sort Pareto front by latency for a continuous line
    pareto_df = pareto_df.sort_values('average_latency')
    
    with nature_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
        
        # Plot all raw sampled points in gray
        if 'average_latency' in raw_df.columns and 'hls_lut' in raw_df.columns:
            ax.scatter(raw_df['average_latency'], raw_df['hls_lut'], 
                       color='gray', alpha=0.3, s=2, label='Raw Configurations')
        
        # Plot HINN Pareto front
        ax.scatter(pareto_df['average_latency'], pareto_df['hls_lut'], 
                   color='#D55E00', s=15, label='HINN Pareto Front', zorder=5)
        ax.plot(pareto_df['average_latency'], pareto_df['hls_lut'], 
                color='#D55E00', linewidth=1.5, zorder=4)
        
        ax.set_xlabel('Average Latency (Cycles)')
        ax.set_ylabel('Area (LUTs)')
        ax.set_title('Multi-Objective Optimization')
        ax.legend()
        
        fig.tight_layout()
        save_figure(fig, out_path)
        print(f"Saved Pareto front plot to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pareto', type=str, default='results/data/pareto_front.csv')
    parser.add_argument('--raw', type=str, default='data/processed/targets.parquet')
    parser.add_argument('--out', type=str, default='results/figs/pareto_front')
    args = parser.parse_args()
    
    if Path(args.pareto).exists():
        plot_pareto(args.pareto, args.raw, args.out)
    else:
        print(f"Error: {args.pareto} not found. Run MOO first.")
