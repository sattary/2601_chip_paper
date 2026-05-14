import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import style
sys.path.append(str(Path(__file__).parent.parent))
from visualize.style import nature_style, save_figure, SINGLE_COL, NATURE_PALETTE

def plot_empirical_monotonicity(audit_csv_path: str, out_path: str):
    """
    Plots a stacked bar chart showing the breakdown of monotonicity compliance.
    """
    df = pd.read_csv(audit_csv_path)
    
    # Calculate global totals
    total_both = df['pairs_both_mono'].sum()
    total_area = df['pairs_area_only'].sum()
    total_lat = df['pairs_lat_only'].sum()
    total_neither = df['pairs_neither'].sum()
    total_pairs = df['pairs_total'].sum()
    
    # Calculate percentages
    pct_both = 100 * total_both / total_pairs
    pct_area = 100 * total_area / total_pairs
    pct_lat = 100 * total_lat / total_pairs
    pct_neither = 100 * total_neither / total_pairs
    
    categories = ['Both Hold\n(Expected Physics)', 'Area Only', 'Latency Only', 'Neither']
    percentages = [pct_both, pct_area, pct_lat, pct_neither]
    
    with nature_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))
        
        colors = [NATURE_PALETTE["teal"], NATURE_PALETTE["blue"], NATURE_PALETTE["vermillion"], NATURE_PALETTE["pink"]]
        
        bars = ax.bar(categories, percentages, color=colors, edgecolor='black', alpha=0.8)
        
        # Add percentage labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
        ax.set_ylabel('Percentage of Configuration Pairs (%)')
        ax.set_ylim(0, 100)
        ax.set_title('Empirical Monotonicity Audit (db4hls)', fontsize=9, pad=15)
        
        # Add a subtle grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        fig.tight_layout()
        save_figure(fig, out_path)
        print(f"Saved empirical monotonicity plot to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=str, default='results/data/monotonicity_audit.csv')
    parser.add_argument('--out', type=str, default='results/figs/empirical_monotonicity')
    args = parser.parse_args()
    
    if Path(args.audit).exists():
        plot_empirical_monotonicity(args.audit, args.out)
    else:
        print(f"Error: {args.audit} not found. Run analysis/monotonicity_check.py first.")
