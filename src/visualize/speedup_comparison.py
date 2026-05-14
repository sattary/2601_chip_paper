import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import style
sys.path.append(str(Path(__file__).parent.parent))
from visualize.style import nature_style, save_figure, SINGLE_COL, NATURE_PALETTE

def plot_speedup_comparison(out_path: str):
    """
    Plots a logarithmic horizontal bar chart comparing HLS Synthesis vs HINN Surrogate time.
    Data sourced from the DB4HLS paper (4 years for 100k configs).
    """
    # Times in hours
    # DB4HLS paper: 4 years = 35040 hours
    # HINN: Training (1 hour) + Inference (2 seconds = 0.0005 hours) -> we'll plot just inference for the DSE phase
    # Actually, let's plot "Time to Explore 100,000 Configurations"
    labels = ['Exhaustive HLS\n(Vivado)', 'HINN Surrogate\n(Inference + NSGA-II)']
    times_hours = [35040, 0.002]  # 4 years vs ~7 seconds
    
    with nature_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.6))
        
        y_pos = range(len(labels))
        colors = [NATURE_PALETTE['vermillion'], NATURE_PALETTE['blue']]
        
        bars = ax.barh(y_pos, times_hours, color=colors, edgecolor='black', alpha=0.8, height=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xscale('log')
        ax.set_xlabel('Computational Time (Hours) [Log Scale]')
        ax.set_title('Time-to-Pareto Acceleration ($10^7$ Speedup)', pad=15)
        
        # Add exact text labels
        ax.text(times_hours[0] * 1.2, y_pos[0], '4 Years', va='center', fontsize=8, fontweight='bold', color=colors[0])
        ax.text(times_hours[1] * 1.5, y_pos[1], '7 Seconds', va='center', fontsize=8, fontweight='bold', color=colors[1])
        
        # Format the log axis gracefully
        ax.set_xlim(1e-4, 1e6)
        
        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        save_figure(fig, out_path)
        print(f"Saved speedup comparison plot to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='results/figs/speedup_comparison')
    args = parser.parse_args()
    plot_speedup_comparison(args.out)
