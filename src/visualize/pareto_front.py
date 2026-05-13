"""
Pareto Front Visualization.

Plots the HINN surrogate Pareto front alongside the ground truth Pareto front
and all raw configurations. The visual proximity of the two Pareto fronts is
the central validation plot of the paper.
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent))
from style import nature_style, save_figure, DOUBLE_COL, NATURE_PALETTE


def plot_pareto(
    pareto_csv: str,
    raw_data_parquet: str,
    out_path: str,
    gt_pareto_csv: Optional[str] = None,
) -> None:
    pareto_df = pd.read_csv(pareto_csv).sort_values("pred_avg_latency")
    raw_df = pd.read_parquet(raw_data_parquet)

    with nature_style():
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.6))

        # All raw ground truth configurations (background scatter)
        if "average_latency" in raw_df.columns and "hls_lut" in raw_df.columns:
            ax.scatter(
                raw_df["average_latency"], raw_df["hls_lut"],
                color=NATURE_PALETTE["gray"], alpha=0.2, s=1.5,
                label="All configurations (GT)",
                rasterized=True,
            )

        # Ground truth Pareto front (if available)
        if gt_pareto_csv and Path(gt_pareto_csv).exists():
            gt_df = pd.read_csv(gt_pareto_csv).sort_values("gt_avg_latency")
            ax.scatter(
                gt_df["gt_avg_latency"], gt_df["gt_total_area"],
                color=NATURE_PALETTE["teal"], s=20,
                label="Ground Truth Pareto", zorder=5,
            )
            ax.plot(
                gt_df["gt_avg_latency"], gt_df["gt_total_area"],
                color=NATURE_PALETTE["teal"], linewidth=1.0,
                linestyle="--", zorder=4, alpha=0.7,
            )

        # HINN surrogate Pareto front
        ax.scatter(
            pareto_df["pred_avg_latency"], pareto_df["pred_total_area"],
            color=NATURE_PALETTE["vermillion"], s=20,
            label="HINN Surrogate Pareto", zorder=6,
        )
        ax.plot(
            pareto_df["pred_avg_latency"], pareto_df["pred_total_area"],
            color=NATURE_PALETTE["vermillion"], linewidth=1.5, zorder=5,
        )

        ax.set_xlabel("Average Latency (Cycles)")
        ax.set_ylabel("Total Area (LUT + FF)")
        ax.set_title("Multi-Objective Pareto Front: Surrogate vs. Ground Truth")
        ax.legend(loc="upper right")

        fig.tight_layout()
        save_figure(fig, out_path)
        print(f"Saved Pareto front plot to {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto", default="results/data/pareto_front.csv")
    parser.add_argument("--gt-pareto", default="results/data/gt_pareto_front.csv")
    parser.add_argument("--raw", default="data/processed/targets.parquet")
    parser.add_argument("--out", default="results/figs/pareto_front")
    args = parser.parse_args()

    if Path(args.pareto).exists():
        plot_pareto(args.pareto, args.raw, args.out, args.gt_pareto)
    else:
        print(f"Error: {args.pareto} not found. Run 'cli.py moo run' first.")
