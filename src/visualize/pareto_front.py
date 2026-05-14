"""
Pareto Front Visualization: Discrete vs. Continuous Search.

Plots the HINN surrogate Pareto frontier discovered via NSGA-II search
alongside the ground truth and discrete synthesis samples.
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
    discrete_csv: str,
    continuous_csv: str,
    gt_csv: str,
    out_path: str,
) -> None:
    """
    Renders a triple-overlay Pareto front:
    1. GT Pareto (Teal)
    2. Surrogate Discrete (Gray line)
    3. Surrogate Continuous/NSGA-II (Vermillion line)
    """
    with nature_style():
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.7))

        # 1. Ground Truth Pareto
        if Path(gt_csv).exists():
            gt_df = pd.read_csv(gt_csv).sort_values("avg_latency")
            ax.plot(
                gt_df["avg_latency"], gt_df["total_area"],
                color=NATURE_PALETTE["teal"], linewidth=1.5,
                marker='o', markersize=4, label="Ground Truth Front",
                alpha=0.6, zorder=5
            )

        # 2. Surrogate Discrete (HINN on existing configs)
        if Path(discrete_csv).exists():
            d_df = pd.read_csv(discrete_csv).sort_values("avg_latency")
            ax.plot(
                d_df["avg_latency"], d_df["total_area"],
                color="#7F7F7F", linewidth=1.0, linestyle="--",
                label="Surrogate Discrete Front", zorder=4
            )

        # 3. Surrogate Continuous (NSGA-II on HINN manifold)
        if Path(continuous_csv).exists():
            c_df = pd.read_csv(continuous_csv).sort_values("avg_latency")
            ax.plot(
                c_df["avg_latency"], c_df["total_area"],
                color=NATURE_PALETTE["vermillion"], linewidth=2.0,
                label="HINN Continuous Frontier (NSGA-II)", zorder=10
            )

        ax.set_xlabel("Latency (Cycles) — Lower is better")
        ax.set_ylabel("Area (LUT + FF) — Lower is better")
        ax.set_title("HINN-Guided Discovery: Theoretical vs. Discrete Pareto Frontiers")
        ax.legend(loc="upper right", frameon=True, facecolor='white', framealpha=0.9)

        # Log scale if range is massive (common in HLS)
        if Path(continuous_csv).exists():
            if (c_df["avg_latency"].max() / c_df["avg_latency"].min()) > 100:
                ax.set_xscale("log")
            if (c_df["total_area"].max() / c_df["total_area"].min()) > 50:
                ax.set_yscale("log")

        fig.tight_layout()
        save_figure(fig, out_path)
        print(f"Saved Pareto front plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--discrete", default="results/data/pareto_front_discrete.csv")
    parser.add_argument("--continuous", default="results/data/pareto_front_continuous.csv")
    parser.add_argument("--gt", default="results/data/gt_pareto_front.csv")
    parser.add_argument("--out", default="results/figs/pareto_front")
    args = parser.parse_args()

    plot_pareto(args.discrete, args.continuous, args.gt, args.out)
