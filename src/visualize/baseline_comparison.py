"""
Baseline Comparison Figure.

Generates a grouped bar chart comparing per-target R² across:
  - XGBoost (canonical HLS DSE baseline)
  - Vanilla MLP (physics-free ablation)
  - HINN (proposed method)

Also exports results/data/comparison_table.csv for direct copy-paste into LaTeX.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from style import nature_style, save_figure, DOUBLE_COL, NATURE_PALETTE


TARGET_DISPLAY = {
    "hls_lut": "LUT",
    "hls_ff": "FF",
    "average_latency": "Avg. Latency",
    "best_latency": "Best Latency",
}

MODEL_COLORS = {
    "XGBoost": NATURE_PALETTE["gray"],
    "VanillaMLP": NATURE_PALETTE["sky_blue"],
    "HINN": NATURE_PALETTE["blue"],
}


def plot_comparison(
    baseline_csv: str = "results/models/baseline_results.csv",
    hinn_csv: str = "results/models/test_results.csv",
    out_path: str = "results/figs/baseline_comparison",
) -> None:
    """
    Merge baseline and HINN test R² results and plot grouped bar chart.

    Expects:
        baseline_csv: output of baselines.train_baselines()
        hinn_csv:     output of train.train_hinn() (test_results.csv)
    """
    # Load and merge
    baseline_df = pd.read_csv(baseline_csv)
    hinn_df = pd.read_csv(hinn_csv)

    # Keep per-target rows only (drop 'all' aggregate rows)
    baseline_df = baseline_df[baseline_df["target"].isin(TARGET_DISPLAY.keys())]
    hinn_df = hinn_df[hinn_df["target"].isin(TARGET_DISPLAY.keys())]

    combined = pd.concat([baseline_df, hinn_df], ignore_index=True)

    # Pivot for plotting: rows = targets, cols = models
    pivot = combined.pivot_table(index="target", columns="model", values="r2", aggfunc="mean")
    # Order models and targets
    model_order = [m for m in ["XGBoost", "VanillaMLP", "HINN"] if m in pivot.columns]
    target_order = [t for t in TARGET_DISPLAY.keys() if t in pivot.index]
    pivot = pivot.loc[target_order, model_order]

    # Save comparison table for LaTeX
    table_path = Path("results/data/comparison_table.csv")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(table_path)
    print(f"Comparison table saved to {table_path}")

    # Plot
    n_targets = len(target_order)
    n_models = len(model_order)
    x = np.arange(n_targets)
    width = 0.25
    offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

    with nature_style():
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))

        for i, (model, offset) in enumerate(zip(model_order, offsets)):
            values = pivot[model].values
            bars = ax.bar(
                x + offset, values,
                width=width * 0.9,
                color=MODEL_COLORS.get(model, "#333333"),
                label=model,
                zorder=3,
            )
            # Annotate bar tops
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=6,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([TARGET_DISPLAY[t] for t in target_order])
        ax.set_ylabel("Test $R^2$")
        ax.set_ylim(0, 1.05)
        ax.set_title("Surrogate Accuracy: HINN vs. Baselines (Test Set)")
        ax.legend(loc="lower right")
        ax.axhline(y=0.9, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.text(n_targets - 0.1, 0.91, "R²=0.9 target", fontsize=6, ha="right", alpha=0.6)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        save_figure(fig, out_path)
        print(f"Comparison figure saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", default="results/models/baseline_results.csv")
    parser.add_argument("--hinn", default="results/models/test_results.csv")
    parser.add_argument("--out", default="results/figs/baseline_comparison")
    args = parser.parse_args()

    plot_comparison(args.baselines, args.hinn, args.out)
