"""
HINN Design Space Exploration CLI

Structured dynamically based on ali_proj. Each sub-app maps 1:1 to a src module.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

# Ensure src is in Python path for all subcommand imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

app = typer.Typer(
    name="hinn-cli",
    help="Hardware-Informed Neural Network (HINN) CLI for High-Level Synthesis DSE.",
    add_completion=False,
)

DataApp = typer.Typer(help="Data parsing and preprocessing commands.")
TrainApp = typer.Typer(help="Training and architecture commands.")
PlotApp = typer.Typer(help="Visualization and plotting commands.")
MooApp = typer.Typer(help="Multi-objective optimization commands.")
AnalysisApp = typer.Typer(help="Dataset analysis and audit commands.")

app.add_typer(DataApp, name="data")
app.add_typer(TrainApp, name="train")
app.add_typer(PlotApp, name="plot")
app.add_typer(MooApp, name="moo")
app.add_typer(AnalysisApp, name="analysis")


# ============================================================================
# DATA COMMANDS
# ============================================================================

@DataApp.command("prep")
def data_prep_cmd(
    sql_path: str = typer.Option("data/db4hls.sql", "--sql", help="Path to .sql dump file."),
    out_dir: str = typer.Option("data/processed", "--out-dir", help="Output directory for parquet files."),
    fmt: str = typer.Option("parquet", "--format", help="Output format: parquet or csv."),
) -> None:
    """Parse raw SQL design space dump into ML-ready tabular formats."""
    from data_prep import load_and_prepare, save_ml_data, save_to_parquet
    from pathlib import Path as _Path

    typer.echo(f"Preparing data from {sql_path} -> {out_dir}")
    master, X, y = load_and_prepare(sql_path)
    save_ml_data(X, y, out_dir, format=fmt)
    ext = "parquet" if fmt == "parquet" else "csv"
    save_to_parquet(master, _Path(out_dir) / f"master.{ext}")
    typer.echo("Done.")


# ============================================================================
# TRAINING COMMANDS
# ============================================================================

@TrainApp.command("train")
def train_cmd(
    epochs: int = typer.Option(200, "--epochs", help="Number of training epochs."),
    batch_size: int = typer.Option(1024, "--batch-size", help="Batch size."),
    seed: int = typer.Option(42, "--seed", help="Global random seed. Run with multiple seeds for statistical reporting."),
) -> None:
    """Train the multi-task HINN with physics-informed monotonicity regularization."""
    from train import train_hinn
    train_hinn(epochs=epochs, batch_size=batch_size, seed=seed)


@TrainApp.command("baselines")
def baselines_cmd(
    epochs: int = typer.Option(200, "--epochs", help="Epochs for Vanilla MLP baseline."),
    seed: int = typer.Option(42, "--seed", help="Global random seed."),
) -> None:
    """Train XGBoost and Vanilla MLP baselines (same data split as HINN)."""
    from baselines import train_baselines
    train_baselines(epochs=epochs, seed=seed)


# ============================================================================
# ANALYSIS COMMANDS
# ============================================================================

@AnalysisApp.command("monotonicity")
def monotonicity_cmd(
    master: str = typer.Option("data/processed/master.parquet", "--master"),
    out: str = typer.Option("results/data/monotonicity_audit.csv", "--out"),
    max_groups: Optional[int] = typer.Option(None, "--max-groups", help="Limit groups for quick debug run."),
) -> None:
    """Audit ground truth monotonicity of param_3 in db4hls (paper Section 4.1)."""
    from analysis.monotonicity_check import run_monotonicity_audit
    run_monotonicity_audit(master, out, max_groups)


# ============================================================================
# MOO COMMANDS
# ============================================================================

@MooApp.command("run")
def moo_run_cmd(
    model: str = typer.Option("results/models/hinn_best.pt", "--model", help="Trained surrogate .pt path."),
    scalers: str = typer.Option("results/models/", "--scalers", help="Directory containing scaler pickles."),
    features: str = typer.Option("data/processed/features.parquet", "--features"),
    targets: str = typer.Option("data/processed/targets.parquet", "--targets"),
    out: str = typer.Option("results/data/", "--out", help="Output directory for Pareto CSVs."),
) -> None:
    """Extract Pareto front via HINN surrogate over all known discrete configurations."""
    from moo import run_moo
    run_moo(model, scalers, features, targets, out)


# ============================================================================
# PLOT COMMANDS
# ============================================================================

@PlotApp.command("training-dynamics")
def plot_training_cmd(
    metrics: str = typer.Option("results/models/metrics.csv", "--metrics"),
    out: str = typer.Option("results/figs/training_dynamics", "--out"),
) -> None:
    """Plot convergence (MSE + lambda annealing) and CVR over training epochs."""
    from visualize.training_dynamics import plot_training_dynamics
    plot_training_dynamics(metrics, out)


@PlotApp.command("pareto")
def plot_pareto_cmd(
    pareto: str = typer.Option("results/data/pareto_front.csv", "--pareto"),
    gt_pareto: str = typer.Option("results/data/gt_pareto_front.csv", "--gt-pareto"),
    raw: str = typer.Option("data/processed/targets.parquet", "--raw"),
    out: str = typer.Option("results/figs/pareto_front", "--out"),
) -> None:
    """Visualize surrogate Pareto front vs. ground truth Pareto front."""
    from visualize.pareto_front import plot_pareto
    plot_pareto(pareto, raw, out)


@PlotApp.command("monotonicity")
def plot_monotonicity_cmd(
    model: str = typer.Option("results/models/hinn_best.pt", "--model"),
    features: str = typer.Option("data/processed/features.parquet", "--features"),
    out: str = typer.Option("results/figs/monotonicity_proof", "--out"),
) -> None:
    """Sweep parallelism factors through trained HINN to prove soft monotonicity prior."""
    from visualize.monotonicity_proof import plot_monotonicity
    plot_monotonicity(model, "results/models/", features, out)


@PlotApp.command("comparison")
def plot_comparison_cmd(
    baselines: str = typer.Option("results/models/baseline_results.csv", "--baselines"),
    hinn: str = typer.Option("results/models/test_results.csv", "--hinn"),
    out: str = typer.Option("results/figs/baseline_comparison", "--out"),
) -> None:
    """Grouped bar chart: test R² for XGBoost vs Vanilla MLP vs HINN (paper Table 2)."""
    from visualize.baseline_comparison import plot_comparison
    plot_comparison(baselines, hinn, out)


if __name__ == "__main__":
    app()
