"""
HINN Design Space Exploration CLI
Structured dynamically based on ali_proj.
"""
from __future__ import annotations
import typer
from typing import Optional
from pathlib import Path
import sys

# Ensure src is in pythonpath
sys.path.append(str(Path(__file__).parent))

app = typer.Typer(
    name="hinn-cli",
    help="Hardware-Informed Neural Network (HINN) CLI for High-Level Synthesis.",
    add_completion=False,
)

DataApp = typer.Typer(help="Data parsing and preprocessing commands.")
TrainApp = typer.Typer(help="Training and architecture commands.")
PlotApp = typer.Typer(help="Visualization and plotting commands.")
MooApp = typer.Typer(help="Multi-objective optimization commands.")

app.add_typer(DataApp, name="data")
app.add_typer(TrainApp, name="train")
app.add_typer(PlotApp, name="plot")
app.add_typer(MooApp, name="moo")

# ============================================================================
# DATA COMMANDS
# ============================================================================

@DataApp.command("prep")
def data_prep_cmd(
    sql_dir: str = typer.Option("data/raw", "--sql-dir", help="Directory containing raw SQL dumps."),
    out_dir: str = typer.Option("data/processed", "--out-dir", help="Output directory for parquet files.")
) -> None:
    """Parse raw SQL design space dumps into ML-ready tabular formats."""
    # Currently data_prep runs on import or at module level, so we just run it via subprocess or direct import
    import subprocess
    typer.echo(f"Running data preparation on {sql_dir} -> {out_dir}")
    subprocess.run([sys.executable, "src/data_prep.py"], check=True)

# ============================================================================
# TRAINING COMMANDS
# ============================================================================

@TrainApp.command("train")
def train_cmd(
    epochs: int = typer.Option(200, "--epochs", help="Number of training epochs."),
    batch_size: int = typer.Option(1024, "--batch-size", help="Batch size for the DataLoader.")
) -> None:
    """Train the multi-task HINN with dynamic physics constraints."""
    from src.train import train_hinn
    train_hinn(epochs=epochs, batch_size=batch_size)

# ============================================================================
# PLOT COMMANDS
# ============================================================================

@PlotApp.command("training-dynamics")
def plot_training_cmd(
    metrics: str = typer.Option("results/models/metrics.csv", "--metrics", help="Path to metrics CSV."),
    out: str = typer.Option("results/figs/training_dynamics", "--out", help="Output path for the figure.")
) -> None:
    """Plot convergence (MSE) and Constraint Violation Ratio (CVR) over epochs."""
    from src.visualize.training_dynamics import plot_training_dynamics
    plot_training_dynamics(metrics, out)

@PlotApp.command("pareto")
def plot_pareto_cmd(
    pareto: str = typer.Option("results/data/pareto_front.csv", "--pareto", help="MOO results CSV."),
    raw: str = typer.Option("data/processed/targets.parquet", "--raw", help="Raw dataset targets."),
    out: str = typer.Option("results/figs/pareto_front", "--out", help="Output path for the figure.")
) -> None:
    """Visualize MOO Pareto boundary against raw sampled designs."""
    from src.visualize.pareto_front import plot_pareto
    plot_pareto(pareto, raw, out)

@PlotApp.command("monotonicity")
def plot_monotonicity_cmd(
    model: str = typer.Option("results/models/hinn_best.pt", "--model", help="Trained surrogate path."),
    features: str = typer.Option("data/processed/features.parquet", "--features", help="Feature dataset."),
    out: str = typer.Option("results/figs/monotonicity_proof", "--out", help="Output path for the figure.")
) -> None:
    """Sweep parallelism factors through the network to prove strict hardware bounding."""
    from src.visualize.monotonicity_proof import plot_monotonicity
    plot_monotonicity(model, "results/models/", features, out)

if __name__ == "__main__":
    app()
