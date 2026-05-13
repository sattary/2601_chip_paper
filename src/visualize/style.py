"""
Shared matplotlib/seaborn style for Nature-level publication figures for HINN.
Adapted from ali_proj.
"""
from __future__ import annotations
import contextlib
from pathlib import Path
from typing import Generator
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

MM_TO_INCH = 1.0 / 25.4
SINGLE_COL = 89 * MM_TO_INCH
DOUBLE_COL = 183 * MM_TO_INCH
DPI = 300

NATURE_PALETTE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "teal": "#009E73",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "sky_blue": "#56B4E9",
    "purple": "#9C27B0",
    "gray": "#7F7F7F",
}
LINE_COLORS = list(NATURE_PALETTE.values())[:6]

NATURE_RC = {
    "font.family": "serif",
    "font.serif": ["Computer Modern", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "mathtext.fontset": "dejavuserif",
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

@contextlib.contextmanager
def nature_style() -> Generator[None, None, None]:
    old_rc = mpl.rcParams.copy()
    mpl.rcParams.update(NATURE_RC)
    sns.set_context("paper", font_scale=1.0)
    sns.set_palette(LINE_COLORS)
    try:
        yield
    finally:
        mpl.rcParams.update(old_rc)
        sns.set()

def save_figure(fig: plt.Figure, path: str) -> None:
    path_obj = Path(path)
    base_path = path_obj.with_suffix("")
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base_path}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{base_path}.svg", bbox_inches="tight")
    plt.close(fig)
