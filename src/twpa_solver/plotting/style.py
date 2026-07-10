"""Shared plotting style helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt

FIG_DPI = 200
SAVE_FORMATS = ("png",)
THESIS_FIGSIZE_MAP = (9, 7)
THESIS_FIGSIZE_SPECTRUM = (12, 7)


def apply_thesis_style() -> None:
    """Apply a LaTeX/serif look to every matplotlib figure in the process.

    Uses a real LaTeX backend when a ``latex`` binary is on PATH; otherwise
    falls back to matplotlib's Computer-Modern mathtext so figures still render
    with LaTeX-style fonts (and ``$...$`` math in labels) without a TeX install.
    """
    rc: dict[str, object] = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
    }
    if shutil.which("latex"):
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.rcParams.update(rc)


def save_figure(
    fig: plt.Figure,
    outpath: Path | str,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
    dpi: int = FIG_DPI,
) -> None:
    """Save a figure as PNG plus optional vector formats."""
    path = Path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
