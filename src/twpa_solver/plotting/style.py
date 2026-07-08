"""Shared plotting style helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

FIG_DPI = 200
SAVE_FORMATS = ("png",)
THESIS_FIGSIZE_MAP = (9, 7)
THESIS_FIGSIZE_SPECTRUM = (12, 7)


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
