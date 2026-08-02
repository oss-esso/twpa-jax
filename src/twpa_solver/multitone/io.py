"""Serialization for multitone compression results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_compression_outputs(
    outdir: str | Path,
    points: list[dict[str, Any]],
    *,
    arrays: dict[str, np.ndarray] | None = None,
    summary: dict[str, Any] | None = None,
    states: dict[str, np.ndarray] | None = None,
    save_states: str = "none",
    spatial_rows: list[dict[str, Any]] | None = None,
) -> Path:
    """Write the stable compression artifact set and return its directory."""
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    if points:
        columns = sorted({key for point in points for key in point})
        with (destination / "compression_points.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(points)
    else:
        (destination / "compression_points.csv").write_text("\n", encoding="utf-8")
    np.savez_compressed(destination / "compression_arrays.npz", **(arrays or {}))
    payload = dict(summary or {})
    payload.setdefault("stability_status", "NOT_CHECKED")
    (destination / "compression_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    if save_states not in {"none", "last", "selected", "all"}:
        raise ValueError(f"unknown save_states={save_states!r}")
    if states and save_states != "none":
        selected = states if save_states == "all" else {key: value for key, value in states.items() if key in {"last", "zero_signal", "mid", "p1db"}}
        for name, state in selected.items():
            value = np.asarray(state)
            np.savez_compressed(
                destination / f"multitone_solution_{name}.npz",
                # Validation and continuation diagnostics need the converged
                # state at the solver's precision; float32 quantizes tiny
                # signal sectors enough to spoil JVP finite differences.
                X_real=np.asarray(value.real, dtype=np.float64),
                X_imag=np.asarray(value.imag, dtype=np.float64),
            )
    if spatial_rows is not None:
        columns = sorted({key for row in spatial_rows for key in row})
        with (destination / "spatial_profiles.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(spatial_rows)
    return destination
