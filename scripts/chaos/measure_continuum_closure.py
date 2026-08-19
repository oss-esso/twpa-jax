"""Measure whether the broadband residue admits a low-parameter closure.

This module reads existing ``result.json``/``spectrum.npz`` artifacts.  It
does not integrate a circuit or modify solver state.  The default report is a
falsification record: a linear family in the measured coherent-side scalars
is accepted only when every residue-shape metric has a bounded normalized
fit residual.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.chaos.measure_ansatz_validity import (
    BAND_HIGH_MULTIPLE,
    BAND_LOW_FRACTION,
    GENERATOR_ORDER,
    PUMP_ORDER,
    SIGNAL_ORDER,
    _reduce_lattice,
    _nodes,
    _within_window,
)


PUMP_ONLY_COLUMNS = (
    "on_lattice_pump_only",
    "off_lattice_pump_only",
    "top20_of_off_lattice_pump_only",
    "generator_share_pump_only",
    "generator_over_pump_pump_only",
)


@dataclass(frozen=True)
class ResidueMetrics:
    """Power and shape statistics for one coherent/residue split."""

    total_power: float
    coherent_power: float
    residue_power: float
    residue_fraction: float
    centroid_hz: float
    width_hz: float
    decay_exponent: float
    coherent_bins: int
    residue_bins: int


def _validate_spectrum(
    frequency_hz: np.ndarray,
    power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.asarray(frequency_hz, dtype=float).reshape(-1)
    values = np.asarray(power, dtype=float).reshape(-1)
    if frequency.size != values.size or frequency.size < 2:
        raise ValueError("frequency_hz and power must have equal length >= 2")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(values)):
        raise ValueError("frequency_hz and power must be finite")
    if np.any(values < 0.0):
        raise ValueError("power must be non-negative")
    if not np.all(np.diff(frequency) > 0.0):
        raise ValueError("frequency_hz must be strictly increasing")
    if float(values.sum()) <= 0.0:
        raise ValueError("power must contain a positive total")
    return frequency, values


def coherent_frequency_mask(
    frequency_hz: np.ndarray,
    *,
    pump_hz: float,
    signal_hz: float,
    window_hz: float,
    generator_frequency_hz: float | None = None,
) -> np.ndarray:
    """Return bins assigned to the pump/signal lattice and fitted generator."""
    frequency = np.asarray(frequency_hz, dtype=float).reshape(-1)
    pump = float(pump_hz)
    signal = float(signal_hz)
    window = float(window_hz)
    if pump <= 0.0 or signal <= 0.0 or window <= 0.0:
        raise ValueError("pump_hz, signal_hz, and window_hz must be positive")
    nodes = _nodes(pump, signal, 1)
    mask = _within_window(frequency, nodes, window)
    if generator_frequency_hz is not None:
        generator = float(generator_frequency_hz)
        if math.isfinite(generator) and generator > 0.0:
            base = np.asarray([
                n * pump + m * signal
                for n in range(-PUMP_ORDER + 2, PUMP_ORDER - 1)
                for m in range(-SIGNAL_ORDER + 1, SIGNAL_ORDER)
            ])
            orders = np.arange(-GENERATOR_ORDER, GENERATOR_ORDER + 1)
            generator_nodes = np.abs(
                (base[:, None] + generator * orders[None, :]).reshape(-1)
            )
            generator_nodes = np.unique(generator_nodes[generator_nodes > 0.0])
            if generator_nodes.size:
                mask |= _within_window(frequency, generator_nodes, window)
    return mask


def split_coherent_residue(
    frequency_hz: np.ndarray,
    power: np.ndarray,
    *,
    pump_hz: float,
    signal_hz: float,
    window_hz: float,
    generator_frequency_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a spectrum into coherent power, residue power, and a bin mask."""
    frequency, values = _validate_spectrum(frequency_hz, power)
    mask = coherent_frequency_mask(
        frequency,
        pump_hz=pump_hz,
        signal_hz=signal_hz,
        window_hz=window_hz,
        generator_frequency_hz=generator_frequency_hz,
    )
    coherent = np.where(mask, values, 0.0)
    residue = np.where(mask, 0.0, values)
    return coherent, residue, mask


def _decay_exponent(
    frequency_hz: np.ndarray,
    residue_power: np.ndarray,
    *,
    pump_hz: float,
    window_hz: float,
) -> float:
    offset = np.abs(frequency_hz - pump_hz)
    usable = (
        (residue_power > 0.0)
        & (offset > max(window_hz, np.finfo(float).eps))
    )
    if int(np.count_nonzero(usable)) < 2:
        return float("nan")
    slope, _intercept = np.polyfit(
        np.log(offset[usable]), np.log(residue_power[usable]), 1,
    )
    return float(slope)


def residue_metrics(
    frequency_hz: np.ndarray,
    coherent_power: np.ndarray,
    residue_power: np.ndarray,
    *,
    pump_hz: float,
    window_hz: float,
    coherent_mask: np.ndarray | None = None,
) -> ResidueMetrics:
    """Compute residue shape metrics while preserving the power identity."""
    frequency = np.asarray(frequency_hz, dtype=float).reshape(-1)
    coherent = np.asarray(coherent_power, dtype=float).reshape(-1)
    residue = np.asarray(residue_power, dtype=float).reshape(-1)
    if coherent.size != frequency.size or residue.size != frequency.size:
        raise ValueError("split spectra and frequency_hz must have equal length")
    if (
        not np.all(np.isfinite(frequency))
        or not np.all(np.isfinite(coherent))
        or not np.all(np.isfinite(residue))
        or np.any(coherent < 0.0)
        or np.any(residue < 0.0)
        or not np.all(np.diff(frequency) > 0.0)
    ):
        raise ValueError("split spectra must be finite, non-negative, and ordered")
    if coherent_mask is None:
        mask = coherent > 0.0
    else:
        mask = np.asarray(coherent_mask, dtype=bool).reshape(-1)
        if mask.size != frequency.size:
            raise ValueError("coherent_mask has the wrong length")
    total = float((coherent + residue).sum())
    coherent_total = float(coherent.sum())
    residue_total = float(residue.sum())
    if total <= 0.0:
        raise ValueError("split spectrum has zero total power")
    if residue_total > 0.0:
        centroid = float(np.sum(frequency * residue) / residue_total)
        width = float(
            math.sqrt(max(
                0.0,
                np.sum(((frequency - centroid) ** 2) * residue) / residue_total,
            ))
        )
        decay = _decay_exponent(
            frequency, residue, pump_hz=pump_hz, window_hz=window_hz,
        )
    else:
        centroid, width, decay = float("nan"), 0.0, float("nan")
    return ResidueMetrics(
        total_power=total,
        coherent_power=coherent_total,
        residue_power=residue_total,
        residue_fraction=residue_total / total,
        centroid_hz=centroid,
        width_hz=width,
        decay_exponent=decay,
        coherent_bins=int(np.count_nonzero(mask)),
        residue_bins=int(mask.size - np.count_nonzero(mask)),
    )


def analyse_spectrum(
    frequency_hz: np.ndarray,
    spectrum_db_relative_pump: np.ndarray,
    *,
    pump_hz: float,
    signal_hz: float,
    generator_over_pump: float | None,
) -> ResidueMetrics:
    """Reduce one stored dB spectrum using the existing ansatz conventions."""
    frequency = np.asarray(frequency_hz, dtype=float)
    db = np.asarray(spectrum_db_relative_pump, dtype=float)
    band = (
        (frequency > BAND_LOW_FRACTION * pump_hz)
        & (frequency < BAND_HIGH_MULTIPLE * pump_hz)
        & np.isfinite(db)
    )
    frequency = frequency[band]
    power = 10.0 ** (db[band] / 10.0)
    if frequency.size < 2:
        raise ValueError("stored spectrum has fewer than two usable bins")
    window_hz = 3.0 * float(np.median(np.diff(frequency)))
    generator_hz = (
        None
        if generator_over_pump is None or not math.isfinite(generator_over_pump)
        else float(generator_over_pump) * pump_hz
    )
    coherent, residue, mask = split_coherent_residue(
        frequency,
        power,
        pump_hz=pump_hz,
        signal_hz=signal_hz,
        window_hz=window_hz,
        generator_frequency_hz=generator_hz,
    )
    return residue_metrics(
        frequency,
        coherent,
        residue,
        pump_hz=pump_hz,
        window_hz=window_hz,
        coherent_mask=mask,
    )


def _control_key(device: str, axis: str, value: float) -> tuple[str, str, str]:
    return device, axis, f"{float(value):.12g}"


def _result_index(roots: Iterable[Path]) -> dict[tuple[str, str, str], Path]:
    index: dict[tuple[str, str, str], Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for result_path in root.rglob("result.json"):
            try:
                record = json.loads(result_path.read_text(encoding="utf-8"))
                device = str(record.get("device", ""))
                axis = str(record.get("control_axis", ""))
                value = record.get("control_value")
                spectrum_path = result_path.with_name("spectrum.npz")
                if not device or value is None or not spectrum_path.exists():
                    continue
                key = _control_key(device, axis, float(value))
                index.setdefault(key, result_path.parent)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return index


def _load_row_spectrum(point_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(point_dir / "spectrum.npz", allow_pickle=False) as data:
        if "frequency_hz" in data:
            frequency = np.asarray(data["frequency_hz"], dtype=float)
        elif "frequency_ghz" in data:
            frequency = np.asarray(data["frequency_ghz"], dtype=float) * 1.0e9
        else:
            raise KeyError("spectrum.npz has no frequency_hz/frequency_ghz")
        if "spectrum_db_relative_pump" in data:
            level = np.asarray(data["spectrum_db_relative_pump"], dtype=float)
        elif "spectrum_dbm" in data:
            level = np.asarray(data["spectrum_dbm"], dtype=float)
        else:
            raise KeyError("spectrum.npz has no supported dB spectrum")
    return frequency, level


def _pump_only_metrics(
    frequency_hz: np.ndarray,
    spectrum_db_relative_pump: np.ndarray,
    *,
    pump_hz: float,
    signal_hz: float,
) -> dict[str, float]:
    """Recompute the pump-only reduction using the stored spectrum."""
    frequency = np.asarray(frequency_hz, dtype=float)
    level = np.asarray(spectrum_db_relative_pump, dtype=float)
    band = (
        (frequency > BAND_LOW_FRACTION * pump_hz)
        & (frequency < BAND_HIGH_MULTIPLE * pump_hz)
        & np.isfinite(level)
    )
    frequency, level = frequency[band], level[band]
    if frequency.size < 2:
        raise ValueError("stored spectrum has fewer than two usable bins")
    power = 10.0 ** (level / 10.0)
    window_hz = 3.0 * float(np.median(np.diff(frequency)))
    reduced = _reduce_lattice(
        frequency,
        power,
        pump_hz,
        signal_hz,
        window_hz,
        400,
        pump_only=True,
    )
    return {f"{key}_pump_only": float(value) for key, value in reduced.items()
            if key in {name.removesuffix("_pump_only") for name in PUMP_ONLY_COLUMNS}}


def _reconcile_pump_only_schema(
    row: dict[str, str],
    recomputed: dict[str, float],
) -> tuple[dict[str, float], str, list[str], float, dict[str, float]]:
    """Use CSV pump-only fields only after checking them against the artifact."""
    absent = [name for name in PUMP_ONLY_COLUMNS if not row.get(name)]
    if absent:
        return recomputed, "recomputed_from_spectrum", absent, float("nan"), {}
    try:
        provided = {name: float(row[name]) for name in PUMP_ONLY_COLUMNS}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid pump-only CSV fields: {exc}") from exc
    if not all(math.isfinite(value) for value in provided.values()):
        raise ValueError("pump-only CSV fields must be finite")
    differences = {
        name: abs(provided[name] - recomputed[name])
        for name in PUMP_ONLY_COLUMNS
        if abs(provided[name] - recomputed[name]) > 1.0e-10
    }
    difference = max(differences.values(), default=0.0)
    if difference > 1.0e-10:
        return (
            recomputed,
            "recomputed_after_csv_mismatch",
            [],
            difference,
            differences,
        )
    return provided, "csv_verified_against_spectrum", [], difference, {}


def _fit_family(
    rows: list[dict[str, Any]],
    *,
    lattice_feature: str = "on_lattice_pump_only",
) -> dict[str, Any]:
    names = [
        "residue_fraction",
        "centroid_over_pump",
        "width_over_pump",
        "decay_exponent",
    ]
    feature_names = [
        "drive",
        lattice_feature,
        "branch_current_max_over_ic",
        "min_cos_phi",
    ]
    features: list[list[float]] = []
    targets: dict[str, list[float]] = {name: [] for name in names}
    fit_rows: list[dict[str, Any]] = []
    for row in rows:
        values = [row.get(name) for name in feature_names]
        outputs = [row.get(name) for name in names]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            continue
        if any(value is None or not math.isfinite(float(value)) for value in outputs):
            continue
        features.append([1.0, *[float(value) for value in values]])
        fit_rows.append(row)
        for name, value in zip(names, outputs):
            targets[name].append(float(value))
    if len(features) < len(feature_names) + 2:
        return {"verdict": "INSUFFICIENT_DATA", "n_fit": len(features)}

    design = np.asarray(features, dtype=float)
    metrics: dict[str, Any] = {}
    relative_errors: list[float] = []
    normalized_residuals: list[np.ndarray] = []
    for name in names:
        target = np.asarray(targets[name], dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        prediction = design @ coefficients
        scale = max(float(np.ptp(target)), abs(float(np.mean(target))), 1.0e-12)
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        normalized = rmse / scale
        relative_errors.append(normalized)
        normalized_residuals.append((prediction - target) / scale)
        metrics[name] = {
            "rmse": rmse,
            "normalized_rmse": normalized,
            "coefficients": coefficients.tolist(),
        }
    threshold = 0.25
    aggregate_residual = np.sqrt(
        np.mean(np.square(np.asarray(normalized_residuals, dtype=float)), axis=0)
    )
    metrics.update({
        "verdict": "PASS" if max(relative_errors) <= threshold else "NO_GO",
        "threshold": threshold,
        "max_normalized_rmse": max(relative_errors),
        "n_fit": len(features),
        "features": feature_names,
        "residual_distribution": {
            "metric_normalized_rmse": {
                name: metrics[name]["normalized_rmse"] for name in names
            },
            "aggregate_normalized_rms": {
                "n": int(aggregate_residual.size),
                "min": float(np.min(aggregate_residual)),
                "p25": float(np.percentile(aggregate_residual, 25.0)),
                "median": float(np.median(aggregate_residual)),
                "p75": float(np.percentile(aggregate_residual, 75.0)),
                "p90": float(np.percentile(aggregate_residual, 90.0)),
                "p95": float(np.percentile(aggregate_residual, 95.0)),
                "p99": float(np.percentile(aggregate_residual, 99.0)),
                "max": float(np.max(aggregate_residual)),
            },
            "row_ids": [
                f"{row['device']}:{row['control_axis']}:{row['control_value']:g}"
                for row in fit_rows
            ],
            "aggregate_normalized_rms_values": aggregate_residual.tolist(),
        },
    })
    return metrics


def analyse_corpus(
    csv_path: Path | Iterable[Path],
    campaign_roots: Iterable[Path],
    *,
    lattice_denominator: str = "pump_only",
) -> dict[str, Any]:
    """Analyse every CSV row with an explicit pump-only schema reconciliation."""
    if lattice_denominator not in {"pump_only", "signal"}:
        raise ValueError("lattice_denominator must be 'pump_only' or 'signal'")
    csv_paths = [csv_path] if isinstance(csv_path, Path) else list(csv_path)
    if not csv_paths:
        raise ValueError("at least one CSV path is required")
    roots = list(campaign_roots)
    measurements: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    schema_counts: dict[str, dict[str, int]] = {}
    schema_missing_columns: dict[str, dict[str, int]] = {}
    schema_mismatch_controls: dict[str, list[float]] = {}
    total_rows = 0
    for current_csv in csv_paths:
        with current_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        total_rows += len(rows)
        # Prefer artifacts next to a source CSV.  This keeps duplicate controls
        # from being silently resolved to a different campaign tree.
        index = _result_index([current_csv.parent, *roots])
        for row in rows:
            source = str(current_csv)
            device = str(row.get("device", ""))
            axis = str(row.get("control_axis", ""))
            value = row.get("control_value")
            if not device or value is None:
                missing.append({"row": row, "source": source,
                                "reason": "missing identity fields"})
                continue
            key = _control_key(device, axis, float(value))
            point_dir = index.get(key)
            if point_dir is None:
                missing.append({"row": row, "source": source,
                                "reason": "matching stored spectrum not found"})
                continue
            try:
                with (point_dir / "result.json").open(encoding="utf-8") as handle:
                    record = json.load(handle)
                frequency, level = _load_row_spectrum(point_dir)
                pump_hz = float(row["pump_hz"])
                signal_hz = float(row["signal_hz"])
                generator_ratio = float(row["generator_over_pump"])
                signal_on_lattice = float(row["on_lattice"])
                if not math.isfinite(signal_on_lattice):
                    raise ValueError("signal on_lattice must be finite")
                metrics = analyse_spectrum(
                    frequency,
                    level,
                    pump_hz=pump_hz,
                    signal_hz=signal_hz,
                    generator_over_pump=generator_ratio,
                )
                recomputed = _pump_only_metrics(
                    frequency, level, pump_hz=pump_hz, signal_hz=signal_hz,
                )
                pump_only, schema_source, absent, max_difference, discrepancies = (
                    _reconcile_pump_only_schema(row, recomputed)
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                missing.append({"row": row, "source": source, "reason": str(exc)})
                continue
            counts = schema_counts.setdefault(device, {})
            counts[schema_source] = counts.get(schema_source, 0) + 1
            if absent:
                absent_counts = schema_missing_columns.setdefault(device, {})
                for name in absent:
                    absent_counts[name] = absent_counts.get(name, 0) + 1
            if discrepancies:
                schema_mismatch_controls.setdefault(device, []).append(float(value))
            item = {
                "device": device,
                "control_axis": axis,
                "control_value": float(value),
                "drive": float(value),
                "on_lattice": (
                    pump_only["on_lattice_pump_only"]
                    if lattice_denominator == "pump_only"
                    else signal_on_lattice
                ),
                "on_lattice_signal": signal_on_lattice,
                **pump_only,
                "pump_only_schema_source": schema_source,
                "pump_only_missing_columns": absent,
                "pump_only_max_abs_difference": max_difference,
                "pump_only_csv_discrepancies": discrepancies,
                "branch_current_max_over_ic": (
                    None
                    if not row.get("branch_current_max_over_ic")
                    else float(row["branch_current_max_over_ic"])
                ),
                "min_cos_phi": (
                    None if not row.get("min_cos_phi") else float(row["min_cos_phi"])
                ),
                "pump_hz": pump_hz,
                "point_dir": str(point_dir),
                "csv_source": source,
                **asdict(metrics),
            }
            item["centroid_over_pump"] = metrics.centroid_hz / pump_hz
            item["width_over_pump"] = metrics.width_hz / pump_hz
            measurements.append(item)
            del record
    csv_value: str | list[str] = (
        str(csv_paths[0]) if len(csv_paths) == 1 else [str(path) for path in csv_paths]
    )
    return {
        "csv_path": csv_value,
        "csv_paths": [str(path) for path in csv_paths],
        "n_csv_rows": total_rows,
        "n_measured": len(measurements),
        "n_missing": len(missing),
        "lattice_denominator": lattice_denominator,
        "pump_only_columns": list(PUMP_ONLY_COLUMNS),
        "pump_only_schema_counts_by_device": schema_counts,
        "pump_only_missing_columns_by_device": schema_missing_columns,
        "pump_only_mismatch_controls_by_device": schema_mismatch_controls,
        "measurements": measurements,
        "missing": missing,
        "family_fit": _fit_family(
            measurements,
            lattice_feature=(
                "on_lattice_pump_only"
                if lattice_denominator == "pump_only"
                else "on_lattice_signal"
            ),
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=Path("outputs/chaos/ansatz_validity/ansatz_validity.csv"),
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        action="append",
        default=None,
        help="root to search for matching result.json/spectrum.npz pairs",
    )
    parser.add_argument(
        "--lattice-denominator",
        choices=("pump_only", "signal"),
        default="pump_only",
        help="lattice fraction used as the closure feature (default: pump_only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/chaos/ansatz_validity/continuum_closure.json"),
    )
    args = parser.parse_args(argv)
    roots = args.campaign_root or [Path("outputs/chaos")]
    csv_paths = args.csv if isinstance(args.csv, list) else [args.csv]
    report = analyse_corpus(
        csv_paths,
        roots,
        lattice_denominator=args.lattice_denominator,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "verdict": report["family_fit"].get("verdict"),
        "n_measured": report["n_measured"],
        "n_missing": report["n_missing"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
