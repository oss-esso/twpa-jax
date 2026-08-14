"""Reclassify and replot Phase B using only persisted traces."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.chaos import run_phaseB_pump_only as phase_b
from scripts.chaos.phaseA_regression import load_committed_module


ROOT = Path(__file__).resolve().parents[2]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(phase_b._json_safe(row) for row in rows)


def _load_trace(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / str(row["trace_path"])
    data = np.load(path, allow_pickle=False)
    return np.asarray(data["t"], dtype=float), np.asarray(data["v_out"], dtype=float)


def _floor_from_trace(path: Path, pump_hz: float) -> tuple[float, float]:
    data = np.load(path, allow_pickle=False)
    orders = phase_b.classifier.symmetry_order_parameters(
        np.asarray(data["t"], dtype=float), np.asarray(data["v_out"], dtype=float), pump_hz,
    )
    return float(orders["q_even"]), float(orders["q_dc"])


def _device_reduction(name: str, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = sorted(_read_rows(root / "summary.csv"), key=lambda row: float(row["pump_power_dbm"]))
    pump_hz = 7.0e9 if name == "guarcello" else (7.12e9 if name == "jc_jtwpa" else 7.90e9)
    pump_off = root / "pump_off" / "trace.npz"
    pump_off_floor = None
    if pump_off.exists():
        pump_off_floor = _floor_from_trace(pump_off, pump_hz)
    low_rows = rows[: min(5, len(rows))]
    low_floor = (
        float(np.mean([float(row.get("q_even", "nan")) for row in low_rows])),
        float(np.mean([float(row.get("q_dc", "nan")) for row in low_rows])),
    )
    baseline = (
        max(low_floor[0], pump_off_floor[0]) if pump_off_floor else low_floor[0],
        max(low_floor[1], pump_off_floor[1]) if pump_off_floor else low_floor[1],
    )
    low_spectral_lines = []
    for low_row in low_rows:
        spectrum = np.load(ROOT / str(low_row["trace_path"]).replace("trace.npz", "spectrum.npz"), allow_pickle=False)
        frequency = np.asarray(spectrum["frequency_hz"], dtype=float)
        level = np.asarray(spectrum["spectrum_db_relative_pump"], dtype=float)
        low_spectral_lines.append(max(
            float(level[np.abs(frequency - 0.5 * pump_hz) <= 80e6].max()),
            float(level[np.abs(frequency - 1.5 * pump_hz) <= 80e6].max()),
        ))
    spectral_floor_db = float(np.median(low_spectral_lines))
    legacy = load_committed_module() if name == "guarcello" else None
    before_after: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") == "FAILED":
            continue
        t, v = _load_trace(row)
        old_verdict = None
        if legacy is not None:
            old_verdict = legacy.classify_trace(t, v, drive_hz=pump_hz).verdict
        reduced = phase_b._reduce_trace(
        t, v, pump_hz, baseline_q_even=baseline[0], baseline_q_dc=baseline[1],
            symmetry_floor_factor=20.0, spectral_floor_db=spectral_floor_db,
        )
        row.update({key: value for key, value in reduced.items() if key not in {"spectrum_frequency_hz", "spectrum_db_relative_pump", "upward_branch"}})
        row["reduction_source"] = "trace.npz; no solver import"
        if legacy is not None:
            before_after.append({
                "pump_power_dbm": float(row["pump_power_dbm"]),
                "before_verdict": old_verdict,
                "after_verdict": row["verdict"],
                "changed": old_verdict != row["verdict"],
            })
    metadata = {
        "device": name,
        "pump_off_floor": pump_off_floor,
        "low_power_floor_first_five": low_floor,
        "chosen_floor": baseline,
        "spectral_half_integer_floor_db": spectral_floor_db,
        "symmetry_floor_factor": 20.0,
        "minimum_samples_per_period": phase_b.classifier.MIN_SAMPLES_PER_PERIOD,
        "smaller_stride_alias_check": "NOT_AVAILABLE: persisted traces have record_stride=20 and no unstrided samples",
    }
    _write_rows(root / "summary.csv", rows)
    if before_after:
        (root / "classifier_before_after.json").write_text(
            json.dumps({
                "baseline": "git show HEAD:scripts/chaos/attractor_classify.py",
                "fixed": "working scripts/chaos/attractor_classify.py",
                "rows": before_after,
                "changed_count": sum(row["changed"] for row in before_after),
            }, indent=2), encoding="utf-8",
        )
    (root / "reduction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return rows, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "chaos" / "phaseB")
    args = parser.parse_args()
    all_rows: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] = {}
    for name in ("guarcello", "jc_jtwpa", "jc_fqjtwpa"):
        root = args.output / name
        if not (root / "summary.csv").exists():
            continue
        rows, device_metadata = _device_reduction(name, root)
        all_rows[name] = rows
        metadata[name] = device_metadata
        phase_b._plot_device(name, root, rows)
    if all_rows:
        phase_b._plot_cross_device(all_rows, args.output)
    (args.output / "reduction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"devices": list(all_rows), "metadata": metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
