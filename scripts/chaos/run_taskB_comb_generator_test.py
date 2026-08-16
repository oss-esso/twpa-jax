"""Read-only null-control and fixed-comb analysis for Phase B spectra."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from scripts.chaos import measure_ansatz_validity as measure


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs" / "chaos" / "phaseB_signal"
ARTIFACTS = INPUT / "taskB_scans"
SUMMARY = INPUT / "taskB_comb_test.csv"
PROMINENCE_THRESHOLD = 3.0
N_TRIALS = 400
COMB_HZ = {"ipm_2c_fixed": 241.7e6}


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
        prefix=path.name + ".", suffix=".tmp",
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def _scan(
    freq: np.ndarray, power: np.ndarray, pump_hz: float, signal_hz: float,
    window_hz: float,
) -> tuple[list[dict[str, float]], np.ndarray]:
    base = np.array([
        n * pump_hz + m * signal_hz
        for n in range(-measure.PUMP_ORDER + 2, measure.PUMP_ORDER - 1)
        for m in range(-measure.SIGNAL_ORDER + 1, measure.SIGNAL_ORDER)
    ])
    orders = np.arange(-measure.GENERATOR_ORDER, measure.GENERATOR_ORDER + 1)
    total = float(power.sum())
    ratios = np.linspace(0.01, 0.5, N_TRIALS)
    shares: list[dict[str, float]] = []
    for ratio in ratios:
        f_a = ratio * pump_hz
        nodes = np.abs((base[:, None] + f_a * orders[None, :]).ravel())
        nodes = nodes[nodes > 0.0]
        share = float(power[measure._within_window(freq, nodes, window_hz)].sum() / total)
        shares.append({"trial": float(len(shares)), "f_a_over_f_p": float(ratio), "share": share})
    return shares, ratios


def _comb_share(
    freq: np.ndarray, power: np.ndarray, pump_hz: float, signal_hz: float,
    comb_hz: float, window_hz: float,
) -> float:
    base = np.array([
        n * pump_hz + m * signal_hz
        for n in range(-measure.PUMP_ORDER + 2, measure.PUMP_ORDER - 1)
        for m in range(-measure.SIGNAL_ORDER + 1, measure.SIGNAL_ORDER)
    ])
    orders = np.arange(-measure.GENERATOR_ORDER, measure.GENERATOR_ORDER + 1)
    nodes = np.abs((base[:, None] + comb_hz * orders[None, :]).ravel())
    nodes = nodes[nodes > 0.0]
    return float(power[measure._within_window(freq, nodes, window_hz)].sum() / power.sum())


def _point_data(point: Path) -> tuple[dict, np.ndarray, np.ndarray, float, float, float, float] | None:
    record_path = point / "result.json"
    spectrum_path = point / "spectrum.npz"
    if not record_path.exists() or not spectrum_path.exists():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    device = str(record.get("device", ""))
    pump_hz = float(record.get("pump_hz") or (7.0e9 if device == "guarcello" else 0.0))
    signal_hz = float(record.get("signal_hz") or 0.0)
    if signal_hz <= 0.0:
        legacy = measure.LEGACY_SIGNAL_GHZ.get(str(record.get("device", "")))
        signal_hz = legacy * 1e9 if legacy else 0.0
    if pump_hz <= 0.0 or signal_hz <= 0.0:
        return None
    with np.load(spectrum_path) as data:
        freq = np.asarray(data["frequency_hz"])
        magnitude_db = np.asarray(data["spectrum_db_relative_pump"])
    band = (
        (freq > measure.BAND_LOW_FRACTION * pump_hz)
        & (freq < measure.BAND_HIGH_MULTIPLE * pump_hz)
        & np.isfinite(magnitude_db)
    )
    freq, magnitude_db = freq[band], magnitude_db[band]
    if freq.size < 64:
        return None
    power = 10.0 ** (magnitude_db / 10.0)
    window_hz = measure.WINDOW_BINS * float(np.median(np.diff(freq)))
    integer_mask = measure._within_window(
        freq, measure._nodes(pump_hz, signal_hz, 1), window_hz,
    )
    return (
        record, freq[~integer_mask], power[~integer_mask], float(power.sum()),
        window_hz, pump_hz, signal_hz,
    )


def main() -> int:
    rows: list[dict[str, object]] = []
    excluded = 0
    for point in sorted(INPUT.glob("*/*/")):
        data = _point_data(point)
        if data is None:
            continue
        record, off_freq, off_power, total_power, window_hz, pump_hz, signal_hz = data
        device = str(record["device"])
        off_lattice = float(off_power.sum() / total_power)
        control_value = record.get("control_value", record.get("pump_power_dbm"))
        payload: dict[str, object] = {
            "device": device,
            "point": point.name,
            "control_value": control_value,
            "off_lattice": off_lattice,
            "prominence_threshold": PROMINENCE_THRESHOLD,
        }
        if off_lattice <= 0.01:
            excluded += 1
            payload["excluded"] = True
            payload["excluded_reason"] = "off_lattice <= 0.01; generator statistic is numerical-floor degenerate"
            best = median = prominence = best_ratio = comb = None
            verdict = "EXCLUDED_OFF_LATTICE_FLOOR"
        else:
            scans, _ = _scan(
                off_freq, off_power, pump_hz, signal_hz, window_hz,
            )
            shares = np.asarray([item["share"] for item in scans])
            best_index = int(np.argmax(shares))
            best = float(shares[best_index])
            median = float(np.median(shares))
            p90 = float(np.percentile(shares, 90.0))
            prominence = float(best / median) if median > 0.0 else None
            best_ratio = float(scans[best_index]["f_a_over_f_p"])
            comb_hz = COMB_HZ.get(device)
            comb = (
                _comb_share(
                    off_freq, off_power, pump_hz, signal_hz,
                    comb_hz, window_hz,
                ) if comb_hz is not None else None
            )
            if comb is None:
                verdict = "NOT_RUN_NO_DEVICE_COMB_SPACING"
            elif comb >= best:
                verdict = "COMB_SIDEBAND"
            else:
                verdict = "INCOMMENSURATE_GENERATOR"
            payload.update({
                "best_share": best,
                "best_f_a_over_f_p": best_ratio,
                "median_share": median,
                "p90_share": p90,
                "peak_prominence_best_over_median": prominence,
                "comb_spacing_hz": comb_hz,
                "comb_share": comb,
                "generator_scan": scans,
            })
        payload["verdict"] = verdict
        artifact = ARTIFACTS / device / (point.name + ".json")
        _atomic_text(artifact, json.dumps(payload, indent=2, allow_nan=False))
        rows.append({
            "device": device,
            "point": point.name,
            "control_value": control_value,
            "off_lattice": off_lattice,
            "best_share": best,
            "median_share": median,
            "p90_share": payload.get("p90_share"),
            "best_over_median": prominence,
            "f_a_over_f_p": best_ratio,
            "comb_share": comb,
            "comb_verdict": verdict,
            "prominence_clears_threshold": (
                None if prominence is None else prominence >= PROMINENCE_THRESHOLD
            ),
        })

    rows.sort(key=lambda row: (str(row["device"]), float(row["control_value"])))
    if not rows:
        raise SystemExit("no reducible Phase B points found")
    fieldnames = list(rows[0])
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=SUMMARY.parent,
        delete=False, prefix=SUMMARY.name + ".", suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(SUMMARY)

    recommendations: dict[str, str] = {}
    for device in sorted({str(row["device"]) for row in rows}):
        device_rows = [row for row in rows if row["device"] == device]
        strong = [row for row in device_rows if row["prominence_clears_threshold"] is True]
        comb_rows = [row for row in strong if row["comb_verdict"] in {"COMB_SIDEBAND", "INCOMMENSURATE_GENERATOR"}]
        if comb_rows and sum(row["comb_verdict"] == "COMB_SIDEBAND" for row in comb_rows) >= len(comb_rows) / 2:
            recommendations[device] = "more sidebands"
        elif comb_rows:
            recommendations[device] = "auxiliary generator"
        elif strong and device not in COMB_HZ:
            ratios = np.asarray([float(row["f_a_over_f_p"]) for row in strong])
            recommendations[device] = "auxiliary generator" if ratios.size >= 2 and float(np.ptp(ratios)) <= 0.01 else "neither"
        else:
            recommendations[device] = "neither"
    report = {
        "prominence_threshold": PROMINENCE_THRESHOLD,
        "generator_trials": N_TRIALS,
        "excluded_off_lattice_floor_points": excluded,
        "comb_spacing_hz": COMB_HZ,
        "recommendations": recommendations,
    }
    _atomic_text(INPUT / "taskB_report.json", json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
