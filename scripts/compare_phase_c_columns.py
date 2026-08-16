"""Build measured Phase C comparisons on the FDTD achieved-current grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".hybrid_outputs" / "phase_c_three_way"
SIDEBANDS = 6

DEVICES = {
    "jc_jtwpa": {"timestep_status": "MEASURED_TIMESTEP_NOT_CONVERGED"},
    "jc_fqjtwpa": {"timestep_status": "MEASURED_TIMESTEP_NOT_CONVERGED"},
    "ipm_2c_fixed": {"timestep_status": "MEASURED_TIMESTEP_CONVERGED"},
    "guarcello": {"timestep_status": "MEASURED_PAPER_TIMESTEP"},
}


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _fdtd_rows(device: str) -> list[dict[str, Any]]:
    root = ROOT / "outputs" / "chaos" / "phaseB_signal" / device
    rows: list[dict[str, Any]] = []
    for path in root.rglob("result.json"):
        row = _read_json(path)
        if row is None:
            continue
        current = _number(row.get("pump_current_peak_a_achieved"))
        provenance = "measured"
        if current is None and device == "guarcello":
            power_w = 1.0e-3 * 10.0 ** (float(row["pump_power_dbm"]) / 10.0)
            current = math.sqrt(2.0 * power_w / 50.0)
            provenance = "derived_legacy_50_ohm"
        if current is None:
            continue
        row["_current"] = current
        row["_current_provenance"] = provenance
        row["_path"] = str(path)
        rows.append(row)
    return sorted(rows, key=lambda item: float(item["_current"]))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _nearest(rows: list[dict[str, Any]], current: float, key: str) -> dict[str, Any] | None:
    candidates = [(row, _number(row.get(key))) for row in rows]
    candidates = [(row, value) for row, value in candidates if value is not None]
    if not candidates:
        return None
    row, value = min(candidates, key=lambda item: abs(float(item[1]) - current))
    return row if math.isclose(float(value), current, rel_tol=2.0e-9, abs_tol=1.0e-18) else None


def _summaries(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in root.rglob("compression_summary.json"):
        row = _read_json(path)
        if row is None:
            continue
        row["_path"] = str(path)
        rows.append(row)
    return rows


def _new_status(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NO_DATA_NOT_RUN"
    if _number(row.get("small_signal_gain_vs_off_db")) is not None and int(row.get("n_failed_power_points", 0)) == 0:
        return "MEASURED"
    return str(row.get("status") or "NO_DATA_UNKNOWN_FAILURE")


def _diagnostic_rj(rows: list[dict[str, Any]], current: float) -> tuple[float | None, str | None]:
    row = _nearest(rows, current, "pump_current_a")
    if row is None:
        return None, None
    summary = row.get("pump_summary") or {}
    return _number(summary.get("branch_current_max_over_ic")), row.get("_path")


def _stats(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return {"count": 0, "median_s": None, "worst_s": None, "total_s": None}
    midpoint = count // 2
    median = (
        ordered[midpoint]
        if count % 2
        else 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
    )
    return {
        "count": count,
        "median_s": median,
        "worst_s": max(ordered),
        "total_s": sum(ordered),
    }


def _difference_stats(records: list[dict[str, Any]], left: str, right: str) -> dict[str, float | int | None]:
    differences = [
        float(row[right]) - float(row[left])
        for row in records
        if row.get(left) is not None and row.get(right) is not None
    ]
    if not differences:
        return {"count": 0, "median_signed": None, "median_abs": None, "worst_abs": None}
    signed = sorted(differences)
    absolute = sorted(abs(value) for value in differences)
    midpoint = len(signed) // 2
    median_signed = signed[midpoint] if len(signed) % 2 else 0.5 * (signed[midpoint - 1] + signed[midpoint])
    median_abs = absolute[midpoint] if len(absolute) % 2 else 0.5 * (absolute[midpoint - 1] + absolute[midpoint])
    return {
        "count": len(differences), "median_signed": median_signed,
        "median_abs": median_abs, "worst_abs": max(absolute),
    }


def reduce_device(device: str, output: Path) -> dict[str, Any]:
    fdtd = _fdtd_rows(device)
    single_path = ROOT / ".hybrid_outputs" / "phase_c_single_tone_exact" / device / "map_points.csv"
    single = _csv_rows(single_path)
    new = _summaries(ROOT / ".hybrid_outputs" / "phase_c_new_hb" / device)
    timed = _summaries(ROOT / ".hybrid_outputs" / "phase_c_new_hb_timed" / device)
    diagnostics = _summaries(ROOT / ".hybrid_outputs" / "phase_c_new_hb_diagnostics" / device)
    records: list[dict[str, Any]] = []
    for index, frow in enumerate(fdtd):
        current = float(frow["_current"])
        srow = _nearest(single, current, "pump_current_peak_a")
        nrow = _nearest(new, current, "pump_current_a")
        trow = _nearest(timed, current, "pump_current_a")
        new_rj, diagnostic_path = _diagnostic_rj(diagnostics, current)
        single_measured = bool(srow and srow.get("status") == "PASS" and _number(srow.get("gain_vs_off_db")) is not None)
        new_status = _new_status(nrow)
        record: dict[str, Any] = {
            "device": device,
            "point_index": index,
            "control_axis": frow.get("control_axis"),
            "control_value": frow.get("control_value"),
            "pump_power_dbm": frow.get("pump_power_dbm"),
            "pump_current_peak_a_achieved": current,
            "pump_current_provenance": frow["_current_provenance"],
            "fdtd_status": "MEASURED" if _number(frow.get("gain_vs_off_db")) is not None else "NO_DATA_MISSING_GAIN",
            "fdtd_timestep_status": DEVICES[device]["timestep_status"],
            "fdtd_gain_vs_off_db": _number(frow.get("gain_vs_off_db")),
            "fdtd_sidebands": 0,
            "fdtd_dt_s": _number(frow.get("dt_s")),
            "fdtd_runtime_s": _number(frow.get("runtime_s")),
            "fdtd_source_path": frow["_path"],
            "single_tone_status": "MEASURED" if single_measured else str((srow or {}).get("status") or "NO_DATA_NOT_RUN"),
            "single_tone_gain_vs_off_db": _number((srow or {}).get("gain_vs_off_db")),
            "single_tone_sidebands": SIDEBANDS,
            "single_tone_runtime_s": _number((srow or {}).get("elapsed_s")),
            "single_tone_source_path": str(single_path) if srow is not None else None,
            "new_hb_status": new_status,
            "new_hb_gain_vs_off_db": _number((nrow or {}).get("small_signal_gain_vs_off_db")),
            "new_hb_sidebands": SIDEBANDS,
            "new_hb_runtime_s": _number((trow or {}).get("total_wall_runtime_s")),
            "new_hb_source_path": (nrow or {}).get("_path"),
            "new_hb_runtime_source_path": (trow or {}).get("_path"),
            "new_hb_failure_reason": (nrow or {}).get("failure_reason") or (nrow or {}).get("message"),
            "new_hb_pump_continuation_method": (nrow or {}).get("pump_continuation_method"),
            "new_hb_pump_continuation_steps": (nrow or {}).get("pump_continuation_steps"),
        }
        if device != "guarcello":
            record.update({
                "fdtd_r_j": _number(frow.get("r_j")),
                "fdtd_r_j_state": "signal_installed",
                "single_tone_r_j": (
                    _number((srow or {}).get("pump_branch_current_max_over_ic"))
                    if single_measured else None
                ),
                "new_hb_r_j": new_rj,
                "new_hb_r_j_source_path": diagnostic_path,
            })
        records.append(record)

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{device}.csv"
    fields = list(records[0]) if records else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    import matplotlib.pyplot as plt

    panel_count = 3 if device != "guarcello" else 2
    fig, axes = plt.subplots(panel_count, 1, figsize=(8.5, 3.2 * panel_count), sharex=True)
    if panel_count == 2:
        gain_ax, runtime_ax = axes
        rj_ax = None
    else:
        gain_ax, rj_ax, runtime_ax = axes
    x = [1.0e6 * float(row["pump_current_peak_a_achieved"]) for row in records]
    for prefix, label, marker in (("fdtd", "FDTD", "^-"), ("single_tone", "single-tone HB", "o-"), ("new_hb", "new HB", "s-")):
        y = [row.get(f"{prefix}_gain_vs_off_db") for row in records]
        pairs = [(xx, yy) for xx, yy in zip(x, y) if yy is not None]
        if pairs:
            gain_ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker, label=label)
    gain_ax.set_ylabel("gain_vs_off_db (dB)")
    gain_ax.legend()
    if rj_ax is not None:
        for prefix, label, marker in (("fdtd", "FDTD signal-on", "^-"), ("single_tone", "single-tone HB", "o-"), ("new_hb", "new-HB pump diagnostic", "s-")):
            y = [row.get(f"{prefix}_r_j") for row in records]
            pairs = [(xx, yy) for xx, yy in zip(x, y) if yy is not None]
            if pairs:
                rj_ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker, label=label)
        rj_ax.set_ylabel("junction utilization")
        rj_ax.legend()
    for prefix, label, marker in (("single_tone", "single-tone HB", "o-"), ("new_hb", "new HB", "s-")):
        y = [
            row.get(f"{prefix}_runtime_s")
            if row["single_tone_status"] == "MEASURED"
            and row["new_hb_status"] == "MEASURED"
            and row["single_tone_runtime_s"] is not None
            and row["new_hb_runtime_s"] is not None
            else None
            for row in records
        ]
        pairs = [(xx, yy) for xx, yy in zip(x, y) if yy is not None]
        if pairs:
            runtime_ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker, label=label)
    runtime_ax.set_ylabel("wall time (s)")
    runtime_ax.set_xlabel("achieved pump current (microampere)")
    runtime_ax.legend()
    fig.suptitle(f"Phase C measured columns: {device}, S={SIDEBANDS}")
    fig.tight_layout()
    figure_path = output / f"{device}.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    counts = {
        "fdtd_gain": sum(row["fdtd_gain_vs_off_db"] is not None for row in records),
        "single_tone_gain": sum(row["single_tone_status"] == "MEASURED" for row in records),
        "new_hb_gain": sum(row["new_hb_status"] == "MEASURED" for row in records),
        "runtime_both": sum(row["single_tone_runtime_s"] is not None and row["new_hb_runtime_s"] is not None for row in records),
    }
    if device != "guarcello":
        counts.update({
            "fdtd_r_j": sum(row["fdtd_r_j"] is not None for row in records),
            "single_tone_r_j": sum(row["single_tone_r_j"] is not None for row in records),
            "new_hb_r_j": sum(row["new_hb_r_j"] is not None for row in records),
        })
    paired = [
        row for row in records
        if row["single_tone_status"] == "MEASURED"
        and row["new_hb_status"] == "MEASURED"
        and row["single_tone_runtime_s"] is not None
        and row["new_hb_runtime_s"] is not None
    ]
    timing = {
        "single_tone": _stats([float(row["single_tone_runtime_s"]) for row in paired]),
        "new_hb": _stats([float(row["new_hb_runtime_s"]) for row in paired]),
    }
    agreement = {
        "gain_single_minus_fdtd_db": _difference_stats(
            records, "fdtd_gain_vs_off_db", "single_tone_gain_vs_off_db"
        ),
        "gain_new_minus_fdtd_db": _difference_stats(
            records, "fdtd_gain_vs_off_db", "new_hb_gain_vs_off_db"
        ),
    }
    if device != "guarcello":
        agreement.update({
            "r_j_single_minus_fdtd_signal_on": _difference_stats(
                records, "fdtd_r_j", "single_tone_r_j"
            ),
            "r_j_new_minus_fdtd_signal_on": _difference_stats(
                records, "fdtd_r_j", "new_hb_r_j"
            ),
        })
    distinct_sources = all(
        len({row["fdtd_source_path"], row["single_tone_source_path"], row["new_hb_source_path"]} - {None})
        == sum(value is not None for value in (row["fdtd_source_path"], row["single_tone_source_path"], row["new_hb_source_path"]))
        for row in records
    )
    return {
        "device": device, "points": len(records), "counts": counts,
        "paired_runtime": timing,
        "agreement": agreement,
        "arm_source_paths_distinct": distinct_sources,
        "csv": str(csv_path), "figure": str(figure_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", default=",".join(DEVICES))
    args = parser.parse_args()
    summary = [reduce_device(device.strip(), args.output) for device in args.devices.split(",") if device.strip()]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
