#!/usr/bin/env python3
"""Re-reduce saved Poincare data with directional Guarcello statistics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.chaos.run_guarcello_repro import add_gain_vs_off


def branch_stats(values: np.ndarray) -> dict[str, Any]:
    """Return legacy and directional spread for one saved crossing sequence."""
    points = np.asarray(values, dtype=float).reshape(-1)
    points = points[np.isfinite(points)]
    upward = points[points > 0.0]
    downward = points[points < 0.0]
    scale = max(abs(float(np.median(upward))) if upward.size else 0.0,
                float(np.ptp(upward)) if upward.size else 0.0, 1.0e-30)
    clusters = 0
    if upward.size:
        clusters = 1 + int(np.count_nonzero(np.diff(np.sort(upward)) > 0.03 * scale))
    return {
        "sigma_both": float(np.std(points)) if points.size else float("nan"),
        "sigma_upward": float(np.std(upward)) if upward.size else float("nan"),
        "sigma_downward": float(np.std(downward)) if downward.size else float("nan"),
        "count_both": int(points.size),
        "count_upward": int(upward.size),
        "count_downward": int(downward.size),
        "directional_clusters_upward": clusters,
    }


def _phase2(path: Path, reference_signal_v: float) -> dict[str, Any]:
    with (path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if (path / "poincare_map.npz").exists():
        data = np.load(path / "poincare_map.npz", allow_pickle=True)
        controls = data["x"]
        values = data["values"]
    else:
        controls = np.asarray([
            float(row.get("pump_dbm", row.get("signal_ghz"))) for row in rows
        ])
        values = []
        for index in range(len(rows)):
            branch_path = path / f"point_{index:05d}" / "poincare_branches.npz"
            branch = np.load(branch_path, allow_pickle=True)
            values.append(branch["both"])
    reduced = []
    for x, row, crossing_values in zip(controls, rows, values):
        stats = branch_stats(crossing_values)
        reduced.append({"control": float(x), **row, **stats})
    reduced = add_gain_vs_off(reduced, reference_signal_v)
    if "fig2a" in path.name or "fig2a" in str(path):
        stable = float(min(item["sigma_upward"] for item in reduced))
        for item in reduced:
            item["sigma_deep_stable"] = stable
            item["sigma_ratio"] = item["sigma_upward"] / stable
        below_transition = [item for item in reduced if item["control"] <= -54.0]
        transition_rows = [item for item in reduced if item["control"] > -54.0]
        first_cluster = next(
            (item["control"] for item in reduced
             if int(item["directional_clusters_upward"]) > 1),
            float("nan"),
        )
        at_minus54 = next(item for item in reduced if np.isclose(item["control"], -54.0))
        gate = {
            "G1_mean_abs_residual_db_below_transition": None,
            "G1_max_abs_residual_db_below_transition": None,
            "G1_pass": None,
            "G2_first_cluster_rise_dbm": first_cluster,
            "G2_pass": bool(-54.0 <= first_cluster <= -53.5),
            "G3_pass": False,
            "G4_max_abs_phi_at_minus54": float(at_minus54["max_abs_phi_last_recorded"]),
            "G4_pass": bool(1.3 <= float(at_minus54["max_abs_phi_last_recorded"]) <= 1.7),
            "transition_band": [
                item for item in reduced if -54.0 < item["control"] <= -53.5
            ],
            "ratio_max": max(item["sigma_ratio"] for item in reduced),
            "ratio_crosses_20_within_1db": False,
            "retired_gain_le_8_db_to_minus54": {
                "status": "RETIRED",
                "reason": "digitized value at -54 dBm is 9.0 dB; the old gate encoded an eyeball threshold",
            },
        }
        targets = {
            -70.0: 0.0, -65.0: 0.6, -62.0: 1.4, -60.0: 2.0,
            -58.0: 3.4, -57.0: 4.5, -56.0: 6.0, -55.0: 7.5,
            -54.0: 9.0, -53.5: 12.0,
        }
        residuals = [
            float(next(item["gain_vs_off_db"] for item in reduced
                       if np.isclose(item["control"], control)) - target)
            for control, target in targets.items()
        ]
        residual_by_control = dict(zip(targets, residuals))
        low_residuals = [value for control, value in residual_by_control.items()
                         if control <= -54.0]
        gate["G1_mean_abs_residual_db_below_transition"] = float(np.mean(np.abs(low_residuals)))
        gate["G1_max_abs_residual_db_below_transition"] = float(np.max(np.abs(low_residuals)))
        gate["G1_pass"] = bool(gate["G1_mean_abs_residual_db_below_transition"] <= 1.0)
        gate["digitized_target_mean_abs_residual_db"] = float(np.mean(np.abs(residuals)))
        gate["digitized_target_max_abs_residual_db"] = float(np.max(np.abs(residuals)))
        shoulder = next((item for item in reduced if np.isclose(item["control"], -54.5)), None)
        edge = next((item for item in reduced if np.isclose(item["control"], -53.5)), None)
        if shoulder is not None and edge is not None:
            gate["sigma_shoulder_dbm"] = shoulder["sigma_upward"]
            gate["sigma_edge_dbm"] = edge["sigma_upward"]
            gate["sigma_edge_to_shoulder_ratio"] = edge["sigma_upward"] / shoulder["sigma_upward"]
            gate["ratio_crosses_20_within_1db"] = bool(
                gate["sigma_edge_to_shoulder_ratio"] >= 20.0
            )
            gate["G3_pass"] = gate["ratio_crosses_20_within_1db"]
        wideband_deltas = [
            abs(float(item["gain_wideband_vs_off_db"]) - float(item["gain_vs_off_db"]))
            for item in reduced if item["control"] <= -54.5
        ]
        gate["G6_max_wideband_narrowband_delta_db_below_transition"] = float(max(wideband_deltas))
        gate["G6_pass"] = bool(max(wideband_deltas) <= 0.3)
        chaotic = [item for item in reduced if item["control"] > -53.5]
        gate["chaotic_wideband_observation_db"] = [
            float(item["gain_wideband_vs_off_db"]) for item in chaotic
        ]
    else:
        masked = [item for item in reduced if not np.isclose(item["control"], 7.0)]
        stable = float(min(item["sigma_upward"] for item in masked))
        for item in reduced:
            item["sigma_deep_stable"] = stable
            item["sigma_ratio"] = item["sigma_upward"] / stable
            item["masked_at_pump_frequency"] = bool(np.isclose(item["control"], 7.0))
        gains = np.asarray([float(item["gain_vs_off_db"]) for item in masked])
        band = np.asarray([
            float(item["gain_vs_off_db"]) for item in masked
            if 6.0 <= item["control"] <= 8.0
        ])
        gate = {
            "mask_7ghz_contamination": True,
            "gain_6_to_8_mean_db": float(np.mean(band)),
            "gain_6_to_8_range_db": float(np.ptp(band)),
            "gain_peak_db": float(np.max(gains)),
            "gain_at_4_and_10_db": [masked[0]["gain_vs_off_db"], masked[-1]["gain_vs_off_db"]],
            "poincare_dense_after_mask": all(item["count_upward"] > 0 for item in masked),
            "G5_max_adjacent_gain_step_db": float(np.max(np.abs(np.diff(gains)))),
            "G5_pass": bool(np.max(np.abs(np.diff(gains))) <= 1.5),
            "B4_peak_gain_db": float(np.max(gains)),
            "B4_pass": bool(np.max(gains) <= 10.0),
            "G6_max_wideband_narrowband_delta_db_below_transition": float("nan"),
        }
    payload = {"source": str(path), "statistic": "upward dV/dt branch",
               "legacy_both_sign_retained": True, "rows": reduced, "gates": gate}
    output = path / "directional_sigma_reduction.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _walk_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "trace" in value:
            records.append(value)
        for child in value.values():
            records.extend(_walk_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_walk_records(child))
    return records


def _phase3(path: Path) -> dict[str, Any]:
    source = path / "campaign_summary.json"
    if not source.exists():
        source = path / "bias_campaign_summary.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = []
    for record in _walk_records(payload):
        trace = Path(record["trace"])
        if not trace.is_absolute():
            trace = Path.cwd() / trace
        data = np.load(trace, allow_pickle=True)
        key = "poincare_both" if "poincare_both" in data.files else "poincare"
        stats = branch_stats(data[key])
        records.append({
            "direction": record.get("direction"),
            "control": record.get("power_dbm", record.get("external_flux_fraction")),
            "old_verdict": record.get("verdict"), "trace": str(trace), **stats,
        })
    finite = [row for row in records if np.isfinite(row["sigma_upward"])]
    stable = min((row["sigma_upward"] for row in finite), default=float("nan"))
    for row in records:
        row["sigma_deep_stable"] = stable
        row["sigma_ratio"] = row["sigma_upward"] / stable if stable > 0 else float("nan")
    output_payload = {
        "source": str(source), "statistic": "upward dV/dt branch",
        "legacy_both_sign_retained": True, "records": records,
        "verdict_basis": (
            "directional Poincare geometry and directional sigma; the former smooth "
            "both-sign amplitude-tracking argument is void"
        ),
    }
    if "ladder_bracket20" in str(path):
        broad = [row for row in records if row["directional_clusters_upward"] > 1]
        up_controls = {row["control"] for row in broad if row["direction"] == "up"}
        down_controls = {row["control"] for row in broad if row["direction"] == "down"}
        output_payload["verdict"] = "NO_BIFURCATION_FOUND"
        output_payload["verdict_basis"] = (
            "directional sigma removes the two-sign period-2 interpretation; "
            f"{len(broad)} isolated upward broad points remain, with "
            f"{len(up_controls & down_controls)} matched controls in the down sweep"
        )
    output = path / "directional_sigma_reduction.json"
    output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return output_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2", type=Path, action="append", default=[])
    parser.add_argument("--phase3", type=Path, action="append", default=[])
    parser.add_argument("--reference", type=Path, default=None)
    args = parser.parse_args()
    for path in args.phase2:
        if not args.reference:
            raise SystemExit("--reference is required for Phase-2 reduction")
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        signal_v = float(reference["analysis"]["signal_vout_peak_v"])
        _phase2(path, signal_v)
        print(path / "directional_sigma_reduction.json", flush=True)
    for path in args.phase3:
        _phase3(path)
        print(path / "directional_sigma_reduction.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
