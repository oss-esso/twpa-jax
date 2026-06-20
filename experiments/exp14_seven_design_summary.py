# experiments/exp14_seven_design_summary.py
"""Build the 7-design JC parity summary table for the pump-mode-policy work.

Reads, for each JC doc design, the exp08 pump report and exp09 gain sweep
produced under outputs/exp14_*/, optionally compares to a JC reference 21pt
curve, and emits:

    outputs/exp14_seven_design_summary/summary.csv
    outputs/exp14_seven_design_summary/summary.json

Columns:
    case, status, jc_max, py_max, jc_peak, py_peak, max_abs_err_db,
    mean_abs_err_db, rms_err_db, pump_runtime_s, gain_runtime_s, mode_policy,
    pump_modes

Statuses are honest: SOLVED_MATCHED_JC (parity vs JC reference), SOLVED_NO_JC_REF
(pump+gain valid but no working JC reference curve to compare), UNSUPPORTED
(design family the scalar real-time-domain pump-mode policy cannot represent),
FINITE_NONCONVERGED (pump finished but did not reach tolerance), FAIL.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

OUTDIR = Path("outputs/exp14_seven_design_summary")
PARITY_RMS_THRESHOLD_DB = 0.3


@dataclass
class CaseSpec:
    case: str
    base: str | None  # outputs/exp14_<...> run dir, or None for unsupported
    mode_policy: str
    jc_ref_csv: str | None
    unsupported_reason: str | None = None


CASES: list[CaseSpec] = [
    CaseSpec(
        "jc_jpa",
        "outputs/exp14_jpa_odd10_scale2",
        "positive_odd_jc",
        "outputs/exp13_compare/jc_jpa_curve.csv",
    ),
    CaseSpec(
        "jc_jtwpa",
        "outputs/exp14_jtwpa_odd10_scale2",
        "positive_odd_jc",
        "outputs/exp13_jtwpa_fast_scale2/jc_jtwpa_curve_21pt.csv",
    ),
    CaseSpec(
        "jc_fqjtwpa",
        "outputs/exp14_fqjtwpa_odd10_scale2",
        "positive_odd_jc",
        "outputs/exp14_jc_refs/jc_fqjtwpa_curve_21pt.csv",
    ),
    CaseSpec(
        "jc_fxjpa",
        "outputs/exp14_fxjpa_dense8_scale2",
        "dense_real",
        "outputs/exp14_jc_refs/jc_fxjpa_curve_21pt.csv",
    ),
    CaseSpec(
        "jc_fxjtwpa",
        "outputs/exp14_fxjtwpa_dense4_scale2",
        "dense_real",
        "outputs/exp14_jc_refs/jc_fxjtwpa_curve_21pt.csv",
    ),
    CaseSpec(
        "jc_dpjpa",
        "outputs/exp14_dpjpa_multitone",
        "multi_tone_lattice",
        "outputs/exp14_jc_refs/jc_dpjpa_curve_21pt.csv",
    ),
    CaseSpec(
        "jc_fqjtwpa_diss",
        "outputs/exp14_fqjtwpa_diss_odd10",
        "positive_odd_jc",
        "outputs/exp14_jc_refs/jc_fqjtwpa_diss_curve_21pt.csv",
    ),
]


def load_curve(path: str | Path) -> dict[float, float]:
    out: dict[float, float] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[round(float(r["signal_ghz"]), 3)] = float(r["gain_db"])
    return out


def peak_of(curve: dict[float, float]) -> tuple[float, float]:
    f = max(curve, key=lambda k: curve[k])
    return curve[f], f


def summarize(spec: CaseSpec) -> dict[str, object]:
    row: dict[str, object] = {
        "case": spec.case,
        "status": "",
        "jc_max": "",
        "py_max": "",
        "jc_peak": "",
        "py_peak": "",
        "max_abs_err_db": "",
        "mean_abs_err_db": "",
        "rms_err_db": "",
        "pump_runtime_s": "",
        "gain_runtime_s": "",
        "mode_policy": spec.mode_policy,
        "pump_modes": "",
    }

    if spec.base is None:
        row["status"] = "UNSUPPORTED"
        row["pump_modes"] = spec.unsupported_reason or ""
        return row

    base = Path(spec.base)
    pump_report = base / "pump" / "pump_report.json"
    gain_csv = base / "gain" / "gain_sweep.csv"
    gain_report = base / "gain" / "gain_report.json"

    pump_converged = False
    pump_coeff_rel = None
    if pump_report.exists():
        rep = json.loads(pump_report.read_text(encoding="utf-8"))
        md = rep.get("metadata", {})
        row["mode_policy"] = md.get("pump_mode_policy", spec.mode_policy)
        row["pump_modes"] = ";".join(str(m) for m in md.get("pump_modes", []))
        reports = rep.get("reports", [])
        if reports:
            row["pump_runtime_s"] = round(float(reports[-1].get("runtime_s", 0.0)), 3)
            pump_coeff_rel = float(reports[-1].get("coeff_rel", math.inf))
        pump_converged = rep.get("final_status") == "VALID_CONVERGED" or (
            pump_coeff_rel is not None and pump_coeff_rel < 1e-6
        )

    if not pump_converged:
        row["status"] = "FINITE_NONCONVERGED"
        if pump_coeff_rel is not None:
            row["rms_err_db"] = f"pump_coeff_rel={pump_coeff_rel:.3e}"
        return row

    if not gain_csv.exists():
        row["status"] = "FAIL"
        row["rms_err_db"] = "missing gain_sweep.csv"
        return row

    py = load_curve(gain_csv)
    py_max, py_peak = peak_of(py)
    row["py_max"] = round(py_max, 4)
    row["py_peak"] = py_peak

    if gain_report.exists():
        grep = json.loads(gain_report.read_text(encoding="utf-8"))
        row["gain_runtime_s"] = round(
            float(grep.get("metadata", {}).get("total_runtime_s", 0.0)), 3
        )

    if spec.jc_ref_csv and Path(spec.jc_ref_csv).exists():
        jc = load_curve(spec.jc_ref_csv)
        jc_max, jc_peak = peak_of(jc)
        row["jc_max"] = round(jc_max, 4)
        row["jc_peak"] = jc_peak
        common = sorted(set(py) & set(jc))
        if common:
            d = np.array([py[f] - jc[f] for f in common])
            rms = float(np.sqrt(np.mean(d**2)))
            row["max_abs_err_db"] = round(float(np.max(np.abs(d))), 5)
            row["mean_abs_err_db"] = round(float(np.mean(np.abs(d))), 5)
            row["rms_err_db"] = round(rms, 5)
            row["status"] = (
                "SOLVED_MATCHED_JC"
                if rms < PARITY_RMS_THRESHOLD_DB
                else "SOLVED_JC_MISMATCH"
            )
        else:
            row["status"] = "SOLVED_NO_JC_REF"
            row["rms_err_db"] = "no overlapping frequencies"
    else:
        row["status"] = "SOLVED_NO_JC_REF"

    return row


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize(s) for s in CASES]

    cols = [
        "case",
        "status",
        "jc_max",
        "py_max",
        "jc_peak",
        "py_peak",
        "max_abs_err_db",
        "mean_abs_err_db",
        "rms_err_db",
        "pump_runtime_s",
        "gain_runtime_s",
        "mode_policy",
        "pump_modes",
    ]

    csv_path = OUTDIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    (OUTDIR / "summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    width = max(len(c) for c in cols)
    print(f"wrote {csv_path}")
    for r in rows:
        print(f"{r['case']:16s} {r['status']:20s} "
              f"py_max={r['py_max']} jc_max={r['jc_max']} rms={r['rms_err_db']}")


if __name__ == "__main__":
    main()
