from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"
BASE_REPORT = ROOT / "outputs" / "exp14_fqjtwpa_diss_odd10" / "gain" / "gain_report.json"
JC_REF = ROOT / "outputs" / "exp14_jc_refs" / "jc_fqjtwpa_diss_curve_21pt.csv"


def find_existing_baseline_report() -> Path:
    candidates = [
        BASE_REPORT,
        ROOT / "outputs" / "exp14_diss_loss_study" / "_diss_A" / "gain_report.json",
        ROOT.parent / "outputs" / "_diss_A" / "gain_report.json",
        ROOT.parent / "outputs" / "exp14_diss_loss_study" / "_diss_A" / "gain_report.json",
        Path(r"D:\Projects\outputs") / "_diss_A" / "gain_report.json",
        Path(r"D:\Projects\outputs") / "exp14_diss_loss_study" / "_diss_A" / "gain_report.json",
    ]

    for c in candidates:
        if c.exists():
            return c

    roots = [
        ROOT / "outputs",
        ROOT.parent / "outputs",
        Path(r"D:\Projects\outputs"),
        Path(r"D:\Projects\Thesis"),
    ]

    hits: list[Path] = []
    for r in roots:
        if r.exists():
            hits.extend(r.rglob("gain_report.json"))

    scored = []
    for h in hits:
        s = str(h).lower()
        score = 0
        if "diss" in s:
            score += 10
        if "_diss_a" in s or "lossyp_lossyg" in s or "lossy" in s:
            score += 10
        if "fqjtwpa" in s:
            score += 5
        scored.append((score, h))

    scored.sort(key=lambda x: (-x[0], str(x[1])))
    if scored and scored[0][0] > 0:
        print("AUTO_BASELINE_REPORT", scored[0][1])
        return scored[0][1]

    msg = ["Could not find dissipative baseline gain_report.json. Candidates searched:"]
    msg.extend(str(c) for c in candidates)
    msg.append("All gain_report hits:")
    msg.extend(str(h) for _, h in scored[:30])
    raise SystemExit("\n".join(msg))
OUT = ROOT / "outputs" / "exp14_diss_linearization_probe_odd10"

MODELS = [
    "current_complex_c",
    "conjugate_complex_c",
    "complex_c_sign_omega",
    "conductance_signed_omega",
    "conductance_abs_omega",
    "conductance_abs_omega_opposite",
    "real_capacitance",
]


def resolve_path(x: str | None) -> Path:
    if not x:
        raise ValueError("missing path in baseline metadata")
    p = Path(x)
    if p.is_absolute():
        return p
    return ROOT / p


def load_curve(path: Path) -> dict[float, float]:
    d: dict[float, float] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[round(float(r["signal_ghz"]), 6)] = float(r["gain_db"])
    return d


def metrics(py_csv: Path, jc_csv: Path) -> dict[str, float]:
    py = load_curve(py_csv)
    jc = load_curve(jc_csv)
    common = sorted(set(py) & set(jc))
    if not common:
        raise RuntimeError(f"no common frequencies between {py_csv} and {jc_csv}")
    diff = np.array([py[f] - jc[f] for f in common], dtype=float)
    return {
        "n": float(len(common)),
        "py_peak": float(max(py.values())),
        "jc_peak": float(max(jc.values())),
        "py_peak_freq": float(max(py, key=py.get)),
        "jc_peak_freq": float(max(jc, key=jc.get)),
        "rms_db": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs_db": float(np.mean(np.abs(diff))),
        "max_abs_db": float(np.max(np.abs(diff))),
        "mean_signed_db": float(np.mean(diff)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    baseline_report = find_existing_baseline_report()

    if not JC_REF.exists():
        # Try common non-repo output roots.
        alt_refs = [
            ROOT.parent / "outputs" / "exp14_jc_refs" / "jc_fqjtwpa_diss_curve_21pt.csv",
            Path(r"D:\Projects\outputs") / "exp14_jc_refs" / "jc_fqjtwpa_diss_curve_21pt.csv",
        ]
        for alt in alt_refs:
            if alt.exists():
                globals()["JC_REF"] = alt
                break
        else:
            raise SystemExit(f"Missing JC ref: {JC_REF}")

    print(f"USING_BASELINE_REPORT {baseline_report}")
    print(f"USING_JC_REF {JC_REF}")

    report = json.loads(baseline_report.read_text(encoding="utf-8"))
    meta = report["metadata"]

    ipm_dir = resolve_path(meta["ipm_dir"])
    pump_dir = resolve_path(meta["pump_dir"])

    base_cmd = [
        sys.executable,
        str(EXP09),
        "--ipm-dir", str(ipm_dir),
        "--pump-dir", str(pump_dir),
        "--sweep",
        "--signal-start-ghz", "4.0",
        "--signal-stop-ghz", "8.0",
        "--points", "21",
        "--sidebands", str(meta.get("sidebands", 10)),
        "--signal-m", str(meta.get("signal_m", 0)),
        "--idler-m", str(meta.get("idler_m", -2)),
        "--source-port", str(meta.get("source_port", 1)),
        "--out-port", str(meta.get("out_port", 2)),
        "--source-current-a", str(meta.get("source_current_a", 1.0)),
        "--z0-ohm", str(meta.get("z0_ohm", 50.0)),
        "--gamma-nt", str(meta.get("gamma_nt", 128)),
    ]

    rows = []
    for model in MODELS:
        outdir = OUT / model
        cmd = base_cmd + [
            "--outdir", str(outdir),
            "--loss-linearization-model", model,
        ]

        print("\n" + "=" * 100)
        print("RUN", model)
        print(" ".join(cmd))
        print("=" * 100)

        proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
        if proc.returncode != 0:
            rows.append({
                "model": model,
                "status": f"FAIL_RETURN_{proc.returncode}",
                "n": "",
                "py_peak": "",
                "jc_peak": "",
                "py_peak_freq": "",
                "jc_peak_freq": "",
                "rms_db": "",
                "mean_abs_db": "",
                "max_abs_db": "",
                "mean_signed_db": "",
            })
            continue

        m = metrics(outdir / "gain_sweep.csv", JC_REF)
        rows.append({"model": model, "status": "OK", **m})
        print(
            f"RESULT {model}: "
            f"py_peak={m['py_peak']:.6f} jc_peak={m['jc_peak']:.6f} "
            f"rms={m['rms_db']:.6f} max_abs={m['max_abs_db']:.6f} "
            f"mean_signed={m['mean_signed_db']:.6f}"
        )

    summary = OUT / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model", "status", "n", "py_peak", "jc_peak",
            "py_peak_freq", "jc_peak_freq",
            "rms_db", "mean_abs_db", "max_abs_db", "mean_signed_db",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n" + "=" * 100)
    print(f"WROTE {summary}")
    print("=" * 100)
    for r in sorted([r for r in rows if r["status"] == "OK"], key=lambda x: float(x["rms_db"])):
        print(
            f"{r['model']:32s} "
            f"rms={float(r['rms_db']):.6f} "
            f"max_abs={float(r['max_abs_db']):.6f} "
            f"py_peak={float(r['py_peak']):.6f} "
            f"mean_signed={float(r['mean_signed_db']):+.6f}"
        )


if __name__ == "__main__":
    main()
