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
OUT = ROOT / "outputs" / "exp14_diss_pump_state_scale_fine_probe"

SCALES = [1.0005, 1.0010, 1.0015, 1.0020, 1.0025, 1.0030, 1.0035]


def load_curve(path: Path) -> dict[float, float]:
    d = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[round(float(r["signal_ghz"]), 6)] = float(r["gain_db"])
    return d


def metrics(py_csv: Path) -> dict[str, float]:
    py = load_curve(py_csv)
    jc = load_curve(JC_REF)
    common = sorted(set(py) & set(jc))
    diff = np.array([py[f] - jc[f] for f in common], dtype=float)
    return {
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

    meta = json.loads(BASE_REPORT.read_text(encoding="utf-8"))["metadata"]

    rows = []
    for scale in SCALES:
        outdir = OUT / f"scale_{scale:.3f}".replace(".", "p")
        cmd = [
            sys.executable, str(EXP09),
            "--ipm-dir", str(ROOT / meta["ipm_dir"]),
            "--pump-dir", str(ROOT / meta["pump_dir"]),
            "--sweep",
            "--signal-start-ghz", "4.0",
            "--signal-stop-ghz", "8.0",
            "--points", "21",
            "--sidebands", str(meta["sidebands"]),
            "--signal-m", str(meta["signal_m"]),
            "--idler-m", str(meta["idler_m"]),
            "--source-port", str(meta["source_port"]),
            "--out-port", str(meta["out_port"]),
            "--source-current-a", str(meta["source_current_a"]),
            "--z0-ohm", str(meta["z0_ohm"]),
            "--gamma-nt", "96",
            "--loss-linearization-model", "conductance_abs_omega",
            "--pump-state-scale", str(scale),
            "--outdir", str(outdir),
        ]

        print("\nRUN scale", scale)
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
        if proc.returncode != 0:
            rows.append({"scale": scale, "status": f"FAIL_{proc.returncode}"})
            continue

        m = metrics(outdir / "gain_sweep.csv")
        rows.append({"scale": scale, "status": "OK", **m})
        print(
            f"scale={scale:.6f} rms={m['rms_db']:.6f} "
            f"max_abs={m['max_abs_db']:.6f} "
            f"py_peak={m['py_peak']:.6f} "
            f"mean_signed={m['mean_signed_db']:+.6f}"
        )

    summary = OUT / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scale", "status", "py_peak", "jc_peak", "py_peak_freq", "jc_peak_freq",
            "rms_db", "mean_abs_db", "max_abs_db", "mean_signed_db",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\nSUMMARY")
    for r in sorted([r for r in rows if r["status"] == "OK"], key=lambda x: x["rms_db"]):
        print(
            f"scale={r['scale']:.6f} "
            f"rms={r['rms_db']:.6f} "
            f"max_abs={r['max_abs_db']:.6f} "
            f"py_peak={r['py_peak']:.6f} "
            f"mean_signed={r['mean_signed_db']:+.6f}"
        )


if __name__ == "__main__":
    main()
