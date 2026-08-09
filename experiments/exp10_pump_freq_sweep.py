"""Pump-frequency gain sweep at a fixed pump power (line plot).

For a fixed external pump power, sweep the pump frequency and read the gain at a
fixed signal tone. Solves exp08 (pump) + exp09 (gain) per pump frequency,
warm-starting each pump solve from the previous frequency (adjacent pump
frequencies share a similar harmonic-balance solution), so the sweep is cheap.

Runs one curve per ``--jc-scale`` so the JC positive-phasor 2x convention can be
overlaid against the bare physical current and compared to a measurement.

Example (matches the -22 dBm measurement point, Ip = 8.93 uA peak):
    python experiments/exp10_pump_freq_sweep.py --power-dbm -22 \
        --pump-freq-min-ghz 5 --pump-freq-max-ghz 10 --n-freq 26 \
        --signal-ghz 6 --jc-scales 1.0,2.0
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP08 = "experiments/exp08_full_ipm_pump_solve.py"
EXP09 = "experiments/exp09_full_ipm_gain_from_pump.py"


def dbm_to_peak_current_a(power_dbm: float, *, attenuation_db: float, z0_ohm: float) -> float:
    source_dbm = float(power_dbm) - float(attenuation_db)
    power_w = 1.0e-3 * 10.0 ** (source_dbm / 10.0)
    return math.sqrt(2.0 * power_w / float(z0_ohm))


def read_json(path: Path) -> dict[str, Any] | None:
    import json

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def signal_ghz_for(pump_freq_ghz: float, args: argparse.Namespace) -> float:
    """Readout signal frequency: ws = wp - detuning (default 100 MHz) per point,
    unless an explicit fixed --signal-ghz overrides it."""
    if getattr(args, "signal_ghz", None) is not None:
        return float(args.signal_ghz)
    return float(pump_freq_ghz) - float(args.signal_detuning_mhz) / 1000.0


def run(cmd: list[str], log: Path, timeout_s: float) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        try:
            p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                               text=True, timeout=timeout_s, check=False)
            return int(p.returncode)
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout_s}s\n")
            return 124


def solve_point(
    freq_ghz: float,
    inject_current_a: float,
    out_dir: Path,
    args: argparse.Namespace,
    promote_from: Path | None,
) -> dict[str, Any]:
    pump_dir = out_dir / "pump"
    gain_dir = out_dir / "gain"
    pump_cmd = [
        sys.executable, EXP08,
        "--ipm-dir", str(args.ipm_dir),
        "--outdir", str(pump_dir),
        "--pump-port", str(args.pump_port),
        "--pump-freq-ghz", f"{freq_ghz:.12g}",
        "--pump-current-a", f"{inject_current_a:.17g}",
        "--pump-mode-policy", "positive_odd_jc",
        "--pump-mode-count", str(args.pump_mode_count),
        "--nt", str(args.nt),
        "--quiet",
    ]
    if promote_from is not None:
        pump_cmd += ["--initial-guess", "zero", "--promote-from-pump-dir", str(promote_from)]
    else:
        pump_cmd += ["--initial-guess", "linear_phasor", "--continuation-mode", "fixed",
                     "--continuation-steps", str(args.continuation_steps)]

    pump_rc = run(pump_cmd, out_dir / "pump.log", args.pump_timeout_s)
    pump_report = read_json(pump_dir / "pump_report.json")
    pump_ok = pump_rc == 0 and pump_report is not None and pump_report.get("final_status") == "VALID_CONVERGED"

    gain_db = math.nan
    if pump_ok:
        gain_cmd = [
            sys.executable, EXP09,
            "--ipm-dir", str(args.ipm_dir),
            "--pump-dir", str(pump_dir),
            "--outdir", str(gain_dir),
            "--source-port", str(args.source_port),
            "--out-port", str(args.out_port),
            "--signal-ghz", f"{signal_ghz_for(freq_ghz, args):.12g}",
            "--sidebands", str(args.sidebands),
            "--gamma-nt", str(args.gamma_nt),
            "--fallback-pump-freq-ghz", f"{freq_ghz:.12g}",
        ]
        gain_rc = run(gain_cmd, out_dir / "gain.log", args.gain_timeout_s)
        gain_report = read_json(gain_dir / "gain_report.json")
        if gain_rc == 0 and gain_report is not None:
            results = gain_report.get("results", [])
            vals = [r.get("gain_db") for r in results if r.get("gain_db") is not None]
            if vals:
                gain_db = float(vals[0])

    return {
        "pump_freq_ghz": freq_ghz,
        "pump_ok": pump_ok,
        "gain_db": gain_db,
        "pump_dir": pump_dir if pump_ok else None,
    }


def sweep_scale(scale: float, args: argparse.Namespace, freqs: np.ndarray) -> list[dict[str, Any]]:
    physical = dbm_to_peak_current_a(args.power_dbm, attenuation_db=args.attenuation_db, z0_ohm=args.z0_ohm)
    inject = physical * scale
    print(f"=== scale {scale}: inject {inject*1e6:.3f} uA "
          f"({inject/args.ic_a:.3f} Ic) at {args.power_dbm} dBm ===", flush=True)
    scale_dir = args.outdir / f"scale_{scale:g}"
    rows: list[dict[str, Any]] = []
    prev: Path | None = None
    for k, f in enumerate(freqs):
        t0 = time.perf_counter()
        r = solve_point(float(f), inject, scale_dir / f"fp_{f:.4g}".replace(".", "p"), args, prev)
        r["scale"] = scale
        r["inject_current_a"] = inject
        r["inject_over_ic"] = inject / args.ic_a
        r["elapsed_s"] = time.perf_counter() - t0
        rows.append(r)
        prev = r["pump_dir"] if r["pump_ok"] else None  # only chain off a converged solve
        print(f"  [{k+1}/{len(freqs)}] fp={f:.3g} GHz pump_ok={r['pump_ok']} "
              f"gain={r['gain_db']:.3f} dB ({r['elapsed_s']:.1f}s)", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "exp10_pump_freq_sweep_m22dbm")
    p.add_argument("--ipm-dir", type=Path, default=ROOT / "outputs" / "ipm_python_design")
    p.add_argument("--power-dbm", type=float, default=-22.0)
    p.add_argument("--attenuation-db", type=float, default=35.0)
    p.add_argument("--z0-ohm", type=float, default=50.0)
    p.add_argument("--ic-a", type=float, default=4.1658984303797465e-6)
    p.add_argument("--pump-freq-min-ghz", type=float, default=5.0)
    p.add_argument("--pump-freq-max-ghz", type=float, default=10.0)
    p.add_argument("--n-freq", type=int, default=26)
    # Default: signal tracks the pump at ws = wp - 100 MHz per swept frequency.
    p.add_argument("--signal-ghz", type=float, default=None,
                   help="Fixed absolute signal frequency (GHz). If omitted, the "
                   "signal tracks the pump at ws = wp - --signal-detuning-mhz.")
    p.add_argument("--signal-detuning-mhz", type=float, default=100.0,
                   help="Signal detuning below the pump when --signal-ghz is not "
                   "set: ws = wp - detuning (default 100 MHz).")
    p.add_argument("--jc-scales", type=str, default="1.0,2.0")
    p.add_argument("--pump-port", type=int, default=4)
    p.add_argument("--source-port", type=int, default=1)
    p.add_argument("--out-port", type=int, default=2)
    p.add_argument("--pump-mode-count", type=int, default=10)
    p.add_argument("--nt", type=int, default=40)
    p.add_argument("--sidebands", type=int, default=10)
    p.add_argument("--gamma-nt", type=int, default=96)
    p.add_argument("--continuation-steps", type=int, default=20)
    p.add_argument("--pump-timeout-s", type=float, default=240.0)
    p.add_argument("--gain-timeout-s", type=float, default=120.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    freqs = np.linspace(args.pump_freq_min_ghz, args.pump_freq_max_ghz, args.n_freq)
    scales = [float(s) for s in args.jc_scales.split(",")]

    all_rows: list[dict[str, Any]] = []
    series: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for scale in scales:
        rows = sweep_scale(scale, args, freqs)
        all_rows.extend(rows)
        series[scale] = (
            np.array([r["pump_freq_ghz"] for r in rows]),
            np.array([r["gain_db"] for r in rows]),
        )

    # CSV
    csv_path = args.outdir / "pump_freq_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scale", "inject_current_a", "inject_over_ic", "pump_freq_ghz",
            "pump_ok", "gain_db", "elapsed_s"], extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {csv_path}", flush=True)

    # Line plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for scale in scales:
        fx, gy = series[scale]
        inj_uA = dbm_to_peak_current_a(args.power_dbm, attenuation_db=args.attenuation_db, z0_ohm=args.z0_ohm) * scale * 1e6
        ax.plot(fx, gy, "-o", markersize=4,
                label=f"inject {inj_uA:.2f} uA = {inj_uA*1e-6/args.ic_a:.2f} Ic (scale {scale:g})")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("pump frequency (GHz)")
    _sig_lbl = (f"{args.signal_ghz:g} GHz" if args.signal_ghz is not None
               else f"wp - {args.signal_detuning_mhz:g} MHz")
    ax.set_ylabel(f"gain S21 at signal {_sig_lbl} (dB)")
    ax.set_title(f"IPM JTWPA gain vs pump frequency @ {args.power_dbm:g} dBm "
                 f"(Ip={dbm_to_peak_current_a(args.power_dbm, attenuation_db=args.attenuation_db, z0_ohm=args.z0_ohm)*1e6:.2f} uA)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = args.outdir / "pump_freq_sweep.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"wrote {png}", flush=True)


if __name__ == "__main__":
    main()
