"""Signal-frequency gain profile at a fixed pump (line plot).

Solves the pump once at a fixed external power and pump frequency, then sweeps
the signal frequency with exp09 to produce the S21 gain profile -- the standard
JTWPA gain curve (cf. JosephsonCircuits IPM_JTWPA.jl, which sweeps the signal at
a fixed pump). One curve per ``--jc-scale`` so the positive-phasor 2x convention
can be overlaid against the bare physical current.

Example (-22 dBm measurement point, pump 7.9 GHz, Ip = 8.93 uA peak):
    python experiments/exp10_signal_sweep_at_pump.py --power-dbm -22 \
        --pump-freq-ghz 7.9 --signal-min-ghz 5 --signal-max-ghz 10 \
        --signal-points 51 --jc-scales 1.0,2.0
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

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


def read_sweep(gain_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = gain_dir / "gain_sweep.csv"
    fx: list[float] = []
    gy: list[float] = []
    if path.exists():
        for r in csv.DictReader(path.open(encoding="utf-8")):
            try:
                fx.append(float(r["signal_ghz"]))
                gy.append(float(r["gain_db"]))
            except (KeyError, ValueError):
                continue
    order = np.argsort(fx)
    return np.array(fx)[order], np.array(gy)[order]


def sweep_scale(scale: float, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, bool]:
    physical = dbm_to_peak_current_a(args.power_dbm, attenuation_db=args.attenuation_db, z0_ohm=args.z0_ohm)
    inject = physical * scale
    scale_dir = args.outdir / f"scale_{scale:g}"
    pump_dir = scale_dir / "pump"
    gain_dir = scale_dir / "gain"
    print(f"=== scale {scale}: inject {inject*1e6:.3f} uA ({inject/args.ic_a:.3f} Ic), "
          f"pump {args.pump_freq_ghz} GHz @ {args.power_dbm} dBm ===", flush=True)

    pump_cmd = [
        sys.executable, EXP08,
        "--ipm-dir", str(args.ipm_dir),
        "--outdir", str(pump_dir),
        "--pump-port", str(args.pump_port),
        "--pump-freq-ghz", f"{args.pump_freq_ghz:.12g}",
        "--pump-current-a", f"{inject:.17g}",
        "--pump-mode-policy", "positive_odd_jc",
        "--pump-mode-count", str(args.pump_mode_count),
        "--nt", str(args.nt),
        "--initial-guess", "linear_phasor",
        "--continuation-mode", "fixed",
        "--continuation-steps", str(args.continuation_steps),
        "--quiet",
    ]
    pump_rc = run(pump_cmd, scale_dir / "pump.log", args.pump_timeout_s)
    pump_ok = pump_rc == 0 and (pump_dir / "pump_solution.npz").exists()
    if not pump_ok:
        print(f"  pump FAILED (rc={pump_rc}) -- no gain curve for scale {scale}", flush=True)
        return np.array([]), np.array([]), inject, False

    gain_cmd = [
        sys.executable, EXP09,
        "--ipm-dir", str(args.ipm_dir),
        "--pump-dir", str(pump_dir),
        "--outdir", str(gain_dir),
        "--source-port", str(args.source_port),
        "--out-port", str(args.out_port),
        "--sweep",
        "--signal-start-ghz", f"{args.signal_min_ghz:.12g}",
        "--signal-stop-ghz", f"{args.signal_max_ghz:.12g}",
        "--points", str(args.signal_points),
        "--sidebands", str(args.sidebands),
        "--gamma-nt", str(args.gamma_nt),
        "--fallback-pump-freq-ghz", f"{args.pump_freq_ghz:.12g}",
    ]
    run(gain_cmd, scale_dir / "gain.log", args.gain_timeout_s)
    fx, gy = read_sweep(gain_dir)
    if gy.size:
        print(f"  gain peak {np.nanmax(gy):.3f} dB at {fx[int(np.nanargmax(gy))]:.3g} GHz", flush=True)
    return fx, gy, inject, True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "exp10_signal_sweep_m22dbm")
    p.add_argument("--ipm-dir", type=Path, default=ROOT / "outputs" / "ipm_python_design")
    p.add_argument("--power-dbm", type=float, default=-22.0)
    p.add_argument("--pump-freq-ghz", type=float, default=7.9)
    p.add_argument("--attenuation-db", type=float, default=35.0)
    p.add_argument("--z0-ohm", type=float, default=50.0)
    p.add_argument("--ic-a", type=float, default=4.1658984303797465e-6)
    p.add_argument("--signal-min-ghz", type=float, default=5.0)
    p.add_argument("--signal-max-ghz", type=float, default=10.0)
    p.add_argument("--signal-points", type=int, default=51)
    p.add_argument("--jc-scales", type=str, default="1.0,2.0")
    p.add_argument("--pump-port", type=int, default=4)
    p.add_argument("--source-port", type=int, default=1)
    p.add_argument("--out-port", type=int, default=2)
    p.add_argument("--pump-mode-count", type=int, default=10)
    p.add_argument("--nt", type=int, default=40)
    p.add_argument("--sidebands", type=int, default=10)
    p.add_argument("--gamma-nt", type=int, default=96)
    p.add_argument("--continuation-steps", type=int, default=20)
    p.add_argument("--pump-timeout-s", type=float, default=300.0)
    p.add_argument("--gain-timeout-s", type=float, default=600.0)
    p.add_argument("--jc-curve-csv", type=Path, default=None,
                   help="Optional JosephsonCircuits reference CSV (signal_ghz,gain_db) to overlay.")
    p.add_argument("--replot-only", action="store_true",
                   help="Skip solving; rebuild the plot from an existing signal_sweep.csv.")
    p.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Clip the y-axis (dB). A curve that falls entirely below LO is "
                   "annotated as off-scale instead of squashing the readable curves.")
    p.add_argument("--plot-name", default="signal_sweep.png",
                   help="Output PNG filename within --outdir.")
    return p.parse_args()


def read_python_series(csv_path: Path) -> dict[float, tuple[np.ndarray, np.ndarray, float]]:
    """Group an existing signal_sweep.csv by scale -> (freqs, gains, inject_current_a)."""
    by_scale: dict[float, list[tuple[float, float, float]]] = {}
    for r in csv.DictReader(csv_path.open(encoding="utf-8")):
        by_scale.setdefault(float(r["scale"]), []).append(
            (float(r["signal_ghz"]), float(r["gain_db"]), float(r["inject_current_a"]))
        )
    out: dict[float, tuple[np.ndarray, np.ndarray, float]] = {}
    for scale, rows in by_scale.items():
        rows.sort()
        out[scale] = (
            np.array([x[0] for x in rows]),
            np.array([x[1] for x in rows]),
            rows[0][2],
        )
    return out


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    scales = [float(s) for s in args.jc_scales.split(",")]

    csv_path = args.outdir / "signal_sweep.csv"
    fig, ax = plt.subplots(figsize=(8, 5))

    if args.replot_only:
        series = read_python_series(csv_path)
        for scale in sorted(series):
            fx, gy, inject = series[scale]
            ax.plot(fx, gy, "-", lw=1.8,
                    label=f"Python inject {inject*1e6:.2f} uA = {inject/args.ic_a:.2f} Ic (scale {scale:g})")
    else:
        all_rows: list[dict[str, float]] = []
        for scale in scales:
            fx, gy, inject, ok = sweep_scale(scale, args)
            if not ok or gy.size == 0:
                continue
            ax.plot(fx, gy, "-", lw=1.8,
                    label=f"Python inject {inject*1e6:.2f} uA = {inject/args.ic_a:.2f} Ic (scale {scale:g})")
            for f, g in zip(fx, gy):
                all_rows.append({"scale": scale, "inject_current_a": inject,
                                 "signal_ghz": f, "gain_db": g})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["scale", "inject_current_a", "signal_ghz", "gain_db"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {csv_path}", flush=True)

    jc_max = None
    if args.jc_curve_csv is not None and args.jc_curve_csv.exists():
        jc = list(csv.DictReader(args.jc_curve_csv.open(encoding="utf-8")))
        jf = np.array([float(r["signal_ghz"]) for r in jc])
        jg = np.array([float(r["gain_db"]) for r in jc])
        order = np.argsort(jf)
        jc_max = float(np.nanmax(jg))
        ax.plot(jf[order], jg[order], "k--", lw=2.0, label="JosephsonCircuits (IPM_JTWPA.jl)")

    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("signal frequency (GHz)")
    ax.set_ylabel("gain S21 (dB)")
    ax.set_title(f"IPM JTWPA gain profile @ {args.power_dbm:g} dBm, pump {args.pump_freq_ghz:g} GHz "
                 f"(Ip={dbm_to_peak_current_a(args.power_dbm, attenuation_db=args.attenuation_db, z0_ohm=args.z0_ohm)*1e6:.2f} uA)")

    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
        # Note when the JC curve is entirely below the clip (non-convergent here).
        if jc_max is not None and jc_max < args.ylim[0]:
            ax.annotate(
                f"JC off-scale (~{jc_max:.0f} dB): hbsolve diverges at this drive",
                xy=(0.5, 0.04), xycoords="axes fraction", ha="center",
                fontsize=8, color="0.3",
            )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = args.outdir / args.plot_name
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"wrote {png}", flush=True)


if __name__ == "__main__":
    main()
