"""Isolate which IPM design parameter(s) drive pump-frequency fringe periodicity.

Complements scripts/plot_lj_periodicity.py's existing Lj-only sweep
(outputs/lj_periodicity_maps/map_lj*_cg66, Cg fixed at 66 fF) with two more
single-parameter sweeps, holding everything else at the exp07 reference
(Lj=123.9 pH, Cg=66 fF, array_length=418, num_rows=6):

  --sweep cg           : vary Cg, Lj fixed.
  --sweep array_length : vary array_length (cells per row), Lj/Cg fixed.
  --sweep num_rows     : vary num_rows, Lj/Cg fixed.

For each design point: build the circuit directly via the exp07 builder
module (so array_length/num_rows overrides are possible, unlike the CLI),
run a fast single-power/51-frequency-point gain map (7-8 GHz), then count
peaks (scipy.signal.find_peaks, prominence-gated, matching the metric
established interactively) on the resulting gain-vs-frequency curve.

Usage:
    python scripts/periodicity_campaign.py --sweep cg --values 22 33 44 55 66 77 88 99
    python scripts/periodicity_campaign.py --sweep array_length --values 150 250 350 418 500 600
    python scripts/periodicity_campaign.py --sweep num_rows --values 2 4 6 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import exp07_python_ipm_design_builder as ipm  # noqa: E402
import plot_lj_periodicity as ljp  # noqa: E402


def build_design(outdir: Path, **overrides) -> None:
    if (outdir / "C.npz").exists():
        return
    params = ipm.IPMParams(**overrides)
    coupler = ipm.make_coupler_discrete(params, "cached")
    circuit, ends = ipm.make_ipm(params, coupler)
    mats = ipm.build_matrices(circuit)
    ipm.write_outputs(str(outdir), circuit, params, coupler, ends, mats)


def run_map(design_dir: Path, map_dir: Path, log_path: Path, power_dbm: float,
            n_frequency: int, freq_lo: float, freq_hi: float, overwrite: bool) -> None:
    if (map_dir / "map_arrays.npz").exists() and not overwrite:
        return
    command = [
        sys.executable, "scripts/run_gain_map.py",
        "--executor", "inprocess",
        "--mode", "warmstart",
        "--inproc-pump-backend", "schur_cpu_mt",
        "--inproc-preconditioner", "real_coupled_fast",
        "--inproc-fold-predictor", "secant",
        "--inproc-fail-fast",
        "--fold-skip-patience", "2",
        "--pump-current-jc-scale", "1.0",
        "--circuit-dir", str(design_dir),
        "--n-power", "1",
        "--n-frequency", str(n_frequency),
        "--pump-power-min-dbm", str(power_dbm),
        "--pump-power-max-dbm", str(power_dbm),
        "--pump-freq-min-ghz", str(freq_lo),
        "--pump-freq-max-ghz", str(freq_hi),
        "--signal-detuning-mhz", "100",
        "--attenuation-db", "35",
        "--no-signal-spectrum",
        "--outdir", str(map_dir),
        "--overwrite",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def peak_count(map_dir: Path, freq_lo: float, freq_hi: float, prominence_db: float) -> dict:
    freq, gain, power = ljp.load_simulation(map_dir, None)
    order = np.argsort(freq)
    freq, gain = freq[order], gain[order]
    band = (freq >= freq_lo) & (freq <= freq_hi)
    freq, gain = freq[band], gain[band]
    finite = np.isfinite(gain)
    freq, gain = freq[finite], gain[finite]
    if freq.size < 2:
        return {"n_peaks": 0, "spacing_ghz": float("nan"), "power_dbm": power, "n_finite": int(freq.size)}
    peaks, _ = find_peaks(gain, prominence=prominence_db)
    spacing = float((freq[-1] - freq[0]) / (len(peaks) - 1)) if len(peaks) >= 2 else float("nan")
    return {
        "n_peaks": int(len(peaks)),
        "spacing_ghz": spacing,
        "peak_freqs_ghz": [float(x) for x in freq[peaks]],
        "power_dbm": float(power),
        "n_finite": int(freq.size),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", choices=["cg", "array_length", "num_rows"], required=True)
    p.add_argument("--values", type=float, nargs="+", required=True)
    p.add_argument("--reference-lj-ph", type=float, default=123.9)
    p.add_argument("--reference-cg-ff", type=float, default=66.0)
    p.add_argument("--reference-power-dbm", type=float, default=-25.0)
    p.add_argument("--n-frequency", type=int, default=51)
    p.add_argument("--freq-range-ghz", type=float, nargs=2, default=(7.0, 8.0))
    p.add_argument("--peak-prominence-db", type=float, default=2.0)
    p.add_argument("--design-root", type=Path, default=Path("outputs/periodicity_campaign_designs"))
    p.add_argument("--map-root", type=Path, default=Path("outputs/periodicity_campaign_maps"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    args.design_root.mkdir(parents=True, exist_ok=True)
    args.map_root.mkdir(parents=True, exist_ok=True)
    lo, hi = args.freq_range_ghz
    out = args.out or Path(f"plots/periodicity_campaign_{args.sweep}.json")

    results: dict[str, dict] = {}
    for value in args.values:
        tag = f"{args.sweep}{value:g}".replace(".", "p")
        overrides = {
            "Lj": args.reference_lj_ph * 1e-12,
            "Cg": args.reference_cg_ff * 1e-15,
        }
        if args.sweep == "cg":
            overrides["Cg"] = value * 1e-15
        elif args.sweep == "array_length":
            overrides["array_length"] = int(value)
        elif args.sweep == "num_rows":
            overrides["num_rows"] = int(value)

        design_dir = args.design_root / tag
        map_dir = args.map_root / tag
        log_path = args.map_root / f"{tag}.log"

        print(f"[{tag}] building design ...", flush=True)
        build_design(design_dir, **overrides)
        print(f"[{tag}] running map ...", flush=True)
        run_map(design_dir, map_dir, log_path, args.reference_power_dbm,
                args.n_frequency, lo, hi, args.overwrite)
        r = peak_count(map_dir, lo, hi, args.peak_prominence_db)
        results[tag] = {"value": value, **r}
        print(f"[{tag}] n_peaks={r['n_peaks']} spacing={r['spacing_ghz']:.4f} GHz "
              f"n_finite={r['n_finite']}/{args.n_frequency}", flush=True)

    print(f"\n{'value':>10} {'n_peaks':>8} {'spacing(GHz)':>13} {'n_finite':>9}")
    for tag, r in results.items():
        print(f"{r['value']:10g} {r['n_peaks']:8d} {r['spacing_ghz']:13.4f} {r['n_finite']:9d}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
