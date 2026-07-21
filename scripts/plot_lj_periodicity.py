"""Compare pump-frequency periodicity and optionally sweep the IPM Lj.

The measurement is a collection of ``105C5_<pump>GHz.npy`` files containing
response versus pump power and signal frequency.  At one requested pump power,
the measured curve is reduced to the maximum response over a signal-frequency
band.  Simulation curves are read from ``map_arrays.npz`` at the nearest pump
power row.  The second panel removes each curve's mean, making periodicity
comparable when absolute gain differs.

Examples
--------
Plot the existing measurement and map::

    python scripts/plot_lj_periodicity.py \
      --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20_123p9_cg66_halfcurrent_run_gain_map

Generate one-frequency-slice maps for several Lj values and plot them.  The
simulation power is adjusted to preserve approximately the same ``I/Ic`` as
the reference ``Lj``::

    python scripts/plot_lj_periodicity.py --run-sweep \
      --lj-values 70 80 90 100 110 123.9 140
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

FREQ_RE = re.compile(r"105C5_([0-9.]+)GHz\.npy$")


def load_measurement(
    measurement_dir: Path,
    target_power_dbm: float,
    signal_band_ghz: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    files = sorted(measurement_dir.glob("105C5_*GHz.npy"))
    if not files:
        raise FileNotFoundError(f"No measurement files in {measurement_dir}")

    frequencies: list[float] = []
    gains: list[float] = []
    selected_power: float | None = None
    lo, hi = signal_band_ghz

    for path in files:
        match = FREQ_RE.search(path.name)
        if not match:
            continue
        data = np.load(path, allow_pickle=True).item()
        powers = np.asarray(data["PumpPower"], dtype=float)
        power_index = int(np.argmin(np.abs(powers - target_power_dbm)))
        power = float(powers[power_index])
        if selected_power is None:
            selected_power = power
        signal_frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
        response = np.asarray(data["Response"], dtype=float)
        band = (signal_frequency >= lo) & (signal_frequency <= hi)
        if not np.any(band):
            raise ValueError(f"Empty signal band {signal_band_ghz} for {path.name}")
        frequencies.append(float(match.group(1)))
        gains.append(float(np.nanmax(response[power_index, band])))

    order = np.argsort(frequencies)
    assert selected_power is not None
    return (
        np.asarray(frequencies, dtype=float)[order],
        np.asarray(gains, dtype=float)[order],
        selected_power,
    )


def load_simulation(
    map_dir: Path,
    target_power_dbm: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    arrays = np.load(map_dir / "map_arrays.npz")
    powers = np.asarray(arrays["pump_power_dbm"], dtype=float)
    frequencies = np.asarray(arrays["pump_frequency_ghz"], dtype=float)
    gains = np.asarray(arrays["gain_db_warm"], dtype=float)
    # Lj sweeps deliberately generate one power row per map.  For a full map,
    # select the requested reference power instead.
    if powers.size == 1:
        power_index = 0
    elif target_power_dbm is not None:
        power_index = int(np.argmin(np.abs(powers - target_power_dbm)))
    else:
        raise ValueError(f"{map_dir} has multiple power rows; target power is required")
    return frequencies, gains[power_index], float(powers[power_index])


def matched_power_dbm(reference_power_dbm: float, reference_lj_ph: float, lj_ph: float) -> float:
    """Power giving approximately the same I/Ic when Lj changes.

    Ic = Phi0/Lj, while the map's peak current is proportional to 10**(P/20).
    Therefore P(Lj) = P(ref) + 20 log10(ref_Lj/Lj).
    """
    return float(reference_power_dbm + 20.0 * np.log10(reference_lj_ph / lj_ph))


def run_lj_sweep(args: argparse.Namespace) -> list[Path]:
    design_root = args.design_root
    map_root = args.sweep_root
    design_root.mkdir(parents=True, exist_ok=True)
    map_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for lj in args.lj_values:
        tag = f"lj{lj:g}".replace(".", "p")
        target_power_dbm = matched_power_dbm(
            args.reference_power_dbm, args.reference_lj_ph, lj
        )
        design_dir = design_root / f"ipm_{tag}_cg{args.cg_ff:g}".replace(".", "p")
        map_dir = map_root / f"map_{tag}_cg{args.cg_ff:g}".replace(".", "p")
        log_path = map_root / f"{tag}.log"
        if not (design_dir / "C.npz").exists():
            subprocess.run(
                [
                    sys.executable,
                    "experiments/exp07_python_ipm_design_builder.py",
                    "--outdir",
                    str(design_dir),
                    "--coupler-mode",
                    args.coupler_mode,
                    "--lj-ph",
                    str(lj),
                    "--cg-ff",
                    str(args.cg_ff),
                    "--write-matrices",
                ],
                check=True,
            )
        if not (map_dir / "map_arrays.npz").exists() or args.overwrite:
            command = [
                sys.executable,
                "scripts/run_gain_map.py",
                "--executor",
                "inprocess",
                "--mode",
                "warmstart",
                "--inproc-pump-backend",
                "schur_cpu_mt",
                "--inproc-preconditioner",
                "real_coupled_fast",
                "--inproc-fold-predictor",
                "secant",
                "--inproc-fail-fast",
                "--fold-skip-patience",
                "2",
                "--pump-current-jc-scale",
                "1.0",
                "--circuit-dir",
                str(design_dir),
                "--n-power",
                "1",
                "--n-frequency",
                str(args.n_frequency),
                "--pump-power-min-dbm",
                str(target_power_dbm),
                "--pump-power-max-dbm",
                str(target_power_dbm),
                "--pump-freq-min-ghz",
                str(args.pump_freq_min_ghz),
                "--pump-freq-max-ghz",
                str(args.pump_freq_max_ghz),
                "--signal-detuning-mhz",
                "100",
                "--attenuation-db",
                str(args.attenuation_db),
                "--no-signal-spectrum",
                "--outdir",
                str(map_dir),
                "--overwrite",
            ]
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        outputs.append(map_dir)
    return outputs


def plot_curves(
    measurement_dir: Path,
    map_dirs: list[Path],
    output: Path,
    target_power_dbm: float,
    signal_band_ghz: tuple[float, float],
    frequency_shift_ghz: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meas_freq, meas_gain, meas_power = load_measurement(
        measurement_dir, target_power_dbm, signal_band_ghz
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(
        meas_freq + frequency_shift_ghz,
        meas_gain,
        "o-",
        ms=3,
        lw=1.2,
        label=f"measurement ({meas_power:.2f} dBm)",
    )
    axes[1].plot(
        meas_freq + frequency_shift_ghz,
        meas_gain - np.nanmean(meas_gain),
        "o-",
        ms=3,
        lw=1.2,
        label="measurement, mean-centered",
    )

    for map_dir in map_dirs:
        sim_freq, sim_gain, sim_power = load_simulation(map_dir, target_power_dbm)
        label = map_dir.name
        axes[0].plot(sim_freq, sim_gain, "-", lw=1.2, label=f"{label} ({sim_power:.2f} dBm)")
        axes[1].plot(sim_freq, sim_gain - np.nanmean(sim_gain), "-", lw=1.2, label=label)

    axes[0].set_ylabel("gain / peak response (dB)")
    axes[1].set_ylabel("mean-centered gain (dB)")
    axes[1].set_xlabel("pump frequency (GHz)")
    axes[0].set_title(
        f"Pump-frequency periodicity near {target_power_dbm:g} dBm "
        f"(measurement frequency shift +{frequency_shift_ghz:g} GHz)"
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    metadata = {
        "measurement_target_power_dbm": target_power_dbm,
        "measurement_selected_power_dbm": meas_power,
        "signal_band_ghz": list(signal_band_ghz),
        "frequency_shift_ghz": frequency_shift_ghz,
        "map_dirs": [str(path) for path in map_dirs],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurement-dir",
        type=Path,
        default=Path("docs/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK"),
    )
    parser.add_argument("--map-dir", type=Path, action="append", default=[])
    parser.add_argument("--target-power-dbm", type=float, default=-25.0)
    parser.add_argument(
        "--reference-power-dbm",
        type=float,
        default=-25.0,
        help="Measurement/reference power. Lj-sweep simulation powers are shifted from this.",
    )
    parser.add_argument(
        "--reference-lj-ph",
        type=float,
        default=123.9,
        help="Lj corresponding to --reference-power-dbm (pH).",
    )
    parser.add_argument("--signal-band-ghz", type=float, nargs=2, default=(4.0, 12.0))
    parser.add_argument("--frequency-shift-ghz", type=float, default=0.99)
    parser.add_argument("--output", type=Path, default=Path("plots/lj_periodicity.png"))
    parser.add_argument("--run-sweep", action="store_true")
    parser.add_argument("--lj-values", type=float, nargs="+", default=[79.0, 100.0, 123.9, 150.0])
    parser.add_argument("--cg-ff", type=float, default=66.0)
    parser.add_argument("--coupler-mode", choices=["cached", "optimize"], default="cached")
    parser.add_argument("--design-root", type=Path, default=Path("outputs/lj_periodicity_designs"))
    parser.add_argument("--sweep-root", type=Path, default=Path("outputs/lj_periodicity_maps"))
    parser.add_argument("--n-frequency", type=int, default=51)
    parser.add_argument("--pump-freq-min-ghz", type=float, default=7.0)
    parser.add_argument("--pump-freq-max-ghz", type=float, default=8.0)
    parser.add_argument("--attenuation-db", type=float, default=35.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    maps = list(args.map_dir)
    if args.run_sweep:
        maps.extend(run_lj_sweep(args))
    if not maps:
        parser.error("provide --map-dir or use --run-sweep")
    plot_curves(
        args.measurement_dir,
        maps,
        args.output,
        args.target_power_dbm if not args.run_sweep else args.reference_power_dbm,
        tuple(args.signal_band_ghz),
        args.frequency_shift_ghz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
