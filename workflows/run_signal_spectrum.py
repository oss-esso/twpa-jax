"""Run one fixed pump point and plot its signal spectrum plus port-1 S-parameters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twpa_solver.signal.passive import db20, passive_s_matrix
from scripts import plot_gain_map, run_gain_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", "--ipm-dir", type=Path, required=True)
    parser.add_argument("--pump-power-dbm", type=float, required=True)
    parser.add_argument("--pump-frequency-ghz", type=float, required=True)
    parser.add_argument("--signal-start-ghz", type=float, required=True)
    parser.add_argument("--signal-stop-ghz", type=float, required=True)
    parser.add_argument("--signal-points", type=int, default=501)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/signal_spectrum"))
    args, extra = parser.parse_known_args(argv)
    if args.signal_stop_ghz <= args.signal_start_ghz or args.signal_points < 2:
        raise SystemExit("signal range must be increasing and contain at least 2 points")

    offsets = np.linspace(args.signal_start_ghz - args.pump_frequency_ghz,
                          args.signal_stop_ghz - args.pump_frequency_ghz,
                          args.signal_points)
    # run_gain_map stores a symmetric offset ladder. Keep the requested range
    # as the canonical output and use the map backend for the actual pump solve.
    run_args = [
        "--circuit-dir", str(args.design), "--outdir", str(args.run_dir),
        "--executor", "inprocess", "--n-power", "1", "--n-frequency", "1",
        "--pump-power-min-dbm", str(args.pump_power_dbm),
        "--pump-power-max-dbm", str(args.pump_power_dbm),
        "--pump-freq-min-ghz", str(args.pump_frequency_ghz),
        "--pump-freq-max-ghz", str(args.pump_frequency_ghz),
        "--signal-spectrum", "--signal-offset-count-per-side", str(max(1, args.signal_points // 2)),
        "--signal-offset-start-mhz", str(abs(offsets[1]) * 1000.0 if offsets.size > 1 else 1.0),
        "--signal-offset-step-mhz", str(abs(offsets[1] - offsets[0]) * 1000.0),
        *extra,
    ]
    result = run_gain_map.main(run_args)
    if result != 0:
        return result
    plot_gain_map.main(["--run-dir", str(args.run_dir), "--outdir", str(args.run_dir / "plots"),
                        "--ipm-dir", str(args.design)])

    freq = np.linspace(args.signal_start_ghz, args.signal_stop_ghz, args.signal_points)
    matrix = passive_s_matrix(args.design, freq * 1e9)
    np.savez_compressed(args.run_dir / "port1_sparameters.npz", freq_ghz=freq,
                        **{name + "_db": db20(matrix[:, out - 1, 0])
                           for name, out in (("s11", 1), ("s21", 2), ("s31", 3), ("s41", 4))})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, out in (("s11", 1), ("s21", 2), ("s31", 3), ("s41", 4)):
        ax.plot(freq, db20(matrix[:, out - 1, 0]), label=name.upper())
    ax.set_xlabel("Signal frequency (GHz)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title(f"Pump-off port-1 S-parameters: {args.design.name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 200}), ("pdf", {}), ("svg", {})):
        fig.savefig(args.run_dir / f"port1_sparameters.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote fixed-pump spectrum and port-1 S-parameters to {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
