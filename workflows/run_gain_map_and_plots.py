"""Run a gain map and produce its complete standard plot catalogue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import plot_gain_map, run_gain_map
from scripts.plot_gain_vs_pumpfreq_signalfreq import plot_one as plot_pump_frequency
from scripts.plot_gain_vs_pumppower_signalfreq import plot_one as plot_pump_power

# Standard production gain-map engine flags (see CLAUDE.md "Standard gain-map
# flag set"). Applied unless the caller already passed the same flag through
# to run_gain_map.py, so a caller can still override any single one.
DEFAULT_ENGINE_FLAGS: list[tuple[str, str | None]] = [
    ("--mode", "warmstart"),
    ("--inproc-pump-backend", "schur_cpu_mt"),
    ("--inproc-preconditioner", "real_coupled_fast"),
    ("--inproc-fold-predictor", "secant"),
    ("--inproc-fail-fast", None),
    ("--fold-skip-patience", "2"),
    ("--pump-current-jc-scale", "1.0"),
    ("--frequency-chunk-size", "10"),
    ("--signal-detuning-mhz", "150"),
    ("--no-signal-spectrum", None),
    ("--log-level", "INFO"),
]


def _apply_default_engine_flags(run_args: list[str]) -> list[str]:
    """Prepend DEFAULT_ENGINE_FLAGS for any flag the caller did not already pass."""
    present = set(run_args)
    defaults: list[str] = []
    for flag, value in DEFAULT_ENGINE_FLAGS:
        if flag in present:
            continue
        defaults.append(flag)
        if value is not None:
            defaults.append(value)
    return defaults + run_args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Unrecognised options are forwarded to run_gain_map.py.",
    )
    parser.add_argument("--design", "--ipm-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/gain_map_workflow"))
    parser.add_argument("--plot-top-k", type=int, default=5)
    parser.add_argument("--plot-min-gain-db", type=float, default=10.0)
    parser.add_argument("--plot-save-pdf", action="store_true")
    parser.add_argument("--plot-save-svg", action="store_true")
    args, run_args = parser.parse_known_args(argv)

    run_args = _apply_default_engine_flags(run_args)
    run_args.extend([
        "--circuit-dir", str(args.design), "--outdir", str(args.run_dir),
        "--executor", "inprocess",
    ])
    result = run_gain_map.main(run_args)
    if result != 0:
        return result

    plot_args = [
        "--run-dir", str(args.run_dir), "--outdir", str(args.run_dir / "plots"),
        "--ipm-dir", str(args.design), "--top-k", str(args.plot_top_k),
        "--min-gain-db", str(args.plot_min_gain_db),
    ]
    if args.plot_save_pdf:
        plot_args.append("--save-pdf")
    if args.plot_save_svg:
        plot_args.append("--save-svg")
    result = plot_gain_map.main(plot_args)

    spectrum = args.run_dir / "map_spectrum.npz"
    if spectrum.exists():
        axes_dir = args.run_dir / "plots" / "maps"
        try:
            plot_pump_frequency(args.run_dir, args.run_dir.name, axes_dir / "gain_vs_pump_frequency_signal.png")
        except (KeyError, ValueError, IndexError) as exc:
            print(f"skipped pump-frequency projection: {exc}")
        try:
            plot_pump_power(args.run_dir, args.run_dir.name, axes_dir / "gain_vs_pump_power_signal.png")
        except (KeyError, ValueError, IndexError) as exc:
            print(f"skipped pump-power projection: {exc}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
