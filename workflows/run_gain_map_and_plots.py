"""Run a gain map and produce its complete standard plot catalogue."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import plot_gain_map, run_gain_map, run_hybrid_gain_map
from scripts.plot_gain_vs_pumpfreq_signalfreq import plot_one as plot_pump_frequency
from scripts.plot_gain_vs_pumppower_signalfreq import plot_one as plot_pump_power
from twpa_solver.core import load_circuit
from twpa_solver.multitone.resources import available_memory_gb, fast_coupled_footprint

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
    ("--signal-detuning-mhz", "150"),
    ("--no-signal-spectrum", None),
    ("--log-level", "INFO"),
]

# The production Schur/real-coupled map footprint is approximately 2.5 GiB per
# isolated frequency worker for the standard 31-tone basis on ipm_2c_fixed.
# ``fast_coupled_footprint`` scales that measured value with the nonlinear and
# port-node count of the input circuit. Only half of currently free physical
# memory is assigned to workers; the other half remains available to the OS,
# the parent process, and unrelated applications.
_FAST_RAM_FRACTION = 0.5
_FAST_DEFAULT_TONES = 31


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


def _translate_slow_flags(run_args: list[str]) -> list[str]:
    """Translate shared public map flags to the slow runner's names."""
    aliases = {
        "--pump-power-min-dbm": "--power-min-dbm",
        "--pump-power-max-dbm": "--power-max-dbm",
        "--pump-freq-min-ghz": "--freq-min-ghz",
        "--pump-freq-max-ghz": "--freq-max-ghz",
    }
    return [aliases.get(token, token) for token in run_args]


def _defer_fast_cleanup(run_args: list[str]) -> list[str]:
    """Keep pump solutions until the workflow has finished plotting."""
    retained = [
        token
        for token in run_args
        if token not in {"--compact-output", "--no-compact-output"}
    ]
    return [*retained, "--no-compact-output"]


def _option_present(run_args: list[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in run_args)


def _option_int(run_args: list[str], option: str, default: int) -> int:
    for index, token in enumerate(run_args):
        if token == option and index + 1 < len(run_args):
            try:
                return int(run_args[index + 1])
            except ValueError:
                break
        if token.startswith(f"{option}="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                break
    return default


def _fast_worker_memory_gb(design: Path) -> float:
    """Estimate the peak memory of one isolated fast-map worker."""

    circuit = load_circuit(design)
    # The Schur retained system consists of the nonlinear branches and port
    # nodes. This is the same dimension used by the measured fast-coupled
    # resource model; use a conservative lower bound for unusual circuits.
    retained = max(1, int(circuit.Ic.size) + len(circuit.port_to_index))
    return fast_coupled_footprint(_FAST_DEFAULT_TONES, retained).peak_gb


def _fast_parallel_defaults(
    run_args: list[str],
    design: Path,
) -> list[str]:
    """Add safe fast-map worker and chunk defaults for one design."""

    result = list(run_args)
    workers_explicit = _option_present(result, "--frequency-workers")
    chunk_explicit = _option_present(result, "--frequency-chunk-size")
    if workers_explicit:
        workers = max(1, _option_int(result, "--frequency-workers", 1))
    else:
        available_gb = available_memory_gb()
        if available_gb is None:
            workers = 1
            print(
                "fast-map worker sizing: available RAM is unavailable; using 1 worker",
                flush=True,
            )
        else:
            try:
                worker_gb = _fast_worker_memory_gb(design)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                worker_gb = 2.5
                print(
                    f"fast-map worker sizing: footprint estimate failed ({exc}); "
                    f"using {worker_gb:.1f} GiB per worker",
                    flush=True,
                )
            ram_budget_gb = _FAST_RAM_FRACTION * float(available_gb)
            workers = max(1, int(ram_budget_gb // worker_gb))
            cpu_count = os.cpu_count() or 1
            workers = min(workers, cpu_count)
            n_frequency = _option_int(result, "--n-frequency", 50)
            workers = min(workers, max(1, n_frequency))
            print(
                f"fast-map worker sizing: available_ram={available_gb:.2f} GiB "
                f"budget={ram_budget_gb:.2f} GiB per_worker={worker_gb:.2f} GiB "
                f"workers={workers}",
                flush=True,
            )
        result.extend(["--frequency-workers", str(workers)])

    if not chunk_explicit:
        # Keep one explicit chunk-size value per worker. The lower-level
        # runner preserves this value rather than recalculating it.
        result.extend(["--frequency-chunk-size", str(workers)])
    return result


def _cleanup_pump_solutions(run_dir: Path) -> int:
    """Remove per-point pump solutions after all plots have been generated."""
    solutions = list(run_dir.rglob("pump_solution.npz"))
    for solution in solutions:
        solution.unlink()
    if solutions:
        print(f"cleaned up {len(solutions)} pump solution(s) from {run_dir}")
    return len(solutions)


def _job_directories(
    design_paths: list[Path], run_root: Path,
) -> list[tuple[Path, Path]]:
    """Return stable, non-overlapping run directories for a design batch."""

    names = [path.name for path in design_paths]
    if len(set(names)) != len(names):
        raise ValueError(
            "multiple designs have the same directory name; use unique inputs"
        )
    return [(path, run_root / name) for path, name in zip(design_paths, names)]


def _run_one(
    design: Path,
    run_dir: Path,
    args: argparse.Namespace,
    base_run_args: list[str],
    slow: bool,
) -> int:
    """Run and plot one compiled design before returning to the batch loop."""

    run_args = list(base_run_args)
    run_args.extend([
        "--circuit-dir", str(design), "--outdir", str(run_dir),
    ])
    if not slow:
        run_args.extend(["--executor", "inprocess"])
    result = (run_hybrid_gain_map.main(run_args) if slow
              else run_gain_map.main(run_args))
    if result != 0:
        return result

    plot_args = [
        "--run-dir", str(run_dir), "--outdir", str(run_dir / "plots"),
        "--ipm-dir", str(design), "--top-k", str(args.plot_top_k),
        "--min-gain-db", str(args.plot_min_gain_db),
    ]
    if args.plot_save_pdf:
        plot_args.append("--save-pdf")
    if args.plot_save_svg:
        plot_args.append("--save-svg")
    try:
        result = plot_gain_map.main(plot_args)

        spectrum = run_dir / "map_spectrum.npz"
        if spectrum.exists():
            axes_dir = run_dir / "plots" / "maps"
            try:
                plot_pump_frequency(
                    run_dir,
                    run_dir.name,
                    axes_dir / "gain_vs_pump_frequency_signal.png",
                )
            except (KeyError, ValueError, IndexError) as exc:
                print(f"skipped pump-frequency projection: {exc}")
            try:
                plot_pump_power(
                    run_dir,
                    run_dir.name,
                    axes_dir / "gain_vs_pump_power_signal.png",
                )
            except (KeyError, ValueError, IndexError) as exc:
                print(f"skipped pump-power projection: {exc}")
    finally:
        if not slow:
            _cleanup_pump_solutions(run_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Unrecognised options are forwarded to run_gain_map.py.",
    )
    parser.add_argument(
        "--design", "--design-dir", "--ipm-dir", "--circuit-dir", type=Path,
        nargs="+", required=True,
    )
    parser.add_argument(
        "--output-dir", "--run-dir", "--outdir", dest="run_dir",
        type=Path, default=Path("outputs/gain_map_workflow"),
        help="Output root; batched designs are written to one subdirectory each",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fast", action="store_true", help="baseline HB map")
    modes.add_argument("--slow", action="store_true", help="HB + TD physical-boundary map")
    parser.add_argument("--plot-top-k", "--top-k", dest="plot_top_k",
                        type=int, default=10)
    parser.add_argument("--plot-min-gain-db", type=float, default=10.0)
    parser.add_argument("--plot-save-pdf", action="store_true")
    parser.add_argument("--plot-save-svg", action="store_true")
    args, run_args = parser.parse_known_args(argv)

    design_paths = list(args.design)
    if len(design_paths) == 1:
        jobs = [(design_paths[0], args.run_dir)]
    else:
        try:
            jobs = _job_directories(design_paths, args.run_dir)
        except ValueError as exc:
            parser.error(str(exc))

    slow = bool(args.slow)
    if not slow:
        run_args = _apply_default_engine_flags(run_args)
        run_args = _defer_fast_cleanup(run_args)
    elif "--log-level" not in run_args:
        run_args.extend(["--log-level", "WARNING"])
    run_args = _translate_slow_flags(run_args) if slow else run_args
    for design, run_dir in jobs:
        job_args = (
            _fast_parallel_defaults(run_args, design)
            if not slow else run_args
        )
        result = _run_one(design, run_dir, args, job_args, slow)
        if result != 0:
            return result
        print(f"completed gain-map workflow for {design} in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
