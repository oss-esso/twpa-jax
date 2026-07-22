"""Build an IPM design and generate its pump-off S-parameter plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from twpa_solver.builders import ipm
from twpa_solver.signal.passive import db20, passive_s_matrix


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="All unrecognised options are forwarded to the IPM builder.",
    )
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--passive-start-ghz", type=float, default=4.0)
    parser.add_argument("--passive-stop-ghz", type=float, default=11.0)
    parser.add_argument("--passive-points", type=int, default=1401)
    parser.add_argument("--passive-z0-ohm", type=float, default=50.0)
    return parser


def _write_passive(design_dir: Path, args: argparse.Namespace) -> None:
    if args.passive_points < 2 or args.passive_stop_ghz <= args.passive_start_ghz:
        raise ValueError("passive frequency range must be increasing and have at least 2 points")
    freq_ghz = np.linspace(
        args.passive_start_ghz, args.passive_stop_ghz, args.passive_points
    )
    matrix = passive_s_matrix(
        design_dir, freq_ghz * 1e9, z0_ohm=args.passive_z0_ohm
    )
    names = ("s11", "s21", "s31", "s41", "s14", "s24", "s34")
    pairs = ((1, 1), (2, 1), (3, 1), (4, 1), (1, 4), (2, 4), (3, 4))
    traces = {
        name: db20(matrix[:, out - 1, source - 1])
        for name, (out, source) in zip(names, pairs)
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    np.savez_compressed(design_dir / "passive_sparameters.npz", freq_ghz=freq_ghz, **{
        f"{name}_db": values for name, values in traces.items()
    })

    def save(names_to_plot: tuple[str, ...], stem: str, title: str) -> None:
        fig, axes = plt.subplots(len(names_to_plot), 1, figsize=(11, 3.2 * len(names_to_plot)), sharex=True)
        for axis, name in zip(np.atleast_1d(axes), names_to_plot):
            axis.plot(freq_ghz, traces[name], lw=1.2)
            axis.set_ylabel(f"|{name.upper()}| (dB)")
            axis.grid(True, alpha=0.3)
        np.atleast_1d(axes)[-1].set_xlabel("Signal frequency (GHz)")
        fig.suptitle(title)
        fig.tight_layout()
        for suffix, kwargs in (("png", {"dpi": 200}), ("pdf", {}), ("svg", {})):
            fig.savefig(design_dir / f"{stem}.{suffix}", **kwargs)
        plt.close(fig)

    save(("s21", "s24"), "passive_s21_s24", f"Pump-off response: {design_dir.name}")
    save(("s11", "s21", "s31", "s41"), "passive_s11_s21_s31_s41",
         f"Pump-off response from port 1: {design_dir.name}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    workflow_args, builder_args = parser.parse_known_args(argv)
    builder_args.extend(["--outdir", str(workflow_args.design_dir), "--write-matrices"])
    ipm.main(builder_args)
    _write_passive(workflow_args.design_dir, workflow_args)
    print(f"wrote design and passive plots to {workflow_args.design_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
