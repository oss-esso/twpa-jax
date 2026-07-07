from __future__ import annotations

from pathlib import Path
import argparse
import sys

from twpa_solver.builders import scattered


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build scattered 2c/3c IPM design matrices."
    )
    p.add_argument("--design", choices=["2c", "3c"], required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--lj-scatter-sigma", type=float, default=0.0)
    p.add_argument("--lj-scatter-seed", type=int, default=1)
    return p.parse_args()


def main() -> int:
    # Preferred production function names.
    if hasattr(scattered, "build_scattered_design"):
        args = parse_args()
        scattered.build_scattered_design(
            design=args.design,
            outdir=args.outdir,
            lj_scatter_sigma=args.lj_scatter_sigma,
            lj_scatter_seed=args.lj_scatter_seed,
        )
        return 0

    if hasattr(scattered, "build_scattered_ipm_design"):
        args = parse_args()
        scattered.build_scattered_ipm_design(
            design=args.design,
            outdir=args.outdir,
            lj_scatter_sigma=args.lj_scatter_sigma,
            lj_scatter_seed=args.lj_scatter_seed,
        )
        return 0

    # Compatibility path: if the copied experiment module still exposes its own
    # argparse main(), let it parse the same CLI flags.
    if hasattr(scattered, "main"):
        result = scattered.main()
        return 0 if result is None else int(result)

    names = ", ".join(sorted(n for n in dir(scattered) if not n.startswith("_")))
    raise RuntimeError(
        "twpa_solver.builders.scattered has no supported entry point. "
        "Expected build_scattered_design(...), build_scattered_ipm_design(...), "
        f"or main(). Available public names: {names}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
