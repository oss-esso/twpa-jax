"""Small, reproducible multitone basis-convergence study driver."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from twpa_solver.multitone.basis import build_lattice_basis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omega-p", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--omega-max", type=float, required=True)
    parser.add_argument("--pump-modes", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--max-order", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["signal_order_max", "n_tones", "n_p", "n_delta"])
        writer.writeheader()
        for order in args.max_order:
            basis = build_lattice_basis(args.pump_modes, order, args.omega_p, args.delta, args.omega_max)
            writer.writerow({"signal_order_max": order, "n_tones": basis.n_tones, "n_p": basis.n_p, "n_delta": basis.n_delta})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
