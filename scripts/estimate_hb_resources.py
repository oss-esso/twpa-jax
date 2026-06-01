"""Estimate dense HB Jacobian storage without allocating numerical arrays."""

from __future__ import annotations

import argparse
import json


def estimate(n_cells: int, n_tones: int) -> dict[str, float | int]:
    real_unknowns = 2 * n_tones * (2 * n_cells + 1)
    jacobian_bytes = real_unknowns**2 * 8
    return {
        "n_cells": n_cells,
        "n_tones": n_tones,
        "real_unknowns": real_unknowns,
        "one_float64_jacobian_mib": jacobian_bytes / 1024**2,
        "one_float64_jacobian_gib": jacobian_bytes / 1024**3,
        "three_matrix_working_set_gib": 3 * jacobian_bytes / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, nargs="+", required=True)
    parser.add_argument("--n-tones", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps([estimate(n, args.n_tones) for n in args.n_cells], indent=2))


if __name__ == "__main__":
    main()
