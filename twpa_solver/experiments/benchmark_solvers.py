"""Reduced solver comparison benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from twpa_solver.solvers.jax_dense_newton import solve_jax_dense_newton
from twpa_solver.solvers.jax_newton_krylov import solve_jax_newton_krylov
from twpa_solver.solvers.pseudo_transient import solve_pseudo_transient
from twpa_solver.solvers.scipy_least_squares import solve_least_squares
from twpa_solver.solvers.scipy_root import solve_newton_krylov, solve_root


def main(argv: list[str] | None = None) -> None:
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def residual_np(x: np.ndarray) -> np.ndarray:
        return np.asarray([x[0] ** 3 - 1.0])

    def residual_jax(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([x[0] ** 3 - 1.0])

    rows = []
    solvers = [
        ("scipy least_squares", lambda: solve_least_squares(residual_np, np.asarray([0.2]))),
        ("scipy root", lambda: solve_root(residual_np, np.asarray([0.2]))),
        ("scipy newton_krylov", lambda: solve_newton_krylov(residual_np, np.asarray([0.2]))),
        ("jax dense Newton", lambda: solve_jax_dense_newton(residual_jax, np.asarray([0.2]))),
        ("jax matrix-free Newton-Krylov", lambda: solve_jax_newton_krylov(residual_jax, np.asarray([0.2]))),
        ("pseudo-transient + LS", lambda: solve_pseudo_transient(residual_np, np.asarray([0.2]))),
    ]
    for name, run in solvers:
        try:
            result = run()
            rows.append(
                {
                    "solver": name,
                    "status": "implemented and passed" if result.success else "implemented but failed",
                    "tested_problem": "x^3 - 1 = 0",
                    "success_rate": 1.0 if result.success else 0.0,
                    "runtime_s": result.runtime_s,
                    "notes": result.message,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "solver": name,
                    "status": "failed with reason",
                    "tested_problem": "x^3 - 1 = 0",
                    "success_rate": 0.0,
                    "runtime_s": np.nan,
                    "notes": str(exc),
                }
            )
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
