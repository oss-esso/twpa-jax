"""Generate solver production-readiness classification."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SolverReadiness:
    solver: str
    readiness_class: str
    actual_twpa_tested: bool
    largest_actual_twpa_test: str
    convergence_metadata: bool
    gpu_jax_native: bool
    map_used: bool
    notes: str


def readiness_rows() -> list[SolverReadiness]:
    return [
        SolverReadiness(
            "scipy-least-squares",
            "PRODUCTION_MAP_SOLVER",
            True,
            "ipm_jtwpa_physical_coupler, cells_per_line=32, pump_harmonics=5, sidebands=3, 25x25",
            True,
            False,
            True,
            "Current production baseline for reduced/physical Python maps.",
        ),
        SolverReadiness(
            "jax-dense-newton",
            "VALIDATED_SMALL_TWPA",
            True,
            "tiny reduced JTWPA residual, one harmonic",
            True,
            True,
            False,
            "Dense Jacobian; suitable for small validation problems only.",
        ),
        SolverReadiness(
            "jax-newton-krylov",
            "VALIDATED_SMALL_TWPA",
            True,
            "tiny reduced JTWPA residual with passive preconditioner",
            True,
            True,
            False,
            "JVP/GMRES path works on small TWPA residual; not yet map baseline.",
        ),
        SolverReadiness(
            "scipy-root",
            "TOY_ONLY",
            False,
            "scalar toy equations",
            True,
            False,
            False,
            "Wrapper exists; not validated on TWPA residuals.",
        ),
        SolverReadiness(
            "scipy-newton-krylov",
            "FAILED_OR_WEAK",
            False,
            "scalar toy equation had weak convergence behavior",
            True,
            False,
            False,
            "Not production; needs globalization/preconditioning.",
        ),
        SolverReadiness(
            "pseudo-transient",
            "TOY_ONLY",
            False,
            "toy scalar/nonlinear residual only",
            True,
            False,
            False,
            "Wrapper is not validated as a real TWPA globalization strategy.",
        ),
        SolverReadiness(
            "arclength",
            "SCAFFOLD_ONLY",
            False,
            "toy fold equation",
            False,
            False,
            False,
            "No TWPA branch continuation yet.",
        ),
        SolverReadiness(
            "shooting",
            "SCAFFOLD_ONLY",
            False,
            "toy stationary periodic state",
            False,
            False,
            False,
            "No TWPA periodic-orbit validation yet.",
        ),
        SolverReadiness(
            "anderson",
            "TOY_ONLY",
            False,
            "cos fixed-point toy",
            False,
            False,
            False,
            "Acceleration utility only; not wired to production TWPA map.",
        ),
        SolverReadiness(
            "deflation-multistart",
            "SCAFFOLD_ONLY",
            False,
            "solution clustering utility",
            False,
            False,
            False,
            "No TWPA branch discovery validation.",
        ),
        SolverReadiness(
            "mor",
            "SCAFFOLD_ONLY",
            False,
            "placeholder transfer-function API",
            False,
            False,
            False,
            "No reduced model fitting for TWPA conversion yet.",
        ),
        SolverReadiness(
            "two-tone-hb",
            "SCAFFOLD_ONLY",
            False,
            "frequency grid generation",
            False,
            False,
            False,
            "No production two-tone nonlinear residual.",
        ),
    ]


def write_readiness_report(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = readiness_rows()
    with (outdir / "solver_readiness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    lines = [
        "# New TWPA Solver Production Readiness",
        "",
        "| Solver | readiness class | largest actual TWPA test | map-used? | GPU/JAX-ready? | notes |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.solver} | {row.readiness_class} | {row.largest_actual_twpa_test} | "
            f"{row.map_used} | {row.gpu_jax_native} | {row.notes} |"
        )
    (outdir / "new_twpa_solver_production_readiness.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    write_readiness_report(Path(args.outdir))


if __name__ == "__main__":
    main()
