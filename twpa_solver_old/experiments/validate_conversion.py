"""Generate conversion S-parameter validation artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from twpa_solver_old.experiments.run_ipm_25x25_gain_map import _row_from_solution_status
from twpa_solver_old.model.ipm import IPMConfig, build_ipm_jtwpa_reduced_marker
from twpa_solver_old.model.nonlinearities import JosephsonNonlinearity
from twpa_solver_old.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver_old.residuals.conversion import build_conversion_sparameters
from twpa_solver_old.residuals.linear import solve_linear_sparameters


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = _validation_rows()
    with (outdir / "conversion_validation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(row["result"] == "PASS" for row in rows)
    (outdir / "conversion_validation_summary.md").write_text(
        "\n".join(
            [
                "# Conversion Validation",
                "",
                f"- checks passed: {passed} / {len(rows)}",
                "- rows: `conversion_validation_rows.csv`",
            ]
        ),
        encoding="utf-8",
    )


def _validation_rows() -> list[dict[str, Any]]:
    model = build_ipm_jtwpa_reduced_marker(IPMConfig(cells_per_line=1))
    residual = PumpAFTResidual(model, PumpAFTConfig(pump_frequency_hz=6e9, harmonics=3))
    zero = build_conversion_sparameters(
        model,
        residual,
        np.zeros(residual.size),
        5e9,
        2,
        pump_success=True,
        pump_status="converged",
    )
    sideband_count = len(zero.sidebands)
    port_count = len(model.ports)
    y = zero.y_conversion_s.reshape(port_count, sideband_count, port_count, sideband_count)
    offdiag = []
    for out_sideband in range(sideband_count):
        for in_sideband in range(sideband_count):
            if out_sideband != in_sideband:
                offdiag.append(np.max(np.abs(y[:, out_sideband, :, in_sideband])))
    offdiag_max = float(max(offdiag))

    linear = solve_linear_sparameters(model, 5e9)
    sideband_0 = zero.sidebands.index(0)
    s21 = zero.s_conversion[
        1 * sideband_count + sideband_0,
        0 * sideband_count + sideband_0,
    ]
    s21_error = float(abs(s21 - linear.s[1, 0]))

    small = _conversion_for_amplitude(model, residual, 1e-19)
    smaller = _conversion_for_amplitude(model, residual, 1e-21)
    sideband_2 = small.sidebands.index(2)
    idler_small = abs(small.s_conversion[1 * sideband_count + sideband_2, sideband_0])
    idler_smaller = abs(smaller.s_conversion[1 * sideband_count + sideband_2, sideband_0])

    derivative_error = _josephson_derivative_error()
    masked = _row_from_solution_status(
        {"signal_gain_db": 1.0, "idler_gain_db": 2.0},
        success=False,
        status="diagnostic",
    )
    return [
        _row("zero_pump_block_diagonal", offdiag_max, 1e-18, offdiag_max <= 1e-18),
        _row("zero_pump_s21_consistency", s21_error, 1e-8, s21_error <= 1e-8),
        _row(
            "small_pump_idler_tends_to_zero",
            float(idler_smaller / idler_small),
            1e-3,
            idler_smaller < idler_small,
        ),
        _row("josephson_derivative_finite_difference", derivative_error, 1e-5, derivative_error <= 1e-5),
        _row(
            "pump_status_masks_gain",
            float(np.isnan(masked["signal_gain_db"]) and np.isnan(masked["idler_gain_db"])),
            1.0,
            bool(np.isnan(masked["signal_gain_db"]) and np.isnan(masked["idler_gain_db"])),
        ),
    ]


def _conversion_for_amplitude(model, residual: PumpAFTResidual, amplitude: float):
    x = np.zeros(residual.size)
    x.reshape(residual.harmonics, 2, residual.num_nodes)[0, 0, 0] = amplitude
    return build_conversion_sparameters(
        model,
        residual,
        x,
        5e9,
        2,
        pump_success=True,
        pump_status="converged",
    )


def _josephson_derivative_error() -> float:
    law = JosephsonNonlinearity(np.asarray([8e-6]))
    psi = np.asarray([1e-18])
    eps = 1e-21
    finite_difference = (law.current(psi + eps) - law.current(psi - eps)) / (2.0 * eps)
    return float(abs((law.derivative(psi) - finite_difference)[0]) / abs(finite_difference[0]))


def _row(name: str, value: float, tolerance: float, passed: bool) -> dict[str, Any]:
    return {
        "validation": name,
        "value": value,
        "tolerance": tolerance,
        "result": "PASS" if passed else "FAIL",
    }


if __name__ == "__main__":
    main()
