"""Gates for the derived quantities behind the NS onset figure set."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.plot_ns_onset_figures import (
    control_to_dbm,
    fit_linear_intercept,
    interpolate_crossing,
    load_branch,
    run,
    parse_args,
    twist_coefficient,
)


def test_crossing_is_interpolated_between_the_bracketing_drives() -> None:
    drive = np.array([-24.25, -24.20])
    magnitude = np.array([0.9997399515, 1.0005276786])

    found = interpolate_crossing(drive, magnitude, 1.0)

    assert found is not None
    crossing, bracket = found
    assert crossing == pytest.approx(-24.2334939, abs=1e-6)
    assert bracket == (-24.25, -24.20)


def test_a_branch_that_never_reaches_one_reports_no_crossing() -> None:
    drive = np.linspace(-24.6, -23.6, 21)
    magnitude = np.full(drive.size, 0.9275)

    assert interpolate_crossing(drive, magnitude, 1.0) is None


def test_first_crossing_is_returned_when_the_branch_recrosses() -> None:
    drive = np.array([0.0, 1.0, 2.0, 3.0])
    magnitude = np.array([0.5, 1.5, 0.5, 1.5])

    found = interpolate_crossing(drive, magnitude, 1.0)

    assert found is not None
    assert found[0] == pytest.approx(0.5)


def test_normal_form_intercept_recovers_the_seeded_onset() -> None:
    p_c, slope = -24.2435, 0.9202
    drive = np.array([-24.05, -23.90])

    measured_slope, measured_pc, r_squared = fit_linear_intercept(
        drive, slope * (drive - p_c)
    )

    assert measured_slope == pytest.approx(slope)
    assert measured_pc == pytest.approx(p_c)
    assert r_squared == pytest.approx(1.0)


def test_twist_splits_the_two_measured_frequency_slopes() -> None:
    # a alone below onset, a + b*dr2/dP above it.
    assert twist_coefficient(0.0404, -0.01513, 0.9202) == pytest.approx(
        -0.0604, abs=5e-4
    )


def test_twist_is_undefined_without_amplitude_growth() -> None:
    assert twist_coefficient(0.04, -0.02, 0.0) is None


def test_control_axis_maps_onto_dbm_through_its_anchor() -> None:
    control = np.array([0.587852, 0.5875, 0.5950])

    drive = control_to_dbm(control, 0.587852, -24.05)

    assert drive[0] == pytest.approx(-24.05)
    assert drive[1] == pytest.approx(-24.0552, abs=1e-3)
    assert drive[2] == pytest.approx(-23.9450, abs=1e-3)


def _write_floquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _floquet_row(drive: float, magnitude: float, converged: bool = True) -> dict:
    return {
        "requested_drive_dbm": drive,
        "multiplier_magnitude": magnitude,
        "multiplier_phase_rad": 0.586,
        "omega_real_ghz": 0.70 + 0.05 * (drive + 24.6),
        "omega_imag_ghz": 0.001,
        "mode_overlap": 0.999,
        "converged": converged,
    }


def test_unconverged_rows_are_dropped_unless_requested(tmp_path: Path) -> None:
    path = tmp_path / "floquet.csv"
    _write_floquet(path, [
        _floquet_row(-24.3, 0.998),
        _floquet_row(-24.2, float("nan"), converged=False),
    ])

    assert load_branch([path], True).drive.size == 1
    assert load_branch([path], False).drive.size == 2


def test_several_csvs_merge_and_deduplicate_on_drive(tmp_path: Path) -> None:
    first = tmp_path / "down.csv"
    second = tmp_path / "up.csv"
    _write_floquet(first, [_floquet_row(-24.3, 0.998), _floquet_row(-24.25, 0.9997)])
    _write_floquet(second, [_floquet_row(-24.25, 0.9997), _floquet_row(-24.2, 1.0005)])

    branch = load_branch([first, second], True)

    assert branch.drive.tolist() == [-24.3, -24.25, -24.2]


def test_run_writes_the_figure_set_and_a_summary(tmp_path: Path) -> None:
    floquet = tmp_path / "floquet.csv"
    _write_floquet(floquet, [
        _floquet_row(-24.30, 0.99898),
        _floquet_row(-24.25, 0.99974),
        _floquet_row(-24.20, 1.00053),
        _floquet_row(-24.15, 1.00134),
    ])
    torus = tmp_path / "torus.csv"
    with torus.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["drive_dbm", "radius_squared", "omega_a_over_omega_p"]
        )
        writer.writeheader()
        writer.writerows([
            {"drive_dbm": -24.05, "radius_squared": 0.17809,
             "omega_a_over_omega_p": 0.09150053},
            {"drive_dbm": -23.90, "radius_squared": 0.31612,
             "omega_a_over_omega_p": 0.09121317},
        ])

    outdir = tmp_path / "figs"
    payload = run(parse_args([
        "--floquet-csv", str(floquet),
        "--torus-csv", str(torus),
        "--pump-ghz", "7.9",
        "--outdir", str(outdir),
    ]))

    assert payload["summary"]["hill_crossing_dbm"] == pytest.approx(-24.2335, abs=1e-3)
    assert payload["summary"]["torus_p_c_dbm"] == pytest.approx(-24.2435, abs=1e-3)
    assert payload["summary"]["onset_agreement_db"] == pytest.approx(0.010, abs=2e-3)
    for stem in ("fig1_onset_agreement", "fig2_root_locus", "fig3_frequency_twist"):
        assert (outdir / f"{stem}.png").is_file()
    assert (outdir / "onset_summary.json").is_file()


def test_a_stable_column_still_produces_figures_and_names_the_omission(
    tmp_path: Path,
) -> None:
    floquet = tmp_path / "floquet.csv"
    _write_floquet(floquet, [_floquet_row(-24.3, 0.93), _floquet_row(-24.2, 0.94)])
    outdir = tmp_path / "figs"

    payload = run(parse_args([
        "--floquet-csv", str(floquet),
        "--pump-ghz", "7.9",
        "--outdir", str(outdir),
    ]))

    assert "hill_crossing_dbm" not in payload["summary"]
    assert any("crossing" in note for note in payload["skipped"])
    assert (outdir / "fig1_onset_agreement.png").is_file()
