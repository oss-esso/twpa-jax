import numpy as np
import pytest

from twpa_solver.builders.cpw_coupler import CPWConformalCoupler, optimize_cpw_coupler
from twpa_solver.builders.ipm import (
    IPMParams,
    calculate_discrete_params,
    make_coupler_discrete,
)


def test_explicit_v1_geometry_uses_the_conformal_path() -> None:
    width = 39.897
    gap_to_ground = 10.5973385055
    gap_between_lines = 44.762
    length = 3787.7
    explicit = CPWConformalCoupler(
        [gap_to_ground, gap_between_lines, gap_to_ground],
        [width, width],
        length,
    ).parameters()

    assert explicit["coupling_db"] == pytest.approx(-13.0049342233, abs=1e-6)
    assert explicit["Z_eff"] == pytest.approx(48.6685048868, abs=1e-6)


def test_maxwell_fallback_v1_matches_legacy_discrete_parameters() -> None:
    coupler = CPWConformalCoupler(
        [10.5973385055, 44.762, 10.5973385055],
        [39.897, 39.897],
        3787.7,
    )
    coupler.C, coupler.L = coupler._fallback_matrices()
    fallback = coupler.parameters()
    legacy = calculate_discrete_params(
        39.897, 10.5973385055, 44.762, 3787.7
    ).geometry

    assert fallback["coupling_db"] == pytest.approx(legacy.k_db, abs=1e-6)
    assert fallback["Z_eff"] == pytest.approx(legacy.z_input_ohm, abs=1e-6)


def test_explicit_v3_geometry_uses_three_conductor_conformal_path() -> None:
    coupler = CPWConformalCoupler(
        [5.5, 5.0, 5.0, 5.5],
        [9.186, 15.0, 9.186],
        2738.2160926784595,
    )
    parameters = coupler.parameters()

    assert parameters["coupling_db"] == pytest.approx(-28.2496555093, abs=1e-6)
    assert parameters["Z_eff"] == pytest.approx(49.3393827325, abs=1e-6)
    assert np.all(np.isfinite(coupler.C))
    assert np.all(np.linalg.eigvalsh(coupler.C) > 0.0)


def test_v3_conformal_root_succeeds_for_all_three_conductors() -> None:
    coupler = CPWConformalCoupler(
        [5.5, 5.0, 5.0, 5.5],
        [9.186, 15.0, 9.186],
        1000.0,
    )
    a, b = coupler._branch_points()

    for metal in range(3):
        roots = coupler._find_c(a, b, metal)
        assert len(roots) == 2


def test_auto_selects_two_line_for_stronger_coupling() -> None:
    result = optimize_cpw_coupler(-14.0, 8.0e9)
    assert result.model == "two_line"
    assert len(result.widths_um) == 2
    assert result.length_um > 0.0


def test_auto_selects_three_line_for_weaker_coupling() -> None:
    result = optimize_cpw_coupler(-25.0, 8.0e9)
    assert result.model == "three_line"
    assert len(result.widths_um) == 3
    assert len(result.gaps_um) == 4
    assert result.length_um > 0.0


def test_auto_uses_two_line_at_minus_nineteen_db() -> None:
    result = optimize_cpw_coupler(-19.0, 8.0e9)
    assert result.model == "two_line"


def test_auto_mode_returns_the_normal_distributed_coupler_ir() -> None:
    coupler = make_coupler_discrete(
        IPMParams(coupling_dB=-25.0, coupler_freq_hz=8.0e9), "auto"
    )
    assert coupler.geometry.model == "three_line"
    assert coupler.N_coupled > 0
    assert coupler.L_cell > 0.0
    assert coupler.C_gnd_cell > 0.0
