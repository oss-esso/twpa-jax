"""Tests for the pump-mode basis policy layer (experiments/pump_basis.py).

These define the JC-compatible pump-mode contract used by exp08 (pump solve)
and exp09 (gain): policy resolution, the positive-phasor reconstruction
metadata, warm-start promotion across bases, and round-trip persistence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXPERIMENTS))

import pump_basis as pb  # noqa: E402


def test_positive_odd_modes_matches_jc_jtwpa_list() -> None:
    # JC introspection: sol.nonlinear.modes = [1,3,5,...,19] for K=10.
    assert pb.positive_odd_modes(10) == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


def test_parse_explicit_modes() -> None:
    assert pb.parse_explicit_modes("1, 3,5 ,7") == [1, 3, 5, 7]
    with pytest.raises(ValueError):
        pb.parse_explicit_modes("")


def test_dense_real_preserves_legacy_behavior() -> None:
    basis = pb.resolve_pump_basis(
        policy="dense_real",
        omega_p=1.0,
        harmonics=4,
        mode_count=None,
        explicit_modes=None,
    )
    assert basis.modes == [1, 2, 3, 4]
    assert basis.real_reconstruction_factor == 2
    assert basis.phase_convention == "exp_plus_i_k_omega_t"


def test_positive_odd_jc_uses_mode_count() -> None:
    basis = pb.resolve_pump_basis(
        policy="positive_odd_jc",
        omega_p=2.0,
        harmonics=3,
        mode_count=10,
        explicit_modes=None,
    )
    assert basis.modes == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    md = basis.to_metadata()
    assert md["pump_modes"] == basis.modes
    assert md["pump_basis"] == "positive_phasor"
    assert md["real_reconstruction_factor"] == 2
    assert md["phase_convention"] == "exp_plus_i_k_omega_t"
    assert md["pump_mode_policy"] == "positive_odd_jc"


def test_positive_phasor_explicit_requires_modes() -> None:
    with pytest.raises(ValueError):
        pb.resolve_pump_basis(
            policy="positive_phasor_explicit",
            omega_p=1.0,
            harmonics=3,
            mode_count=None,
            explicit_modes=None,
        )
    basis = pb.resolve_pump_basis(
        policy="positive_phasor_explicit",
        omega_p=1.0,
        harmonics=3,
        mode_count=None,
        explicit_modes="1,3,5,7",
    )
    assert basis.modes == [1, 3, 5, 7]


def test_auto_jc_reads_nmodulationharmonics() -> None:
    design_meta = {
        "metadata": {
            "Nmodulationharmonics": [10],
            "features": {"single_pump": True, "multi_pump": False},
        }
    }
    basis = pb.resolve_pump_basis(
        policy="auto_jc",
        omega_p=1.0,
        harmonics=3,
        mode_count=None,
        explicit_modes=None,
        design_meta=design_meta,
    )
    assert basis.modes == pb.positive_odd_modes(10)


def test_auto_jc_uses_dense_for_dc_biased_designs() -> None:
    design_meta = {
        "metadata": {
            "Nmodulationharmonics": [8],
            "pump_sources": [
                {"port": 2, "mode": [0], "current_a": 140.3e-6},
                {"port": 2, "mode": [1], "current_a": 0.7e-6},
            ],
            "features": {
                "single_pump": True,
                "multi_pump": False,
                "needs_dc": True,
            },
        }
    }
    basis = pb.resolve_pump_basis(
        policy="auto_jc",
        omega_p=1.0,
        harmonics=8,
        mode_count=None,
        explicit_modes=None,
        design_meta=design_meta,
    )
    assert basis.modes == [1, 2, 3, 4, 5, 6, 7, 8]


def test_auto_jc_rejects_multi_pump() -> None:
    design_meta = {"metadata": {"features": {"multi_pump": True}}}
    with pytest.raises(ValueError, match="multi-pump"):
        pb.resolve_pump_basis(
            policy="auto_jc",
            omega_p=1.0,
            harmonics=3,
            mode_count=None,
            explicit_modes=None,
            design_meta=design_meta,
        )


def test_auto_jc_rejects_multi_pump_after_saved_summary_nesting() -> None:
    design_meta = {
        "metadata": {
            "nodes": 4,
            "jj_branches": 2,
            "metadata": {
                "Nmodulationharmonics": [8, 8],
                "features": {"multi_pump": True},
            },
        }
    }
    with pytest.raises(ValueError, match="multi-pump"):
        pb.resolve_pump_basis(
            policy="auto_jc",
            omega_p=1.0,
            harmonics=3,
            mode_count=None,
            explicit_modes=None,
            design_meta=design_meta,
        )


def test_pump_basis_validation() -> None:
    with pytest.raises(ValueError):
        pb.PumpBasis(modes=[1, 1, 3], policy="x", omega_p=1.0)
    with pytest.raises(ValueError):
        pb.PumpBasis(modes=[0, 1], policy="x", omega_p=1.0)
    with pytest.raises(ValueError):
        pb.PumpBasis(modes=[3, 5], policy="x", omega_p=1.0, source_mode=1)


def test_promote_solution_to_basis_copies_shared_modes() -> None:
    src = pb.PumpBasis(modes=[1, 3, 5], policy="positive_odd_jc", omega_p=1.0)
    dst = pb.PumpBasis(modes=[1, 3, 5, 7, 9], policy="positive_odd_jc", omega_p=1.0)
    X_src = np.arange(3 * 4, dtype=np.complex128).reshape(3, 4) + 1j
    X_dst = pb.promote_solution_to_basis(X_src, src, dst)

    assert X_dst.shape == (5, 4)
    np.testing.assert_array_equal(X_dst[0], X_src[0])  # mode 1
    np.testing.assert_array_equal(X_dst[1], X_src[1])  # mode 3
    np.testing.assert_array_equal(X_dst[2], X_src[2])  # mode 5
    np.testing.assert_array_equal(X_dst[3], np.zeros(4))  # new mode 7
    np.testing.assert_array_equal(X_dst[4], np.zeros(4))  # new mode 9


def test_load_pump_basis_round_trip(tmp_path: Path) -> None:
    import json

    modes = [1, 3, 5]
    X = np.array([[1 + 1j, 2 - 1j], [0.5j, -0.5], [3, 4j]], dtype=np.complex128)
    np.savez(
        tmp_path / "pump_solution.npz",
        X_real=X.real,
        X_imag=X.imag,
        harmonics=np.asarray(modes, dtype=np.int64),
        pump_modes=np.asarray(modes, dtype=np.int64),
    )
    meta = {
        "metadata": {
            "omega_p": 44.7e9,
            "pump_mode_policy": "positive_odd_jc",
            "pump_source_mode": 1,
            "pump_modes": modes,
        }
    }
    with open(tmp_path / "pump_report.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    X_load, basis = pb.load_pump_basis_from_solution(tmp_path)
    np.testing.assert_allclose(X_load, X)
    assert basis.modes == modes
    assert basis.policy == "positive_odd_jc"
    assert basis.omega_p == pytest.approx(44.7e9)


def test_load_pump_basis_legacy_dense_fallback(tmp_path: Path) -> None:
    # Legacy solution: only a `harmonics` array, no pump_modes / no report.
    X = np.ones((4, 2), dtype=np.complex128)
    np.savez(
        tmp_path / "pump_solution.npz",
        X_real=X.real,
        X_imag=X.imag,
        harmonics=np.arange(1, 5, dtype=np.int64),
    )
    X_load, basis = pb.load_pump_basis_from_solution(tmp_path, fallback_omega_p=10.0)
    assert basis.modes == [1, 2, 3, 4]
    assert basis.omega_p == pytest.approx(10.0)
