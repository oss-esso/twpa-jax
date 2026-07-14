from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from twpa_solver.pump.basis import load_pump_basis_from_solution
from twpa_solver.pump.io import write_results


def _write(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    modes = np.array([1, 3, 5, 7, 9], dtype=np.int64)
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((5, 300)) + 1j * rng.standard_normal((5, 300))) * 1e-16
    metadata = {
        "pump_modes": modes.tolist(),
        "omega_p": 2.0 * np.pi * 8e9,
        "pump_mode_policy": "positive_odd_jc",
    }
    write_results(tmp_path, X, reports=[], solution_summary={}, metadata=metadata)
    return tmp_path / "pump_solution.npz", X, modes


def test_solution_stored_float32(tmp_path: Path) -> None:
    sol_path, _, _ = _write(tmp_path)
    with np.load(sol_path) as z:
        assert z["X_real"].dtype == np.float32
        assert z["X_imag"].dtype == np.float32


def test_solution_is_compressed(tmp_path: Path) -> None:
    sol_path, _, _ = _write(tmp_path)
    with zipfile.ZipFile(sol_path) as zf:
        # savez_compressed uses DEFLATE; savez uses STORED.
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in zf.infolist())


def test_roundtrip_loads_complex128_within_float32_tol(tmp_path: Path) -> None:
    _, X, modes = _write(tmp_path)
    loaded, basis = load_pump_basis_from_solution(tmp_path)
    # float32 on disk must not leak complex64 into downstream scipy solves.
    assert loaded.dtype == np.complex128
    assert basis.k.tolist() == modes.tolist()
    rel_err = np.abs(loaded - X).max() / np.abs(X).max()
    assert rel_err < 1e-6  # float32 relative precision ~1e-7
