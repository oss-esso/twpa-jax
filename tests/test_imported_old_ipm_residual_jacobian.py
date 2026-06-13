import numpy as np

from twpa_solver.importers.julia_circuit_json import import_julia_circuit_json
from twpa_solver.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def _tiny_residual(tmp_path):
    imported = import_julia_circuit_json(write_tiny_export(tmp_path / "tiny.json"))
    return PumpAFTResidual(
        imported.model,
        PumpAFTConfig(
            pump_frequency_hz=6e9,
            harmonics=1,
            source_current_peak_a=6.3e-6,
            residual_scale_a=1e-6,
        ),
    )


def test_imported_old_ipm_sparse_jacobian_shape(tmp_path):
    residual = _tiny_residual(tmp_path)
    jac = residual.jacobian_sparse(residual.initial_guess())
    assert jac.shape == (residual.size, residual.size)
    assert jac.nnz > 0


def test_imported_old_ipm_residual_jvp_matches_finite_difference(tmp_path):
    residual = _tiny_residual(tmp_path)
    rng = np.random.default_rng(123)
    x = rng.normal(scale=1e-18, size=residual.size)
    v = rng.normal(size=residual.size)
    v /= np.linalg.norm(v)
    eps = 1e-20
    finite_difference = (residual(x + eps * v) - residual(x - eps * v)) / (2.0 * eps)
    analytic = residual.jvp(x, v)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-3, atol=2e-3)
