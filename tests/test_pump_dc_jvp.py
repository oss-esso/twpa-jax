from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from twpa_solver.pump import hb


def test_spectral_real_coupled_jvp_matches_aft_with_dynamic_dc() -> None:
    grid = hb.HarmonicGrid(
        modes=np.asarray([0, 1, 2]),
        nt=16,
        omega=2.0 * math.pi,
    )
    problem = hb.FullPumpProblem(
        C=sp.csr_matrix([[1.0 + 0.0j]]),
        G=sp.csr_matrix([[0.02 + 0.0j]]),
        K=sp.csr_matrix([[1.0 + 0.0j]]),
        Bphi=sp.csr_matrix([[1.0]]),
        branch=hb.JosephsonBranchArray(
            Ic=np.asarray([1.0]),
            phi0=1.0,
        ),
        grid=grid,
        pump_node_index=0,
        pump_current_a=0.3,
        dc_branch_flux=np.asarray([0.35]),
    )
    X = np.asarray([[0.04], [0.08 + 0.02j], [0.01 - 0.03j]], dtype=np.complex128)
    V = np.asarray([[0.03 + 0.07j], [0.02 - 0.01j], [-0.04 + 0.05j]], dtype=np.complex128)

    aft = problem.jvp_coeffs(X, V)
    spectral = problem.spectral_tangent_state(problem.tangent_state(X))
    coupled = problem.jvp_coeffs_with_spectral_tangent(V, spectral)
    matrix = problem.real_coupled_jacobian(spectral)

    np.testing.assert_allclose(
        hb.pack_complex(coupled),
        matrix @ hb.pack_complex(V),
        rtol=2e-11,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        hb.pack_complex(aft),
        hb.pack_complex(coupled),
        rtol=2e-11,
        atol=2e-12,
    )
