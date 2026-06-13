from __future__ import annotations

import numpy as np

from twpa_solver.model.nonlinearities import JosephsonNonlinearity


def test_josephson_derivative_matches_finite_difference() -> None:
    law = JosephsonNonlinearity(np.asarray([8e-6]))
    psi = np.asarray([1e-18])
    eps = 1e-21
    fd = (law.current(psi + eps) - law.current(psi - eps)) / (2.0 * eps)
    np.testing.assert_allclose(law.derivative(psi), fd, rtol=1e-6)
