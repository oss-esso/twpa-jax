from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from scripts.debug_n16_hb_td_departure import mode_audit


class _Branch:
    critical_current = np.ones(1)


def test_mode_audit_reports_finite_fastest_mode():
    class _Circuit:
        K = sp.csr_matrix([[1.0]])
        Bphi = sp.csr_matrix([[1.0]])
        C = sp.csr_matrix([[1.0]])

    class _System:
        circuit = _Circuit()
        branch = _Branch()
        phi0 = 1.0
        omega = 2.0

    result = mode_audit(_System())

    assert np.isfinite(result["omega_max_over_omega_p"])
    assert result["omega_max_over_omega_p"] > 0.0
    assert result["omega_max_dt_at_step_0p01"] > 0.0
