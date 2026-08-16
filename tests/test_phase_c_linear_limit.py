from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from scripts.chaos.run_guarcello_jc_phase5 import (
    JcDevice,
    finite_linear_inductance_device,
)


def test_finite_linear_inductance_variant_preserves_junction_stiffness():
    device = JcDevice(
        name="fixture", n_nodes=2, C=sp.eye(2, format="csr"),
        G=sp.csr_matrix((2, 2)), K=sp.eye(2, format="csr"),
        Bphi=sp.csr_matrix(np.asarray([[1.0], [-1.0]])),
        Ic=np.asarray([2.0]), Lj=np.asarray([0.5]), Cj=np.asarray([0.0]),
        Cg=np.asarray([0.0, 0.0]), phi0=1.0, natural_bandwidth=1,
        rcm_bandwidth=1, selected_bandwidth=1, selected_ordering="natural",
        permutation=np.arange(2), ic_uniform=False, pump_node=0,
        pump_output_node=1, signal_node=0, output_node=1,
        has_parallel_geometric_inductor=False, implicit_linear_stiffness=True,
        dc_bias_current_a=0.0, phi_dc_rad=0.0,
    )
    variant = finite_linear_inductance_device(device)
    assert np.allclose(variant.Ic, 0.0)
    assert np.allclose(variant.K.toarray(), [[3.0, -2.0], [-2.0, 3.0]])


def test_jc_linear_gate_has_first_order_timestep_convergence():
    from scripts.chaos.phaseC_e_validation import measure_jc_finite_linear_gate

    result = measure_jc_finite_linear_gate(
        "jc_jtwpa", dt_norms=(0.01, 0.005, 0.0025)
    )
    errors = [
        result["dt_norm_0p01"]["relative_error"],
        result["dt_norm_0p005"]["relative_error"],
        result["dt_norm_0p0025"]["relative_error"],
    ]
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    assert all(1.7 <= ratio <= 2.3 for ratio in ratios), ratios
