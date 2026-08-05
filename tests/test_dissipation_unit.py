from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from twpa_solver.builders.ipm import (
    Element,
    IPMParams,
    LossSpec,
    build_component_plan,
    build_matrices,
)
from twpa_solver.builders.scatter import ScatterSpec
from twpa_solver.core import CircuitMatrices, default_loss_model_for, save_circuit
from twpa_solver.core.linear import (
    LOSS_MODELS,
    dynamic_block,
    dynamic_block_from_parts,
    require_real,
)
from twpa_solver.pump.problem import (
    FullPumpProblem,
    HarmonicGrid,
    JosephsonBranchArray,
)


def _circuit(C: sp.spmatrix) -> CircuitMatrices:
    return CircuitMatrices(
        C=C,
        G=sp.eye(2, format="csr") * 1e-3,
        K=sp.eye(2, format="csr"),
        Bphi=sp.csr_matrix((2, 1)),
        Ic=np.array([1.0]),
    )


def test_loss_model_resolver_and_conjugate_symmetry() -> None:
    circuit = _circuit(sp.csr_matrix(np.diag([1.0 - 1j * 1e-3, 2.0])))
    assert circuit.has_loss
    assert default_loss_model_for(circuit) == "conductance_abs_omega"
    for model in LOSS_MODELS:
        positive = dynamic_block(circuit, 3.0, loss_model=model).toarray()
        negative = dynamic_block(circuit, -3.0, loss_model=model).toarray()
        if model == "conductance_abs_omega":
            assert np.array_equal(negative, positive.conj())
        elif model == "current_complex_c":
            assert not np.array_equal(negative, positive.conj())


def test_require_real_rejects_loss() -> None:
    with pytest.raises(ValueError, match="imaginary part"):
        require_real(np.array([1.0 + 2.0j]), what="test quantity")
    assert np.array_equal(require_real(np.array([1.0 + 0.0j]), what="real"), [1.0])


def test_dynamic_block_entry_points_are_identical() -> None:
    circuit = _circuit(sp.eye(2, format="csr"))
    for model in LOSS_MODELS:
        expected = dynamic_block(circuit, 4.0, loss_model=model).toarray()
        actual = dynamic_block_from_parts(
            circuit.C, circuit.G, circuit.K, 4.0, loss_model=model
        ).toarray()
        assert np.array_equal(expected, actual)


def test_builder_excludes_junction_capacitance_from_loss() -> None:
    elements = [
        Element("ground", 1, 0, 2.0, "capacitor", "jtl_cg"),
        Element("junction", 1, 2, 3.0, "capacitor", "jj_cj"),
    ]
    matrices = build_matrices(elements, LossSpec(default=1e-3))
    assert np.any(matrices["C"].data.imag != 0.0)
    junction_only = build_matrices(
        [Element("junction", 1, 2, 3.0, "capacitor", "jj_cj")],
        LossSpec(default=1e-3),
    )
    assert np.count_nonzero(junction_only["C"].data.imag) == 0
    assert np.all(matrices["C"].data.imag <= 0.0)


def test_plasma_locked_scatter_preserves_product_and_lj_stream() -> None:
    params = IPMParams(array_length=4, num_rows=1)
    lj = ScatterSpec(sigma=0.03)
    independent = build_component_plan(params, lj_scatter=lj, seed=7)
    locked = build_component_plan(
        params,
        lj_scatter=lj,
        cj_scatter=ScatterSpec(mode="plasma_locked"),
        seed=7,
    )
    assert np.array_equal(independent.lj, locked.lj)
    assert np.allclose(locked.lj * locked.cj, params.Lj * params.Cj, rtol=0, atol=1e-28)
    with pytest.raises(ValueError, match="sigma"):
        build_component_plan(
            params,
            lj_scatter=lj,
            cj_scatter=ScatterSpec(sigma=1e-3, mode="plasma_locked"),
        )


def test_complex_circuit_round_trip_preserves_loss(tmp_path) -> None:
    circuit = _circuit(sp.csr_matrix(np.diag([1.0 - 1j * 1e-3, 2.0])))
    save_circuit(circuit, tmp_path)
    from twpa_solver.core import load_circuit

    loaded = load_circuit(tmp_path)
    assert loaded.has_loss
    assert np.array_equal(loaded.C.data, circuit.C.data)
