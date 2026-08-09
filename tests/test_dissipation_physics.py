"""Physics gates for dielectric dissipation.

These are the checks that justify trusting a lossy solve. They are analytic and
self-contained -- no second simulator is involved, per the standing rule that
JosephsonCircuits.jl is a drift check and not a reference.

Every gate here has been verified to fail under the mutation named in its
docstring. A gate that cannot be made to fail is not measuring anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from twpa_solver.builders.ipm import Element, LossSpec, build_matrices
from twpa_solver.core import CircuitMatrices
from twpa_solver.core.linear import solve_linear_scattering
from twpa_solver.pump.problem import (
    FullPumpProblem,
    HarmonicGrid,
    JosephsonBranchArray,
)

FREQ_HZ = 6.0e9
N_CELLS = 200
TAN_DELTAS = (0.0, 1e-5, 1e-4, 1e-3)


def lc_ladder(n_cells: int, tan_delta: float) -> CircuitMatrices:
    """A 50 ohm terminated LC ladder built through the production stamper.

    Deliberately not a hand-written matrix: the point is to exercise the same
    ``build_matrices`` loss path the IPM builder uses.
    """
    elements: list[Element] = []
    for i in range(1, n_cells + 1):
        elements.append(Element(f"L{i}", i, i + 1, 4.0e-11, "linear_inductor", "tl_l"))
        elements.append(Element(f"C{i}", i, 0, 4.0e-14, "capacitor", "jtl_cg"))
    elements.append(
        Element(f"C{n_cells + 1}", n_cells + 1, 0, 4.0e-14, "capacitor", "jtl_cg")
    )
    elements.append(Element("Rin", 1, 0, 50.0, "resistor", ""))
    elements.append(Element("Rout", n_cells + 1, 0, 50.0, "resistor", ""))
    elements.append(Element("P1", 1, 0, 1, "port", ""))
    elements.append(Element("P2", n_cells + 1, 0, 2, "port", ""))
    m = build_matrices(elements, LossSpec(default=tan_delta))
    return CircuitMatrices(
        C=m["C"], G=m["G"], K=m["K"], Bphi=m["Bphi"], Ic=m["Ic"],
        port_to_index={int(p): int(i) for p, i in m["port_vectors"].items()},
    )


def s21_magnitude(tan_delta: float, *, loss_model: str) -> float:
    circuit = lc_ladder(N_CELLS, tan_delta)
    result = solve_linear_scattering(
        circuit, frequency_hz=FREQ_HZ, source_port=1, out_port=2,
        loss_model=loss_model,
    )
    return float(abs(result.s))


# --------------------------------------------------------------------------
# Gate 1 -- attenuation obeys the exp(-alpha L) law over three decades
# --------------------------------------------------------------------------

def test_gate1_excess_attenuation_is_linear_in_tan_delta() -> None:
    """ln|S21| must scale as 1:10:100 across tan_delta = 1e-5, 1e-4, 1e-3.

    Small-loss dielectric attenuation is alpha ~ (omega/2) tan_delta sqrt(LC),
    linear in tan_delta, so the excess attenuation over the lossless reference
    must gain a factor of ten per decade. This is the strongest self-contained
    check available for a lossy line.

    Mutation: swap the sign in dynamic_block's ``conductance_abs_omega`` branch
    (i.e. use ``conductance_abs_omega_opposite``) -- see the mutation test below.
    """
    magnitudes = [
        s21_magnitude(td, loss_model="conductance_abs_omega") for td in TAN_DELTAS
    ]
    assert all(m < 1.0 for m in magnitudes), f"line is not passive: {magnitudes}"

    excess = [np.log(magnitudes[0] / m) for m in magnitudes[1:]]
    assert all(e > 0.0 for e in excess), "loss did not attenuate"

    for index in range(len(excess) - 1):
        ratio = excess[index + 1] / excess[index]
        assert abs(ratio - 10.0) / 10.0 < 0.02, (
            f"decade {index} attenuation ratio {ratio:.4f} is not 10 within 2%"
        )


def test_gate1_mutation_opposite_sign_produces_gain() -> None:
    """The gate has teeth: the opposite-sign convention must amplify.

    If this ever passes with |S21| <= lossless, the sign of the loss stamp is no
    longer observable and Gate 1 is vacuous.
    """
    lossless = s21_magnitude(0.0, loss_model="conductance_abs_omega")
    wrong_sign = s21_magnitude(1e-3, loss_model="conductance_abs_omega_opposite")
    assert wrong_sign > lossless, (
        "opposite-sign loss convention did not produce gain; Gate 1 is vacuous"
    )


# --------------------------------------------------------------------------
# Gate 2 -- passivity, and the negative-sideband sign trap
# --------------------------------------------------------------------------

def test_gate2_passive_line_is_monotone_in_tan_delta() -> None:
    """|S21| strictly decreasing in tan_delta at every sampled frequency."""
    for frequency in (4.0e9, 6.0e9, 8.0e9):
        magnitudes = []
        for td in TAN_DELTAS:
            circuit = lc_ladder(N_CELLS, td)
            result = solve_linear_scattering(
                circuit, frequency_hz=frequency, source_port=1, out_port=2,
                loss_model="conductance_abs_omega",
            )
            magnitudes.append(abs(result.s))
        assert magnitudes == sorted(magnitudes, reverse=True), (
            f"|S21| not monotone in tan_delta at {frequency / 1e9:g} GHz: {magnitudes}"
        )


def test_gate2_negative_frequency_stays_dissipative() -> None:
    """At omega < 0 the loss must still dissipate.

    This is the trap that made ``current_complex_c`` unusable for lossy circuits:
    with a complex C the loss term flips sign with omega^2 -> the lower Floquet
    sidebands would be amplified rather than attenuated.
    """
    circuit = lc_ladder(20, 1e-3)
    for model, expect_symmetric in (
        ("conductance_abs_omega", True),
        ("current_complex_c", False),
    ):
        from twpa_solver.core.linear import dynamic_block

        positive = dynamic_block(circuit, 2.0 * np.pi * FREQ_HZ, loss_model=model)
        negative = dynamic_block(circuit, -2.0 * np.pi * FREQ_HZ, loss_model=model)
        symmetric = np.allclose(
            negative.toarray(), positive.toarray().conj(), rtol=0, atol=0
        )
        assert symmetric is expect_symmetric, (
            f"loss_model={model!r} conjugate symmetry was {symmetric}, "
            f"expected {expect_symmetric}"
        )


# --------------------------------------------------------------------------
# Gate 3 -- the lossless path is bit-identical
# --------------------------------------------------------------------------

def test_gate3_zero_tan_delta_is_bitwise_identical() -> None:
    """tan_delta = 0 must reproduce the lossless build exactly, not approximately.

    This is what protects every published result from the refactor.
    """
    elements = [
        Element("L1", 1, 2, 4.0e-11, "linear_inductor", "tl_l"),
        Element("C1", 1, 0, 4.0e-14, "capacitor", "jtl_cg"),
        Element("Cj1", 1, 2, 1.4e-13, "capacitor", "jj_cj"),
        Element("Cc1", 1, 2, 3.0e-16, "coupling_capacitor", "coupling_cap"),
    ]
    lossless = build_matrices(elements, LossSpec(default=0.0))
    assert not np.iscomplexobj(lossless["C"].data), (
        "a zero loss tangent must leave C in its real dtype"
    )

    lossy = build_matrices(elements, LossSpec(default=1e-4))
    assert np.array_equal(lossy["C"].data.real, lossless["C"].data)
    assert np.array_equal(lossy["C"].indices, lossless["C"].indices)


# --------------------------------------------------------------------------
# Gate 4 -- the pump time residual rewrite is faithful
# --------------------------------------------------------------------------

def test_gate4_time_residual_matches_stamped_form_for_real_c() -> None:
    """The synthesized residual must equal the old stamped one for a real C.

    The rewrite exists so a frequency-domain loss is not silently dropped. For a
    lossless circuit the two forms are algebraically identical, so any deviation
    here is a bug introduced by the rewrite rather than a physical difference.
    """
    rng = np.random.default_rng(0)
    n, nb = 6, 3
    C = sp.random(n, n, density=0.5, random_state=1, format="csr")
    C = (C + C.T).tocsr() * 1e-13
    G = sp.eye(n, format="csr") * 1e-3
    K = sp.eye(n, format="csr") * 1e9
    Bphi = sp.random(n, nb, density=0.6, random_state=2, format="csr")
    grid = HarmonicGrid(modes=np.array([1.0, 3.0]), nt=16, omega=2 * np.pi * 7e9)
    problem = FullPumpProblem(
        C=C, G=G, K=K, Bphi=Bphi,
        branch=JosephsonBranchArray(Ic=np.full(nb, 2.5e-6), phi0=3.29e-16),
        grid=grid, pump_node_index=0, pump_current_a=1.0e-6, source_mode=1,
    )
    X = (rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))) * 1e-17

    synthesized = problem.time_residual(X, 1.0)

    x_t = grid.synthesize(X)
    dx_t = grid.synthesize_derivative(X, order=1)
    ddx_t = grid.synthesize_derivative(X, order=2)
    stamped = (problem.C @ ddx_t.T).T + (problem.G @ dx_t.T).T + (problem.K @ x_t.T).T
    stamped = stamped + problem.nonlinear_current_time(X) - problem.source_time(1.0)
    stamped = np.asarray(stamped, dtype=float)

    relative = np.linalg.norm(synthesized - stamped) / np.linalg.norm(stamped)
    assert relative < 1e-12, f"time_residual rewrite is not faithful: {relative:.3e}"


def test_gate4_time_residual_sees_loss_that_the_stamped_form_drops() -> None:
    """On a lossy circuit the two forms must disagree.

    The old stamped form cast a complex product to float, discarding exactly the
    dissipative term. If this test ever passes with the two forms agreeing, the
    loss has stopped reaching the pump residual.
    """
    n, nb = 4, 2
    C = sp.eye(n, format="csr").astype(np.complex128) * (1.0e-13 * (1 - 1j * 1e-3))
    G = sp.eye(n, format="csr") * 1e-3
    K = sp.eye(n, format="csr") * 1e9
    Bphi = sp.random(n, nb, density=0.8, random_state=3, format="csr")
    grid = HarmonicGrid(modes=np.array([1.0]), nt=8, omega=2 * np.pi * 7e9)
    problem = FullPumpProblem(
        C=C, G=G, K=K, Bphi=Bphi,
        branch=JosephsonBranchArray(Ic=np.full(nb, 2.5e-6), phi0=3.29e-16),
        grid=grid, pump_node_index=0, pump_current_a=1.0e-6, source_mode=1,
        loss_model="conductance_abs_omega",
    )
    X = np.full((1, n), 1e-17 + 0j)

    synthesized = problem.time_residual(X, 1.0)

    ddx_t = grid.synthesize_derivative(X, order=2)
    dx_t = grid.synthesize_derivative(X, order=1)
    x_t = grid.synthesize(X)
    with pytest.warns(np.exceptions.ComplexWarning):
        stamped = np.asarray(
            (problem.C @ ddx_t.T).T + (problem.G @ dx_t.T).T + (problem.K @ x_t.T).T,
            dtype=float,
        )
    stamped = stamped + problem.nonlinear_current_time(X) - problem.source_time(1.0)

    # The discarded term is the dissipative one, so the two forms separate at
    # roughly the loss-tangent scale. A default-tolerance np.allclose would not
    # resolve that, which is why this is quantitative.
    relative = np.linalg.norm(synthesized - stamped) / np.linalg.norm(stamped)
    assert relative > 1e-7, (
        f"the loss term is not reaching the pump time residual (rel={relative:.3e})"
    )
