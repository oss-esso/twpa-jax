from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from twpa_solver.core.circuit import CircuitMatrices


LOSS_MODELS = (
    "current_complex_c",
    "real_capacitance",
    "conjugate_complex_c",
    "complex_c_sign_omega",
    "conductance_signed_omega",
    "conductance_abs_omega",
    "conductance_abs_omega_opposite",
)


def dynamic_block(
    circuit: CircuitMatrices,
    omega: float,
    *,
    loss_model: str = "current_complex_c",
    extra_K: sp.spmatrix | None = None,
) -> sp.csr_matrix:
    """Build D(w) = K - w^2 C + i w G, with optional loss convention."""
    Cfull = circuit.C.astype(np.complex128).tocsr()
    Gfull = circuit.G.astype(np.complex128).tocsr()
    K = circuit.K.astype(np.complex128).tocsr()

    if extra_K is not None:
        K = K + extra_K.astype(np.complex128).tocsr()

    Cre = Cfull.real.astype(np.complex128).tocsr()
    Cim = Cfull.imag.astype(np.complex128).tocsr()

    if loss_model == "current_complex_c":
        C = Cfull
        G = Gfull
    elif loss_model == "real_capacitance":
        C = Cre
        G = Gfull
    elif loss_model == "conjugate_complex_c":
        C = Cfull.conjugate().astype(np.complex128).tocsr()
        G = Gfull
    elif loss_model == "complex_c_sign_omega":
        sgn = 1.0 if omega >= 0.0 else -1.0
        C = (Cre + 1j * sgn * Cim).astype(np.complex128).tocsr()
        G = Gfull
    elif loss_model == "conductance_signed_omega":
        C = Cre
        G = (Gfull - omega * Cim).astype(np.complex128).tocsr()
    elif loss_model == "conductance_abs_omega":
        C = Cre
        G = (Gfull - abs(omega) * Cim).astype(np.complex128).tocsr()
    elif loss_model == "conductance_abs_omega_opposite":
        C = Cre
        G = (Gfull + abs(omega) * Cim).astype(np.complex128).tocsr()
    else:
        raise ValueError(f"unknown loss_model={loss_model!r}")

    return (K - omega * omega * C + 1j * omega * G).tocsr()


def port_s_from_unit_current_response(
    response_voltage: complex,
    *,
    source_port: int,
    out_port: int,
    z0_ohm: float = 50.0,
) -> complex:
    """Convert unit-current port voltage response into an S-like port quantity."""
    s = 2.0 * response_voltage / z0_ohm
    if int(source_port) == int(out_port):
        s -= 1.0
    return complex(s)


@dataclass
class LinearScatteringResult:
    frequency_hz: float
    source_port: int
    out_port: int
    phi_out: complex
    v_out: complex
    s: complex

    @property
    def s_abs(self) -> float:
        return float(abs(self.s))

    @property
    def s_db(self) -> float:
        return float(20.0 * np.log10(max(abs(self.s), 1e-300)))


def solve_linear_scattering(
    circuit: CircuitMatrices,
    *,
    frequency_hz: float,
    source_port: int,
    out_port: int,
    source_current_a: float = 1.0,
    z0_ohm: float = 50.0,
    loss_model: str = "current_complex_c",
    extra_K: sp.spmatrix | None = None,
) -> LinearScatteringResult:
    """Solve the linear single-frequency response between two ports."""
    if source_port not in circuit.port_to_index:
        raise ValueError(f"source_port={source_port} not in {circuit.port_to_index}")
    if out_port not in circuit.port_to_index:
        raise ValueError(f"out_port={out_port} not in {circuit.port_to_index}")

    omega = 2.0 * np.pi * float(frequency_hz)
    A = dynamic_block(
        circuit,
        omega,
        loss_model=loss_model,
        extra_K=extra_K,
    ).tocsc()

    b = np.zeros(circuit.node_count, dtype=np.complex128)
    b[circuit.port_to_index[int(source_port)]] = source_current_a

    y = spla.spsolve(A, b)

    phi_out = complex(y[circuit.port_to_index[int(out_port)]])
    v_out = complex(1j * omega * phi_out)
    s = port_s_from_unit_current_response(
        v_out / source_current_a,
        source_port=source_port,
        out_port=out_port,
        z0_ohm=z0_ohm,
    )

    return LinearScatteringResult(
        frequency_hz=float(frequency_hz),
        source_port=int(source_port),
        out_port=int(out_port),
        phi_out=phi_out,
        v_out=v_out,
        s=s,
    )
