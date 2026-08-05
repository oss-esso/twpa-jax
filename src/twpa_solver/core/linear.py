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

LOSSY_DEFAULT_LOSS_MODEL = "conductance_abs_omega"


def default_loss_model_for(circuit: CircuitMatrices) -> str:
    """Return the canonical convention for this circuit's capacitance."""
    return LOSSY_DEFAULT_LOSS_MODEL if circuit.has_loss else "current_complex_c"


def require_real(matrix_or_array: object, *, what: str) -> np.ndarray:
    """Return real data, refusing to silently discard a loss contribution."""
    if sp.issparse(matrix_or_array):
        data = matrix_or_array.data
        if np.iscomplexobj(data) and np.any(data.imag != 0.0):
            raise ValueError(
                f"{what} has a non-zero imaginary part; largest discarded magnitude "
                f"would be {float(np.max(np.abs(data.imag))):.6g}"
            )
        return matrix_or_array.real
    value = np.asarray(matrix_or_array)
    if np.iscomplexobj(value) and np.any(value.imag != 0.0):
        raise ValueError(
            f"{what} has a non-zero imaginary part; largest discarded magnitude "
            f"would be {float(np.max(np.abs(value.imag))):.6g}"
        )
    return value.real if np.iscomplexobj(value) else value


def dynamic_block_from_parts(
    C: sp.spmatrix,
    G: sp.spmatrix,
    K: sp.spmatrix,
    omega: float,
    *,
    loss_model: str = "current_complex_c",
    extra_K: sp.spmatrix | None = None,
) -> sp.csr_matrix:
    """Build one dynamic block from matrix parts and a loss convention."""
    Cfull = C.astype(np.complex128).tocsr()
    Gfull = G.astype(np.complex128).tocsr()
    Kfull = K.astype(np.complex128).tocsr()
    if extra_K is not None:
        Kfull = Kfull + extra_K.astype(np.complex128).tocsr()
    Cre = Cfull.real.astype(np.complex128).tocsr()
    Cim = Cfull.imag.astype(np.complex128).tocsr()
    if loss_model == "current_complex_c":
        selected_C, selected_G = Cfull, Gfull
    elif loss_model == "real_capacitance":
        selected_C, selected_G = Cre, Gfull
    elif loss_model == "conjugate_complex_c":
        selected_C, selected_G = Cfull.conjugate().tocsr(), Gfull
    elif loss_model == "complex_c_sign_omega":
        selected_C = (Cre + 1j * (1.0 if omega >= 0.0 else -1.0) * Cim).tocsr()
        selected_G = Gfull
    elif loss_model == "conductance_signed_omega":
        selected_C, selected_G = Cre, Gfull - omega * Cim
    elif loss_model == "conductance_abs_omega":
        selected_C, selected_G = Cre, Gfull - abs(omega) * Cim
    elif loss_model == "conductance_abs_omega_opposite":
        selected_C, selected_G = Cre, Gfull + abs(omega) * Cim
    else:
        raise ValueError(f"unknown loss_model={loss_model!r}")
    return (Kfull - omega * omega * selected_C + 1j * omega * selected_G).tocsr()


def dynamic_block(
    circuit: CircuitMatrices,
    omega: float,
    *,
    loss_model: str = "current_complex_c",
    extra_K: sp.spmatrix | None = None,
) -> sp.csr_matrix:
    """Build D(w) = K - w^2 C + i w G, with optional loss convention."""
    return dynamic_block_from_parts(
        circuit.C, circuit.G, circuit.K, omega,
        loss_model=loss_model, extra_K=extra_K,
    )


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


def port_waves(
    voltage: complex, current: complex, z0_ohm: float = 50.0
) -> tuple[complex, complex]:
    """Return incident and outgoing power-wave amplitudes ``(a, b)``."""
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    scale = 2.0 * np.sqrt(z0_ohm)
    return (voltage + z0_ohm * current) / scale, (voltage - z0_ohm * current) / scale


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
