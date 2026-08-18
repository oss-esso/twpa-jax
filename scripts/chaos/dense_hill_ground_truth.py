"""Run the small dense ground-truth gate for Hill-root continuation.

The reference is a deliberately truncated two-node circuit with three nearly
degenerate sidebands.  It has the same conversion-matrix assembly path as the
production solver, while its six-by-six matrix is small enough for a complete
``numpy.linalg.eig`` spectrum at every ladder point.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from twpa_solver.core.circuit import CircuitMatrices
from twpa_solver.signal.floquet import assemble_conversion_matrix
from twpa_solver.signal.stability import refine_singular_omega
import twpa_solver.signal.stability as stability


def _reference_circuit() -> CircuitMatrices:
    omega_0 = 2.0 * math.pi * 1.0e9
    capacitance = sp.identity(2, format="csr", dtype=np.complex128)
    stiffness = sp.diags(
        [omega_0**2, (omega_0 + 3.0) ** 2], format="csr"
    )
    return CircuitMatrices(
        C=capacitance,
        G=sp.csr_matrix((2, 2), dtype=np.complex128),
        K=stiffness,
        Bphi=sp.csr_matrix((2, 0), dtype=np.float64),
        Ic=np.empty(0, dtype=float),
    )


def _assemble_factory(
    circuit: CircuitMatrices, amplitude: float
) -> tuple[Callable[[complex], sp.csc_matrix], float, list[int]]:
    omega_0 = 2.0 * math.pi * 1.0e9
    omega_p = 2.0 * math.pi * 1.0e-9
    ms = [0, 1, 2]
    shift = 2.0 * omega_0 * amplitude * 0.1
    khat = {0: sp.diags([shift, 0.0], format="csr", dtype=np.complex128)}

    def assemble(omega: complex) -> sp.csc_matrix:
        return assemble_conversion_matrix(
            circuit=circuit,
            khat=khat,
            omega_s=omega,
            omega_p=omega_p,
            ms=ms,
            loss_model="current_complex_c",
        )

    return assemble, omega_p, ms


def _run_ladder(amplitudes: list[float], seeded: bool) -> tuple[list[float], list[int]]:
    circuit = _reference_circuit()
    target_vector = np.zeros(6, dtype=np.complex128)
    target_vector[0] = 1.0
    original_eigs = stability.spla.eigs
    blind_calls = 0

    def choose_cluster_member(
        matrix: sp.spmatrix, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal blind_calls
        vector = kwargs.get("v0")
        if vector is None:
            index = 1 + blind_calls % 3
            blind_calls += 1
        else:
            vector_array = np.asarray(vector).reshape(-1)
            index = int(np.argmax(np.abs(vector_array)))
        diagonal = np.asarray(matrix.diagonal(), dtype=np.complex128)
        eigenvector = np.zeros((6, 1), dtype=np.complex128)
        eigenvector[index, 0] = 1.0
        return np.array([diagonal[index]]), eigenvector

    stability.spla.eigs = choose_cluster_member
    try:
        overlaps: list[float] = []
        indices: list[int] = []
        seed = 1.0
        mode = target_vector if seeded else None
        for amplitude in amplitudes:
            assemble, omega_p, _ = _assemble_factory(circuit, amplitude)
            result = refine_singular_omega(
                assemble,
                seed * 2.0 * math.pi * 1.0e9 + 1.0e-2j,
                seed * 2.0 * math.pi * 1.0e9 + 2.0e-2j,
                max_iters=20,
                tol=1.0e-12,
                v0=mode,
            )
            dense_values, dense_vectors = np.linalg.eig(
                assemble(result.omega).toarray()
            )
            order = np.argsort(np.abs(dense_values))
            tracked = result.mode_vector / np.linalg.norm(result.mode_vector)
            dense_modes = dense_vectors[:, order]
            dense_modes /= np.linalg.norm(dense_modes, axis=0, keepdims=True)
            overlap_values = np.abs(np.conj(tracked) @ dense_modes)
            match = int(np.argmax(overlap_values))
            overlaps.append(float(overlap_values[match]))
            indices.append(match)
            if not result.converged:
                raise RuntimeError(
                    f"reference refinement failed at amplitude {amplitude}: "
                    f"iterations={result.iterations} residual={result.residual}"
                )
            seed = float(result.signal_ghz.real)
            mode = result.mode_vector if seeded else None
        return overlaps, indices
    finally:
        stability.spla.eigs = original_eigs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amplitudes",
        default="0.10,0.20,0.30,0.40,0.50,0.60",
    )
    args = parser.parse_args()
    amplitudes = [float(item) for item in args.amplitudes.split(",")]
    circuit = _reference_circuit()
    dimension = circuit.node_count * 3
    print(
        json.dumps(
            {
                "reference": "deliberately_truncated_two_node_circuit",
                "node_count": circuit.node_count,
                "sideband_count": 3,
                "conversion_dimension": dimension,
                "dense_bytes": dimension * dimension * 16,
            },
            sort_keys=True,
        )
    )
    seeded_overlap, seeded_index = _run_ladder(amplitudes, seeded=True)
    blind_overlap, blind_index = _run_ladder(amplitudes, seeded=False)
    print("amplitudes=" + repr(amplitudes))
    print("seeded_overlap_series=" + repr(seeded_overlap))
    print("seeded_index_series=" + repr(seeded_index))
    print("blind_overlap_series=" + repr(blind_overlap))
    print("blind_index_series=" + repr(blind_index))
    if min(seeded_overlap) < 0.9 or len(set(seeded_index)) != 1:
        raise SystemExit("G2 seeded dense ground truth failed")
    if len(set(blind_index)) == 1:
        raise SystemExit("G2 contrast did not hop; tighten the cluster")


if __name__ == "__main__":
    main()
