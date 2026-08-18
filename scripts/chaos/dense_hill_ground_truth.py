"""Run the dense ground-truth gate for Hill-root continuation.

The fixture uses the 2,560-node dimension of the JTWPA builder, which is the
largest conversion dimension that fits a dense eigensolve in this environment.
Its diagonal pencil is deliberately constructed with a measured near-
degenerate cluster.  The production conversion-matrix assembly and the
production complex-root refinement are still used; only the linear reference
pencil is controlled so that the branch identity has a closed form.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from twpa_solver.builders.jc_doc import build_jtwpa
from twpa_solver.core.circuit import CircuitMatrices
from twpa_solver.signal.floquet import assemble_conversion_matrix
from twpa_solver.signal.stability import refine_singular_omega
import twpa_solver.signal.stability as stability


def _reference_circuit() -> CircuitMatrices:
    """Build a dense-compatible pencil with the largest device dimension."""
    builder, _ = build_jtwpa()
    dimension = int(builder.assemble()["C"].shape[0])
    epsilon = 1.0e-9
    omega_0 = 2.0 * math.pi * 1.0e0
    return CircuitMatrices(
        C=sp.identity(dimension, format="csr", dtype=np.complex128),
        G=sp.csr_matrix((dimension, dimension), dtype=np.complex128),
        K=sp.diags(
            omega_0**2 + epsilon * np.arange(dimension),
            format="csr",
            dtype=np.complex128,
        ),
        Bphi=sp.csr_matrix((dimension, 0), dtype=np.float64),
        Ic=np.empty(0, dtype=float),
    )


def _assemble_factory(
    circuit: CircuitMatrices, amplitude: float
) -> tuple[Callable[[complex], sp.csc_matrix], float]:
    """Return a conversion pencil whose first mode is the tracked target."""
    omega_0 = 2.0 * math.pi * 1.0e0
    omega_p = 2.0 * math.pi * 1.0e-9
    shift = 2.0 * omega_0 * amplitude * 0.1
    target_shift = sp.csr_matrix(
        (np.asarray([shift], dtype=np.complex128), ([0], [0])),
        shape=circuit.C.shape,
    )

    def assemble(omega: complex) -> sp.csc_matrix:
        return assemble_conversion_matrix(
            circuit=circuit,
            khat={0: target_shift},
            omega_s=omega,
            omega_p=omega_p,
            ms=[0],
            loss_model="current_complex_c",
        )

    return assemble, omega_p


def _run_ladder(
    amplitudes: list[float], seeded: bool, dimension: int
) -> tuple[list[float], list[int]]:
    """Run one dense ladder and return overlaps and sorted-spectrum indices."""
    circuit = _reference_circuit()
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
        eigenvector = np.zeros((dimension, 1), dtype=np.complex128)
        eigenvector[index, 0] = 1.0
        return np.array([diagonal[index]]), eigenvector

    stability.spla.eigs = choose_cluster_member
    try:
        overlaps: list[float] = []
        indices: list[int] = []
        seed = 2.0 * math.pi * 1.0e0
        mode = None if not seeded else np.eye(dimension, dtype=np.complex128)[:, 0]
        for amplitude in amplitudes:
            assemble, _ = _assemble_factory(circuit, amplitude)
            result = refine_singular_omega(
                assemble,
                seed + 1.0e-2j,
                seed + 2.0e-2j,
                max_iters=20,
                tol=1.0e-12,
                v0=mode,
            )
            dense_values, dense_vectors = np.linalg.eig(
                assemble(result.omega).toarray()
            )
            order = np.argsort(dense_values.real)
            dense_modes = dense_vectors[:, order]
            dense_modes /= np.linalg.norm(dense_modes, axis=0, keepdims=True)
            tracked = result.mode_vector / np.linalg.norm(result.mode_vector)
            overlap_values = np.abs(np.conj(tracked) @ dense_modes)
            match = int(np.argmax(overlap_values))
            overlaps.append(float(overlap_values[match]))
            indices.append(match)
            if not result.converged:
                raise RuntimeError(
                    f"reference refinement failed at amplitude {amplitude}: "
                    f"iterations={result.iterations} residual={result.residual}"
                )
            seed = float(result.omega.real)
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
    dimension = circuit.node_count
    epsilon = 1.0e-9
    dense_bytes = dimension * dimension * 16
    print(
        json.dumps(
            {
                "reference": "full_jc_jtwpa_node_dimension_controlled_pencil",
                "node_count": circuit.node_count,
                "sideband_count": 1,
                "conversion_dimension": dimension,
                "dense_bytes": dense_bytes,
                "cluster_min_eigenvalue_separation": epsilon,
                "cluster_relative_separation": epsilon / (2.0 * math.pi * 1e0) ** 2,
            },
            sort_keys=True,
        )
    )
    seeded_overlap, seeded_index = _run_ladder(
        amplitudes, seeded=True, dimension=dimension
    )
    blind_overlap, blind_index = _run_ladder(
        amplitudes, seeded=False, dimension=dimension
    )
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
