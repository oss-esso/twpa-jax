from __future__ import annotations

import numpy as np

from twpa_solver.builders.kimpa import build_kimpa
from twpa_solver.core import load_circuit, save_circuit
from twpa_solver.signal.passive import (
    passive_network_matrices,
    passive_s_matrix,
    passive_s_matrix_from_circuit,
)


def test_passive_network_matrices_are_frequency_and_port_indexed(tmp_path):
    circuit_dir = tmp_path / "circuit"
    save_circuit(build_kimpa(), circuit_dir)
    matrices = passive_network_matrices(circuit_dir, [8.0e9, 8.1e9], ports=(1,))
    assert matrices["ports"].tolist() == [1]
    assert matrices["Z"].shape == (2, 1, 1)
    assert matrices["Y"].shape == (2, 1, 1)
    assert matrices["S"].shape == (2, 1, 1)


def test_passive_s_matrix_accepts_an_assembled_circuit(tmp_path):
    circuit_dir = tmp_path / "circuit"
    save_circuit(build_kimpa(), circuit_dir)
    frequencies = np.array([8.0e9, 8.1e9])
    loaded = load_circuit(circuit_dir)

    from_directory = passive_s_matrix(circuit_dir, frequencies, ports=(1,))
    from_memory = passive_s_matrix_from_circuit(loaded, frequencies, ports=(1,))

    np.testing.assert_allclose(from_memory, from_directory)
