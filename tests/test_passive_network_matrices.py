from __future__ import annotations

from twpa_solver.builders.kimpa import build_kimpa
from twpa_solver.core import save_circuit
from twpa_solver.signal.passive import passive_network_matrices


def test_passive_network_matrices_are_frequency_and_port_indexed(tmp_path):
    circuit_dir = tmp_path / "circuit"
    save_circuit(build_kimpa(), circuit_dir)
    matrices = passive_network_matrices(circuit_dir, [8.0e9, 8.1e9], ports=(1,))
    assert matrices["ports"].tolist() == [1]
    assert matrices["Z"].shape == (2, 1, 1)
    assert matrices["Y"].shape == (2, 1, 1)
    assert matrices["S"].shape == (2, 1, 1)
