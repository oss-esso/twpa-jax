import numpy as np
import scipy.sparse as sp

from twpa_solver.core import CircuitMatrices, solve_linear_scattering, save_circuit, load_circuit


def make_one_node_linear_circuit():
    # One damped node to ground. No Josephson branches.
    C = sp.csr_matrix([[1.0e-12]])
    G = sp.csr_matrix([[1.0 / 50.0]])
    K = sp.csr_matrix([[1.0e-6]])
    Bphi = sp.csr_matrix((1, 0))
    Ic = np.asarray([], dtype=float)
    return CircuitMatrices(
        C=C,
        G=G,
        K=K,
        Bphi=Bphi,
        Ic=Ic,
        nodes=np.asarray([1], dtype=np.int64),
        port_to_index={1: 0},
        metadata={"test": "one_node_linear"},
    )


def test_linear_scattering_smoke():
    circuit = make_one_node_linear_circuit()
    r = solve_linear_scattering(
        circuit,
        frequency_hz=6.0e9,
        source_port=1,
        out_port=1,
    )
    assert np.isfinite(r.s_abs)
    assert np.isfinite(r.s_db)


def test_circuit_save_load_round_trip(tmp_path):
    circuit = make_one_node_linear_circuit()
    save_circuit(circuit, tmp_path)
    loaded = load_circuit(tmp_path)

    assert loaded.node_count == 1
    assert loaded.branch_count == 0
    assert loaded.port_to_index == {1: 0}
    assert loaded.C.shape == (1, 1)
