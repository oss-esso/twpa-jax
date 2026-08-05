from __future__ import annotations

import numpy as np

from scripts.run_kimpa_gain import build_parser, run
from twpa_solver.builders.kimpa import build_kimpa
from twpa_solver.signal.gamma import compute_gamma_hat
from twpa_solver.signal.io import PumpSolution
from twpa_solver.pump.basis import PumpBasis


def test_pump_solve_converges_and_writes_waveforms(tmp_path) -> None:
    args = build_parser().parse_args([
        "--output-dir", str(tmp_path), "--pump-dbm", "-60",
        "--pump-nt", "8", "--sidebands", "1", "--max-ell", "2",
        "--dc-current-a", "0.00055",
    ])
    result = run(args)
    assert result["pump_converged"] is True
    assert result["pump_coeff_rel"] < 1e-10
    waveform = np.load(tmp_path / "kimpa_gain_waveforms.npz")
    assert waveform["branch_current_time"].shape[1] == 1


def test_paper_environment_mode_is_explicit(tmp_path) -> None:
    args = build_parser().parse_args([
        "--output-dir", str(tmp_path), "--environment", "paper_standing_wave",
        "--no-solve",
    ])
    assert run(args)["environment"] == "paper_standing_wave"


def test_dc_bias_creates_odd_pump_order_content(tmp_path) -> None:
    args = build_parser().parse_args([
        "--output-dir", str(tmp_path), "--pump-dbm", "-60",
        "--pump-nt", "8", "--sidebands", "1", "--max-ell", "2",
        "--dc-current-a", "0.00055",
    ])
    run(args)
    state = np.load(tmp_path / "kimpa_gain_waveforms.npz")["pump_state"]
    circuit = build_kimpa(args.fixture)
    pump = PumpSolution(
        X=state,
        omega_p=2.0 * np.pi * args.pump_ghz * 1e9,
        pump_freq_ghz=args.pump_ghz,
        harmonics=state.shape[0],
        nt_original=args.pump_nt,
        metadata={},
        modes=[1, 2, 3],
        basis=PumpBasis([1, 2, 3], "dense_real", 2.0 * np.pi * args.pump_ghz * 1e9),
    )
    biased = compute_gamma_hat(
        circuit, pump, 2, args.pump_nt,
        np.asarray([4.645737027209366e-13]),
    )
    unbiased = compute_gamma_hat(circuit, pump, 2, args.pump_nt, np.zeros(1))
    assert np.abs(biased[1][0]) > 1e-12 * np.abs(biased[0][0])
    assert np.abs(unbiased[1][0]) < 1e-10 * np.abs(unbiased[0][0])
    np.testing.assert_allclose(biased[-1], np.conj(biased[1]), rtol=0.0, atol=1e-12)
