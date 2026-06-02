from pathlib import Path

import numpy as np

from twpa.calibration.dataset_objectives import evaluate_harmonia_jtl_linear_dataset


def test_target_sample_ranks_first(tmp_path: Path) -> None:
    dataset = tmp_path / "harmonia_jtl_linear_dataset.npz"
    output = tmp_path / "objective_summary.json"
    s = np.zeros((3, 2, 2, 2), dtype=np.complex128)
    s[:, :, 1, 0] = np.asarray([[0.7, 0.8], [0.8, 0.9], [0.9, 1.0]])
    s[:, :, 0, 1] = s[:, :, 1, 0]
    gain = 20.0 * np.log10(np.abs(s[:, :, 1, 0]))
    np.savez_compressed(dataset, parameter_names=np.asarray(["Lj_H"]), parameters=np.asarray([[0.8e-9], [1.0e-9], [1.2e-9]]), frequency_hz=np.asarray([4e9, 5e9]), s_real=s.real, s_imag=s.imag, gain_db=gain)

    summary = evaluate_harmonia_jtl_linear_dataset(dataset_npz=dataset, output_json=output, target_lj_h=1.01e-9)

    assert output.exists()
    assert summary["target_index"] == 1
    assert summary["ranking"][0]["index"] == 1
    assert summary["ranking"][0]["total_loss"] == 0.0
