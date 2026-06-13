import json

from twpa_solver.experiments.solve_old_ipm_backend_point import main

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def test_old_ipm_scipy_backend_point_not_placeholder(tmp_path):
    circuit = write_tiny_export(tmp_path / "tiny.json")
    outdir = tmp_path / "point"
    main(
        [
            "--circuit-json",
            str(circuit),
            "--backend",
            "scipy-least-squares",
            "--pump-frequency-ghz",
            "6.0",
            "--external-pump-power-dbm",
            "-28.0",
            "--source-power-dbm",
            "-60.0",
            "--pump-current-a",
            "6.3e-6",
            "--pump-harmonics",
            "10",
            "--modulation-harmonics",
            "5",
            "--outdir",
            str(outdir),
            "--max-effective-pump-harmonics",
            "1",
            "--max-nfev",
            "1",
        ]
    )
    result = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
    assert result["backend"] == "scipy-least-squares"
    assert result["status"] != "BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN"
    assert result["residual_norm"] is not None
    assert result["infinity_norm"] is not None
    assert (outdir / "residual_history.csv").exists()
    assert (outdir / "pump_solution_coefficients.npz").exists()
