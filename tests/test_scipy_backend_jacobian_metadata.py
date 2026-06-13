import json

from twpa_solver.experiments.solve_old_ipm_backend_point import main

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def test_scipy_backend_result_reports_residual_reduction(tmp_path):
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
            "20",
            "--no-continuation-enabled",
        ]
    )
    result = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
    assert result["jacobian_strategy"] == "analytic_sparse_aft"
    assert result["initial_residual_inf"] > result["final_residual_inf"]
    assert result["residual_reduction_factor"] > 1.0
    assert result["num_jacobian_evals"] > 0
    assert (outdir / "continuation_history.csv").exists()


def test_backend_adapter_parses_jacobian_metadata():
    text = (
        r"D:\Projects\Thesis\Harmonia.jl\experiments\jc_setup_cache"
        r"\run_report_old_ipm_power_map_backend_compare.jl"
    )
    with open(text, encoding="utf-8") as f:
        contents = f.read()
    assert 'row_metadata["backend_metadata"] = result.metadata' in contents
    assert "metadata_json" in contents
