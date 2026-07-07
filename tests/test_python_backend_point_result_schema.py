import json

from twpa_solver_old.experiments.solve_old_ipm_backend_point import main

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def test_python_backend_point_schema_for_non_least_squares_backend(tmp_path):
    circuit = write_tiny_export(tmp_path / "tiny.json")
    outdir = tmp_path / "point"
    main(
        [
            "--circuit-json",
            str(circuit),
            "--backend",
            "scipy-root",
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
        ]
    )
    result = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
    assert result["backend"] == "scipy-root"
    assert result["status"] != "BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN"
    assert result["status"] in {
        "VALID_CONVERGED",
        "FINITE_NONCONVERGED",
        "RESIDUAL_REDUCED_NOT_CONVERGED",
        "FAILED_MAX_NFEV",
        "FAILED_NONFINITE_RESIDUAL",
        "FAILED_NUMERICALLY",
        "FAILED_EXCEPTION",
        "FAILED_TIMEOUT",
        "FAILED_MEMORY",
    }
    for key in (
        "convergence_mask_value",
        "finite_mask_value",
        "solver_warning_mask_value",
        "initial_residual_norm",
        "initial_infinity_norm",
        "residual_reduction_factor",
        "function_evals",
        "jacobian_evals",
        "linear_solves",
        "point_runtime_s",
        "error_type",
        "error_message",
    ):
        assert key in result
    assert result["metadata"]["surrogate_topology_used"] is False
    assert result["metadata"]["requested_backend"] == "scipy-root"
