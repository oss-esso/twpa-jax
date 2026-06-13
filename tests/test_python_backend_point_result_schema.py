import json

from twpa_solver.experiments.solve_old_ipm_backend_point import main

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def test_python_backend_point_not_implemented_schema(tmp_path):
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
    assert result["status"] == "BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN"
    assert result["success"] is False
    assert result["gain_db_max"] is None
    assert result["metadata"]["surrogate_topology_used"] is False
