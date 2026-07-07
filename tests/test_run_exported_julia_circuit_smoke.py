import csv

from twpa_solver_old.experiments.run_exported_julia_circuit_map import main

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def test_run_exported_julia_circuit_smoke(tmp_path):
    circuit = write_tiny_export(tmp_path / "tiny.json")
    outdir = tmp_path / "out"
    main(
        [
            "--circuit-json",
            str(circuit),
            "--outdir",
            str(outdir),
            "--points",
            "1",
            "--max-linear-nodes",
            "20",
        ]
    )
    rows_path = outdir / "report_old_ipm_python_backend_rows.csv"
    assert rows_path.exists()
    rows = list(csv.DictReader(rows_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["backend"] == "python-exported-netlist"
    assert rows[0]["status"] in {
        "LINEAR_IMPORTED_SMOKE_OK_HB_NOT_IMPLEMENTED",
        "IMPORTED_LINEAR_SMOKE_FAILED",
    }
