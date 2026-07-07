import csv

from twpa_solver_old.experiments.compare_backend_5x5_to_jc_reference import main


def _write_rows(path, status="VALID_CONVERGED", gain="1.0"):
    rows = []
    for p in range(5):
        for f in range(5):
            rows.append(
                {
                    "pump_frequency_ghz": str(6.0 + 0.5 * f),
                    "external_power_dbm": str(-28.0 + p),
                    "status": status,
                    "gain_db_max": gain,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_backend_comparison_tiny_fake_5x5(tmp_path):
    ref = tmp_path / "ref"
    root = tmp_path / "backend"
    out = root / "comparison"
    ref.mkdir()
    jc = root / "josephsoncircuits"
    nb = root / "scipy_least_squares"
    jc.mkdir(parents=True)
    nb.mkdir(parents=True)
    _write_rows(ref / "report_old_ipm_power_map_rows.csv")
    _write_rows(jc / "report_old_ipm_power_map_rows.csv")
    _write_rows(
        nb / "report_old_ipm_power_map_rows.csv",
        status="BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN",
        gain="",
    )
    main(["--reference-root", str(ref), "--backend-root", str(root), "--outdir", str(out)])
    summary = (out / "backend_comparison_summary.md").read_text(encoding="utf-8")
    assert "REFERENCE_REPRODUCED" in summary
    assert "NOT_IMPLEMENTED" in summary
