from pathlib import Path


SCRIPT = Path(
    r"D:\Projects\Thesis\Harmonia.jl\experiments\jc_setup_cache\run_ipm_twpa_35x35_all_backends.jl"
)


def test_all_backend_script_has_incremental_resume_artifacts():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "rows.jsonl" in text
    assert "load_incremental_rows" in text
    assert "append_incremental_row" in text
    assert "--resume" in text
    assert "--overwrite" in text


def test_all_backend_script_writes_required_backend_grids():
    text = SCRIPT.read_text(encoding="utf-8")
    for filename in (
        "rows.csv",
        "gain_db_grid.csv",
        "convergence_mask_grid.csv",
        "finite_mask_grid.csv",
        "solver_warning_mask_grid.csv",
        "residual_norm_grid.csv",
        "infinity_norm_grid.csv",
        "point_runtime_grid.csv",
        "status_grid.csv",
        "map_timing.json",
        "report.md",
    ):
        assert filename in text
