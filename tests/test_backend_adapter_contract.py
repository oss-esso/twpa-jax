from pathlib import Path


def test_julia_backend_adapter_contract_exists():
    path = Path(r"D:\Projects\Thesis\Harmonia.jl\experiments\jc_setup_cache\run_report_old_ipm_power_map_backend_compare.jl")
    text = path.read_text(encoding="utf-8")
    assert "struct BackendMapPointResult" in text
    for field in ("backend", "status", "success", "gain_db_max", "residual_norm", "infinity_norm"):
        assert field in text
    assert "run_case_backend" in text
    assert "hbsolve" not in text or "run_case(" in text
