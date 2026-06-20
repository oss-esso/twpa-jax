from pathlib import Path


JULIA_SCRIPT = Path(
    r"D:\Projects\Thesis\Harmonia.jl\experiments\jc_setup_cache\run_ipm_twpa_35x35_all_backends.jl"
)
PYTHON_POINT = Path(
    r"D:\Projects\Thesis\twpa_jax\twpa_solver\experiments\solve_old_ipm_backend_point.py"
)


def test_all_backend_runner_does_not_reference_surrogate_topologies():
    text = JULIA_SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "ipm_jtwpa_reduced_marker",
        "ipm_jtwpa_physical_coupler",
        "ipm_jtwpa_old_constants_compact_surrogate",
        "ipm_jtwpa_old_julia_parity",
    )
    for name in forbidden:
        assert name not in text
    assert "run_case_backend(" in text
    assert "export_circuit_once" in text


def test_python_point_backend_reports_no_surrogate_topology():
    text = PYTHON_POINT.read_text(encoding="utf-8")
    assert '"surrogate_topology_used": False' in text
    assert "import_julia_circuit_json" in text
