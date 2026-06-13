from pathlib import Path


def test_julia_map_backend_selection_lists_required_backends():
    path = Path(r"D:\Projects\Thesis\Harmonia.jl\experiments\jc_setup_cache\run_report_old_ipm_power_map_backend_compare.jl")
    text = path.read_text(encoding="utf-8")
    for backend in (
        "josephsoncircuits",
        "scipy-least-squares",
        "scipy-root",
        "scipy-newton-krylov",
        "jax-dense-newton",
        "jax-newton-krylov",
        "pseudo-transient",
    ):
        assert backend in text
    assert "build_old_ipm_circuit()" in text
    assert "write_outputs(outdir, \"mapN\", rows)" in text
