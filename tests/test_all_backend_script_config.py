from pathlib import Path


SCRIPT = Path(
    r"D:\Projects\Thesis\Harmonia.jl\experiments\jc_setup_cache\run_ipm_twpa_35x35_all_backends.jl"
)


def test_all_backend_script_exists_and_uses_canonical_builder():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "run_ipm_twpa_35x35_all_backends.jl" in text
    assert "run_case_backend(" in text
    assert "build_old_ipm_circuit()" not in text or "Circuit source" in text
    assert "export_circuit_once" in text
    assert "surrogate_topologies_used\" => false" in text


def test_all_backend_script_default_backends_and_axes():
    text = SCRIPT.read_text(encoding="utf-8")
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
    assert '"--points" => "35"' in text
    assert '"--pump-freq-min-ghz" => "6.0"' in text
    assert '"--pump-freq-max-ghz" => "8.0"' in text
    assert '"--pump-power-min-dbm" => "-28.0"' in text
    assert '"--pump-power-max-dbm" => "-19.0"' in text
    assert '"--power-offset-db" => "32.0"' in text
