import json

from twpa_solver_old.experiments.solve_old_ipm_backend_point import main

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


FORBIDDEN = {
    "BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN",
    "TODO",
    "PLACEHOLDER",
    "SCAFFOLD_ONLY",
}


def test_all_independent_backends_return_non_placeholder_status(tmp_path):
    circuit = write_tiny_export(tmp_path / "tiny.json")
    for backend in (
        "scipy-least-squares",
        "scipy-root",
        "scipy-newton-krylov",
        "jax-dense-newton",
        "jax-newton-krylov",
        "pseudo-transient",
    ):
        outdir = tmp_path / backend
        main(
            [
                "--circuit-json",
                str(circuit),
                "--backend",
                backend,
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
                "10",
                "--no-continuation-enabled",
            ]
        )
        result = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
        assert result["status"] not in FORBIDDEN
        assert result["metadata"]["surrogate_topology_used"] is False
        assert result["point_runtime_s"] >= 0.0
        assert result["infinity_norm"] is not None
