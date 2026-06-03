from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_harmonia_benchmark_suite import (
    BenchmarkCase,
    select_cases,
    run_benchmark_suite,
)


def test_select_cases_prefers_modern_architecture() -> None:
    cases = select_cases(suite="modern", include_legacy=False)

    assert cases
    assert all(case.architecture == "CircuitIR" for case in cases)

    names = {case.name for case in cases}

    assert "jtl_linear" in names
    assert "ethz_jtl_linear" in names
    assert "lumped_jpa_linear" in names
    assert "jc_jpa_reflection_legacy" not in names


def test_select_cases_can_include_legacy() -> None:
    cases = select_cases(suite="modern", include_legacy=True)
    names = {case.name for case in cases}

    assert "jc_jpa_reflection_legacy" in names
    assert "matched_tl_linear_reference" in names


def test_actual_minimal_harmonia_benchmark_suite_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    required_configs = [
        harmonia_root / "examples" / "configs" / "harmonia_jtl_linear_jc_smoke.json",
        harmonia_root / "examples" / "configs" / "harmonia_lumped_jpa_linear_jc_smoke.json",
    ]

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    if not all(path.exists() for path in required_configs):
        pytest.skip("Required benchmark configs are not available.")

    cases = [
        BenchmarkCase(
            name="jtl_linear",
            config_filename="harmonia_jtl_linear_jc_smoke.json",
            architecture="CircuitIR",
            circuit_family="JTL",
            expected_simulation_type="harmonia_jtl_linear_jc_smoke",
            expected_circuit_template="circuit_ir_jtl_chain_linear_jc",
            timeout_s=180.0,
            notes="Minimal benchmark test case.",
        ),
        BenchmarkCase(
            name="lumped_jpa_linear",
            config_filename="harmonia_lumped_jpa_linear_jc_smoke.json",
            architecture="CircuitIR",
            circuit_family="JPA",
            expected_simulation_type="harmonia_lumped_jpa_linear_jc_smoke",
            expected_circuit_template="circuit_ir_lumped_jpa_reflection_linear_jc",
            timeout_s=180.0,
            notes="Minimal benchmark test case.",
        ),
    ]

    summary = run_benchmark_suite(
        harmonia_root=harmonia_root,
        benchmark_dir=tmp_path / "benchmark",
        cases=cases,
        repetitions=1,
        force=True,
    )

    assert summary["summary"]["n_rows"] == 2
    assert summary["summary"]["by_status"] == {"PASS": 2}

    rows = summary["rows"]

    assert len(rows) == 2

    for row in rows:
        assert row["architecture"] == "CircuitIR"
        assert row["status"] == "PASS"
        assert row["runner_ok"] is True
        assert row["matches_expected_simulation_type"] is True
        assert row["matches_expected_circuit_template"] is True
        assert row["python_wall_time_s"] > 0.0
        assert row["julia_status_runtime_s"] is not None
        assert row["h5_size_bytes"] is not None
        assert row["h5_size_bytes"] > 0
        assert row["frequency_points"] is not None
        assert row["s_shape"] is not None
        assert row["s_all_finite"] is True
        assert row["frequency_all_finite"] is True
        assert row["gain_db_all_finite"] is True
        assert row["h5_backend"] is not None

    benchmark_dir = tmp_path / "benchmark"

    assert (benchmark_dir / "benchmark_runs.csv").exists()
    assert (benchmark_dir / "benchmark_summary.json").exists()
    assert (benchmark_dir / "benchmark_cases.json").exists()
    assert (benchmark_dir / "environment.json").exists()