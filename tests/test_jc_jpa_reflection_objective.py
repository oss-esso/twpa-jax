from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_jc_jpa_reflection_objective import (
    build_jc_reflection_objective_summary,
)
from scripts.run_jc_jpa_reflection_campaign import run_campaign
from twpa.calibration.objectives import (
    evaluate_jc_reflection_dataset_against_target,
    evaluate_one_port_reflection_objective,
)
from twpa.io.dataset_builder import build_jc_jpa_reflection_dataset


def test_one_port_reflection_objective_zero_for_identical_curves() -> None:
    f = np.linspace(4.5e9, 5.0e9, 7)
    s11 = np.exp(1j * np.linspace(0.0, 1.0, 7))
    refl = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-300))

    result = evaluate_one_port_reflection_objective(
        frequency_hz=f,
        candidate_s11=s11,
        target_s11=s11,
        candidate_reflection_db=refl,
        target_reflection_db=refl,
    )

    assert result.total_loss == pytest.approx(0.0)
    assert result.s11_complex_loss == pytest.approx(0.0)
    assert result.reflection_db_loss == pytest.approx(0.0)
    assert result.peak_frequency_loss == pytest.approx(0.0)
    assert result.peak_reflection_db_loss == pytest.approx(0.0)


def test_one_port_reflection_objective_penalizes_different_curves() -> None:
    f = np.linspace(4.5e9, 5.0e9, 7)

    target_s11 = np.ones(7, dtype=np.complex128)
    candidate_s11 = 0.8 * np.ones(7, dtype=np.complex128)

    target_refl = 20.0 * np.log10(np.maximum(np.abs(target_s11), 1e-300))
    candidate_refl = 20.0 * np.log10(np.maximum(np.abs(candidate_s11), 1e-300))

    result = evaluate_one_port_reflection_objective(
        frequency_hz=f,
        candidate_s11=candidate_s11,
        target_s11=target_s11,
        candidate_reflection_db=candidate_refl,
        target_reflection_db=target_refl,
    )

    assert result.total_loss > 0.0
    assert result.s11_complex_loss > 0.0
    assert result.reflection_db_loss > 0.0


def test_jc_reflection_dataset_objective_ranks_target_first() -> None:
    f = np.linspace(4.5e9, 5.0e9, 7)

    s0 = 0.8 * np.ones(7, dtype=np.complex128)
    s1 = 1.0 * np.ones(7, dtype=np.complex128)
    s2 = 1.2 * np.ones(7, dtype=np.complex128)

    s = np.stack([s0, s1, s2], axis=0)
    refl = 20.0 * np.log10(np.maximum(np.abs(s), 1e-300))

    params = np.asarray(
        [
            [50.0, 100e-15, 1000e-12, 1000e-15, 4.75001e9, 0.0, 4.0, 4.0],
            [50.0, 100e-15, 1000e-12, 1000e-15, 4.75001e9, 2.0e-9, 4.0, 4.0],
            [50.0, 100e-15, 1000e-12, 1000e-15, 4.75001e9, 5.65e-9, 4.0, 4.0],
        ]
    )

    rows = evaluate_jc_reflection_dataset_against_target(
        parameters=params,
        frequency_hz=f,
        s11_complex=s,
        reflection_db=refl,
        target_index=1,
    )

    ranked = sorted(rows, key=lambda row: row["total_loss"])

    assert ranked[0]["sample_index"] == 1
    assert ranked[0]["total_loss"] == pytest.approx(0.0)


def test_actual_jc_reflection_objective_pipeline_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    campaign_dir = tmp_path / "campaign"
    dataset_dir = tmp_path / "dataset"
    objective_dir = tmp_path / "objective"

    summary = run_campaign(
        pump_currents_a=[0.0, 2.0e-9, 5.65e-9],
        harmonia_root=harmonia_root,
        campaign_dir=campaign_dir,
        force=True,
        timeout_s=180.0,
        n_frequency=7,
    )

    assert summary["registry"]["by_status"] == {"PASS": 3}

    built = build_jc_jpa_reflection_dataset(
        registry_csv=campaign_dir / "runs.csv",
        output_dir=dataset_dir,
    )

    objective_summary = build_jc_reflection_objective_summary(
        dataset_npz=built.dataset_npz,
        output_dir=objective_dir,
        target_pump_current_a=5.65e-9,
    )

    ranked = objective_summary["ranked_losses"]

    assert ranked[0]["sample_index"] == objective_summary["target_index"]
    assert ranked[0]["total_loss"] == pytest.approx(0.0)
    assert objective_summary["n_samples"] == 3
    assert objective_summary["n_frequency"] == 7
    assert (objective_dir / "objective_summary.json").exists()