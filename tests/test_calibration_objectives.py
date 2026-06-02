from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_linear_sparams_objective import build_objective_summary
from scripts.run_linear_sparams_campaign import run_campaign
from twpa.calibration.objectives import (
    evaluate_dataset_against_target,
    evaluate_sparameter_objective,
)
from twpa.io.dataset_builder import build_linear_sparams_dataset


def _matched_s(n: int = 5) -> np.ndarray:
    s = np.zeros((n, 2, 2), dtype=np.complex128)
    s[:, 1, 0] = 1.0
    s[:, 0, 1] = 1.0
    return s


def _mismatched_s(n: int = 5, refl: float = 0.1) -> np.ndarray:
    s = _matched_s(n)
    s[:, 0, 0] = refl
    s[:, 1, 1] = refl
    s[:, 1, 0] = np.sqrt(1.0 - refl**2)
    s[:, 0, 1] = np.sqrt(1.0 - refl**2)
    return s


def test_objective_zero_for_identical_sparameters() -> None:
    f = np.linspace(4e9, 8e9, 5)
    s = _matched_s(5)
    gain = np.zeros(5)

    result = evaluate_sparameter_objective(
        frequency_hz=f,
        candidate_s=s,
        target_s=s,
        candidate_gain_db=gain,
        target_gain_db=gain,
    )

    assert result.total_loss == pytest.approx(0.0)
    assert result.s21_complex_loss == pytest.approx(0.0)
    assert result.s11_match_loss == pytest.approx(0.0)
    assert result.s22_match_loss == pytest.approx(0.0)


def test_objective_penalizes_mismatch() -> None:
    f = np.linspace(4e9, 8e9, 5)
    target = _matched_s(5)
    candidate = _mismatched_s(5, refl=0.1)

    result = evaluate_sparameter_objective(
        frequency_hz=f,
        candidate_s=candidate,
        target_s=target,
    )

    assert result.total_loss > 0.0
    assert result.s11_match_loss > 0.0
    assert result.s22_match_loss > 0.0


def test_dataset_evaluation_ranks_target_first() -> None:
    f = np.linspace(4e9, 8e9, 5)
    s0 = _mismatched_s(5, refl=0.1)
    s1 = _matched_s(5)
    s2 = _mismatched_s(5, refl=0.2)

    s = np.stack([s0, s1, s2], axis=0)
    gain = np.zeros((3, 5))
    params = np.asarray(
        [
            [50.0, 45.0, 0.1, 1.2e8, 0.0],
            [50.0, 50.0, 0.1, 1.2e8, 0.0],
            [50.0, 55.0, 0.1, 1.2e8, 0.0],
        ]
    )

    rows = evaluate_dataset_against_target(
        parameters=params,
        frequency_hz=f,
        s_complex=s,
        gain_db=gain,
        target_index=1,
    )

    ranked = sorted(rows, key=lambda row: row["total_loss"])

    assert ranked[0]["sample_index"] == 1
    assert ranked[0]["total_loss"] == pytest.approx(0.0)


def test_actual_linear_objective_pipeline_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    campaign_dir = tmp_path / "campaign"
    dataset_dir = tmp_path / "dataset"
    objective_dir = tmp_path / "objective"

    summary = run_campaign(
        z_lines_ohm=[45.0, 50.0, 55.0],
        harmonia_root=harmonia_root,
        campaign_dir=campaign_dir,
        force=True,
        timeout_s=120.0,
        n_frequency=11,
    )

    assert summary["registry"]["by_status"] == {"PASS": 3}

    built = build_linear_sparams_dataset(
        registry_csv=campaign_dir / "runs.csv",
        output_dir=dataset_dir,
    )

    objective_summary = build_objective_summary(
        dataset_npz=built.dataset_npz,
        output_dir=objective_dir,
        target_z_line_ohm=50.0,
    )

    ranked = objective_summary["ranked_losses"]

    assert ranked[0]["sample_index"] == objective_summary["target_index"]
    assert ranked[0]["total_loss"] == pytest.approx(0.0)
    assert objective_summary["n_samples"] == 3
    assert objective_summary["n_frequency"] == 11
    assert (objective_dir / "objective_summary.json").exists()