from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.chaos.run_physical_torus_column import _write_period1_checkpoint
from twpa_solver.signal.io import load_pump


def test_period1_checkpoint_round_trips_through_load_pump(
    tmp_path: Path,
) -> None:
    state = np.asarray([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=np.complex128)
    pump = SimpleNamespace(
        X=state,
        modes=[1],
        omega_p=2.0e9,
        pump_freq_ghz=2.0 / (2.0 * np.pi),
        nt_original=40,
        basis=SimpleNamespace(policy="positive_odd_jc", source_mode=1),
        metadata={"pump_power_dbm_requested": -24.2},
    )
    pump_step = SimpleNamespace(
        pump=pump,
        full_state=state,
        source_current_a=2.0e-6,
        achieved_current_a=2.0e-6,
        iterations=3,
        coeff_rel=1.0e-12,
        time_rel=2.0e-12,
    )
    args = SimpleNamespace(pump_port=4)
    audit = {"loss_model": "current_complex_c", "analytic_in_omega": True}

    checkpoint = _write_period1_checkpoint(
        tmp_path,
        2,
        -24.2,
        pump_step,
        args,
        audit,
    )

    loaded = load_pump(checkpoint, fallback_pump_freq_ghz=2.0)
    np.testing.assert_array_equal(loaded.X, state)
    assert loaded.modes == [1]
    report = json.loads(
        (checkpoint / "pump_report.json").read_text(encoding="utf-8")
    )
    assert report["metadata"]["checkpoint_kind"] == (
        "period1_reproducibility"
    )
    assert report["metadata"]["loss_audit"] == audit
