"""Gates for JAX device selection in the batched FDTD backend.

These run on a machine with no accelerator, so they pin the fallback contract
rather than accelerator execution: ``auto`` must degrade to the CPU and say so,
``gpu`` must refuse instead of degrading silently, and ``cpu`` must pin the CPU
even when an accelerator is visible.
"""

from __future__ import annotations

import sys
import types
from inspect import signature
from pathlib import Path

import numpy as np
import pytest

from scripts.chaos.run_guarcello_jc_phase5 import (
    JAX_DEVICE_PREFERENCES,
    PHI0_REDUCED,
    _factor_banded_lu,
    _jax_banded_solve,
    _unpack_banded_factor,
    integrate_jc_banded_batch,
    load_jc_device,
    resolve_jax_device,
)


ROOT = Path(__file__).resolve().parents[1]


class _Device:
    def __init__(self, platform: str) -> None:
        self.platform = platform

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.platform}>"


def _install_fake_jax(monkeypatch: pytest.MonkeyPatch, platforms: list[str]) -> None:
    devices = [_Device(platform) for platform in platforms]

    def jax_devices(kind: str | None = None) -> list[_Device]:
        if kind is None:
            return devices
        return [device for device in devices if device.platform == kind]

    monkeypatch.setitem(
        sys.modules, "jax", types.SimpleNamespace(devices=jax_devices)
    )


def test_auto_falls_back_to_cpu_and_names_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_jax(monkeypatch, ["cpu"])

    device, note = resolve_jax_device("auto")

    assert device.platform == "cpu"
    assert "fell back" in note


def test_auto_prefers_an_accelerator_when_one_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_jax(monkeypatch, ["cpu", "gpu"])

    device, note = resolve_jax_device("auto")

    assert device.platform == "gpu"
    assert "gpu" in note


def test_gpu_refuses_rather_than_degrading_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_jax(monkeypatch, ["cpu"])

    with pytest.raises(RuntimeError, match="no accelerator"):
        resolve_jax_device("gpu")


def test_cpu_pins_the_cpu_even_beside_an_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_jax(monkeypatch, ["cpu", "gpu"])

    device, _ = resolve_jax_device("cpu")

    assert device.platform == "cpu"


def test_an_unknown_preference_is_rejected() -> None:
    with pytest.raises(ValueError, match="jax_device"):
        resolve_jax_device("cuda")


def test_the_preference_list_is_the_documented_one() -> None:
    assert JAX_DEVICE_PREFERENCES == ("auto", "gpu", "cpu")


def test_batched_jax_exposes_the_scan_solve_option() -> None:
    assert "solve_kind" in signature(integrate_jc_banded_batch).parameters


@pytest.mark.parametrize(
    ("name", "source", "expected_bandwidth"),
    [
        ("ipm_2c_fixed", ROOT / "designs/ipm_2c_fixed", 5),
        ("rf_squid_2393_3wm", ROOT / "designs/rf_squid_2393_3wm.yaml", 2),
    ],
)
def test_scan_banded_rhs_matches_sequential_at_device_bandwidth(
    name: str, source: Path, expected_bandwidth: int,
) -> None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    device = load_jc_device(source)
    assert device.name == name
    assert device.selected_bandwidth == expected_bandwidth
    dt_s = 1.0e-13
    lu = _factor_banded_lu(
        device.C / dt_s**2 + device.G / (2.0 * dt_s) + device.K,
        device.selected_bandwidth,
    )
    _kind, factor_arrays = _unpack_banded_factor(lu, device.selected_bandwidth)
    diagonal, *bands = factor_arrays
    lower = tuple(jnp.asarray(value) for value in bands[:5])
    upper = tuple(jnp.asarray(value) for value in bands[5:])
    rhs = jnp.asarray(np.random.default_rng(19).standard_normal(device.n_nodes))
    sequential = np.asarray(
        _jax_banded_solve(
            rhs, jnp.asarray(diagonal), lower, upper,
            device.selected_bandwidth, "sequential",
        )
    )
    scan = np.asarray(
        _jax_banded_solve(
            rhs, jnp.asarray(diagonal), lower, upper,
            device.selected_bandwidth, "scan",
        )
    )
    relative = np.linalg.norm(scan - sequential) / np.linalg.norm(sequential)
    assert relative <= 1.0e-12


def test_scan_trajectory_matches_sequential_on_ipm_batch_two() -> None:
    device = load_jc_device(ROOT / "designs/ipm_2c_fixed")
    dt_s = 0.01 / (3.4e-6 / (PHI0_REDUCED * 90.0e-15)) ** 0.5
    settings = {
        "pump_currents_a": np.asarray([0.8e-6, 1.0e-6]),
        "pump_hz": 7.9e9,
        "signal_current_a": 0.0,
        "signal_hz": 7.4e9,
        "dt_s": dt_s,
        "n_steps": 600,
        "record_stride": 20,
        "backend": "jax",
        "jax_device": "cpu",
    }
    sequential = integrate_jc_banded_batch(
        device, solve_kind="sequential", **settings,
    )
    scan = integrate_jc_banded_batch(device, solve_kind="scan", **settings)
    relative = np.linalg.norm(scan[4] - sequential[4]) / np.linalg.norm(sequential[4])
    assert relative <= 1.0e-11
