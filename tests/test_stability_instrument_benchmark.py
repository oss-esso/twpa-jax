from __future__ import annotations

from scripts.benchmark_stability_instruments import damped_reference


def test_selected_monodromy_control_matches_damped_reference() -> None:
    result = damped_reference(96)
    assert result["absolute_radius_error"] < 5e-4
    assert result["monodromy_spectral_radius"] < 1.0
