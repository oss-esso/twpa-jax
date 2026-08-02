from __future__ import annotations

import math

import pytest

from scripts.run_le_gal_2025_hb import _current_from_dbm


@pytest.mark.parametrize("z0", [50.0, 62.4])
def test_dbm_peak_current_convention(z0: float) -> None:
    requested_dbm = -100.0
    current = _current_from_dbm(requested_dbm, z0)
    recovered_w = current * current * z0 / 2.0
    recovered_dbm = 10.0 * math.log10(recovered_w) + 30.0
    assert recovered_dbm == pytest.approx(requested_dbm, abs=1e-12)
