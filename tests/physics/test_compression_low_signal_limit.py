from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="The default JPA fixture has not yet been found at >3 dB gain.")
def test_low_signal_limit_requires_a_real_gain_operating_point() -> None:
    pytest.fail("No non-degenerate gain operating point is registered yet")
