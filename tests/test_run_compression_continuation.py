from __future__ import annotations

import pytest

from scripts.run_compression import _fallback_fixed_steps_for_span


def test_fallback_fixed_steps_are_sized_from_remaining_span() -> None:
    assert _fallback_fixed_steps_for_span(1.0) == 4
    assert _fallback_fixed_steps_for_span(0.26) == 2


def test_fallback_fixed_steps_reject_invalid_span() -> None:
    with pytest.raises(ValueError, match="span"):
        _fallback_fixed_steps_for_span(0.0)
