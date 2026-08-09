from __future__ import annotations

import shutil
from pathlib import Path

from scripts.run_hybrid_column import prepare_output_dir


def test_prepare_output_dir_creates_requested_nested_path(tmp_path: Path) -> None:
    requested = tmp_path / "outputs" / "test-path" / "nested" / "hybrid-run"
    result = prepare_output_dir(requested, 7.0)

    assert result == requested
    assert result.is_dir()
    assert not (result / ".write_probe").exists()
