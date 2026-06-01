from __future__ import annotations

import os
from pathlib import Path

import jax
import pytest

# Keep pytest temp dirs inside the repo to avoid host-temp ACL failures on Windows.
_TMP_ROOT = Path(__file__).resolve().parents[1] / ".pytest_tmp"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_TMP_ROOT))

jax.config.update("jax_enable_x64", True)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow dense-HB smoke tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow dense-HB smoke test")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
