"""Tests for the setprofile matrix tracer in run_gain_map_column_matrices.

The tracer is a sys.setprofile callback: it must (a) capture whitelisted sparse
matrices from solver/target frames on call/return boundaries, (b) ignore
non-target frames and non-call/return events (so it stays cheap), and (c)
deduplicate by object identity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_gain_map_column_matrices import MatrixTracer  # noqa: E402


class _FakeCode:
    def __init__(self, filename: str, name: str = "solve") -> None:
        self.co_filename = filename
        self.co_name = name


class _FakeFrame:
    def __init__(self, filename: str, lineno: int = 1, locals_: dict | None = None) -> None:
        self.f_code = _FakeCode(filename)
        self.f_lineno = lineno
        self.f_locals = locals_ or {}


def test_is_target_classifies_solver_paths(tmp_path: Path) -> None:
    tracer = MatrixTracer(tmp_path)
    assert tracer._is_target("C:/repo/src/twpa_solver/pump/solver.py")
    assert tracer._is_target("/repo/experiments/exp08.py")
    assert tracer._is_target("/repo/scripts/run_gain_map.py")
    assert not tracer._is_target("/env/site-packages/scipy/sparse/_compressed.py")
    assert not tracer._is_target("/env/lib/python3.11/logging/__init__.py")


def test_dispatch_captures_target_frames_on_return(tmp_path: Path) -> None:
    tracer = MatrixTracer(tmp_path)
    matrix = sp.eye(3, format="csr")
    target = _FakeFrame(
        "/repo/src/twpa_solver/pump/solver.py", lineno=7, locals_={"A": matrix}
    )
    # A whitelisted sparse matrix in a target frame is captured on 'return'.
    tracer.dispatch(target, "return", None)
    assert len(tracer.index) == 1


def test_dispatch_ignores_nontarget_frames_and_non_return_events(tmp_path: Path) -> None:
    tracer = MatrixTracer(tmp_path)
    matrix = sp.eye(3, format="csr")
    # Non-target frame: never scanned even with a whitelisted matrix present.
    other = _FakeFrame(
        "/env/site-packages/scipy/sparse/_compressed.py", locals_={"A": matrix}
    )
    tracer.dispatch(other, "return", None)
    assert tracer.index == []
    # Target frame but a non-return event ('call'/'line'/'c_call'): skipped for speed.
    target = _FakeFrame("/repo/src/twpa_solver/pump/solver.py", locals_={"A": matrix})
    tracer.dispatch(target, "call", None)
    tracer.dispatch(target, "line", None)
    tracer.dispatch(target, "c_call", None)
    assert tracer.index == []


def test_save_captures_whitelisted_sparse_and_dedups(tmp_path: Path) -> None:
    tracer = MatrixTracer(tmp_path)
    frame = _FakeFrame("/repo/src/twpa_solver/pump/solver.py", lineno=42)
    matrix = sp.eye(3, format="csr")

    tracer._save(matrix, frame, 42, "A")
    assert len(tracer.index) == 1
    saved = tmp_path / tracer.index[0]["path"]
    assert saved.exists()
    assert tracer.index[0]["shape"] == [3, 3]

    # Same object at the same site is written once.
    tracer._save(matrix, frame, 42, "A")
    assert len(tracer.index) == 1


def test_dispatch_never_propagates_save_errors(tmp_path: Path) -> None:
    # A failing _save (e.g. a Windows MAX_PATH OSError) must not escape the
    # setprofile callback and abort the solve it observes.
    tracer = MatrixTracer(tmp_path)
    matrix = sp.eye(3, format="csr")
    frame = _FakeFrame("/repo/src/twpa_solver/pump/solver.py", locals_={"A": matrix})

    def boom(*_args, **_kwargs):
        raise OSError(206, "The filename or extension is too long")

    tracer._save = boom  # type: ignore[method-assign]
    tracer.dispatch(frame, "return", None)  # must not raise
    assert tracer._save_errors == 1


def test_save_skips_non_whitelisted_names_and_dense_arrays(tmp_path: Path) -> None:
    import numpy as np

    tracer = MatrixTracer(tmp_path)
    frame = _FakeFrame("/repo/src/twpa_solver/pump/solver.py")
    tracer._save(sp.eye(2, format="csr"), frame, 1, "scratch")  # non-whitelisted name
    tracer._save(np.eye(2), frame, 1, "A")  # dense, not sparse
    assert tracer.index == []
