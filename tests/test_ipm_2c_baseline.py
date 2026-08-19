"""Golden baseline for the trusted IPM 2c design."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from twpa_solver.builders.ipm import Element, build_matrices
from twpa_solver.design import compile_design, load_design


ROOT = Path(__file__).resolve().parents[1]
MATRIX_NAMES = ("C", "G", "K", "Bphi")


def _element_digest(elements: list[Element]) -> str:
    payload = json.dumps(
        [element.__dict__ for element in elements],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_ipm_2c_line_scoped_compiles_to_valid_matrices() -> None:
    compiled = compile_design(load_design(ROOT / "designs" / "ipm_2c_line_scoped.yaml"))
    matrices = build_matrices(compiled.elements)

    assert len(compiled.elements) > 0
    assert {str(number) for number in compiled.ports} == {"1", "2", "3", "4"}

    for name in MATRIX_NAMES[:-1]:
        matrix = matrices[name].tocsr()
        assert matrix.shape[0] == matrix.shape[1]
        assert matrix.nnz > 0

    bphi = matrices["Bphi"].tocsr()
    assert bphi.shape[0] == matrices["C"].shape[0]
    assert bphi.shape[1] == len(matrices["Ic"])
    assert bphi.nnz > 0


def test_ipm_2c_compilation_is_deterministic() -> None:
    design = load_design(ROOT / "designs" / "ipm_2c_line_scoped.yaml")
    first = compile_design(design)
    second = compile_design(load_design(ROOT / "designs" / "ipm_2c_line_scoped.yaml"))

    assert _element_digest(first.elements) == _element_digest(second.elements)
    assert first.cursors == second.cursors
    assert {
        str(number): record.node
        for number, record in sorted(first.ports.items())
    } == {
        str(number): record.node
        for number, record in sorted(second.ports.items())
    }
