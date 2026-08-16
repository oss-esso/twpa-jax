"""Golden baseline for the trusted IPM 2c design."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from twpa_solver.builders.ipm import Element, build_matrices
from twpa_solver.design import compile_design, load_design


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "designs" / "ipm_2c_fixed"
MATRIX_NAMES = ("C", "G", "K", "Bphi")


def _element_digest(elements: list[Element]) -> str:
    payload = json.dumps(
        [element.__dict__ for element in elements],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _matrix_summary(matrix: sp.spmatrix) -> dict[str, Any]:
    csr = matrix.tocsr()
    return {
        "shape": list(csr.shape),
        "nnz": int(csr.nnz),
        "sum": float(csr.sum()),
        "max_abs": float(np.max(np.abs(csr.data))) if csr.nnz else 0.0,
    }


def test_ipm_2c_compiler_matches_golden_baseline() -> None:
    baseline = json.loads(
        (ROOT / "tests" / "data" / "ipm_2c_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    compiled = compile_design(load_design(ROOT / "designs" / "ipm_2c.yaml"))
    matrices = build_matrices(compiled.elements)

    assert hashlib.sha256(
        (REFERENCE / "elements.csv").read_bytes()
    ).hexdigest() == baseline["elements_csv_sha256"]
    assert _element_digest(compiled.elements) == baseline["compiled_elements_sha256"]
    assert len(compiled.elements) == baseline["element_count"]
    assert {
        str(number): record.node
        for number, record in sorted(compiled.ports.items())
    } == baseline["ports"]

    for name in MATRIX_NAMES:
        assert _matrix_summary(matrices[name]) == baseline["matrices"][name]
        stored = sp.load_npz(REFERENCE / f"{name}.npz").tocsr()
        assert (matrices[name].tocsr() != stored).nnz == 0

    current_ic = np.asarray(matrices["Ic"])
    stored_ic = np.asarray(np.load(REFERENCE / "ipm_arrays.npz")["Ic"])
    assert current_ic.shape == (baseline["Ic"]["size"],)
    assert float(current_ic.sum()) == baseline["Ic"]["sum"]
    assert float(np.max(np.abs(current_ic))) == baseline["Ic"]["max_abs"]
    assert hashlib.sha256(current_ic.tobytes()).hexdigest() == baseline["Ic"][
        "sha256"
    ]
    assert np.array_equal(current_ic, stored_ic)


def test_ipm_2c_compilation_is_deterministic() -> None:
    design = load_design(ROOT / "designs" / "ipm_2c.yaml")
    first = compile_design(design)
    second = compile_design(load_design(ROOT / "designs" / "ipm_2c.yaml"))

    assert _element_digest(first.elements) == _element_digest(second.elements)
    assert first.cursors == second.cursors
    assert {
        str(number): record.node
        for number, record in sorted(first.ports.items())
    } == {
        str(number): record.node
        for number, record in sorted(second.ports.items())
    }
