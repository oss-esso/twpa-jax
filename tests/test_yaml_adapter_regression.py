"""Phase 0 snapshots for the Phase 7 public-Circuit YAML adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twpa_solver.builders.ipm import build_matrices
from twpa_solver.design import compile_design, load_design


ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads(
    (ROOT / "tests" / "data" / "yaml_adapter_baseline.json").read_text(
        encoding="utf-8"
    )
)
DESIGN_NAMES = tuple(sorted(BASELINE))


def _digest(value: Any) -> tuple[list[int], str]:
    """Digest sparse matrices and dense branch arrays deterministically."""

    if hasattr(value, "tocsr"):
        matrix = value.tocsr()
        payload = json.dumps(
            {
                "shape": list(matrix.shape),
                "indptr": matrix.indptr.tolist(),
                "indices": matrix.indices.tolist(),
                "data": matrix.data.tolist(),
            },
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return list(matrix.shape), hashlib.sha256(payload).hexdigest()
    array = np.asarray(value)
    return list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()


def _element_digest(elements: list[Any]) -> str:
    """Hash the complete solver-facing element sequence."""

    payload = json.dumps(
        [element.__dict__ for element in elements],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("design_name", DESIGN_NAMES)
def test_yaml_adapter_matches_phase0_snapshot(design_name: str) -> None:
    """Each YAML design remains element- and matrix-identical after adaptation."""

    design = compile_design(load_design(ROOT / "designs" / f"{design_name}.yaml"))
    expected = BASELINE[design_name]
    matrices = build_matrices(design.elements)

    assert len(design.elements) == expected["element_count"]
    assert _element_digest(design.elements) == expected["elements_sha256"]
    assert {
        str(number): record.node for number, record in sorted(design.ports.items())
    } == expected["ports"]
    for matrix_name in ("C", "G", "K", "Bphi", "Ic", "Lj"):
        shape, digest = _digest(matrices[matrix_name])
        assert shape == expected["matrices"][matrix_name]["shape"]
        assert digest == expected["matrices"][matrix_name]["sha256"]


@pytest.mark.parametrize("design_name", DESIGN_NAMES)
def test_yaml_adapter_is_deterministic(design_name: str) -> None:
    """Repeated YAML adaptation emits the same ordered legacy netlist."""

    source = load_design(ROOT / "designs" / f"{design_name}.yaml")
    first = compile_design(source)
    second = compile_design(source)

    assert _element_digest(first.elements) == _element_digest(second.elements)
    assert first.cursors == second.cursors
    assert first.named_nodes == second.named_nodes
    assert first.named_elements == second.named_elements
