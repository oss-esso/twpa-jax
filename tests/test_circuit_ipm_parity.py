"""Phase 6 parity gate for the public IPM 2c design."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from twpa_solver.builders.ipm import (
    IPMParams,
    build_matrices,
    make_coupler_discrete,
    make_ipm,
)
from twpa_solver.builders.ipm import Element


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "data" / "ipm_2c_reference"
HISTORICAL_REFERENCE = ROOT / "designs" / "ipm_2c_fixed"


def _load_python_design() -> Any:
    """Load the repository design module without importing legacy builders."""

    module_path = ROOT / "designs" / "python" / "ipm_2c.py"
    spec = importlib.util.spec_from_file_location("python_ipm_2c", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python design from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _params() -> IPMParams:
    """Return legacy-builder parameters matching the technology preset."""

    return IPMParams(start_node_bot=10_000)


def _element_fields(element: Element) -> dict[str, Any]:
    """Return all element fields used by the parity contract."""

    return {
        "name": element.name,
        "n1": element.n1,
        "n2": element.n2,
        "value": element.value,
        "kind": element.kind,
        "role": element.role,
        "cell_index": element.cell_index,
    }


def _stored_element_fields() -> list[dict[str, Any]]:
    """Read the stored CSV element representation into typed fields."""

    fields: list[dict[str, Any]] = []
    with (REFERENCE / "elements.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            def endpoint(value: str) -> int | str:
                return int(value) if value.lstrip("-").isdigit() else value

            fields.append(
                {
                    "name": row["name"],
                    "n1": endpoint(row["node1"]),
                    "n2": endpoint(row["node2"]),
                    "value": float(row["value"]),
                    "kind": row["kind"],
                    "role": row["role"],
                    "cell_index": (
                        int(row["cell_index"]) if row["cell_index"] else None
                    ),
                }
            )
    return fields


def _public_legacy_compilation() -> Any:
    """Build and compile the Python design using compatibility numbering."""

    params = _params()
    design = _load_python_design().build_ipm_2c()
    return params, design, design.compile(node_numbering="legacy")


def test_public_ipm_matches_make_ipm_and_stored_elements() -> None:
    params, _, compiled = _public_legacy_compilation()
    legacy, _ = make_ipm(params, make_coupler_discrete(params, "auto"))

    assert [_element_fields(element) for element in compiled.elements] == [
        _element_fields(element) for element in legacy
    ]
    assert [_element_fields(element) for element in compiled.elements] == (
        _stored_element_fields()
    )
    assert {number: port.node for number, port in compiled.ports.items()} == {
        1: 1,
        2: 4557,
        3: 10000,
        4: 11538,
    }


def test_public_ipm_matches_make_ipm_and_stored_matrices() -> None:
    params, _, compiled = _public_legacy_compilation()
    legacy, _ = make_ipm(params, make_coupler_discrete(params, "auto"))
    actual = build_matrices(compiled.elements)
    expected = build_matrices(legacy)

    for name in ("C", "G", "K", "Bphi"):
        assert (actual[name] != expected[name]).nnz == 0
        stored = sp.load_npz(REFERENCE / f"{name}.npz").tocsr()
        assert (actual[name] != stored).nnz == 0
    assert np.array_equal(actual["Ic"], expected["Ic"])
    assert np.array_equal(actual["Lj"], expected["Lj"])
    arrays = np.load(REFERENCE / "ipm_arrays.npz")
    assert np.array_equal(actual["Ic"], arrays["Ic"])
    assert np.array_equal(actual["Lj"], arrays["Lj"])


def test_creation_numbering_preserves_emission_order_and_permuted_matrices() -> None:
    params = _params()
    design = _load_python_design().build_ipm_2c(params)
    legacy = design.compile(node_numbering="legacy")
    creation = design.compile(node_numbering="creation")
    assert len(legacy.elements) == len(creation.elements)
    assert [
        (element.kind, element.role, element.value, element.cell_index)
        for element in creation.elements
    ] == [
        (element.kind, element.role, element.value, element.cell_index)
        for element in legacy.elements
    ]

    legacy_matrices = build_matrices(legacy.elements)
    creation_matrices = build_matrices(creation.elements)
    creation_indices = {
        int(node): index
        for index, node in enumerate(creation_matrices["nodes"])
    }
    legacy_to_creation = [
        creation_indices[creation.node_map[legacy.reverse_node_map[int(node)]]]
        for node in legacy_matrices["nodes"]
    ]
    assert len(set(legacy_to_creation)) == len(legacy_to_creation)
    for name in ("C", "G", "K"):
        permuted = creation_matrices[name][legacy_to_creation, :][:, legacy_to_creation]
        assert (permuted != legacy_matrices[name]).nnz == 0
    assert (
        creation_matrices["Bphi"][legacy_to_creation, :]
        != legacy_matrices["Bphi"]
    ).nnz == 0
    assert np.array_equal(creation_matrices["Ic"], legacy_matrices["Ic"])
    assert np.array_equal(creation_matrices["Lj"], legacy_matrices["Lj"])


def test_public_ipm_compilation_is_deterministic() -> None:
    params = _params()
    first = _load_python_design().build_ipm_2c().compile("legacy")
    second = _load_python_design().build_ipm_2c().compile("legacy")
    assert [_element_fields(element) for element in first.elements] == [
        _element_fields(element) for element in second.elements
    ]
    assert {number: port.node for number, port in first.ports.items()} == {
        number: port.node for number, port in second.ports.items()
    }


@pytest.mark.skipif(
    not HISTORICAL_REFERENCE.is_dir(),
    reason="historical designs/ipm_2c_fixed artifact is not present",
)
def test_fixture_matches_historical_artifacts_when_present() -> None:
    """Keep the committed fixture tied to the local historical artifacts."""

    assert (REFERENCE / "elements.csv").read_bytes() == (
        HISTORICAL_REFERENCE / "elements.csv"
    ).read_bytes()
    historical_ipm_elements = HISTORICAL_REFERENCE / "ipm_elements.csv"
    if historical_ipm_elements.exists():
        assert (REFERENCE / "elements.csv").read_bytes() == (
            historical_ipm_elements.read_bytes()
        )
    for name in ("C", "G", "K", "Bphi"):
        current = sp.load_npz(REFERENCE / f"{name}.npz").tocsr()
        historical = sp.load_npz(HISTORICAL_REFERENCE / f"{name}.npz").tocsr()
        assert (current != historical).nnz == 0
    with np.load(REFERENCE / "ipm_arrays.npz") as current, np.load(
        HISTORICAL_REFERENCE / "ipm_arrays.npz"
    ) as historical:
        assert np.array_equal(current["Ic"], historical["Ic"])
        assert np.array_equal(current["Lj"], historical["Lj"])
