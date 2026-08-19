"""Role/cell-index tagging, and byte identity against the live 2c design.

The tagging exists because Cg is not structurally identifiable: filtering for
"capacitor to ground on a Josephson node" returns 2522 elements on 2c, of which
only 2514 are Cg -- six are transmission-line Cl and two are coupler end caps.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
import scipy.sparse as sp

from twpa_solver.builders.ipm import (
    IPMParams,
    build_matrices,
    make_coupler_discrete,
    make_ipm,
)

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "ipm_2c_fixed"
CSV_FIELDS = ("name", "node1", "node2", "kind")


@pytest.fixture(scope="module")
def design_params() -> IPMParams:
    if not DESIGN.exists():
        pytest.skip(f"reference design {DESIGN} is not present")
    params = json.loads((DESIGN / "ipm_summary.json").read_text())["params"]
    return IPMParams(**params)


@pytest.fixture(scope="module")
def nominal_circuit(design_params: IPMParams) -> list:
    coupler = make_coupler_discrete(design_params, "auto")
    circuit, _ = make_ipm(design_params, coupler)
    return circuit


@pytest.fixture(scope="module")
def source_rows() -> list[dict[str, str]]:
    with (DESIGN / "ipm_elements.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_nominal_rebuild_is_element_identical(nominal_circuit, source_rows) -> None:
    assert len(nominal_circuit) == len(source_rows)
    for index, (element, row) in enumerate(zip(nominal_circuit, source_rows), start=1):
        actual = (element.name, str(element.n1), str(element.n2), element.kind)
        assert actual == tuple(row[f] for f in CSV_FIELDS), f"element {index}"
        assert float(element.value) == float(row["value"]), f"element {index}"


@pytest.mark.parametrize("name", ["C", "G", "K", "Bphi"])
def test_nominal_rebuild_reproduces_the_stored_matrices(nominal_circuit, name) -> None:
    rebuilt = build_matrices(nominal_circuit)[name]
    stored = sp.load_npz(DESIGN / f"{name}.npz").tocsr()
    difference = abs(rebuilt - stored)
    assert (difference.max() if difference.nnz else 0.0) == 0.0


def test_role_counts_match_the_device_layout(nominal_circuit, design_params) -> None:
    roles = Counter(element.role for element in nominal_circuit)
    cells = design_params.num_rows * design_params.array_length
    assert roles["jj_lj"] == cells
    assert roles["jj_cj"] == cells
    # One ground cap per JTL node, so one more per row than there are cells.
    assert roles["jtl_cg"] == design_params.num_rows * (design_params.array_length + 1)


def test_every_element_carries_a_role(nominal_circuit) -> None:
    assert all(element.role for element in nominal_circuit)


def test_junction_cell_indices_are_contiguous_in_element_order(
    nominal_circuit, design_params
) -> None:
    # The Lj scatter stream is drawn in cell order, so this ordering is what
    # makes the legacy-parity guarantee meaningful.
    indices = [e.cell_index for e in nominal_circuit if e.role == "jj_lj"]
    assert indices == list(range(design_params.num_rows * design_params.array_length))


def test_paired_junction_elements_share_a_cell_index(nominal_circuit) -> None:
    lj = [e.cell_index for e in nominal_circuit if e.role == "jj_lj"]
    cj = [e.cell_index for e in nominal_circuit if e.role == "jj_cj"]
    assert lj == cj


def test_structural_filter_alone_would_over_count_cg(nominal_circuit) -> None:
    junction_nodes = {
        node
        for element in nominal_circuit
        if element.kind == "josephson_inductor"
        for node in (int(element.n1), int(element.n2))
    }
    structural = [
        element for element in nominal_circuit
        if element.kind == "capacitor"
        and int(element.n2) == 0
        and int(element.n1) in junction_nodes
    ]
    tagged = [e for e in nominal_circuit if e.role == "jtl_cg"]
    assert len(structural) > len(tagged)


def test_written_csv_keeps_the_new_columns(tmp_path: Path, nominal_circuit) -> None:
    from twpa_solver.builders.ipm import IPMParams as _P  # noqa: F401
    header_path = tmp_path / "elements.csv"
    with header_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "name", "node1", "node2", "value", "kind",
                         "role", "cell_index"])
        for index, element in enumerate(nominal_circuit[:20], 1):
            writer.writerow([index, element.name, element.n1, element.n2,
                             element.value, element.kind, element.role,
                             element.cell_index])
    with header_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {"role", "cell_index"} <= set(rows[0])
