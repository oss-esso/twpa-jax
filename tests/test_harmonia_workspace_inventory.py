from __future__ import annotations

from pathlib import Path

import pytest

from scripts.inventory_harmonia_workspace import (
    count_keyword_hits,
    device_tags,
    extract_functions,
    inspect_file,
)


def test_extract_functions_basic() -> None:
    text = """
function add_JTL!(circuit, n_start, ground, Cg, Lj, Cj, N_cell; sigma=0.0)
    return n_start + N_cell
end

make_IPM(circuit, start_node_top, start_node_bot) = circuit
"""
    funcs = extract_functions(text)

    assert "add_JTL!" in funcs
    assert "make_IPM" in funcs


def test_keyword_hits_detect_josephson_hb() -> None:
    text = """
using JosephsonCircuits
sim = hbsolve(ws, wp, sources, Nmodulationharmonics, Npumpharmonics, circuit, circuitdefs)
s = sim.linearized.S(outputport=1, inputport=1)
qe = sim.linearized.QE(outputport=1)
"""
    hits = count_keyword_hits(text)

    assert hits["josephson"] >= 1
    assert hits["hb"] >= 1
    assert hits["sparams"] >= 1
    assert hits["noise"] >= 1


def test_device_tags_detect_ipm_rfsquid() -> None:
    tags = device_tags(
        text="make_IPM(...) add_RF_JTL!(...) Lrf directional coupler",
        rel="IPM_rf_squid.jl",
    )

    assert "IPM" in tags
    assert "RF_SQUID" in tags
    assert "DIRECTIONAL_COUPLER" in tags


def test_inspect_real_harmonia_jl_file_if_available() -> None:
    path = Path(r"D:\Projects\Thesis\Harmonia.jl\src\Transmission_line_block.jl")
    repo = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not path.exists():
        pytest.skip("Local Harmonia.jl source not available.")

    record = inspect_file("Harmonia.jl", repo, path)

    assert record.repo_name == "Harmonia.jl"
    assert record.classification == "STABLE_PACKAGE_CODE"
    assert "add_JTL!" in record.functions or "add_TL!" in record.functions