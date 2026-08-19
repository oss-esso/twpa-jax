"""Rebuilding a stored design as a profiled/scattered variant."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from twpa_solver.builders.ipm import (
    IPMParams,
    assert_source_topology,
    build_variant_design,
)
from twpa_solver.builders.profiles import parse_profile_shorthand as shorthand
from twpa_solver.builders.scatter import ScatterSpec

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "ipm_2c_fixed"
MATRICES = ("C", "G", "K", "Bphi")


@pytest.fixture(scope="module", autouse=True)
def require_design() -> None:
    if not DESIGN.exists():
        pytest.skip(f"reference design {DESIGN} is not present")


def test_stored_params_reproduce_the_stored_netlist() -> None:
    params, mode = assert_source_topology(DESIGN)
    assert isinstance(params, IPMParams)
    assert mode == "auto"


def test_an_ideal_coupler_design_is_detected_not_rejected() -> None:
    # The coupler mode is not stored in the summary, so a design built with
    # --coupler-mode ideal must still be detected by topology matching.
    ideal = DESIGN.parent / "ipm_7c_ideal_m25db_8ghz"
    if not ideal.exists():
        pytest.skip("ideal-coupler reference design is not present")
    assert assert_source_topology(ideal)[1] == "ideal"


def test_an_explicit_wrong_coupler_mode_still_raises() -> None:
    with pytest.raises(ValueError, match="source topology mismatch"):
        assert_source_topology(DESIGN, coupler_mode="optimize")


def test_topology_gate_fires_when_the_params_are_perturbed(tmp_path: Path) -> None:
    summary = json.loads((DESIGN / "ipm_summary.json").read_text())
    summary["params"]["Lj"] = summary["params"]["Lj"] * 1.05
    forged = tmp_path / "forged"
    forged.mkdir()
    (forged / "ipm_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (forged / "ipm_elements.csv").write_bytes((DESIGN / "ipm_elements.csv").read_bytes())
    with pytest.raises(ValueError, match="source topology mismatch"):
        assert_source_topology(forged)


def test_zero_settings_round_trip_the_source_matrices(tmp_path: Path) -> None:
    build_variant_design(DESIGN, tmp_path / "plain", overwrite=True)
    for name in MATRICES:
        rebuilt = sp.load_npz(tmp_path / "plain" / f"{name}.npz").tocsr()
        stored = sp.load_npz(DESIGN / f"{name}.npz").tocsr()
        difference = abs(rebuilt - stored)
        assert (difference.max() if difference.nnz else 0.0) == 0.0


def test_a_profiled_and_scattered_variant_builds(tmp_path: Path) -> None:
    # The gate must run on a nominal rebuild; comparing the variant itself to
    # the source would reject every design that actually changes a value.
    outdir = tmp_path / "variant"
    summary = build_variant_design(
        DESIGN, outdir,
        lj_segments=[shorthand("rows=0-2:const:150p"),
                     shorthand("rows=3-5:linear:123.9p->140p")],
        cg_segments=[shorthand("all:half_cosine:66f->72f")],
        lj_scatter=ScatterSpec(0.01), cj_scatter=ScatterSpec(0.005),
        cg_scatter=ScatterSpec(0.02), seed=7, overwrite=True,
    )
    assert summary["total_elements"] == 16192

    arrays = np.load(outdir / "ipm_arrays.npz")
    assert arrays["Lj_nominal"][0] == pytest.approx(150e-12)
    assert arrays["Lj_nominal"][-1] == pytest.approx(140e-12)
    for name in ("Lj", "Cj", "Cg"):
        relative = np.abs(arrays[name] / arrays[f"{name}_nominal"] - 1.0)
        assert relative.max() > 1e-4, f"{name} scatter was not applied"

    for name in MATRICES[:1]:
        rebuilt = sp.load_npz(outdir / f"{name}.npz").tocsr()
        stored = sp.load_npz(DESIGN / f"{name}.npz").tocsr()
        assert abs(rebuilt - stored).max() > 0.0


def test_variant_summary_can_regenerate_its_own_profile(tmp_path: Path) -> None:
    outdir = tmp_path / "regen"
    build_variant_design(
        DESIGN, outdir, lj_segments=[shorthand("rows=0:const:150p")],
        seed=2, overwrite=True,
    )
    plan = json.loads((outdir / "ipm_summary.json").read_text())["component_plan"]
    assert plan["segments"]["Lj"][0]["shape"] == "const"
    assert plan["segments"]["Lj"][0]["start"] == pytest.approx(150e-12)
    assert plan["segments"]["Lj"][0]["select"]["rows"] == [0, 0]
    assert plan["seed"] == 2


def test_existing_output_is_protected_without_overwrite(tmp_path: Path) -> None:
    outdir = tmp_path / "guarded"
    outdir.mkdir()
    with pytest.raises(FileExistsError):
        build_variant_design(DESIGN, outdir)


def test_summary_stays_small_enough_to_load_per_solve(tmp_path: Path) -> None:
    outdir = tmp_path / "small"
    build_variant_design(DESIGN, outdir, lj_scatter=ScatterSpec(0.01), overwrite=True)
    assert (outdir / "ipm_summary.json").stat().st_size < 32_000
