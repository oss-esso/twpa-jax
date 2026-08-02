"""Per-cell plan wiring: plasma frequency, Cg mapping, blocks, and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from twpa_solver.builders.ipm import (
    ComponentPlan,
    IPMParams,
    apply_lj_scatter,
    build_component_plan,
    build_matrices,
    legacy_scatter_summary,
    make_coupler_discrete,
    make_ipm,
    resolve_scatter_seed,
)
from twpa_solver.builders.profiles import parse_profile_shorthand as shorthand
from twpa_solver.builders.scatter import ScatterSpec

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "ipm_2c_fixed"


@pytest.fixture(scope="module")
def params() -> IPMParams:
    if not DESIGN.exists():
        pytest.skip(f"reference design {DESIGN} is not present")
    return IPMParams(**json.loads((DESIGN / "ipm_summary.json").read_text())["params"])


@pytest.fixture(scope="module")
def coupler(params: IPMParams):
    return make_coupler_discrete(params, "cached")


def cells(params: IPMParams) -> int:
    return params.num_rows * params.array_length


def test_empty_plan_reproduces_the_scalar_design(params) -> None:
    plan = build_component_plan(params)
    assert np.all(plan.lj == params.Lj)
    assert np.all(plan.cj == params.Cj)
    assert np.all(plan.cg == params.Cg)


def test_nominal_cj_holds_plasma_frequency_under_an_arbitrary_lj_profile(params) -> None:
    plan = build_component_plan(params, lj_segments=[shorthand("all:linear:100p->160p")])
    product = plan.lj * plan.cj
    reference = params.Lj * params.Cj
    assert np.max(np.abs(product - reference)) / reference < 1e-14


def test_independent_cj_scatter_breaks_plasma_frequency_on_purpose(params) -> None:
    # Deliberate: the user chose an independent Cj draw over an area-correlated
    # one, so this spread is the specified behavior and must not be "fixed".
    plan = build_component_plan(params, cj_scatter=ScatterSpec(0.02), seed=3)
    product = plan.lj * plan.cj
    assert np.std(product) / np.mean(product) > 1e-3


def test_lj_only_scatter_leaves_cj_alone(params) -> None:
    plan = build_component_plan(params, lj_scatter=ScatterSpec(0.01), seed=1)
    assert np.all(plan.cj == params.Cj)


def test_scatter_reproduces_the_legacy_builder(params, coupler) -> None:
    legacy_circuit, _ = make_ipm(params, coupler)
    apply_lj_scatter(legacy_circuit, sigma=0.01, seed=1)
    legacy = np.array(
        [e.value for e in legacy_circuit if e.kind == "josephson_inductor"]
    )
    plan = build_component_plan(params, lj_scatter=ScatterSpec(0.01), seed=1)
    np.testing.assert_array_equal(plan.lj, legacy)


def test_scattering_one_component_does_not_disturb_another(params) -> None:
    base = build_component_plan(params, lj_scatter=ScatterSpec(0.01), seed=1)
    with_cg = build_component_plan(
        params, lj_scatter=ScatterSpec(0.01), cg_scatter=ScatterSpec(0.05), seed=1
    )
    np.testing.assert_array_equal(base.lj, with_cg.lj)


def test_row_block_profile_touches_only_that_row(params, coupler) -> None:
    plan = build_component_plan(params, lj_segments=[shorthand("rows=0:const:150p")])
    length = params.array_length
    assert np.all(plan.lj[:length] == 150e-12)
    assert np.all(plan.lj[length:] == params.Lj)
    circuit, _ = make_ipm(params, coupler, plan=plan)
    values = [e.value for e in circuit if e.role == "jj_lj"]
    assert values[:length] == [150e-12] * length


def test_half_and_half_profile_splits_at_the_midpoint(params) -> None:
    half = cells(params) // 2
    plan = build_component_plan(
        params,
        lj_segments=[
            shorthand(f"index=0-{half - 1}:const:120p"),
            shorthand(f"index={half}-{cells(params) - 1}:const:150p"),
        ],
    )
    assert np.all(plan.lj[:half] == 120e-12)
    assert np.all(plan.lj[half:] == 150e-12)


def test_cg_boundary_caps_are_halved(params, coupler) -> None:
    plan = build_component_plan(params, cg_segments=[shorthand("all:linear:66f->72f")])
    circuit, _ = make_ipm(params, coupler, plan=plan)
    caps = [(e.value, e.cell_index) for e in circuit if e.role == "jtl_cg"]
    length = params.array_length
    assert caps[0] == (plan.cg[0] / 2.0, 0)
    assert caps[1] == (plan.cg[1], 1)
    assert caps[length] == (plan.cg[length - 1] / 2.0, length - 1)
    assert caps[-1] == (plan.cg[cells(params) - 1] / 2.0, cells(params) - 1)


def test_constant_cg_plan_reproduces_the_scalar_stamps(params, coupler) -> None:
    scalar, _ = make_ipm(params, coupler)
    planned, _ = make_ipm(params, coupler, plan=build_component_plan(params))
    assert [e.value for e in scalar] == [e.value for e in planned]


def test_scatter_changes_values_but_not_the_sparsity_pattern(params, coupler) -> None:
    nominal, _ = make_ipm(params, coupler, plan=build_component_plan(params))
    scattered, _ = make_ipm(
        params, coupler,
        plan=build_component_plan(
            params, lj_scatter=ScatterSpec(0.02), cg_scatter=ScatterSpec(0.02), seed=2
        ),
    )
    before, after = build_matrices(nominal)["C"], build_matrices(scattered)["C"]
    assert (before != 0).nnz == (after != 0).nnz
    assert ((before != 0) != (after != 0)).nnz == 0
    assert abs(before - after).max() > 0.0


def test_make_ipm_still_accepts_the_two_argument_call(params, coupler) -> None:
    circuit, ends = make_ipm(params, coupler)
    assert ends["jj_mod_used"] == cells(params)
    assert circuit


def test_make_ipm_rejects_mod_array_together_with_a_plan(params, coupler) -> None:
    with pytest.raises(ValueError, match="either mod_array or plan"):
        make_ipm(params, coupler, np.ones(cells(params)),
                 plan=build_component_plan(params))


def test_metadata_carries_specs_not_bulk_arrays(params) -> None:
    plan = build_component_plan(
        params, lj_segments=[shorthand("rows=0:const:150p")],
        lj_scatter=ScatterSpec(0.01), seed=4,
    )
    assert plan.metadata["seed"] == 4
    assert plan.metadata["n_cells"] == cells(params)
    assert len(plan.metadata["segments"]["Lj"]) == 1
    encoded = json.dumps(plan.metadata)
    assert len(encoded) < 20_000, "summary metadata must not embed per-cell arrays"


def test_legacy_summary_keys_are_re_emitted(params) -> None:
    plan = build_component_plan(params, lj_scatter=ScatterSpec(0.01), seed=6)
    summary = legacy_scatter_summary(plan)
    assert summary["lj_scatter_enabled"] is True
    assert summary["lj_scatter_sigma"] == pytest.approx(0.01)
    assert summary["lj_scatter_seed"] == 6
    assert summary["lj_scatter_count"] == cells(params)
    assert summary["lj_base_min_h"] == pytest.approx(params.Lj)


@pytest.mark.parametrize(
    "scatter_seed,lj_seed,expected", [(None, None, 1), (None, 7, 7), (5, None, 5), (5, 5, 5)]
)
def test_seed_reconciliation(scatter_seed, lj_seed, expected) -> None:
    assert resolve_scatter_seed(scatter_seed, lj_seed) == expected


def test_conflicting_seeds_are_rejected() -> None:
    with pytest.raises(ValueError, match="disagree"):
        resolve_scatter_seed(1, 7)


def test_plan_arrays_all_share_the_cell_index_space(params) -> None:
    plan = build_component_plan(params)
    assert isinstance(plan, ComponentPlan)
    for array in (plan.lj, plan.cj, plan.cg,
                  plan.lj_nominal, plan.cj_nominal, plan.cg_nominal):
        assert array.shape == (cells(params),)
