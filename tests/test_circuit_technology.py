"""Technology loading and declared builder-default gates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from twpa_solver.circuit import Circuit, Technology, load_technology


ROOT = Path(__file__).resolve().parents[1]


def _technology(
    components: dict[str, Any] | None = None,
    architecture: dict[str, Any] | None = None,
) -> Technology:
    """Construct a small technology fixture for resolution tests."""

    return Technology(
        name="test",
        components=components or {},
        architecture=architecture or {},
        cursors={"signal": 1},
        ground=0,
        coupler_mode="auto",
    )


def _line_values(circuit: Circuit, *, L: float | None = None, C: float | None = None) -> tuple[float, float]:
    """Build one cell and return its inductor and capacitor values."""

    path = circuit.path("signal")
    line = circuit.add_transmission_line(path, cells=1, L=L, C=C)
    return float(line.cells[0].extras["L"].value), float(line.cells[0].Cg.value)


def test_loader_accepts_new_sections_and_legacy_flat_parameters(tmp_path: Path) -> None:
    """Both technology file layouts produce the same public data model."""

    (tmp_path / "new.yaml").write_text(
        """name: new\ncomponents:\n  Ll: 1.0\narchitecture:\n  cells: 2\ncursors:\n  signal: 7\nground: 0\ncoupler_mode: auto\n""",
        encoding="utf-8",
    )
    (tmp_path / "flat.yaml").write_text(
        """name: flat\nparameters:\n  Ll: 1.0\ncursors:\n  signal: 7\nground: 0\ncoupler_mode: auto\n""",
        encoding="utf-8",
    )

    new = load_technology("new", [tmp_path])
    flat = load_technology("flat", [tmp_path])

    assert new.components == {"Ll": 1.0}
    assert new.architecture == {"cells": 2}
    assert flat.components == {"Ll": 1.0}
    assert flat.architecture == {}


def test_loader_rejects_removed_cached_coupler_mode(tmp_path: Path) -> None:
    (tmp_path / "cached.yaml").write_text(
        "name: cached\ncoupler_mode: cached\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unsupported coupler_mode"):
        load_technology("cached", [tmp_path])


def test_builder_resolution_order_has_one_gate_per_level() -> None:
    """Explicit, design, component, architecture, builder, and error resolve distinctly."""

    explicit = Circuit("explicit", _technology({"Ll": 1.0, "Cl": 2.0}))
    explicit.set_design_parameters({"Ll": 3.0, "Cl": 4.0})
    assert _line_values(explicit, L=5.0, C=6.0) == (5.0, 6.0)

    design = Circuit("design", _technology({"Ll": 1.0, "Cl": 2.0}))
    design.set_design_parameters({"Ll": 3.0, "Cl": 4.0})
    assert _line_values(design) == (3.0, 4.0)

    component = Circuit("component", _technology({"Ll": 1.0, "Cl": 2.0}))
    assert _line_values(component) == (1.0, 2.0)

    architecture = Circuit("architecture", _technology(architecture={"Ll": 7.0, "Cl": 8.0}))
    assert _line_values(architecture) == (7.0, 8.0)

    class DefaultCircuit(Circuit):
        """Fixture exposing the shared resolver's final builder-default level."""

        BUILDER_DEFAULTS = {"L": 9.0, "C": 10.0}

    builder = DefaultCircuit("builder")
    assert _line_values(builder) == (9.0, 10.0)

    missing = Circuit("missing")
    with pytest.raises(ValueError, match=r"signal.*parameter 'L'"):
        _line_values(missing)


def test_jj_line_uses_declared_technology_defaults() -> None:
    """The JJ builder resolves all three declared component keys."""

    circuit = Circuit("jj", _technology({"Lj": 1.0, "Cj": 2.0, "Cg": 3.0}))
    path = circuit.path("signal")
    line = circuit.add_jj_line(path, cells=1)
    assert line.cells[0].Lj is not None
    assert line.cells[0].Cj is not None
    assert line.cells[0].Cg is not None
    assert (line.cells[0].Lj.value, line.cells[0].Cj.value,
            line.cells[0].Cg.value) == (1.0, 2.0, 1.5)


def test_parallel_lc_is_extensible_without_central_dispatch() -> None:
    """A new block resolves only its own ``Lk`` declaration."""

    circuit = Circuit("parallel_lc", _technology({"Lk": 11.0}))
    first, second = circuit.add_parallel_lc(
        circuit.node("a"), circuit.ground, C=12.0, name="filter"
    )
    assert first.value == 11.0
    assert second.value == 12.0
    assert [element.name for element in circuit.compile().elements] == [
        "filter.L", "filter.C"
    ]


def test_default_ipm_digest_contract() -> None:
    """The no-argument Python design retains both numbering-policy contracts."""

    module_path = ROOT / "designs" / "python" / "ipm_2c.py"
    spec = importlib.util.spec_from_file_location("technology_ipm_2c", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    expected = {
        "legacy": "efc579029f6a64e0",
        "creation": "39a3bc72da352ab2",
    }
    for policy, digest in expected.items():
        elements = module.build_ipm_2c().compile(policy).elements
        payload = json.dumps(
            [element.__dict__ for element in elements], sort_keys=True
        ).encode()
        assert len(elements) == 16192
        assert hashlib.sha256(payload).hexdigest()[:16] == digest


def test_ipm_entry_point_uses_junction_values_from_second_technology(
    tmp_path: Path,
) -> None:
    """A replacement technology changes Lj without editing the Python design."""

    source = ROOT / "designs" / "technology" / "ipm_default.yaml"
    technology_data = yaml.safe_load(source.read_text(encoding="utf-8"))
    technology_data["components"]["Lj"] = 222.0e-12
    alternate_path = tmp_path / "ipm_default.yaml"
    alternate_path.write_text(yaml.safe_dump(technology_data), encoding="utf-8")

    alternate = load_technology("ipm_default", [tmp_path])
    module_path = ROOT / "designs" / "python" / "ipm_2c.py"
    spec = importlib.util.spec_from_file_location("alternate_ipm_2c", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    compiled = module.build_ipm_2c(technology=alternate).compile().elements
    junctions = [
        element for element in compiled if element.kind == "josephson_inductor"
    ]
    assert junctions
    assert junctions[0].value == 222.0e-12
