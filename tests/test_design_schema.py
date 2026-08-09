import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from twpa_solver.design import compile_design, load_design
from twpa_solver.design.errors import DesignParameterError, DesignSchemaError


def _base() -> dict:
    return {"schema_version": 1, "name": "tiny", "ground": 0,
            "parameters": {}, "cursors": {"signal": 1}, "topology": [
                {"type": "port", "name": "p", "cursor": "signal", "port": 1}
            ]}


def test_unknown_block_field_has_path():
    spec = _base()
    spec["topology"][0]["unexpected"] = 1
    with pytest.raises(DesignSchemaError, match=r"topology\[0\].*unexpected"):
        compile_design(spec)


def test_partial_parameter_interpolation_is_rejected():
    spec = _base()
    spec["topology"][0]["port"] = "${number}e1"
    with pytest.raises(DesignParameterError, match="whole scalar"):
        compile_design(spec)


def test_strict_mode_rejects_unused_parameters():
    spec = _base()
    spec["parameters"] = {"unused": 1}
    with pytest.raises(DesignParameterError, match="unused"):
        compile_design(spec, strict=True)


def test_missing_required_block_field_is_rejected():
    spec = _base()
    spec["topology"][0].pop("port")
    with pytest.raises(DesignSchemaError, match="missing fields"):
        compile_design(spec)


def test_unsafe_yaml_tags_are_rejected():
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        source = Path(directory) / "unsafe.yaml"
        source.write_text("!!python/object/apply:os.system ['echo bad']", encoding="utf-8")
        with pytest.raises(DesignSchemaError, match="invalid YAML"):
            load_design(source)
