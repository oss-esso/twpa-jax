"""Format-neutral technology presets for circuit construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Technology:
    """Electrical and architectural defaults shared by circuit authors."""

    name: str
    components: Mapping[str, Any]
    architecture: Mapping[str, Any]
    cursors: Mapping[str, int]
    ground: int
    coupler_mode: str


def _technology_paths(name: str, search_paths: Sequence[str | Path] | None) -> list[Path]:
    """Return technology candidates in caller order, then repository order."""

    if search_paths is None:
        roots = [Path(__file__).resolve().parents[3] / "designs" / "technology"]
    else:
        roots = [Path(path) for path in search_paths]
    return [
        root / name if root.suffix == ".yaml" else root / f"{name}.yaml"
        for root in roots
    ]


def load_technology(
    name: str,
    search_paths: Sequence[str | Path] | None = None,
) -> Technology:
    """Load one technology preset from ``designs/technology``.

    New presets may separate electrical component values from architecture
    values.  The historical flat ``parameters`` mapping is accepted as a
    component catalogue so existing YAML designs remain valid.
    """

    if not name:
        raise ValueError("technology name must not be empty")
    candidates = _technology_paths(name, search_paths)
    source = next((candidate for candidate in candidates if candidate.exists()), None)
    if source is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"technology preset not found: {name!r} ({searched})")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{source}: invalid technology YAML: {error}") from error
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: technology preset must be a mapping")
    allowed = {"name", "components", "architecture", "parameters", "cursors",
               "ground", "coupler_mode"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"technology {name!r}: unknown keys {sorted(unknown)}")

    legacy = raw.get("parameters", {})
    components = raw.get("components", {})
    architecture = raw.get("architecture", {})
    for label, values in (("parameters", legacy), ("components", components),
                          ("architecture", architecture)):
        if not isinstance(values, Mapping):
            raise ValueError(f"{source}: {label} must be a mapping")
    component_values = {**dict(legacy), **dict(components)}
    cursors = raw.get("cursors", {})
    if not isinstance(cursors, Mapping):
        raise ValueError(f"{source}: cursors must be a mapping")
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in cursors.values()):
        raise ValueError(f"{source}: cursor values must be integers")
    ground = raw.get("ground", 0)
    if not isinstance(ground, int) or isinstance(ground, bool):
        raise ValueError(f"{source}: ground must be an integer")
    coupler_mode = raw.get("coupler_mode", "auto")
    if not isinstance(coupler_mode, str) or not coupler_mode:
        raise ValueError(f"{source}: coupler_mode must be a non-empty string")
    if coupler_mode not in {"auto", "ideal", "optimize"}:
        raise ValueError(
            f"{source}: unsupported coupler_mode {coupler_mode!r}; "
            "expected auto, ideal, or optimize"
        )
    return Technology(
        name=str(raw.get("name", name)),
        components=component_values,
        architecture=dict(architecture),
        cursors={str(key): int(value) for key, value in cursors.items()},
        ground=ground,
        coupler_mode=coupler_mode,
    )


def resolve_builder_parameter(
    parameter: str,
    explicit: Any,
    *,
    design_parameters: Mapping[str, Any],
    technology: Technology | None,
    technology_defaults: Mapping[str, str],
    builder_defaults: Mapping[str, Any] | None = None,
    path: str,
) -> Any:
    """Resolve one builder argument in one shared precedence order.

    The order is explicit call argument, design-level override, technology
    ``components``, technology ``architecture``, builder default, then an
    error that names both the parameter and its hierarchical path.
    """

    if explicit is not None:
        return explicit
    technology_key = technology_defaults.get(parameter)
    if parameter in design_parameters:
        return design_parameters[parameter]
    if technology_key is not None and technology_key in design_parameters:
        return design_parameters[technology_key]
    if technology is not None and technology_key is not None:
        if technology_key in technology.components:
            return technology.components[technology_key]
        if technology_key in technology.architecture:
            return technology.architecture[technology_key]
    if builder_defaults is not None and parameter in builder_defaults:
        return builder_defaults[parameter]
    raise ValueError(
        f"{path}: cannot resolve parameter {parameter!r}"
        f" (technology key {technology_key!r})"
    )
