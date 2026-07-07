"""IPM JTWPA topology assembled from reusable blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from twpa_solver_old.model.blocks import (
    COUPLER_MODEL_COMPACT,
    COUPLER_MODEL_MARKER,
    DirectionalCouplerBlock,
    DirectionalCouplerMarkerBlock,
    JosephsonTransmissionLineBlock,
    PortBlock,
)
from twpa_solver_old.model.topology import CircuitBuilder, CircuitModel
from twpa_solver_old.model.units import CONSTANTS

OLD_JULIA_TARGET_JUNCTIONS = 418 * 6
OLD_JULIA_LJ_H = 2 * 79e-12
OLD_JULIA_CG_F = 66e-15
OLD_JULIA_CJ_F = 145e-15
OLD_JULIA_LL_H = 10 * 4.13e-12
OLD_JULIA_LM_H = 2000 * 4.13e-12
OLD_JULIA_K = 0.999
OLD_JULIA_POWER_OFFSET_DB = 32.0


@dataclass(frozen=True)
class IPMConfig:
    cells_per_line: int = 4
    critical_current_a: float = 8e-6
    shunt_capacitance_f: float = 70e-15
    z0_ohm: float = 50.0
    coupler_loading_s: float = 0.0
    coupler_inductance_top_h: float = 41.3e-12
    coupler_inductance_bottom_h: float = 41.3e-12
    coupler_k: float = 0.25
    coupler_shunt_capacitance_f: float = 17.3e-15
    coupler_mutual_capacitance_f: float = 1.0e-15

    def __post_init__(self) -> None:
        if self.cells_per_line <= 0:
            raise ValueError("cells_per_line must be positive")
        if self.critical_current_a <= 0.0:
            raise ValueError("critical_current_a must be positive")
        if self.shunt_capacitance_f <= 0.0:
            raise ValueError("shunt_capacitance_f must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def build_ipm_jtwpa(config: IPMConfig | None = None) -> CircuitModel:
    """Backward-compatible alias for the reduced marker IPM topology."""
    return build_ipm_jtwpa_reduced_marker(config)


def build_ipm_jtwpa_reduced_marker(config: IPMConfig | None = None) -> CircuitModel:
    """Build reduced Port -> marker -> JTL -> marker -> JTL -> Port topology."""
    cfg = config or IPMConfig()
    total_cells = 2 * cfg.cells_per_line
    builder = CircuitBuilder(total_cells + 1)
    PortBlock("input", 0, cfg.z0_ohm).apply(builder)
    DirectionalCouplerMarkerBlock(1, cfg.coupler_loading_s).apply(builder)
    JosephsonTransmissionLineBlock(
        start_node=0,
        num_cells=cfg.cells_per_line,
        critical_current_a=cfg.critical_current_a,
        shunt_capacitance_f=cfg.shunt_capacitance_f,
        label_prefix="ipm_a",
    ).apply(builder)
    DirectionalCouplerMarkerBlock(cfg.cells_per_line, cfg.coupler_loading_s).apply(builder)
    JosephsonTransmissionLineBlock(
        start_node=cfg.cells_per_line,
        num_cells=cfg.cells_per_line,
        critical_current_a=cfg.critical_current_a,
        shunt_capacitance_f=cfg.shunt_capacitance_f,
        label_prefix="ipm_b",
    ).apply(builder)
    PortBlock("output", total_cells, cfg.z0_ohm).apply(builder)
    builder.metadata.update(
        {
            "topology": "ipm_jtwpa",
            "topology_type": "ipm_jtwpa_reduced_marker",
            "coupler_model": COUPLER_MODEL_MARKER,
            "config": cfg.to_dict(),
            "description": "Reduced marker topology retained for regression only",
        }
    )
    return builder.build()


def build_ipm_jtwpa_physical_coupler(config: IPMConfig | None = None) -> CircuitModel:
    """Build a four-port IPM with two physical coupled-inductor couplers."""
    cfg = config or IPMConfig()
    n = cfg.cells_per_line
    output_node = 2 * n + 2
    pump_input = output_node + 1
    pump_after_c1 = output_node + 2
    pump_before_c2 = output_node + 3
    pump_source = output_node + 4
    builder = CircuitBuilder(output_node + 5)

    PortBlock("input", 0, cfg.z0_ohm).apply(builder)
    PortBlock("output", output_node, cfg.z0_ohm).apply(builder)
    PortBlock("pump_isolation", pump_input, cfg.z0_ohm).apply(builder)
    PortBlock("pump", pump_source, cfg.z0_ohm).apply(builder)

    DirectionalCouplerBlock(
        top_start=0,
        top_end=1,
        bottom_start=pump_input,
        bottom_end=pump_after_c1,
        inductance_top_h=cfg.coupler_inductance_top_h,
        inductance_bottom_h=cfg.coupler_inductance_bottom_h,
        coupling_k=cfg.coupler_k,
        shunt_capacitance_top_f=cfg.coupler_shunt_capacitance_f,
        shunt_capacitance_bottom_f=cfg.coupler_shunt_capacitance_f,
        mutual_capacitance_f=cfg.coupler_mutual_capacitance_f,
        label="ipm_coupler_1",
    ).apply(builder)

    JosephsonTransmissionLineBlock(
        start_node=1,
        num_cells=n,
        critical_current_a=cfg.critical_current_a,
        shunt_capacitance_f=cfg.shunt_capacitance_f,
        label_prefix="ipm_phys_a",
    ).apply(builder)

    builder.add_series_inductor(
        pump_after_c1,
        pump_before_c2,
        cfg.coupler_inductance_bottom_h * max(n, 1),
    )
    builder.add_shunt_capacitor(pump_after_c1, 0.5 * cfg.coupler_shunt_capacitance_f)
    builder.add_shunt_capacitor(pump_before_c2, 0.5 * cfg.coupler_shunt_capacitance_f)

    DirectionalCouplerBlock(
        top_start=n + 1,
        top_end=n + 2,
        bottom_start=pump_before_c2,
        bottom_end=pump_source,
        inductance_top_h=cfg.coupler_inductance_top_h,
        inductance_bottom_h=cfg.coupler_inductance_bottom_h,
        coupling_k=cfg.coupler_k,
        shunt_capacitance_top_f=cfg.coupler_shunt_capacitance_f,
        shunt_capacitance_bottom_f=cfg.coupler_shunt_capacitance_f,
        mutual_capacitance_f=cfg.coupler_mutual_capacitance_f,
        pump_source_node=pump_source,
        label="ipm_coupler_2",
    ).apply(builder)

    JosephsonTransmissionLineBlock(
        start_node=n + 2,
        num_cells=n,
        critical_current_a=cfg.critical_current_a,
        shunt_capacitance_f=cfg.shunt_capacitance_f,
        label_prefix="ipm_phys_b",
    ).apply(builder)

    builder.metadata.update(
        {
            "topology": "ipm_jtwpa_physical_coupler",
            "topology_type": "ipm_jtwpa_physical_coupler",
            "coupler_model": COUPLER_MODEL_COMPACT,
            "pump_source_node": pump_source,
            "pump_source_port": "pump",
            "config": cfg.to_dict(),
            "description": "Four-port physical-coupler IPM with two coupled-inductor couplers",
        }
    )
    return builder.build()


def old_julia_parity_config(config: IPMConfig | None = None) -> IPMConfig:
    """Return the closest Python compact-coupler profile to old Julia constants."""
    cfg = config or IPMConfig()
    return replace(
        cfg,
        critical_current_a=CONSTANTS.reduced_phi0 / OLD_JULIA_LJ_H,
        shunt_capacitance_f=OLD_JULIA_CG_F,
        coupler_inductance_top_h=OLD_JULIA_LL_H,
        coupler_inductance_bottom_h=OLD_JULIA_LM_H,
        coupler_k=OLD_JULIA_K,
        coupler_shunt_capacitance_f=10 * 1.73e-15,
        coupler_mutual_capacitance_f=0.0,
    )


def build_ipm_jtwpa_old_constants_compact_surrogate(config: IPMConfig | None = None) -> CircuitModel:
    """Build a sandbox profile with old Julia constants but surrogate geometry."""
    model = build_ipm_jtwpa_physical_coupler(old_julia_parity_config(config))
    metadata = dict(model.metadata)
    metadata.update(
        {
            "topology": "ipm_jtwpa_old_constants_compact_surrogate",
            "topology_type": "ipm_jtwpa_old_constants_compact_surrogate",
            "geometry_profile": "old_constants_compact_surrogate",
            "coupler_model": COUPLER_MODEL_COMPACT,
            "old_julia_parity_mode": False,
            "surrogate_topology": True,
            "deprecated_alias": "ipm_jtwpa_old_julia_parity",
            "old_julia_source_power_offset_db": OLD_JULIA_POWER_OFFSET_DB,
            "old_julia_port_convention": {
                "input_port_equivalent": 1,
                "output_port_equivalent": 2,
                "pump_port_equivalent": 4,
            },
            "historical_target_cells_or_junctions": OLD_JULIA_TARGET_JUNCTIONS,
            "historical_old_julia_constants": {
                "Nj": OLD_JULIA_TARGET_JUNCTIONS,
                "Lj_H": OLD_JULIA_LJ_H,
                "Cj_F": OLD_JULIA_CJ_F,
                "Cg_F": OLD_JULIA_CG_F,
                "Ll_H": OLD_JULIA_LL_H,
                "Lm_H": OLD_JULIA_LM_H,
                "K": OLD_JULIA_K,
            },
            "parity_limitations": (
                "Uses old constants and pump convention with compact coupled-inductor "
                "surrogate geometry. It is not the old Julia/Harmonia circuit generated "
                "by build_old_ipm_circuit()."
            ),
        }
    )
    return replace(model, metadata=metadata)


def build_ipm_jtwpa_old_julia_parity(config: IPMConfig | None = None) -> CircuitModel:
    """Backward-compatible alias for the old-constants compact surrogate."""
    model = build_ipm_jtwpa_old_constants_compact_surrogate(config)
    metadata = dict(model.metadata)
    metadata["requested_deprecated_topology"] = "ipm_jtwpa_old_julia_parity"
    metadata["warning"] = "This is a compact surrogate, not old-Julia circuit parity."
    return replace(model, metadata=metadata)


def build_ipm_topology(name: str, config: IPMConfig | None = None) -> CircuitModel:
    if name in {"ipm_jtwpa", "ipm_jtwpa_reduced_marker"}:
        return build_ipm_jtwpa_reduced_marker(config)
    if name == "ipm_jtwpa_physical_coupler":
        return build_ipm_jtwpa_physical_coupler(config)
    if name == "ipm_jtwpa_old_constants_compact_surrogate":
        return build_ipm_jtwpa_old_constants_compact_surrogate(config)
    if name == "ipm_jtwpa_old_julia_parity":
        return build_ipm_jtwpa_old_julia_parity(config)
    raise ValueError(f"unknown IPM topology {name}")
