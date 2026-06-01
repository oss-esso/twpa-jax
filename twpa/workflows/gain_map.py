"""
Package-native pump-HB plus small-signal gain-map orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import csv
import json
import numpy as np

from twpa.core.layout import LineLayout
from twpa.core.params import NonlinearParams
from twpa.inference.synthetic import (
    make_gain_frequency_plan,
    make_gain_sweep_config_for_frequencies,
)
from twpa.io.reports import jsonify
from twpa.nonlinear.gain import (
    GainOperatingMap,
    extract_labelled_gain_trace,
    solve_gain_operating_map_from_pump_results,
)
from twpa.nonlinear.pump_hb_ladder import (
    PumpDriveConfig,
    PumpHBLadderConfig,
    PumpHBLadderResult,
    solve_pump_hb_ladder,
)


@dataclass(frozen=True)
class NativeGainMapResult:
    """Compact native gain-map result with full package objects attached."""

    pump_frequencies_hz: np.ndarray
    pump_current_ratios: np.ndarray
    signal_frequencies_hz: np.ndarray
    pump_results: tuple[PumpHBLadderResult, ...]
    operating_map: GainOperatingMap
    signal_gain_db: np.ndarray
    matched_power_gain_db: np.ndarray
    idler_conversion_db: np.ndarray
    converged: np.ndarray
    metadata: Mapping[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return bool(
            self.pump_results
            and all(result.converged for result in self.pump_results)
            and np.all(self.converged)
        )

    @property
    def status(self) -> str:
        return "pass" if self.passed else "partial"

    def to_dict(self) -> dict[str, Any]:
        finite = np.isfinite(self.signal_gain_db)
        return {
            "status": self.status,
            "passed": self.passed,
            "driver": "twpa.workflows.gain_map.solve_native_gain_map",
            "n_pump_points": len(self.pump_results),
            "n_signal_points": int(self.signal_frequencies_hz.size),
            "n_converged": int(np.sum(self.converged)),
            "pump_all_converged": bool(all(result.converged for result in self.pump_results)),
            "gain_all_converged": bool(np.all(self.converged)),
            "max_gain_db": float(np.nanmax(self.signal_gain_db)) if np.any(finite) else None,
            "metadata": jsonify(dict(self.metadata or {})),
        }


def solve_native_gain_map(
    layout: LineLayout,
    nonlinear_params: NonlinearParams,
    *,
    pump_frequencies_hz: Sequence[float],
    pump_current_ratios: Sequence[float],
    signal_frequencies_hz: Sequence[float],
    pump_config: PumpHBLadderConfig | None = None,
    source_impedance_ohm: float = 50.0,
    pump_phase_rad: float = 0.0,
    signal_current_rms_A: complex = 1e-12 + 0j,
    metadata: Mapping[str, Any] | None = None,
) -> NativeGainMapResult:
    """Solve a native pump-frequency/current-ratio gain cube in process."""
    pump_cfg = pump_config or PumpHBLadderConfig()
    pump_f = np.asarray(tuple(pump_frequencies_hz), dtype=float)
    pump_i = np.asarray(tuple(pump_current_ratios), dtype=float)
    signal_f = np.asarray(tuple(signal_frequencies_hz), dtype=float)
    if pump_f.ndim != 1 or pump_f.size == 0 or np.any(pump_f <= 0.0):
        raise ValueError("pump_frequencies_hz must be a non-empty positive 1D sequence")
    if pump_i.ndim != 1 or pump_i.size == 0 or np.any(pump_i < 0.0):
        raise ValueError("pump_current_ratios must be a non-empty non-negative 1D sequence")
    if signal_f.ndim != 1 or signal_f.size == 0 or np.any(signal_f <= 0.0):
        raise ValueError("signal_frequencies_hz must be a non-empty positive 1D sequence")

    pump_results: list[PumpHBLadderResult] = []
    for frequency in pump_f:
        for ratio in pump_i:
            drive = PumpDriveConfig.from_current_rms(
                pump_frequency_hz=float(frequency),
                current_rms_A=float(ratio) * nonlinear_params.I_star_A,
                source_impedance_ohm=source_impedance_ohm,
                phase_rad=pump_phase_rad,
            )
            pump_results.append(
                solve_pump_hb_ladder(
                    layout,
                    nonlinear_params,
                    drive=drive,
                    pump_config=pump_cfg,
                    metadata={
                        "driver": "twpa.workflows.gain_map.solve_native_gain_map",
                        "pump_current_ratio": float(ratio),
                    },
                )
            )

    signal_labels = tuple(f"signal_{index}" for index in range(signal_f.size))
    idler_labels = tuple(f"idler_{index}" for index in range(signal_f.size))

    def target_plan_factory(pump: PumpHBLadderResult):
        idler_f = 2.0 * pump.drive.pump_frequency_hz - signal_f
        if np.any(idler_f <= 0.0):
            raise ValueError("All signal frequencies must produce positive DP4WM idlers")
        return make_gain_frequency_plan(
            pump_frequency_hz=pump.drive.pump_frequency_hz,
            signal_frequency_hz=signal_f,
            idler_frequency_hz=idler_f,
            pump_label=pump.drive.pump_label,
            signal_labels=signal_labels,
            idler_labels=idler_labels,
            n_pump_harmonics=pump_cfg.n_pump_harmonics,
            include_negative=pump_cfg.include_negative_frequencies,
            include_dc=pump_cfg.include_dc,
        )

    output_impedance = (
        1.0 / pump_cfg.distributed.load_conductance_S
        if pump_cfg.distributed.load_conductance_S > 0.0
        else source_impedance_ohm
    )

    def sweep_config_factory(_plan):
        return make_gain_sweep_config_for_frequencies(
            signal_labels=signal_labels,
            idler_labels=idler_labels,
            input_node=pump_cfg.distributed.input_node,
            output_node=None if pump_cfg.distributed.output_node < 0 else pump_cfg.distributed.output_node,
            signal_current_rms_A=signal_current_rms_A,
            input_impedance_ohm=source_impedance_ohm,
            output_impedance_ohm=output_impedance,
        )

    operating_map = solve_gain_operating_map_from_pump_results(
        pump_results,
        target_plan_factory=target_plan_factory,
        sweep_config_factory=sweep_config_factory,
    )
    traces = [extract_labelled_gain_trace(point.sweep) for point in operating_map.points]
    cube_shape = (pump_f.size, pump_i.size, signal_f.size)

    def cube(name: str, *, dtype=float) -> np.ndarray:
        flat = np.asarray([np.asarray(getattr(trace, name)) for trace in traces], dtype=dtype)
        return flat.reshape(cube_shape)

    return NativeGainMapResult(
        pump_frequencies_hz=pump_f,
        pump_current_ratios=pump_i,
        signal_frequencies_hz=signal_f,
        pump_results=tuple(pump_results),
        operating_map=operating_map,
        signal_gain_db=cube("signal_gain_db"),
        matched_power_gain_db=cube("matched_power_gain_db"),
        idler_conversion_db=cube("idler_conversion_db"),
        converged=cube("converged", dtype=bool),
        metadata={
            "layout": layout.summary(),
            "pump_config": pump_cfg.to_dict(),
            **dict(metadata or {}),
        },
    )


def export_native_gain_map_artifacts(
    result: NativeGainMapResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write compact JSON, NPZ, and CSV gain-map artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays_path = out / "full_gain_map_100mm_arrays.npz"
    np.savez_compressed(
        arrays_path,
        pump_frequency_ghz=result.pump_frequencies_hz / 1e9,
        pump_current_ratio=result.pump_current_ratios,
        signal_frequency_ghz=result.signal_frequencies_hz / 1e9,
        signal_gain_db=result.signal_gain_db,
        matched_power_gain_db=result.matched_power_gain_db,
        idler_conversion_db=result.idler_conversion_db,
        converged=result.converged,
    )
    csv_path = out / "full_gain_map_100mm_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["pump_frequency_ghz", "pump_current_ratio", "signal_frequency_ghz", "signal_gain_db", "converged"]
        )
        for i, fp in enumerate(result.pump_frequencies_hz / 1e9):
            for j, ratio in enumerate(result.pump_current_ratios):
                for k, fs in enumerate(result.signal_frequencies_hz / 1e9):
                    writer.writerow([fp, ratio, fs, result.signal_gain_db[i, j, k], result.converged[i, j, k]])
    summary_path = out / "full_gain_map_100mm_summary.json"
    payload = {
        **result.to_dict(),
        "artifact_paths": {
            "arrays_npz": str(arrays_path),
            "points_csv": str(csv_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(jsonify(payload), indent=2), encoding="utf-8")
    return dict(payload["artifact_paths"])


__all__ = [
    "NativeGainMapResult",
    "solve_native_gain_map",
    "export_native_gain_map_artifacts",
]
