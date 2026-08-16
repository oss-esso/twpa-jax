"""Measure the half-period Z2 symmetry of stored pump solutions.

The measurement uses dense Fourier solutions.  An odd-only basis is rejected
because it would force the even-harmonic result to zero by construction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _load_dense_solution(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load coefficients and metadata, requiring an even-capable basis."""
    with np.load(path) as data:
        names = set(data.files)
        modes_name = "pump_modes" if "pump_modes" in names else "harmonics"
        modes = np.asarray(data[modes_name], dtype=int)
        coefficients = np.asarray(data["X_real"] + 1j * data["X_imag"], dtype=complex)
    report_path = path.parent / "pump_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    metadata = report.get("metadata", report)
    policy = str(metadata.get("pump_mode_policy", "unknown"))
    if not np.any(modes % 2 == 0):
        raise ValueError(
            f"{path} is odd-only ({policy}); an even-harmonic measurement would be circular"
        )
    return modes, coefficients, {"pump_mode_policy": policy, "report_path": str(report_path)}


def measure_solution(path: Path, device: str) -> dict[str, Any]:
    """Return even/fundamental power and the direct half-period residual."""
    modes, coefficients, metadata = _load_dense_solution(path)
    if coefficients.shape[0] != modes.size:
        raise ValueError(f"mode/coefficient mismatch in {path}")
    fundamental = np.flatnonzero(modes == 1)
    if fundamental.size != 1:
        raise ValueError(f"fundamental mode 1 is absent or duplicated in {path}")
    fundamental_norm = float(np.linalg.norm(coefficients[fundamental[0]]))
    if fundamental_norm <= 0.0:
        raise ValueError(f"fundamental coefficient is zero in {path}")
    even = modes % 2 == 0
    even_norm = float(np.linalg.norm(coefficients[even]))
    odd = ~even
    # Reconstruct in short theta blocks.  The IPM has 6,446 nodes, so a full
    # node-by-4,096 array would need hundreds of MB despite being unnecessary.
    numerator_sq = 0.0
    denominator_sq = 0.0
    theta = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    for start in range(0, theta.size, 256):
        block = theta[start:start + 256]
        harmonic = np.exp(1j * modes[:, None, None] * block[None, None, :])
        waveform = np.sum(2.0 * np.real(coefficients[:, :, None] * harmonic), axis=0)
        shifted = np.sum(
            2.0 * np.real(
                coefficients[:, :, None]
                * harmonic
                * ((-1.0) ** modes)[:, None, None]
            ),
            axis=0,
        )
        numerator_sq += float(np.sum((shifted + waveform) ** 2))
        denominator_sq += float(np.sum(waveform ** 2))
    residual = float(np.sqrt(numerator_sq / max(denominator_sq, 1e-300)))
    total_norm = float(np.linalg.norm(coefficients))
    return {
        "device": device,
        "solution_path": str(path),
        "metadata_path": metadata["report_path"],
        "measurement_route": "dense_real_or_dense_harmonic_solution; direct Fourier time-domain residual",
        "pump_mode_policy": metadata["pump_mode_policy"],
        "modes": modes.tolist(),
        "fundamental_coefficient_norm": fundamental_norm,
        "even_harmonic_coefficient_norm": even_norm,
        "even_to_fundamental_ratio": even_norm / fundamental_norm,
        "half_period_residual": residual,
        "coefficient_norm": total_norm,
        "status": "MEASURED",
    }


def default_cases(root: Path) -> list[tuple[str, Path]]:
    """Return dense, existing solutions for the requested device classes."""
    return [
        ("jc_jtwpa", root / "outputs/exp13_jtwpa_harmonic_ladder/H5_nt96/pump/pump_solution.npz"),
        ("jc_fqjtwpa", root / "outputs/exp12_fqjtwpa_diss/pump/pump_solution.npz"),
        ("ipm_2c_fixed", root / "outputs/exp08_full_ipm_pump/pump_solution.npz"),
        ("rf_squid_2393_3wm_biased", root / "outputs/rf_squid_3wm_dc_biased/warm/points/point_0006_p_m62dbm_fp_12p08ghz/pump/pump_solution.npz"),
    ]


def analyse(root: Path) -> dict[str, Any]:
    """Measure every available default dense solution."""
    rows: list[dict[str, Any]] = []
    for device, path in default_cases(root):
        if not path.exists():
            rows.append({
                "device": device, "solution_path": str(path),
                "status": "NOT_ESTABLISHED", "reason": "solution file missing",
            })
            continue
        try:
            rows.append(measure_solution(path, device))
        except (ValueError, OSError, KeyError) as error:
            rows.append({
                "device": device, "solution_path": str(path),
                "status": "NOT_ESTABLISHED", "reason": str(error),
            })
    return {
        "claim": "unbiased sinusoidal 4WM Z2 half-period symmetry",
        "route": "dense Fourier coefficients plus reconstructed time-domain residual",
        "rows": rows,
        "consequence_only": (
            "With Z2 present, lambda=+1 admits pitchfork or transcritical branching, "
            "not only a saddle-node fold; period doubling carries the symmetric structure."
        ),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/chaos/symmetry/symmetry.json"))
    args = parser.parse_args(argv)
    result = analyse(args.root.resolve())
    _atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "measured": sum(row.get("status") == "MEASURED" for row in result["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
