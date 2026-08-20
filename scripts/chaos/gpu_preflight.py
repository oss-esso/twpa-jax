"""Self-validating GPU-session preflight with a clean failure boundary.

PAUSED 2026-08-20: this is part of the GPU-session toolchain, which is built
and CPU-smoke-tested but has never been run on a GPU.  No result in this
repository depends on it.  Read
``docs/development/gpu_session_runbook.md`` before reviving it -- in
particular, consumer FP64 is 1/32 (Ampere) or 1/64 (Ada) of FP32, so the
float32 precision gate decides whether an accelerator helps at all.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.chaos.run_guarcello_jc_phase5 import (
    derive_device_spec,
    integrate_jc_banded_batch,
    load_jc_device,
    resolve_jax_device,
    resolve_pump_frequency,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "tests" / "data" / "fdtd_reference" / "ipm_2c_7p9.npz"
EXPECTED_NATURAL_BANDWIDTH = 4558
REFERENCE_TOLERANCE = 1.0e-11


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _nvidia_memory() -> dict[str, int | None]:
    if shutil.which("nvidia-smi") is None:
        return {"total_bytes": None, "free_bytes": None}
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"total_bytes": None, "free_bytes": None}
    first = completed.stdout.splitlines()[0].split(",")
    if len(first) != 2:
        return {"total_bytes": None, "free_bytes": None}
    try:
        return {
            "total_bytes": int(float(first[0].strip()) * 1024 * 1024),
            "free_bytes": int(float(first[1].strip()) * 1024 * 1024),
        }
    except ValueError:
        return {"total_bytes": None, "free_bytes": None}


def _device_memory(device: Any) -> dict[str, int | None]:
    values = {"total_bytes": None, "free_bytes": None, "peak_bytes": None}
    try:
        stats = device.memory_stats() or {}
    except Exception:
        stats = {}
    values["total_bytes"] = stats.get("bytes_limit")
    values["peak_bytes"] = stats.get("peak_bytes_in_use")
    if values["total_bytes"] is not None and stats.get("bytes_in_use") is not None:
        values["free_bytes"] = values["total_bytes"] - stats["bytes_in_use"]
    nvidia = _nvidia_memory()
    for key in ("total_bytes", "free_bytes"):
        if values[key] is None:
            values[key] = nvidia[key]
    return values


def _environment() -> tuple[Any, dict[str, Any]]:
    import jax
    import numba
    import twpa_solver

    devices = jax.devices()
    try:
        x64 = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        x64 = bool(jax.config.jax_enable_x64)
    primary = devices[0] if devices else None
    environment = {
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in devices],
        "device_platform": getattr(primary, "platform", None),
        "device_kind": getattr(primary, "device_kind", None),
        "jax_enable_x64": x64,
        "vram": _device_memory(primary) if primary is not None else _nvidia_memory(),
        "numba_version": numba.__version__,
        "git_sha": _git_sha(),
        "twpa_solver_file": str(Path(twpa_solver.__file__).resolve()),
    }
    return primary, environment


def _relative(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(first) - np.asarray(second)))
    denominator = float(np.linalg.norm(np.asarray(second)))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _load_fixture(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {
            key: np.asarray(data[key])
            for key in (
                "pump_currents_a", "initial_q_previous", "initial_q_current",
                "times", "voltage", "branch_r", "q_final",
            )
        }
    return metadata, arrays


def run_preflight(fixture: Path, output: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "FAILED",
        "stage": "environment",
        "fixture": str(fixture),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def write() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        primary, environment = _environment()
        report["environment"] = environment
        write()
        report["stage"] = "accelerator"
        target, note = resolve_jax_device("gpu")
        report["accelerator"] = {"device": str(target), "note": note}
        write()

        report["stage"] = "fixture"
        if not fixture.exists():
            raise FileNotFoundError(f"fixture does not exist: {fixture}")
        metadata, arrays = _load_fixture(fixture)
        source = ROOT / "designs" / str(metadata["device"])
        spec = derive_device_spec(source)
        live_device = load_jc_device(Path(spec.circuit_dir))
        if int(metadata["natural_bandwidth"]) != EXPECTED_NATURAL_BANDWIDTH:
            raise RuntimeError(
                "fixture natural_bandwidth is not the live-session value: "
                f"{metadata['natural_bandwidth']} != {EXPECTED_NATURAL_BANDWIDTH}"
            )
        if live_device.natural_bandwidth != int(metadata["natural_bandwidth"]):
            raise RuntimeError(
                "fixture/live natural_bandwidth mismatch: "
                f"{metadata['natural_bandwidth']} != {live_device.natural_bandwidth}"
            )
        if resolve_pump_frequency(spec) != float(metadata["pump_hz"]):
            raise RuntimeError("fixture pump frequency does not match the live design")
        report["fixture_metadata"] = metadata
        write()

        report["stage"] = "equivalence"
        base_kwargs = dict(metadata["kwargs"])
        base_kwargs.update({
            "pump_currents_a": arrays["pump_currents_a"],
            "initial_q_previous": arrays["initial_q_previous"],
            "initial_q": arrays["initial_q_current"],
            "backend": "jax",
            "jax_device": "gpu",
        })
        comparisons: list[dict[str, Any]] = []
        for solve_kind in ("sequential", "scan"):
            for dtype in ("float64", "float32"):
                kwargs = dict(base_kwargs)
                kwargs["solve_kind"] = solve_kind
                kwargs["dtype"] = dtype
                started = time.perf_counter()
                result = integrate_jc_banded_batch(live_device, **kwargs)
                elapsed = time.perf_counter() - started
                names = ("times", "voltage", "branch_r", "q_final")
                values = (result[0], result[1], result[2], result[4])
                relative = {
                    name: _relative(value, arrays[name])
                    for name, value in zip(names, values)
                }
                comparisons.append({
                    "solve_kind": solve_kind,
                    "dtype": dtype,
                    "relative_difference": relative,
                    "max_relative_difference": max(relative.values()),
                    "runtime_s": elapsed,
                })
        report["comparisons"] = comparisons
        required = next(
            item for item in comparisons
            if item["solve_kind"] == "sequential" and item["dtype"] == "float64"
        )
        if required["max_relative_difference"] > REFERENCE_TOLERANCE:
            raise RuntimeError(
                "float64/sequential reference mismatch: "
                f"{required['max_relative_difference']:.6e} > {REFERENCE_TOLERANCE:.1e}"
            )
        report["status"] = "PASS"
        report["stage"] = "complete"
        write()
        return report
    except Exception as exc:
        report["error"] = str(exc)
        write()
        raise RuntimeError(f"GPU preflight failed at {report['stage']}: {exc}") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "chaos" / "gpu_session" / "preflight.json",
    )
    args = parser.parse_args(argv)
    try:
        report = run_preflight(args.fixture, args.output)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
