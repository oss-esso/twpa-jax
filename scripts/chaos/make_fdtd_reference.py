"""Create the small, committed CPU reference for the GPU session."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.chaos.run_guarcello_jc_phase5 import (
    PHI0_REDUCED,
    derive_device_spec,
    integrate_jc_banded_batch,
    load_jc_device,
    resolve_pump_frequency,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "tests" / "data" / "fdtd_reference" / "ipm_2c_7p9.npz"
PUMP_HZ = 7.9e9
EXPECTED_NATURAL_BANDWIDTH = 4558


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def make_reference(output: Path, *, force: bool = False) -> dict[str, object]:
    if output.exists() and not force:
        raise FileExistsError(f"reference already exists: {output}; use --force")
    source = ROOT / "designs" / "ipm_2c_fixed"
    spec = derive_device_spec(source)
    if resolve_pump_frequency(spec) != PUMP_HZ:
        raise RuntimeError("reference fixture requires the 7.9 GHz pump column")
    device = load_jc_device(Path(spec.circuit_dir))
    if device.natural_bandwidth != EXPECTED_NATURAL_BANDWIDTH:
        raise RuntimeError(
            "reference fixture requires natural_bandwidth=4558, got "
            f"{device.natural_bandwidth}"
        )

    pump_currents_a = np.asarray([5.0e-6, 7.0e-6], dtype=np.float64)
    n_steps = 3_000
    record_stride = 20
    dt_norm = 0.01
    dt_s = dt_norm / spec.omega_plasma
    kwargs: dict[str, object] = {
        "pump_hz": PUMP_HZ,
        "signal_current_a": 0.0,
        "signal_hz": 7.4e9,
        "dt_s": dt_s,
        "n_steps": n_steps,
        "record_stride": record_stride,
        "backend": "numba",
        "jax_device": "cpu",
        "solve_kind": "sequential",
        "dtype": "float64",
    }
    initial_q_previous = np.zeros((pump_currents_a.size, device.n_nodes))
    initial_q_current = np.zeros((pump_currents_a.size, device.n_nodes))
    started = time.perf_counter()
    result = integrate_jc_banded_batch(
        device,
        pump_currents_a=pump_currents_a,
        initial_q_previous=initial_q_previous,
        initial_q=initial_q_current,
        **kwargs,
    )
    runtime_s = time.perf_counter() - started
    times, voltage, branch_r, _, q_final = result
    metadata = {
        "schema_version": 1,
        "device": "ipm_2c_fixed",
        "pump_hz": PUMP_HZ,
        "natural_bandwidth": int(device.natural_bandwidth),
        "selected_bandwidth": int(device.selected_bandwidth),
        "n_nodes": int(device.n_nodes),
        "git_sha": _git_sha(),
        "dt_norm": dt_norm,
        "dt_s": dt_s,
        "n_steps": n_steps,
        "record_stride": record_stride,
        "pump_currents_a": pump_currents_a.tolist(),
        "initial_state_shape": list(initial_q_current.shape),
        "kwargs": kwargs,
        "runtime_s": runtime_s,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        pump_currents_a=pump_currents_a,
        initial_q_previous=initial_q_previous,
        initial_q_current=initial_q_current,
        times=np.asarray(times),
        voltage=np.asarray(voltage),
        branch_r=np.asarray(branch_r),
        q_final=np.asarray(q_final),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    separation = float(np.linalg.norm(np.asarray(q_final)[1] - np.asarray(q_final)[0]))
    branch_separation = float(
        np.max(np.abs(np.asarray(branch_r)[1] - np.asarray(branch_r)[0]))
    )
    print(json.dumps({
        "output": str(output),
        "natural_bandwidth": device.natural_bandwidth,
        "selected_bandwidth": device.selected_bandwidth,
        "n_nodes": device.n_nodes,
        "n_steps": n_steps,
        "runtime_s": runtime_s,
        "lane_q_final_separation": separation,
        "lane_branch_r_max_separation": branch_separation,
    }, indent=2))
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    make_reference(args.output, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
