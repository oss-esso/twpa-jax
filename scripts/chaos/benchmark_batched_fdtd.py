"""Run the resumable batched-FDTD benchmark matrix on the GPU session host."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.chaos.run_guarcello_jc_phase5 import (
    derive_device_spec,
    integrate_jc_banded_batch,
    load_jc_device,
    resolve_pump_frequency,
)


ROOT = Path(__file__).resolve().parents[2]
PUMP_HZ = 7.9e9
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_SOLVE_KINDS = ("sequential", "scan")
DEFAULT_DTYPES = ("float64", "float32")
CSV_FIELDS = (
    "timestamp", "backend", "jax_device", "batch", "solve_kind", "dtype",
    "n_steps", "record_stride", "status", "runtime_s", "throughput_steps_s",
    "peak_vram_bytes", "natural_bandwidth", "selected_bandwidth", "n_nodes",
    "error",
)


def _nvidia_peak_bytes() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        return int(float(completed.stdout.splitlines()[0].strip()) * 1024 * 1024)
    except ValueError:
        return None


def _peak_bytes(device: Any | None) -> int | None:
    candidates: list[int] = []
    if device is not None:
        try:
            stats = device.memory_stats() or {}
            for key in ("peak_bytes_in_use", "bytes_in_use"):
                value = stats.get(key)
                if value is not None:
                    candidates.append(int(value))
        except Exception:
            pass
    nvidia = _nvidia_peak_bytes()
    if nvidia is not None:
        candidates.append(nvidia)
    return max(candidates) if candidates else None


def _is_oom(error: Exception) -> bool:
    message = str(error).lower()
    return any(token in message for token in ("out of memory", "oom", "resource exhausted"))


def _base_kwargs(device: Any, currents: np.ndarray, n_steps: int, record_stride: int) -> dict[str, Any]:
    spec = derive_device_spec(ROOT / "designs" / "ipm_2c_fixed")
    if resolve_pump_frequency(spec) != PUMP_HZ:
        raise RuntimeError("benchmark requires the 7.9 GHz pump column")
    return {
        "pump_currents_a": currents,
        "pump_hz": PUMP_HZ,
        "signal_current_a": 0.0,
        "signal_hz": 7.4e9,
        "dt_s": 0.01 / spec.omega_plasma,
        "n_steps": n_steps,
        "record_stride": record_stride,
    }


def _row(
    *, backend: str, jax_device: str, batch: int, solve_kind: str, dtype: str,
    n_steps: int, record_stride: int, status: str, runtime_s: float | None,
    throughput: float | None, peak: int | None, device: Any, error: str = "",
) -> dict[str, object]:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": backend,
        "jax_device": jax_device,
        "batch": batch,
        "solve_kind": solve_kind,
        "dtype": dtype,
        "n_steps": n_steps,
        "record_stride": record_stride,
        "status": status,
        "runtime_s": "" if runtime_s is None else runtime_s,
        "throughput_steps_s": "" if throughput is None else throughput,
        "peak_vram_bytes": "" if peak is None else peak,
        "natural_bandwidth": device.natural_bandwidth,
        "selected_bandwidth": device.selected_bandwidth,
        "n_nodes": device.n_nodes,
        "error": error,
    }


def _run_configuration(
    device: Any, *, backend: str, jax_device: str, batch: int,
    solve_kind: str, dtype: str, n_steps: int, record_stride: int,
) -> dict[str, object]:
    currents = np.linspace(5.0e-6, 7.0e-6, batch, dtype=np.float64)
    kwargs = _base_kwargs(device, currents, n_steps, record_stride)
    kwargs.update({
        "backend": backend,
        "jax_device": jax_device,
        "solve_kind": solve_kind,
        "dtype": dtype,
    })
    target = None
    try:
        if backend == "jax":
            import jax

            target = jax.devices(jax_device)[0]
        integrate_jc_banded_batch(device, **kwargs)
        gc.collect()
        started = time.perf_counter()
        result = integrate_jc_banded_batch(device, **kwargs)
        elapsed = time.perf_counter() - started
        del result
        gc.collect()
        peak = _peak_bytes(target)
        throughput = batch * n_steps / elapsed if elapsed > 0.0 else None
        return _row(
            backend=backend, jax_device=jax_device, batch=batch,
            solve_kind=solve_kind, dtype=dtype, n_steps=n_steps,
            record_stride=record_stride, status="complete", runtime_s=elapsed,
            throughput=throughput, peak=peak, device=device,
        )
    except Exception as exc:
        return _row(
            backend=backend, jax_device=jax_device, batch=batch,
            solve_kind=solve_kind, dtype=dtype, n_steps=n_steps,
            record_stride=record_stride,
            status="OOM" if _is_oom(exc) else "ERROR", runtime_s=None,
            throughput=None, peak=_peak_bytes(target), device=device,
            error=str(exc),
        )


def run_benchmark(
    output: Path, *, batches: Iterable[int] = DEFAULT_BATCHES,
    solve_kinds: Iterable[str] = DEFAULT_SOLVE_KINDS,
    dtypes: Iterable[str] = DEFAULT_DTYPES,
    devices: Iterable[str] = ("gpu", "cpu"),
    n_steps: int = 3_000,
    record_stride: int = 20,
) -> None:
    device = load_jc_device(ROOT / "designs" / "ipm_2c_fixed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        stream.flush()

        configurations = (
            ("numba", "cpu", batch, "sequential", "float64")
            for batch in batches
        )
        for backend, jax_device, batch, solve_kind, dtype in configurations:
            row = _run_configuration(
                device, backend=backend, jax_device=jax_device, batch=batch,
                solve_kind=solve_kind, dtype=dtype, n_steps=n_steps,
                record_stride=record_stride,
            )
            writer.writerow(row)
            stream.flush()

        for jax_device in devices:
            for batch in batches:
                for solve_kind in solve_kinds:
                    for dtype in dtypes:
                        row = _run_configuration(
                            device, backend="jax", jax_device=jax_device,
                            batch=batch, solve_kind=solve_kind, dtype=dtype,
                            n_steps=n_steps, record_stride=record_stride,
                        )
                        writer.writerow(row)
                        stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "chaos" / "gpu_session" / "benchmark.csv",
    )
    parser.add_argument("--n-steps", type=int, default=3_000)
    parser.add_argument("--record-stride", type=int, default=20)
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--devices", nargs="+", default=["gpu", "cpu"])
    args = parser.parse_args(argv)
    run_benchmark(
        args.output, batches=args.batches, devices=args.devices,
        n_steps=args.n_steps, record_stride=args.record_stride,
    )
    print(json.dumps({"output": str(args.output), "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
