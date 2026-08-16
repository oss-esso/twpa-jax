"""Re-run the trace reduction for points whose reduction failed.

The integration is the expensive part and its trace is on disk, so a point that
integrated cleanly but whose reduction raised does not need re-integrating: the
section, the spectrum and the classification can all be rebuilt from
``trace.npz`` in seconds.  Points that already reduced are left untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.chaos import run_phaseB_pump_only as PB

ROOT = Path(__file__).resolve().parents[2]


def rereduce(point_dir: Path) -> str:
    result_path = point_dir / "result.json"
    trace_path = point_dir / "trace.npz"
    if not result_path.exists() or not trace_path.exists():
        return "SKIP no trace"
    record = json.loads(result_path.read_text(encoding="utf-8"))
    if record.get("reduction_status") != "DIVERGED":
        return "SKIP already reduced"
    pump_hz = float(record.get("pump_hz", 0.0) or 0.0)
    if pump_hz <= 0.0:
        return "SKIP no pump frequency"
    data = np.load(trace_path)
    t, v = data["t"], data["v_out"]
    try:
        reduced = PB._reduce_trace(t, v, pump_hz)
    except (OverflowError, FloatingPointError, ValueError) as error:
        return f"STILL FAILING {error!r}"
    spectrum_frequency = reduced.pop("spectrum_frequency_hz")
    spectrum_db = reduced.pop("spectrum_db_relative_pump")
    branches = reduced.pop("upward_branch")
    np.savez_compressed(point_dir / "poincare_branches.npz", upward=branches)
    np.savez_compressed(
        point_dir / "spectrum.npz", frequency_hz=spectrum_frequency,
        spectrum_db_relative_pump=spectrum_db,
    )
    record.pop("reduction_status", None)
    record.pop("reduction_error", None)
    record.update(PB._json_safe(reduced))
    result_path.write_text(json.dumps(PB._json_safe(record), indent=2),
                           encoding="utf-8")
    return f"OK {branches.size} section points, verdict {record.get('verdict')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path,
                        help="campaign directory, e.g. outputs/chaos/phaseB_signal")
    args = parser.parse_args()
    for result in sorted(args.root.rglob("result.json")):
        status = rereduce(result.parent)
        if not status.startswith("SKIP"):
            print(f"  {result.parent.relative_to(args.root)}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
