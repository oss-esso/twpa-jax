"""Continue a checkpointed full-IPM hold and compute decay-aware settling metrics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from h1_transient_branch_transfer import (
    build_system,
    decay_aware_stroboscopic_classification,
    implicit_trapezoid_ramp,
    stroboscopic_diagnostics,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--circuit-dir", type=Path, required=True)
    p.add_argument("--restart", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--start-current", type=float, required=True)
    p.add_argument("--target-current", type=float, required=True)
    p.add_argument("--end-period", type=float, required=True)
    p.add_argument("--ramp-periods", type=float, default=10.0)
    p.add_argument("--step", type=float, default=0.01)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    restart = np.load(args.restart)
    initial_theta = float(restart["theta"])
    y0 = np.asarray(restart["y"])
    system = build_system(args.circuit_dir, 7.9, 4)
    total_theta = 2.0 * math.pi * args.end_period
    ramp_theta = 2.0 * math.pi * args.ramp_periods
    theta, states, integrator = implicit_trapezoid_ramp(
        system, y0, args.start_current, args.target_current, total_theta, ramp_theta,
        args.step, 1e-3, 12, checkpoint_dir=args.outdir / "restart_checkpoints",
        checkpoint_periods=10, initial_theta=initial_theta,
    )
    first_period = math.ceil(initial_theta / (2.0 * math.pi))
    strobe_theta = 2.0 * math.pi * np.arange(first_period, args.end_period + 1.0)
    dense = lambda query: np.vstack([np.interp(query, theta, row) for row in states])
    strobe_states = dense(strobe_theta)
    strobe = stroboscopic_diagnostics(system, strobe_theta, strobe_states, len(strobe_theta) - 1)
    decay = decay_aware_stroboscopic_classification(strobe)
    phases = np.asarray([
        system.circuit.Bphi.T @ system.unpack(strobe_states[:, i])[0]
        for i in range(strobe_states.shape[1])
    ])
    unwrapped = np.unwrap(phases, axis=0)
    winding = float(np.mean(unwrapped[-1] - unwrapped[0]) / (2.0 * math.pi))
    result = {
        "initial_theta": initial_theta,
        "initial_period": initial_theta / (2.0 * math.pi),
        "end_period": args.end_period,
        "integrator": integrator,
        "stroboscopic": strobe,
        "decay_aware": decay,
        "mean_phase_winding_cycles": winding,
        "checkpoint": str(args.restart),
    }
    np.savez_compressed(args.outdir / "stroboscopic_states.npz", theta=strobe_theta, states=strobe_states)
    (args.outdir / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
