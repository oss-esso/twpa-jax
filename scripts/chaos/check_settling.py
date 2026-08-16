#!/usr/bin/env python3
"""How many pump periods does a device need before its spectrum means anything?

Integrates one point and reports the off-lattice energy fraction in slices
across the record.  Off-lattice energy is what the bifurcation classification
measures, and residual ringing puts energy there too, so a run whose analysed
window still shows monotone decay is measuring its own transient rather than
the circuit.

Measured 2026-08-14: guarcello settles inside 600 pump periods; jc_jtwpa needs
about 1050.  The near-lossless devices dissipate only through their port
resistors, so ringing traverses the cell chain many times before it leaks out.
Do not assume one device's budget transfers to another.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "phaseB_pump_only", ROOT / "scripts" / "chaos" / "run_phaseB_pump_only.py"
)
assert _SPEC is not None and _SPEC.loader is not None
PHASEB = importlib.util.module_from_spec(_SPEC)
sys.modules["phaseB_pump_only"] = PHASEB
_SPEC.loader.exec_module(PHASEB)

_OVN = importlib.util.spec_from_file_location(
    "ovn", ROOT / "scripts" / "chaos" / "run_phaseB_overnight.py"
)
OVN = importlib.util.module_from_spec(_OVN)
sys.modules["ovn"] = OVN
_argv, sys.argv = sys.argv, ["x", "--dry-run"]
_OVN.loader.exec_module(OVN)
sys.argv = _argv


def pump_hz(device: str) -> float:
    if device == "guarcello":
        return 7.0e9
    source = (
        PHASEB.phase5.phase_c_source_path(device)
        if device in {"ipm_2c_fixed", "rf_squid_2393_3wm"}
        else ROOT / "outputs" / "jc_doc_python_designs" / device
    )
    return float(PHASEB.phase5.resolve_pump_frequency(
        PHASEB.phase5.derive_device_spec(source)
    ))


def off_lattice(t: np.ndarray, v: np.ndarray, drive_hz: float) -> float:
    """Fraction of spectral energy NOT on an integer multiple of the pump."""
    dt = float(np.mean(np.diff(t)))
    power = np.abs(np.fft.rfft(v - np.mean(v))) ** 2
    ratio = np.fft.rfftfreq(t.size, dt) / drive_hz
    on_lattice = (np.abs(ratio - np.round(ratio)) < 0.02) & (np.round(ratio) > 0)
    return float((power.sum() - power[on_lattice].sum()) / max(power.sum(), 1e-300))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--control", type=float, required=True,
                        help="dBm for guarcello/JC devices; I/I_bound for ipm_2c_fixed; "
                             "absolute on-chip pump current in amps for "
                             "rf_squid_2393_3wm")
    parser.add_argument("--periods", type=float, default=2400.0)
    parser.add_argument("--slices", type=int, default=8)
    parser.add_argument("--dt-norm", type=float, default=0.01)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "chaos" / "settling_checks")
    args = parser.parse_args()

    drive_hz = pump_hz(args.device)
    path = args.output / f"{args.device}_{args.periods:.0f}periods"
    row = PHASEB._run_point(
        args.device, args.control, path, args.dt_norm,
        OVN._tmax_norm(args.device, args.periods),
    )
    trace = path / "trace.npz"
    if not trace.exists():
        # _run_point catches its own exceptions and records them in the row, so
        # a missing trace means the integration failed rather than that the file
        # went astray.  Report that instead of a FileNotFoundError several
        # frames away from the real error.
        print(f"{args.device}: integration produced no trace\n"
              f"  status: {row.get('status', 'unknown')}\n"
              f"  error : {row.get('error', 'not recorded')}\n"
              f"  row   : {path / 'result.json'}")
        return 1
    with np.load(trace) as data:
        t, v = data["t"], data["v_out"]

    slices = []
    for index in range(args.slices):
        lo = index * t.size // args.slices
        hi = (index + 1) * t.size // args.slices
        slices.append(off_lattice(t[lo:hi], v[lo:hi], drive_hz))

    span = (t[-1] - t[0]) * drive_hz
    print(f"{args.device}  control={args.control}  "
          f"{span:.0f} pump periods  status={row.get('status', 'OK')}")
    print(f"{'through period':>16} {'off-lattice':>14}")
    settled_at = None
    for index, value in enumerate(slices):
        end = span * (index + 1) / args.slices
        print(f"{end:16.0f} {value:14.3e}")
        if settled_at is None and index >= 1 and value > 0.5 * slices[index - 1]:
            settled_at = span * index / args.slices
    # "Stopped improving" alone cannot tell a settled floor from a curve that
    # never decayed at all: a run flat at 1e-6 and one flat at 0.96 both stop
    # improving on the second slice.  Require the level itself to be low before
    # calling it settled.  The measured floors are 6e-6 (guarcello), 3.1e-6
    # (ipm_2c_fixed) and 1.1e-6 (jc_jtwpa), so 1e-3 is orders clear of all three
    # while still rejecting a record that is essentially all transient.
    settled_floor = 1.0e-3
    if settled_at is not None and slices[-1] > settled_floor:
        print(f"\noff-lattice energy is flat at {slices[-1]:.3e} but never fell "
              f"below {settled_floor:.0e} - this record is transient throughout, "
              f"not settled; run longer")
    elif settled_at is not None:
        print(f"\ndecay stops improving at about {settled_at:.0f} pump periods "
              f"(floor {slices[-1]:.3e})")
    else:
        print("\nstill decaying at the end of the record - run longer")
    print("analysed window must start after that; the driver analyses the last half, "
          "so run at least twice that many periods")

    payload = {"device": args.device, "control": args.control,
               "pump_periods": span, "slices": slices, "settled_at": settled_at}
    (path / "settling.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
