"""Measure stride effects on the Guarcello pump-only symmetry and spectrum."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.chaos.run_phaseB_pump_only import PAPER, _pump_only_paper, _reduce_trace, ROOT


def main() -> int:
    output = ROOT / "outputs" / "chaos" / "phaseB" / "guarcello" / "stride_check"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for power in (-70.0, -54.5):
        for stride in (20, 4, 1):
            device = PAPER.Device()
            tmax_norm = 600.0 * device.omega_plasma / 7.0e9
            t, v, runtime = _pump_only_paper(power, 0.01, tmax_norm, record_stride=stride)
            reduced = _reduce_trace(t, v, 7.0e9)
            result = {
                "pump_power_dbm": power,
                "record_stride_requested": stride,
                "effective_sample_count": int(t.size),
                "runtime_s": runtime,
                "q_even": reduced["q_even"],
                "q_dc": reduced["q_dc"],
                "half_integer_floor_db": reduced["half_integer_floor_db"],
                "half_integer_line_db": reduced["half_integer_line_db"],
            }
            results.append(result)
            (output / f"p{power:g}_s{stride}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
