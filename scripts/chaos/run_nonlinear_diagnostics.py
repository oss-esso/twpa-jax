"""Apply the nonlinear-dynamics diagnostics to a pump-only chaos campaign.

Writes one JSON per device plus a combined summary, per point and atomically,
so an interrupted run keeps everything it has already measured.

Usage::

    python scripts/chaos/run_nonlinear_diagnostics.py \
        --campaign outputs/chaos/phaseB \
        --devices ipm_2c_fixed guarcello \
        --output outputs/chaos/nonlinear
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "chaos"))

import nonlinear_diagnostics as nd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=3000,
                        help="embedded points retained for the correlation integral")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}

    for device in args.devices:
        device_dir = args.campaign / device
        if not device_dir.is_dir():
            print(f"{device}: no directory under {args.campaign}", flush=True)
            continue
        results = sorted(device_dir.glob("*/result.json"))
        rows: list[dict[str, object]] = []
        out_path = args.output / f"{device}.json"
        print(f"\n=== {device}: {len(results)} points ===", flush=True)

        for result_path in results:
            started = time.perf_counter()
            row = nd.analyse_point(result_path.parent)
            rows.append(row)
            elapsed = time.perf_counter() - started

            control = row.get("control_value")
            if row["status"] != "OK":
                print(
                    f"  {control!s:>10}  {row['status']:9s} "
                    f"{row['validity']['reason']}",
                    flush=True,
                )
            else:
                zero_one = row.get("zero_one_test") or {}
                dimension = row.get("correlation_dimension") or {}
                k = zero_one.get("k_median")
                d2 = dimension.get("d2_plateau")
                sigma = row.get("strobe_std")
                print(
                    f"  {control:>10.4f}  "
                    f"K={('%+.4f' % k) if k is not None else '   none':>7s}  "
                    f"D2={('%.3f@m=%s' % (d2, dimension.get('plateau_dimension'))) if d2 else 'none':>12s}  "
                    f"strobe={row.get('strobe_points') or 0:>5d}  "
                    f"sigma={('%.4e' % sigma) if sigma else 'none':>10s}  "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
            # Written every point: seven earlier long runs in this project were
            # lost to end-buffered writes.
            tmp = out_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
            tmp.replace(out_path)

        usable = [
            r for r in rows
            if r.get("status") == "OK" and r.get("strobe_std")
        ]
        usable.sort(key=lambda r: r["control_value"])
        fit = nd.fit_normal_form_exponent(
            [r["control_value"] for r in usable],
            [r["strobe_std"] for r in usable],
        )
        summary[device] = {
            "n_total": len(rows),
            "n_usable": len(usable),
            "n_excluded": sum(1 for r in rows if r.get("status") == "EXCLUDED"),
            "normal_form_fit": asdict(fit),
        }
        print(f"  normal-form fit: {fit.verdict} -- {fit.reason}", flush=True)

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    print(f"\nwrote {args.output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
