"""Summarize HB NS, torus, and unstable-skeleton artifacts per device."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _first_crossing(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    for point in points:
        multiplier = point.get("multiplier", {})
        if float(multiplier.get("magnitude", 0.0)) >= 1.0:
            return point
    return None


def summarize_transition(
    device: str,
    floquet: dict[str, Any],
    torus: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a report without inferring missing boundaries."""
    floquet_points = list(floquet.get("points", []))
    torus_points = list(torus.get("points", []))
    crossing = floquet.get("first_unit_circle_crossing") or _first_crossing(
        floquet_points
    )
    solved_torus = [point for point in torus_points if point.get("converged")]
    plateau = [
        point for point in solved_torus
        if float(point.get("off_comb_norm_fraction", 0.0)) > 0.0
    ]
    return {
        "device": device,
        "ns_crossing": crossing,
        "torus_first_parameter": (
            solved_torus[0].get("point_index") if solved_torus else None
        ),
        "torus_last_parameter": (
            solved_torus[-1].get("point_index") if solved_torus else None
        ),
        "torus_solved_count": len(solved_torus),
        "torus_plateau_points": len(plateau),
        "reference": reference,
        "status": "ESTABLISHED" if crossing is not None else "NOT_ESTABLISHED",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--floquet-json", type=Path, required=True)
    parser.add_argument("--torus-json", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    floquet = json.loads(args.floquet_json.read_text(encoding="utf-8"))
    torus = json.loads(args.torus_json.read_text(encoding="utf-8"))
    reference = None
    if args.reference_json is not None:
        reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
    _atomic_write(
        args.out,
        summarize_transition(args.device, floquet, torus, reference=reference),
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
