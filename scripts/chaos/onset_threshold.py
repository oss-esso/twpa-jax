"""Reference-side squared-radius onset fit for a Neimark--Sacker branch."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OnsetFit:
    """Result of fitting ``radius**2 = slope * (mu - mu_c)``."""

    status: str
    mu_c: float | None
    slope: float | None
    r_squared: float | None
    point_count: int
    bracket: tuple[float, float] | None
    residual_rms: float | None
    rejection_reason: str | None = None


def _r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if total == 0.0:
        return 1.0 if np.allclose(observed, fitted) else 0.0
    return float(1.0 - np.sum((observed - fitted) ** 2) / total)


def fit_squared_radius_threshold(
    control: np.ndarray | list[float],
    radius: np.ndarray | list[float],
    *,
    period1_mask: np.ndarray | list[bool] | None = None,
    max_points: int = 6,
    minimum_r_squared: float = 0.5,
) -> OnsetFit:
    """Fit the first settled non-period-1 radii and reject hard jumps.

    The mask is normally obtained from the return-map period gate.  Supplying
    it explicitly prevents this reference fit from inventing a threshold from
    a transition it has not resolved.  A non-positive slope, poor fit, or
    insufficient points returns ``UNRESOLVED`` rather than a plausible number.
    """
    mu = np.asarray(control, dtype=float).reshape(-1)
    r = np.asarray(radius, dtype=float).reshape(-1)
    if mu.size != r.size or mu.size < 3:
        raise ValueError("control and radius need equal length and at least 3 points")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("control and radius must be finite with non-negative radius")
    if max_points < 3:
        raise ValueError("max_points must be at least 3")

    order = np.argsort(mu)
    mu = mu[order]
    r = r[order]
    if period1_mask is None:
        mask = r <= max(np.finfo(float).eps, 1e-12 * max(float(np.max(r)), 1.0))
        mask = np.asarray(mask, dtype=bool)
    else:
        supplied = np.asarray(period1_mask, dtype=bool).reshape(-1)
        if supplied.size != order.size:
            raise ValueError("period1_mask must match control and radius")
        mask = supplied[order]

    above = np.flatnonzero(~mask)
    if above.size < 3:
        return OnsetFit(
            "UNRESOLVED", None, None, None, int(above.size), None, None,
            "fewer than three settled points above period 1",
        )
    selected = above[:max_points]
    x = mu[selected]
    y = r[selected] ** 2
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    fit_r2 = _r_squared(y, fitted)
    rms = float(np.sqrt(np.mean((y - fitted) ** 2)))
    bracket = (float(x[0]), float(x[-1]))
    if slope <= 0.0:
        reason = "squared-radius slope is not positive"
    elif fit_r2 < minimum_r_squared:
        reason = f"squared-radius fit R^2={fit_r2:.4g} below threshold"
    elif np.any(y <= 0.0):
        reason = "above-transition radius contains a zero value"
    else:
        mu_c = float(-intercept / slope)
        if not np.isfinite(mu_c) or mu_c > x[0] or mu_c < x[0] - 2.0 * (x[-1] - x[0]):
            reason = "extrapolated threshold is outside the resolved onset bracket"
        else:
            return OnsetFit(
                "RESOLVED_CONTINUOUS", mu_c, float(slope), fit_r2,
                int(selected.size), bracket, rms,
            )
    return OnsetFit(
        "UNRESOLVED", None, float(slope), fit_r2, int(selected.size), bracket,
        rms, reason,
    )


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-key", default="control")
    parser.add_argument("--radius-key", default="r_RMS")
    parser.add_argument("--period1-key", default="period1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = fit_squared_radius_threshold(
        payload[args.control_key],
        payload[args.radius_key],
        period1_mask=payload.get(args.period1_key),
    )
    _atomic_write(args.output, asdict(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
