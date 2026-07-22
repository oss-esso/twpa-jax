"""Copy a 2c/3c IPM design and optionally add Josephson Lj scatter.

Examples:
    python -m twpa_solver.builders.scattered --design 2c --outdir outputs/ipm_2c_ljscatter_s1pct_seed1 --lj-scatter-sigma 0.01 --lj-scatter-seed 1
    python -m twpa_solver.builders.scattered --design 3c --outdir outputs/ipm_3c_ljscatter_s1pct_seed1 --lj-scatter-sigma 0.01 --lj-scatter-seed 1

The source IPM directories are treated as authoritative. This matters because
old maps may have been generated from artifacts whose arrays are not reproduced
by the current parametric builder. At sigma=0 this script is an exact artifact
copy, so the resulting map should match the source design.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


DEFAULT_SOURCES = {
    "2c": Path("designs/ipm_2c_fixed"),
    "3c": Path("designs/ipm_3c_fixed"),
}


def _copytree_clean(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _scatter_arrays(
    ipm_dir: Path,
    *,
    sigma: float,
    seed: int,
    clip_min: float,
    clip_max: float,
) -> dict[str, object]:
    if sigma < 0.0:
        raise ValueError("--lj-scatter-sigma must be non-negative")
    if clip_min <= 0.0 or clip_max <= 0.0 or clip_min > clip_max:
        raise ValueError("scatter clip bounds must be positive and ordered")

    arrays_path = ipm_dir / "ipm_arrays.npz"
    with np.load(arrays_path) as z:
        arrays = {name: z[name] for name in z.files}

    if "Lj" not in arrays or "Ic" not in arrays or "phi0_reduced" not in arrays:
        raise KeyError(f"{arrays_path} must contain Lj, Ic, and phi0_reduced")

    base_lj = np.asarray(arrays["Lj"], dtype=float)
    phi0 = float(np.asarray(arrays["phi0_reduced"], dtype=float).reshape(-1)[0])
    if sigma == 0.0:
        factors = np.ones_like(base_lj)
    else:
        rng = np.random.default_rng(int(seed))
        factors = rng.normal(loc=1.0, scale=float(sigma), size=base_lj.size)
        factors = np.clip(factors, float(clip_min), float(clip_max))

    scattered_lj = base_lj * factors
    arrays["Lj"] = scattered_lj
    arrays["Ic"] = phi0 / scattered_lj
    np.savez(arrays_path, **arrays)

    return {
        "lj_scatter_enabled": bool(sigma > 0.0),
        "lj_scatter_sigma": float(sigma),
        "lj_scatter_seed": int(seed),
        "lj_scatter_count": int(base_lj.size),
        "lj_scatter_clip_min": float(clip_min),
        "lj_scatter_clip_max": float(clip_max),
        "lj_scatter_factor_min": float(np.min(factors)) if factors.size else None,
        "lj_scatter_factor_max": float(np.max(factors)) if factors.size else None,
        "lj_scatter_factor_mean": float(np.mean(factors)) if factors.size else None,
        "lj_scatter_factor_std": float(np.std(factors, ddof=0)) if factors.size else None,
        "lj_base_min_h": float(np.min(base_lj)) if base_lj.size else None,
        "lj_base_max_h": float(np.max(base_lj)) if base_lj.size else None,
        "lj_scattered_min_h": float(np.min(scattered_lj)) if scattered_lj.size else None,
        "lj_scattered_max_h": float(np.max(scattered_lj)) if scattered_lj.size else None,
    }


def _zero_scatter_meta(ipm_dir: Path, *, seed: int, clip_min: float, clip_max: float) -> dict[str, object]:
    with np.load(ipm_dir / "ipm_arrays.npz") as z:
        base_lj = np.asarray(z["Lj"], dtype=float)
    return {
        "lj_scatter_enabled": False,
        "lj_scatter_sigma": 0.0,
        "lj_scatter_seed": int(seed),
        "lj_scatter_count": int(base_lj.size),
        "lj_scatter_clip_min": float(clip_min),
        "lj_scatter_clip_max": float(clip_max),
        "lj_scatter_factor_min": 1.0 if base_lj.size else None,
        "lj_scatter_factor_max": 1.0 if base_lj.size else None,
        "lj_scatter_factor_mean": 1.0 if base_lj.size else None,
        "lj_scatter_factor_std": 0.0 if base_lj.size else None,
        "lj_base_min_h": float(np.min(base_lj)) if base_lj.size else None,
        "lj_base_max_h": float(np.max(base_lj)) if base_lj.size else None,
        "lj_scattered_min_h": float(np.min(base_lj)) if base_lj.size else None,
        "lj_scattered_max_h": float(np.max(base_lj)) if base_lj.size else None,
    }


def _scatter_elements_csv(ipm_dir: Path, factors: np.ndarray) -> None:
    elements_path = ipm_dir / "ipm_elements.csv"
    if not elements_path.exists() or factors.size == 0:
        return

    with elements_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # csv.DictReader consumes the header, so preserve the canonical schema.
    header = ["idx", "name", "node1", "node2", "value", "kind"]
    jj_idx = 0
    for row in rows:
        if row.get("kind") == "josephson_inductor":
            row["value"] = repr(float(row["value"]) * float(factors[jj_idx]))
            jj_idx += 1
    if jj_idx != factors.size:
        raise ValueError(
            f"{elements_path} has {jj_idx} Josephson rows but ipm_arrays has {factors.size}"
        )

    with elements_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def _update_summary(ipm_dir: Path, source_ipm_dir: Path, meta: dict[str, object]) -> None:
    summary_path = ipm_dir / "ipm_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary.update(meta)
    summary["source_ipm_dir"] = str(source_ipm_dir)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def scatter_existing_design(
    source_ipm_dir: Path,
    outdir: Path,
    *,
    sigma: float,
    seed: int,
    clip_min: float,
    clip_max: float,
    overwrite: bool,
) -> dict[str, object]:
    _copytree_clean(source_ipm_dir, outdir, overwrite=overwrite)

    arrays_path = outdir / "ipm_arrays.npz"
    with np.load(arrays_path) as z:
        base_lj = np.asarray(z["Lj"], dtype=float)

    if sigma == 0.0:
        meta = _zero_scatter_meta(outdir, seed=seed, clip_min=clip_min, clip_max=clip_max)
    else:
        meta = _scatter_arrays(
            outdir,
            sigma=sigma,
            seed=seed,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        rng = np.random.default_rng(int(seed))
        factors = rng.normal(loc=1.0, scale=float(sigma), size=base_lj.size)
        factors = np.clip(factors, float(clip_min), float(clip_max))
        _scatter_elements_csv(outdir, factors)
    _update_summary(outdir, source_ipm_dir, meta)
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", choices=sorted(DEFAULT_SOURCES), required=True)
    p.add_argument(
        "--source-ipm-dir",
        type=Path,
        default=None,
        help="Source IPM artifact directory. Defaults to the standard 2c/3c output.",
    )
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--lj-scatter-sigma", type=float, default=0.0)
    p.add_argument("--lj-scatter-seed", type=int, default=1)
    p.add_argument("--lj-scatter-clip-min", type=float, default=0.5)
    p.add_argument("--lj-scatter-clip-max", type=float, default=1.5)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    source_ipm_dir = args.source_ipm_dir or DEFAULT_SOURCES[args.design]
    if not source_ipm_dir.exists():
        raise FileNotFoundError(source_ipm_dir)

    meta = scatter_existing_design(
        source_ipm_dir,
        args.outdir,
        sigma=args.lj_scatter_sigma,
        seed=args.lj_scatter_seed,
        clip_min=args.lj_scatter_clip_min,
        clip_max=args.lj_scatter_clip_max,
        overwrite=args.overwrite,
    )
    print(f"wrote {args.outdir}")
    print(f"design={args.design}")
    print(f"source_ipm_dir={source_ipm_dir}")
    print(f"jj={meta['lj_scatter_count']}")
    print(f"lj_scatter_sigma={meta['lj_scatter_sigma']}")
    print(f"lj_scatter_seed={meta['lj_scatter_seed']}")
    print(f"lj_scatter_factor_std={meta['lj_scatter_factor_std']}")


if __name__ == "__main__":
    main()
