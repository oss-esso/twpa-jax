"""Build the eight design variants for the dissipation/scatter campaign.

Dissipation and scatter are deliberately NOT crossed: one arm varies the loss
tangent at zero scatter, the other varies the Lj scatter at zero loss. See
``docs/development/dissipation_plan.md`` Phase 7.

The scatter arm uses ``plasma_locked`` Cj, so the junction plasma frequency
1/sqrt(Lj*Cj) is held constant cell by cell and the four sigma values are the
same realisation at four amplitudes rather than four unrelated draws.

    python scripts/build_dissipation_campaign.py --outroot designs/campaign_diss
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.builders.ipm import (  # noqa: E402
    LossSpec,
    build_variant_design,
)
from twpa_solver.builders.scatter import ScatterSpec  # noqa: E402

DEFAULT_SOURCE = ROOT / "designs" / "ipm_2c_fixed"
SCATTER_SEED = 1


@dataclass(frozen=True)
class Variant:
    run_id: str
    tan_delta: float
    lj_sigma: float


VARIANTS: tuple[Variant, ...] = (
    Variant("2c_base", 0.0, 0.0),
    Variant("2c_td1e5", 1e-5, 0.0),
    Variant("2c_td1e4", 1e-4, 0.0),
    Variant("2c_td1e3", 1e-3, 0.0),
    Variant("2c_sc1pct", 0.0, 0.01),
    Variant("2c_sc3pct", 0.0, 0.03),
    Variant("2c_sc5pct", 0.0, 0.05),
    Variant("2c_sc10pct", 0.0, 0.10),
)


def build_one(variant: Variant, source: Path, outroot: Path, *, overwrite: bool) -> dict:
    outdir = outroot / variant.run_id
    lj_scatter = ScatterSpec(sigma=variant.lj_sigma)
    # A zero-sigma Lj draw leaves Cj at nominal either way, so plasma_locked is
    # only requested when there is a draw to lock to.
    cj_scatter = (
        ScatterSpec(mode="plasma_locked") if variant.lj_sigma > 0.0 else ScatterSpec()
    )
    summary = build_variant_design(
        source,
        outdir,
        lj_scatter=lj_scatter,
        cj_scatter=cj_scatter,
        seed=SCATTER_SEED,
        overwrite=overwrite,
        coupler_mode="cached",
        loss=LossSpec(default=variant.tan_delta),
    )
    return {
        "run_id": variant.run_id,
        "outdir": str(outdir),
        "tan_delta": variant.tan_delta,
        "lj_scatter_sigma": variant.lj_sigma,
        "cj_scatter_mode": "plasma_locked" if variant.lj_sigma > 0.0 else "none",
        "scatter_seed": SCATTER_SEED,
        "total_elements": summary.get("total_elements"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--outroot", type=Path, default=ROOT / "designs" / "campaign_diss")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only", default="", help="Comma-separated run ids; default builds all eight."
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(args.source)
    selected = {v.strip() for v in args.only.split(",") if v.strip()}
    variants = [v for v in VARIANTS if not selected or v.run_id in selected]
    if selected - {v.run_id for v in VARIANTS}:
        raise SystemExit(f"unknown run ids: {sorted(selected - {v.run_id for v in VARIANTS})}")

    args.outroot.mkdir(parents=True, exist_ok=True)
    manifest = []
    for variant in variants:
        record = build_one(variant, args.source, args.outroot, overwrite=args.overwrite)
        manifest.append(record)
        print(
            f"{record['run_id']:<12} elements={record['total_elements']} "
            f"tan_delta={record['tan_delta']:<8g} lj_sigma={record['lj_scatter_sigma']:<5g} "
            f"cj_mode={record['cj_scatter_mode']}"
        )

    manifest_path = args.outroot / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
