"""H0-lite (fold_plan.md, 2026-08-08): frequency reconnaissance around the
mu~0.5253 SIMPLE_FOLD found by Milestone F.5.

NOT a two-parameter bordered fold corrector -- explicitly out of scope per
the user. F.5 already retracted the cusp hypothesis this branch of work was
built to test (alpha=-2.08e21, unambiguous SIMPLE_FOLD). What remains is a
cheap sanity check: does this fold persist and move smoothly with pump
frequency (Hypothesis A, ordinary isolated fold), or does something odd
happen nearby (worth a second look, not built here).

Runs the SAME region_0525 recipe (ds=0.005, mu_max=0.60, max_steps=1200,
max_steps_after_fold=300 -- proven cheap, ~255s/run) at five nearby
frequencies (7.85/7.875/7.90/7.925/7.95 GHz). Each frequency is an
independent cold trace (mu0=0, X0=None) in its OWN process invocation --
the Milestone F OOM lesson (a combined multi-branch single-process run was
killed, exit 137) applies here too, and a cold trace is already proven cheap
enough that cross-frequency warm-seed chaining is not needed to keep this
"lightweight" per the user's own framing.

Records, for the region nearest mu~0.5253 at each frequency: mu_f, I_f (A),
P_f (dBm), sigma1_hat, sigma1/sigma2, tau, alpha_hat, singular-vector
residuals, class. A second pass (--aggregate) reads all five per-frequency
JSON files and prints the table + writes the mu_f(f_p) figure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map  # noqa: E402
from scripts.validate_fold_plan_ad import _build_ref_problem, _current_to_dbm, _engine_args  # noqa: E402
from twpa_solver.pump import hb as exp08  # noqa: E402
from twpa_solver.pump.bifurcation import classify_fold_candidates  # noqa: E402
from twpa_solver.pump.solver import find_fold_candidates, trace_branch  # noqa: E402

OUT_DIR = ROOT / "outputs" / "fold_plan_milestone_h0_lite"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_DBM = -16.0
DS = 0.005
MU_MAX = 0.60
MAX_STEPS = 1200
MAX_STEPS_AFTER_FOLD = 300
FREQS_GHZ = [7.85, 7.875, 7.90, 7.925, 7.95]


def _fmt(x, spec: str = ".4g") -> str:
    if x is None:
        return "n/a"
    try:
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def run_one_frequency(freq_ghz: float) -> dict:
    circuit_dir = ROOT / "designs" / "ipm_2c_fixed"
    scratch = Path("D:/tmp/h0_lite_report") / f"{freq_ghz:.3f}".replace(".", "p")
    engine = run_gain_map.InProcessEngine(_engine_args(circuit_dir, scratch, freq_ghz, REF_DBM))
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    problem, _basis, ref_injected = _build_ref_problem(engine, freq_ghz, REF_DBM)

    t0 = time.perf_counter()
    branch = trace_branch(
        solver, problem, i_ref=ref_injected, mu0=0.0, mu_max=MU_MAX, ds=DS,
        max_steps=MAX_STEPS, max_steps_after_fold=MAX_STEPS_AFTER_FOLD,
        step_control="adaptive", rescale_every=5, refine_fold=True,
    )
    trace_s = time.perf_counter() - t0

    candidates = find_fold_candidates(branch)
    rows = classify_fold_candidates(solver, problem, branch)

    print(f"\n{'=' * 100}")
    print(f"{freq_ghz} GHz: {len(branch.points)} pts, terminal={branch.info.get('terminal_reason')}, "
          f"trace={trace_s:.1f}s, {len(candidates)} candidate(s), {len(rows)} region(s)")

    region = min(
        (r for r in rows if r.get("mu") is not None), key=lambda r: abs(r["mu"] - 0.5253), default=None,
    )
    if region is None:
        print(f"  NO REGION FOUND near mu~0.5253 at {freq_ghz} GHz")
        return {"freq_ghz": freq_ghz, "found": False}

    mu_f = region["mu"]
    i_f = mu_f * ref_injected
    p_f = _current_to_dbm(i_f, freq_ghz, engine.args)
    result = {
        "freq_ghz": freq_ghz, "found": True, "mu_f": mu_f, "i_f_a": i_f, "p_f_dbm": p_f,
        "sigma1_hat": region.get("sigma1_hat"), "sigma_gap": region.get("sigma_gap"),
        "tau": region.get("tau"), "alpha_hat": region.get("alpha_hat"),
        "r_right": region.get("singular_residual_right"),
        "r_left": region.get("singular_residual_left"),
        "class": region.get("class"), "reason": region.get("reason"),
    }
    print(f"  mu_f={mu_f:.6f} I_f={i_f:.4e}A P_f={_fmt(p_f)}dBm "
          f"sigma1_hat={_fmt(region.get('sigma1_hat'))} sigma_gap={_fmt(region.get('sigma_gap'))} "
          f"tau={_fmt(region.get('tau'))} alpha_hat={_fmt(region.get('alpha_hat'))} "
          f"class={region.get('class')}")
    return result


def aggregate(json_paths: list[Path]) -> None:
    results = []
    for p in json_paths:
        with p.open() as fh:
            results.append(json.load(fh))
    results.sort(key=lambda r: r["freq_ghz"])

    header = (
        f"{'f_p GHz':>10}{'mu_f':>12}{'I_f (A)':>14}{'P_f (dBm)':>12}"
        f"{'sigma1_hat':>14}{'sigma1/sigma2':>16}{'tau':>12}{'alpha_hat':>14}{'class':>32}"
    )
    print(header)
    for r in results:
        if not r.get("found"):
            print(f"{r['freq_ghz']:>10.3f}{'NO FOLD FOUND':>70}")
            continue
        print(
            f"{r['freq_ghz']:>10.3f}{_fmt(r['mu_f'], '.6f'):>12}{_fmt(r['i_f_a'], '.4e'):>14}"
            f"{_fmt(r['p_f_dbm'], '.3f'):>12}{_fmt(r['sigma1_hat']):>14}{_fmt(r['sigma_gap']):>16}"
            f"{_fmt(r['tau']):>12}{_fmt(r['alpha_hat']):>14}{r['class']:>32}"
        )

    found = [r for r in results if r.get("found")]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([r["freq_ghz"] for r in found], [r["mu_f"] for r in found], "o-")
    ax.set_xlabel("pump frequency f_p (GHz)")
    ax.set_ylabel("mu_f")
    ax.set_title("H0-lite: mu~0.5253 fold locus vs pump frequency")
    fig.tight_layout()
    out_path = OUT_DIR / "mu_f_vs_freq.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")

    smooth = len(found) >= 2 and all(
        r.get("class") == "SIMPLE_FOLD" for r in found
    )
    print(f"\nall found regions SIMPLE_FOLD: {smooth}")
    print("DONE_H0_LITE_AGGREGATE")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--freq-ghz", type=float, required=True)
    p_run.add_argument("--out-json", type=Path, required=True)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--json-dir", type=Path, required=True)

    args = parser.parse_args()

    if args.cmd == "run":
        result = run_one_frequency(args.freq_ghz)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"wrote {args.out_json}")
        print("DONE_H0_LITE")
    else:
        paths = sorted(args.json_dir.glob("*.json"))
        aggregate(paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
