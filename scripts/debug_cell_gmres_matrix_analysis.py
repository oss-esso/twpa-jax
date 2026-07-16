"""Cold vs warm GMRES/Jacobian analysis for one failing map cell.

Investigates map cell point_index=8 (fp=7.32894736842 GHz,
P=-28.253164556962027 dBm) from the fp=7.329 GHz column
(outputs/measurement_match_debug_02_col3_trim_fixed) -- the first cell in that
column that fails. Reuses the exact production code path (InProcessEngine,
same solver settings) so the numerics match the real map run bit-for-bit.

Runs the cell two ways:
  * cold  -- ``engine.solve_point(mode="seed")``: continuation from X=0.
  * warm  -- ``engine.solve_point(mode="warm")``: a single direct Newton solve
             at lambda=1 seeded with the previous cell's converged solution
             (point 7, P=-28.468354430379748 dBm, hardcoded below, re-solved
             fresh in this script rather than reloaded from disk so the seed
             is in the exact Schur-retained shape the solver expects).

Instrumentation (monkeypatches, restored on exit):
  * ``FastCoupledPreconditioner.refactor`` -- snapshots the exact real-packed
    Jacobian/preconditioner matrix M (see src/twpa_solver/pump/backends/
    fast_coupled.py: "producing the identical exact preconditioner") at every
    Newton iteration, for both runs.
  * ``twpa_solver.pump.solver.gmres_call`` -- snapshots the per-GMRES-iteration
    residual history (pr_norm), not just the final iteration count.

Output (outputs/cell_gmres_analysis_col3_p8/):
  * summary.csv -- one row per Newton iteration: run, iter, lambda, coeff_rel
    before/after, gmres iters, gmres final pr_norm, M nnz/shape/norm/asymmetry.
  * gmres_convergence.png -- pr_norm vs GMRES iteration, one curve per Newton
    step, cold vs warm side by side.
  * coeff_rel_vs_iteration.png -- Newton residual trajectory, cold vs warm.
  * spy_<run>_iter<k>.png -- sparsity pattern of M at the first, a randomly
    picked, and the last captured Newton iteration of each run.
  * condition_estimate.txt -- 1-norm condition number estimate (onenormest,
    Higham-Hager) for the randomly picked M, since a direct dense/eigenvalue
    analysis of a 50360x50360 sparse matrix is not tractable here.
  * per_harmonic_residual.csv / per_harmonic_residual.png -- breaks the scalar
    coeff_rel down by harmonic index k (RMS of R_k over retained nodes,
    normalized by the scalar source RMS) for the last residual_coeffs() call
    of the converged baseline (prev_seed) vs the stalled cold/warm point-8
    iterates -- shows whether the residual mass is broadband or concentrated
    in specific harmonics, since GMRES/preconditioner are already ruled out.
"""
from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gain_map as rgm  # noqa: E402
from twpa_solver.pump.backends.fast_coupled import FastCoupledPreconditioner  # noqa: E402
from twpa_solver.pump.backends.schur_operators import SchurReducedProblem  # noqa: E402
from twpa_solver.pump.hb import pack_complex  # noqa: E402
from twpa_solver.pump import solver as pump_solver  # noqa: E402

OUT_DIR = ROOT / "outputs" / "cell_gmres_analysis_col3_p8"
FREQ_GHZ = 7.32894736842

# Hardcoded from outputs/measurement_match_debug_02_col3_trim_fixed/map_points.csv
PREV_POINT = rgm.GridPoint(
    index=7, i_power=7, j_freq=0,
    power_dbm=-28.468354430379748, pump_freq_ghz=FREQ_GHZ,
    current_a=4.3659520240506854e-06,
)
FAIL_POINT = rgm.GridPoint(
    index=8, i_power=8, j_freq=0,
    power_dbm=-28.253164556962027, pump_freq_ghz=FREQ_GHZ,
    current_a=4.47546796420373e-06,
)

# argv mirrors the CLI command that produced column_debug_col3_trim /
# measurement_match_debug_02_col3_trim_fixed (see CLAUDE.md / prior session).
ARGV = [
    "--executor", "inprocess", "--mode", "warmstart",
    "--circuit-dir", "outputs/ipm_python_design",
    "--outdir", str(OUT_DIR),
    "--n-power", "1", "--n-frequency", "1",
    "--pump-power-min-dbm", "-28.253164556962027",
    "--pump-power-max-dbm", "-28.253164556962027",
    "--pump-freq-min-ghz", str(FREQ_GHZ), "--pump-freq-max-ghz", str(FREQ_GHZ),
    "--inproc-pump-backend", "schur_cpu_mt",
    "--inproc-preconditioner", "real_coupled_fast",
    "--inproc-fold-predictor", "secant",
    "--fold-skip-patience", "2",
    "--column-power-substep",
    "--column-power-substep-min-db", "0.005",
    "--inproc-schur-cache-size", "2",
    "--inproc-max-newton", "16",
    "--inproc-solve-deadline-s", "14",
    "--pump-mode-count", "10", "--nt", "40",
    "--signal-detuning-mhz", "200", "--no-signal-spectrum",
    "--signal-backend", "direct", "--signal-solver", "superlu",
    "--sidebands", "10", "--signal-workers", "6",
]


@dataclass
class NewtonSnapshot:
    run: str
    iter_index: int
    source_scale: float
    coeff_rel_before: float | None
    coeff_rel_after: float | None
    converged: bool
    newton_iterations: int
    gmres_iters: int
    gmres_pr_norms: list[float]
    M: sp.csr_matrix | None


SNAPSHOTS: list[NewtonSnapshot] = []
_CURRENT_RUN = {"label": "unset"}
_LAST_GMRES_PR_NORMS: list[list[float]] = []


def _capturing_refactor(self: FastCoupledPreconditioner, tangent) -> None:
    _ORIG_REFACTOR(self, tangent)
    SNAPSHOTS.append(
        NewtonSnapshot(
            run=_CURRENT_RUN["label"], iter_index=len(SNAPSHOTS),
            source_scale=float("nan"), coeff_rel_before=None, coeff_rel_after=None,
            converged=False, newton_iterations=0, gmres_iters=0, gmres_pr_norms=[],
            M=self.M.copy(),
        )
    )


def _capturing_gmres_call(*args, **kwargs):
    pr_norms: list[float] = []
    user_cb = kwargs.get("callback")

    def cb(pr_norm: float) -> None:
        pr_norms.append(float(pr_norm))
        if user_cb is not None:
            user_cb(pr_norm)

    kwargs["callback"] = cb
    result = _ORIG_GMRES_CALL(*args, **kwargs)
    _LAST_GMRES_PR_NORMS.append(pr_norms)
    return result


@dataclass
class ResidualSnapshot:
    run: str
    call_index: int
    source_scale: float
    coeff_rel: float
    modes: np.ndarray
    retained_full_idx: np.ndarray
    pump_pos: int
    R: np.ndarray = field(repr=False)


RESIDUAL_SNAPSHOTS: list[ResidualSnapshot] = []


def _capturing_residual_coeffs(self: SchurReducedProblem, Xn: np.ndarray, source_scale: float) -> np.ndarray:
    """Wraps SchurReducedProblem.residual_coeffs to snapshot every call.

    Called once per Newton iteration (top of the loop, on the currently
    accepted state) plus once per line-search trial (via problem.norms), so
    consecutive snapshots within a run are not all distinct Newton iterates --
    the last snapshot of a run is always the final (possibly non-converged)
    state, which is what the per-harmonic breakdown below actually wants.
    """
    R = _ORIG_RESIDUAL_COEFFS(self, Xn, source_scale)
    S = self.source_coeffs(source_scale)
    coeff_abs = float(np.linalg.norm(pack_complex(R)) / math.sqrt(pack_complex(R).size))
    src_abs = float(np.linalg.norm(pack_complex(S)) / max(math.sqrt(pack_complex(S).size), 1.0))
    RESIDUAL_SNAPSHOTS.append(
        ResidualSnapshot(
            run=_CURRENT_RUN["label"], call_index=len(RESIDUAL_SNAPSHOTS),
            source_scale=float(source_scale),
            coeff_rel=coeff_abs / max(src_abs, 1e-30),
            modes=np.asarray(self.grid.k, dtype=float).copy(),
            retained_full_idx=np.asarray(self.part.retained, dtype=int).copy(),
            pump_pos=int(self.pump_pos),
            R=R.copy(),
        )
    )
    return R


_ORIG_REFACTOR = FastCoupledPreconditioner.refactor
_ORIG_GMRES_CALL = pump_solver.gmres_call
_ORIG_RESIDUAL_COEFFS = SchurReducedProblem.residual_coeffs


def install_hooks() -> None:
    FastCoupledPreconditioner.refactor = _capturing_refactor
    pump_solver.gmres_call = _capturing_gmres_call
    SchurReducedProblem.residual_coeffs = _capturing_residual_coeffs


def restore_hooks() -> None:
    FastCoupledPreconditioner.refactor = _ORIG_REFACTOR
    pump_solver.gmres_call = _ORIG_GMRES_CALL
    SchurReducedProblem.residual_coeffs = _ORIG_RESIDUAL_COEFFS


def run_cell(engine: rgm.InProcessEngine, point: rgm.GridPoint, *, mode: str,
             warm_X: np.ndarray | None, label: str) -> tuple[dict[str, Any], np.ndarray | None]:
    """Run one cell through the production solve_point path, tagging snapshots."""
    _CURRENT_RUN["label"] = label
    before = len(SNAPSHOTS)
    pass_dir = OUT_DIR / label
    row, X = engine.solve_point(point, pass_dir, mode=mode, warm_X=warm_X)
    reports = _load_reports(pass_dir, point)
    _attach_reports_to_snapshots(before, reports, label)
    return row, X


def _load_reports(pass_dir: Path, point: rgm.GridPoint) -> list[dict[str, Any]]:
    import json

    pdir = pass_dir / "points" / rgm.point_name(point.index, point.power_dbm, point.pump_freq_ghz)
    report_path = pdir / "pump" / "pump_report.json"
    return json.loads(report_path.read_text())["reports"]


def _attach_reports_to_snapshots(before: int, reports: list[dict[str, Any]], label: str) -> None:
    """Match each solve_one's StepReport (one per Newton *solve*, i.e. the whole
    continuation step) back onto the per-Newton-iteration M snapshots.

    Each StepReport covers ``newton_iterations`` Newton steps, so it aligns
    with that many consecutive (M, gmres) snapshots captured since ``before``.
    """
    cursor = before
    for rep in reports:
        n_iters = int(rep["newton_iterations"])
        for k in range(n_iters):
            if cursor >= len(SNAPSHOTS):
                break
            snap = SNAPSHOTS[cursor]
            snap.source_scale = float(rep["source_scale"])
            snap.converged = bool(rep["converged"]) and k == n_iters - 1
            snap.newton_iterations = n_iters
            pr_norms = _LAST_GMRES_PR_NORMS[cursor] if cursor < len(_LAST_GMRES_PR_NORMS) else []
            snap.gmres_pr_norms = pr_norms
            snap.gmres_iters = len(pr_norms)
            cursor += 1


def matrix_stats(M: sp.csr_matrix) -> dict[str, float]:
    frob = float(spla.norm(M, "fro"))
    diff = (M - M.T).tocsr()
    asym = float(spla.norm(diff, "fro")) / frob if frob > 0 else float("nan")
    return {
        "shape0": M.shape[0], "shape1": M.shape[1],
        "nnz": M.nnz, "fill_frac": M.nnz / (M.shape[0] * M.shape[1]),
        "frobenius_norm": frob, "asymmetry_frac": asym,
    }


def condition_estimate(M: sp.csr_matrix) -> dict[str, float]:
    t0 = time.perf_counter()
    norm_m = float(spla.onenormest(M))
    lu = spla.splu(M.tocsc())

    def solve(v: np.ndarray) -> np.ndarray:
        return lu.solve(v)

    def solve_t(v: np.ndarray) -> np.ndarray:
        return lu.solve(v, trans="T")

    minv_op = spla.LinearOperator(M.shape, matvec=solve, rmatvec=solve_t)
    norm_minv = float(spla.onenormest(minv_op))
    return {
        "norm_1_M": norm_m, "norm_1_Minv": norm_minv,
        "cond_1_estimate": norm_m * norm_minv,
        "runtime_s": time.perf_counter() - t0,
    }


def spy_plot(M: sp.csr_matrix, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.spy(M, markersize=0.15, rasterized=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = rgm.parse_args(ARGV)
    engine = rgm.InProcessEngine(args)

    install_hooks()
    try:
        print("=== solving previous cell (point 7, converged) to obtain the warm seed ===")
        row7, X7 = run_cell(engine, PREV_POINT, mode="seed", warm_X=None, label="prev_seed")
        print(f"point 7 status={row7['status']} gain={row7.get('gain_db')}")
        assert row7["pump_status"] == "VALID_CONVERGED", "seed point must converge"

        print("=== COLD: point 8 from X=0 (adaptive_secant continuation, fixed fallback) ===")
        row_cold, _ = run_cell(engine, FAIL_POINT, mode="seed", warm_X=None, label="cold")
        print(f"cold status={row_cold['status']} pump_status={row_cold['pump_status']}")

        print("=== WARM: point 8 direct Newton solve seeded from point 7 ===")
        row_warm, _ = run_cell(engine, FAIL_POINT, mode="warm", warm_X=X7, label="warm")
        print(f"warm status={row_warm['status']} pump_status={row_warm['pump_status']}")
    finally:
        restore_hooks()

    analyze()


def analyze() -> None:
    import csv

    rows = []
    for snap in SNAPSHOTS:
        if snap.run == "prev_seed" or snap.M is None:
            continue
        stats = matrix_stats(snap.M)
        final_pr = snap.gmres_pr_norms[-1] if snap.gmres_pr_norms else None
        rows.append({
            "run": snap.run, "iter_index": snap.iter_index,
            "source_scale": snap.source_scale, "converged": snap.converged,
            "gmres_iters": snap.gmres_iters, "gmres_final_pr_norm": final_pr,
            **stats,
        })

    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} Newton-iteration snapshots)")

    plot_gmres_convergence()
    plot_coeff_rel_trajectory()
    spy_selected()
    random_condition_estimate()
    dump_per_harmonic_residual()
    dump_per_node_residual()


def plot_gmres_convergence() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, label in zip(axes, ["cold", "warm"]):
        for snap in SNAPSHOTS:
            if snap.run != label or not snap.gmres_pr_norms:
                continue
            ax.semilogy(
                range(1, len(snap.gmres_pr_norms) + 1), snap.gmres_pr_norms,
                marker=".", label=f"newton it {snap.iter_index}",
            )
        ax.set_title(f"{label}: GMRES pr_norm per iteration")
        ax.set_xlabel("GMRES iteration")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("pr_norm")
    fig.tight_layout()
    out = OUT_DIR / "gmres_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_coeff_rel_trajectory() -> None:
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in ["cold", "warm"]:
        report_path = (
            OUT_DIR / label / "points"
            / rgm.point_name(FAIL_POINT.index, FAIL_POINT.power_dbm, FAIL_POINT.pump_freq_ghz)
            / "pump" / "pump_report.json"
        )
        reports = json.loads(report_path.read_text())["reports"]
        coeff = []
        cursor = 0
        for rep in reports:
            cursor += int(rep["newton_iterations"])
            coeff.append((cursor, rep["coeff_rel"], rep["source_scale"]))
        xs = [c[0] for c in coeff]
        ys = [c[1] for c in coeff]
        ax.semilogy(xs, ys, marker="o", label=label)
    ax.set_xlabel("cumulative Newton iteration")
    ax.set_ylabel("coeff_rel")
    ax.set_title("point 8: Newton residual trajectory, cold vs warm")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "coeff_rel_vs_iteration.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def spy_selected() -> None:
    for label in ["cold", "warm"]:
        snaps = [s for s in SNAPSHOTS if s.run == label and s.M is not None]
        if not snaps:
            continue
        picks = {"first": snaps[0], "last": snaps[-1]}
        if len(snaps) > 2:
            picks["random"] = random.choice(snaps[1:-1])
        for tag, snap in picks.items():
            out = OUT_DIR / f"spy_{label}_{tag}_iter{snap.iter_index}.png"
            spy_plot(
                snap.M, out,
                f"{label} iter {snap.iter_index} (lambda={snap.source_scale:.4f}) "
                f"nnz={snap.M.nnz}",
            )
            print(f"wrote {out}")


def random_condition_estimate() -> None:
    all_snaps = [s for s in SNAPSHOTS if s.M is not None]
    if not all_snaps:
        return
    snap = random.choice(all_snaps)
    print(
        f"=== randomly picked snapshot for condition estimate: "
        f"run={snap.run} iter={snap.iter_index} lambda={snap.source_scale:.4f} ==="
    )
    stats = matrix_stats(snap.M)
    cond = condition_estimate(snap.M)
    out = OUT_DIR / "condition_estimate.txt"
    with open(out, "w") as f:
        f.write(f"run={snap.run} iter_index={snap.iter_index} "
                f"source_scale={snap.source_scale}\n")
        for k, v in {**stats, **cond}.items():
            f.write(f"{k}={v}\n")
    print(f"wrote {out}")
    for k, v in {**stats, **cond}.items():
        print(f"  {k}={v}")


def per_harmonic_rms(R: np.ndarray) -> np.ndarray:
    """RMS of R_k over retained nodes, for each harmonic row k."""
    return np.linalg.norm(R, axis=1) / math.sqrt(R.shape[1])


def dump_per_harmonic_residual() -> None:
    """Per-harmonic |R_k| for the last residual_coeffs() call of each run.

    "Last call" of a run is the final state the solver evaluated -- the
    converged state for prev_seed, the stalled non-converged state for cold
    and warm (both point 8, at the failing high-power cell).
    """
    import csv

    picks: list[tuple[str, ResidualSnapshot]] = []
    for label in ["prev_seed", "cold", "warm"]:
        snaps = [s for s in RESIDUAL_SNAPSHOTS if s.run == label]
        if snaps:
            picks.append((label, snaps[-1]))

    rows = []
    for tag, snap in picks:
        per_h = per_harmonic_rms(snap.R)
        for h, k in enumerate(snap.modes):
            rows.append({
                "run": tag, "call_index": snap.call_index,
                "source_scale": snap.source_scale, "coeff_rel": snap.coeff_rel,
                "harmonic_k": int(round(k)),
                "R_k_rms": float(per_h[h]),
            })

    csv_path = OUT_DIR / "per_harmonic_residual.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    plot_per_harmonic_residual(picks)


def plot_per_harmonic_residual(picks: list[tuple[str, "ResidualSnapshot"]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    n_picks = len(picks)
    width = 0.8 / max(n_picks, 1)
    # All three runs share the same fixed pump-mode-count-10 basis, so the
    # harmonic list (and its ordering) is identical across picks.
    ref_ks = [int(round(k)) for k in picks[0][1].modes]
    for i, (tag, snap) in enumerate(picks):
        per_h = per_harmonic_rms(snap.R)
        xs = np.arange(len(ref_ks)) + i * width
        ax.bar(
            xs, per_h, width=width,
            label=f"{tag} (lambda={snap.source_scale:.4f}, coeff_rel={snap.coeff_rel:.2e})",
        )
    ax.set_xticks(np.arange(len(ref_ks)) + 0.5 * width * (n_picks - 1))
    ax.set_xticklabels([str(k) for k in ref_ks])
    ax.set_yscale("log")
    ax.set_xlabel("pump harmonic k")
    ax.set_ylabel("RMS |R_k| over retained nodes")
    ax.set_title("point 8: per-harmonic residual, converged baseline vs stalled cold/warm")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "per_harmonic_residual.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def per_node_rms(R: np.ndarray) -> np.ndarray:
    """RMS of R_n over harmonics, for each retained node n."""
    return np.linalg.norm(R, axis=0) / math.sqrt(R.shape[0])


def dump_per_node_residual() -> None:
    """Spatial (per-retained-node) residual breakdown, same three picks as
    dump_per_harmonic_residual: does the stalled residual concentrate on a
    few nodes (a local device nonlinearity) or spread across the whole
    retained set (a global amplitude limit)?
    """
    import csv

    picks: list[tuple[str, ResidualSnapshot]] = []
    for label in ["prev_seed", "cold", "warm"]:
        snaps = [s for s in RESIDUAL_SNAPSHOTS if s.run == label]
        if snaps:
            picks.append((label, snaps[-1]))

    rows = []
    for tag, snap in picks:
        per_n = per_node_rms(snap.R)
        order = np.argsort(per_n)[::-1]
        for rank, pos in enumerate(order[:20]):
            rows.append({
                "run": tag, "rank": rank,
                "retained_position": int(pos),
                "full_node_index": int(snap.retained_full_idx[pos]),
                "is_pump_node": bool(pos == snap.pump_pos),
                "R_n_rms": float(per_n[pos]),
            })

    csv_path = OUT_DIR / "per_node_residual_top20.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    plot_per_node_residual(picks)


def plot_per_node_residual(picks: list[tuple[str, "ResidualSnapshot"]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for tag, snap in picks:
        per_n = per_node_rms(snap.R)
        ax.semilogy(
            np.arange(per_n.size), per_n, linewidth=0.8,
            label=f"{tag} (coeff_rel={snap.coeff_rel:.2e})",
        )
        ax.axvline(snap.pump_pos, color="k", linestyle=":", alpha=0.3)
    ax.set_xlabel("retained-node position (line order; dotted = pump node)")
    ax.set_ylabel("RMS |R_n| over harmonics")
    ax.set_title("point 8: per-node residual, converged baseline vs stalled cold/warm")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "per_node_residual.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
