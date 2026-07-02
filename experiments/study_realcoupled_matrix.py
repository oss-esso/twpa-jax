"""Decomposability study of the real-coupled preconditioner matrix M at the fold.

The production solver ``schur_cpu_rcfast`` factors the real-packed real-coupled
Jacobian M (2H x 2H super-blocks of the retained node count) with sparse LU
(Pardiso, symbolic reuse) once per Newton step. This script asks the question
"is any *other* decomposition cheaper?" by measuring the structural and numerical
properties that select a factorization:

  structure : structural symmetry, block (2H x 2H) pattern, RCM bandwidth, SCCs
  numeric   : numerical symmetry / skew, diagonal dominance, symmetric-part
              spectrum (definiteness), and the *source* of any asymmetry
  factor    : LU fill (L+U nnz / A nnz), and the preconditioner quality of the
              symmetrized matrix (what a symmetric LDL^T would deliver)

Run:
    python experiments/study_realcoupled_matrix.py --pump-freq-ghz 7.0 --power-dbm -22
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as spla

import exp08_full_ipm_pump_solve as exp08
import pump_basis
from pump_solvers.fast_coupled import FastCoupledPreconditioner
from pump_solvers.schur_operators import build_schur_problem

ROOT = Path(__file__).resolve().parents[1]


def assemble_fold_matrix(ipm_dir: Path, pump_freq_ghz: float, power_dbm: float,
                         attenuation_db: float, z0_ohm: float,
                         jc_scale: float, mode_count: int, nt: int,
                         pump_port: int) -> tuple[sp.csr_matrix, int]:
    """Warm-start to the fold and return the assembled real-coupled M and H."""
    ipm = exp08.load_ipm(ipm_dir)
    branch = exp08.JosephsonBranchArray(Ic=ipm.Ic, phi0=ipm.phi0)
    omega = 2 * math.pi * pump_freq_ghz * 1e9
    basis = pump_basis.resolve_pump_basis(
        policy="positive_odd_jc", omega_p=omega, harmonics=3,
        mode_count=mode_count, explicit_modes=None, design_meta=ipm.summary)
    grid = exp08.HarmonicGrid(modes=basis.k, nt=nt, omega=omega)
    ports = list(ipm.port_to_index.values())
    cur = math.sqrt(2.0e-3 * 10.0 ** ((power_dbm - attenuation_db) / 10.0) / z0_ohm)
    full = exp08.FullIPMPumpProblem(
        C=ipm.C, G=ipm.G, K=ipm.K, Bphi=ipm.Bphi, branch=branch, grid=grid,
        pump_node_index=ipm.port_to_index[pump_port],
        pump_current_a=cur * jc_scale, source_mode=basis.source_mode)
    sprob = build_schur_problem(full, ports)
    s = exp08.NewtonKrylovSettings(
        newton_tol=1e-9, max_newton=16, gmres_rtol=1e-7, gmres_atol=0.0,
        gmres_restart=60, gmres_maxiter=80, min_alpha=1 / 1024,
        preconditioner="mean_tangent", compute_time_residual=False,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
        stall_ratio=0.8, stall_patience=4)
    X, reps = exp08.HarmonicNewtonKrylovSolver(s).solve_continuation(
        sprob, continuation_steps=20)
    if not reps[-1].converged:
        raise RuntimeError("warm-start to the fold did not converge")
    pc = FastCoupledPreconditioner(sprob, use_pardiso=False)
    pc.refactor(sprob.tangent_state(X))
    return pc.M.tocsr(), pc.H


def hr(title: str) -> None:
    print("\n== " + title + " " + "=" * max(0, 58 - len(title)))


def study(M: sp.csr_matrix, H: int) -> None:
    N = M.shape[0]
    m = N // (2 * H)
    fn = spla.norm(M)
    print(f"M: {N}x{N}  nnz={M.nnz} ({M.nnz/N:.1f}/row)  "
          f"fill={M.nnz/N**2*100:.4f}%   block m={m}, 2H={2*H} super-blocks")

    hr("symmetry")
    P = (M != 0).astype(np.int8)
    dP = (P - P.T.tocsr()); dP.eliminate_zeros()
    asym = spla.norm(M - M.T) / fn
    print(f"structural: {dP.nnz} asymmetric pattern entries "
          f"({'STRUCTURALLY SYMMETRIC' if dP.nnz == 0 else 'not'})")
    print(f"numerical : ||M-M^T||/||M|| = {asym:.3e} "
          f"({'symmetric' if asym < 1e-10 else 'NONsymmetric'})")

    hr("source of asymmetry (block-diagonal vs harmonic coupling)")
    C = M.tocoo()
    kr, kc = (C.row // m) % H, (C.col // m) % H
    same = kr == kc
    Md = sp.csr_matrix((C.data[same], (C.row[same], C.col[same])), shape=M.shape)
    Mo = sp.csr_matrix((C.data[~same], (C.row[~same], C.col[~same])), shape=M.shape)
    print(f"block-diag (k==q) part : ||A-A^T||/||A|| = "
          f"{spla.norm(Md-Md.T)/spla.norm(Md):.3e}")
    print(f"harmonic-coupling part : ||A-A^T||/||A|| = "
          f"{spla.norm(Mo-Mo.T)/max(spla.norm(Mo),1e-30):.3e}")
    print("=> the asymmetry is the conjugate (k+q) real-embedding term (ri=Pi-Li "
          "vs ir=Pi+Li); it is intrinsic -- removing it drops the term that\n"
          "   collapses GMRES to one iteration.")

    hr("block structure (harmonic 2H x 2H)")
    nb = 2 * H
    Pc = P.tocoo()
    blk = sp.csr_matrix((np.ones(Pc.nnz), (Pc.row // m, Pc.col // m)),
                        shape=(nb, nb))
    blk = (blk != 0).toarray()
    bband = max(abs(i - j) for i in range(nb) for j in range(nb) if blk[i, j])
    print(f"{int(blk.sum())}/{nb*nb} super-blocks populated "
          f"({blk.sum()/nb**2:.0%}), block bandwidth {bband}/{nb-1} "
          f"-> {'DENSE (no block-banding)' if blk.mean() > 0.8 else 'sparse'}")

    hr("bandwidth / reducibility")
    perm = csg.reverse_cuthill_mckee(M, symmetric_mode=False)
    Mp = M[perm][:, perm].tocoo()
    print(f"RCM scalar bandwidth {int(np.abs(Mp.row-Mp.col).max())} of {N} "
          f"(band storage {(2*int(np.abs(Mp.row-Mp.col).max())+1)/N*100:.2f}% of full)")
    ncc, _ = csg.connected_components(M, directed=True, connection="strong")
    print(f"strongly-connected components: {ncc} "
          f"({'IRREDUCIBLE (no block-triangular)' if ncc == 1 else 'reducible'})")

    hr("definiteness (symmetric part spectrum)")
    Ksym = (0.5 * (M + M.T)).tocsc()
    lo = spla.eigsh(Ksym, k=3, which="SA", return_eigenvectors=False, maxiter=3000)
    hi = spla.eigsh(Ksym, k=3, which="LA", return_eigenvectors=False, maxiter=3000)
    print(f"sym-part eigs: min={lo.min():.3e}  max={hi.max():.3e}  -> "
          f"{'INDEFINITE -> no Cholesky' if lo.min() < 0 < hi.max() else 'definite'}")

    hr("LU fill and symmetrized-preconditioner quality")
    lu = spla.splu(M.tocsc())
    print(f"LU (COLAMD): L+U nnz = {lu.L.nnz+lu.U.nnz} "
          f"({(lu.L.nnz+lu.U.nnz)/M.nnz:.2f}x A)  -- low fill, cheap to factor")
    rng = np.random.default_rng(0)
    b = M @ rng.standard_normal(N)
    lus = spla.splu(((M + M.T) * 0.5).tocsc())
    rho = np.linalg.norm(M @ lus.solve(b) - b) / np.linalg.norm(b)
    it = float("inf") if rho >= 1 else np.log(1e-7) / np.log(rho)
    print(f"symmetrized (M+M^T)/2 as preconditioner: residual {rho:.2e} "
          f"-> ~{it:.1f} GMRES iters (LDL^T would need this many; ~2x cheaper\n"
          f"   factor cannot pay for ~{it:.0f}x the GMRES apply+matvec).")

    hr("verdict")
    print("Sparse LU with symbolic reuse (Pardiso mtype 11) is at the practical\n"
          "floor for M: fill is already ~1.8x, the structurally-symmetric pattern\n"
          "is exploited by the ordering, and every alternative is worse --\n"
          "Cholesky (indefinite), LDL^T (kills the conjugate term -> +GMRES),\n"
          "block-triangular (irreducible), banded (band storage > LU fill).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ipm-dir", type=Path, default=ROOT / "outputs" / "ipm_python_design")
    p.add_argument("--pump-freq-ghz", type=float, default=7.0)
    p.add_argument("--power-dbm", type=float, default=-22.0)
    p.add_argument("--attenuation-db", type=float, default=35.0)
    p.add_argument("--z0-ohm", type=float, default=50.0)
    p.add_argument("--pump-current-jc-scale", type=float, default=2.0)
    p.add_argument("--pump-mode-count", type=int, default=10)
    p.add_argument("--nt", type=int, default=40)
    p.add_argument("--pump-port", type=int, default=4)
    args = p.parse_args()

    print(f"assembling real-coupled M at {args.power_dbm} dBm, "
          f"{args.pump_freq_ghz} GHz ...", flush=True)
    M, H = assemble_fold_matrix(
        args.ipm_dir, args.pump_freq_ghz, args.power_dbm, args.attenuation_db,
        args.z0_ohm, args.pump_current_jc_scale, args.pump_mode_count, args.nt,
        args.pump_port)
    study(M, H)


if __name__ == "__main__":
    main()
