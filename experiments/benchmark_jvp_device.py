"""Standalone CPU-vs-accelerator microbenchmark of the retained Schur AFT/JVP.

Per the solver plan: before wiring any accelerator into the Newton/GMRES loop,
isolate the retained Schur JVP on a *converged* high-power pump state and check
(1) the accelerator matvec matches the NumPy matvec, and (2) whether it is
actually faster for this problem size. Wiring a device into GMRES only pays off
if the kernel itself wins here and arrays can stay on-device.

The JVP is the GMRES matvec and the dominant per-iteration cost (~2 ms on CPU,
mostly the AFT). Its data flow is accelerator-shaped:

    V_k -> v(t_j) -> Bphi^T v -> Gamma(t_j) (Bphi^T v) -> Bphi(...) -> dn_k
         (+ assembled sparse Schur S_k V_k)

Backends (``--device``): ``numpy`` (reference), ``jax`` (JIT; runs on whatever
JAX device is present -- GPU if available, else CPU). CuPy is added if installed.

Note: this environment has no CUDA device, so ``jax`` here measures JAX-CPU (a
portability/fusion proxy, not a GPU speedup). The harness is written so the same
code path measures a real GPU when one is present.

Example:
    python experiments/benchmark_jvp_device.py --power-dbm -22 --pump-freq-ghz 7.0 --reps 200
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import exp08_full_ipm_pump_solve as exp08
import pump_basis
from pump_solvers.schur_operators import SchurReducedProblem, build_schur_problem

ROOT = Path(__file__).resolve().parents[1]


def dbm_to_peak_current_a(power_dbm: float, *, attenuation_db: float, z0_ohm: float) -> float:
    return math.sqrt(2.0e-3 * 10.0 ** ((power_dbm - attenuation_db) / 10.0) / z0_ohm)


def warm_solve_to(power_dbm: float, freq_ghz: float, args) -> tuple[SchurReducedProblem, np.ndarray]:
    """Warm-start a power sweep up to ``power_dbm`` and return (problem, X_n)."""
    import io
    import contextlib

    ipm = exp08.load_ipm(args.ipm_dir)
    branch = exp08.JosephsonBranchArray(Ic=ipm.Ic, phi0=ipm.phi0)
    omega = 2.0 * math.pi * freq_ghz * 1e9
    basis = pump_basis.resolve_pump_basis(
        policy="positive_odd_jc", omega_p=omega, harmonics=3,
        mode_count=args.pump_mode_count, explicit_modes=None, design_meta=ipm.summary)
    grid = exp08.HarmonicGrid(modes=basis.k, nt=args.nt, omega=omega)
    ports = list(ipm.port_to_index.values())

    def mkfull(cur):
        return exp08.FullIPMPumpProblem(
            C=ipm.C, G=ipm.G, K=ipm.K, Bphi=ipm.Bphi, branch=branch, grid=grid,
            pump_node_index=ipm.port_to_index[args.pump_port], pump_current_a=cur,
            source_mode=basis.source_mode)

    settings = exp08.NewtonKrylovSettings(
        newton_tol=1e-9, max_newton=16, gmres_rtol=1e-7, gmres_atol=0.0,
        gmres_restart=60, gmres_maxiter=80, min_alpha=1.0 / 1024.0,
        preconditioner="mean_tangent", compute_time_residual=False, verbose=False,
        continuation_predictor="none", jvp_mode="aft", stall_ratio=0.8, stall_patience=4)
    part = build_schur_problem(mkfull(1e-6), ports).part

    warm = None
    powers = list(np.arange(power_dbm - 5.0, power_dbm + 0.01, 0.5))
    for p in powers:
        cur = dbm_to_peak_current_a(p, attenuation_db=args.attenuation_db, z0_ohm=args.z0_ohm)
        s = SchurReducedProblem(full=mkfull(cur * args.pump_current_jc_scale), partition=part)
        solver = exp08.HarmonicNewtonKrylovSolver(settings)
        with contextlib.redirect_stdout(io.StringIO()):
            if warm is None:
                X, rep = solver.solve_continuation(s, continuation_steps=20)
            else:
                X, rep = solver.solve_direct(s, warm)
        if rep[-1].converged:
            warm = X
            last = s
    return last, warm


def build_jax_jvp(problem: SchurReducedProblem, tangent):
    """JIT'd JAX implementation of the retained AFT JVP (device-resident)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse

    E = jnp.asarray(problem.grid.E)                     # (nt, H) complex
    Ec = jnp.asarray(problem.grid.E_conj_T_over_nt)     # (H, nt) complex
    gamma_t = jnp.asarray(tangent.gamma_t)              # (nt, nb)
    BphiT_r = jsparse.BCOO.from_scipy_sparse(problem.BphiT_r.tocoo())  # (nb, m)
    Bphi_r = jsparse.BCOO.from_scipy_sparse(problem.Bphi_r.tocoo())    # (m, nb)
    Sc = [jsparse.BCOO.from_scipy_sparse(problem.part.schur[h].tocoo())
          for h in range(problem.H)]
    Sc_stack = jnp.stack([s.todense() for s in Sc]) if problem.n <= 64 else None

    def jvp(V):  # V: (H, m) complex
        v_t = 2.0 * jnp.real(E @ V)                     # (nt, m)
        dpsi = (BphiT_r @ v_t.T).T                      # (nt, nb)
        di = gamma_t * dpsi
        dn = (Bphi_r @ di.T).T                          # (nt, m)
        DN = Ec @ dn                                    # (H, m)
        lin = jnp.stack([Sc[h] @ V[h] for h in range(problem.H)])
        return lin + DN

    return jax.jit(jvp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ipm-dir", type=Path, default=ROOT / "outputs" / "ipm_python_design")
    ap.add_argument("--power-dbm", type=float, default=-22.0)
    ap.add_argument("--pump-freq-ghz", type=float, default=7.0)
    ap.add_argument("--device", nargs="+", default=["numpy", "jax"],
                    choices=["numpy", "jax", "cupy"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--pump-port", type=int, default=4)
    ap.add_argument("--pump-mode-count", type=int, default=10)
    ap.add_argument("--nt", type=int, default=40)
    ap.add_argument("--attenuation-db", type=float, default=35.0)
    ap.add_argument("--z0-ohm", type=float, default=50.0)
    ap.add_argument("--pump-current-jc-scale", type=float, default=2.0)
    args = ap.parse_args()

    print(f"warm-starting to {args.power_dbm} dBm @ {args.pump_freq_ghz} GHz ...", flush=True)
    problem, Xn = warm_solve_to(args.power_dbm, args.pump_freq_ghz, args)
    tangent = problem.tangent_state(Xn)
    rng = np.random.default_rng(0)
    V = rng.standard_normal((problem.H, problem.n)) + 1j * rng.standard_normal((problem.H, problem.n))
    ref = problem.jvp_coeffs_with_tangent(V, tangent)
    print(f"retained m={problem.n}, H={problem.H}, nt={problem.grid.nt}", flush=True)

    def bench(fn, name, transfer=None):
        # warmup + timed reps
        out = fn(V)
        if transfer is not None:
            out = transfer(out)
        t = time.perf_counter()
        for _ in range(args.reps):
            o = fn(V)
        if transfer is not None:
            o = transfer(o)
        dt = (time.perf_counter() - t) / args.reps * 1000
        err = float(np.linalg.norm(np.asarray(o) - ref) / np.linalg.norm(ref))
        print(f"  {name:10} {dt:7.3f} ms/JVP   rel_err={err:.2e}", flush=True)
        return dt

    print("device microbenchmark (retained Schur AFT/JVP):", flush=True)
    if "numpy" in args.device:
        bench(lambda v: problem.jvp_coeffs_with_tangent(v, tangent), "numpy")
    if "jax" in args.device:
        try:
            import jax
            import jax.numpy as jnp
            print(f"  jax device: {jax.devices()} backend={jax.default_backend()}", flush=True)
            jfn = build_jax_jvp(problem, tangent)
            Vj = jnp.asarray(V)
            bench(lambda _v: jax.block_until_ready(jfn(Vj)), "jax",
                  transfer=lambda o: np.asarray(o))
        except Exception as exc:  # pragma: no cover
            print(f"  jax: unavailable ({exc})", flush=True)
    if "cupy" in args.device:
        try:
            import cupy  # noqa: F401
            print("  cupy present (path not implemented in this harness)", flush=True)
        except Exception:
            print("  cupy: not installed", flush=True)


if __name__ == "__main__":
    main()
