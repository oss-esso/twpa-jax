"""Benchmark GPU sparse LU vs CPU Pardiso/SuperLU on the real-coupled matrix M.

Answers "is LU decomposition faster on the GPU?" for the fold preconditioner.
Warm-starts to the fold, assembles the real-packed real-coupled M, and times
factor + solve on:

  CPU SuperLU (scipy)                    -- reference
  CPU Pardiso phase-23 (numeric + solve) -- the production per-Newton-step cost
  GPU cupyx.scipy.sparse.linalg.splu     -- NOTE: backend is CPU SuperLU + GPU
                                            triangular solve (not a GPU factor!)
  GPU cuSOLVER csrlsvqr (spsolve)        -- a real device factorization (no reuse)

Result (RTX 3060 Laptop, CuPy 14): GPU loses 40-44x. No true GPU sparse-LU with
symbolic reuse is available, the sparse triangular solve is sequential (slow),
and just the host->device transfer of M exceeds the entire CPU solve.

Run (needs a CUDA GPU + cupy):
    python experiments/benchmark_gpu_lu.py --pump-freq-ghz 7.0 --power-dbm -22
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from study_realcoupled_matrix import assemble_fold_matrix

ROOT = Path(__file__).resolve().parents[1]


def timeit(fn, n: int = 5) -> float:
    fn()  # warmup
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


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

    print("assembling real-coupled M at the fold ...", flush=True)
    M, _ = assemble_fold_matrix(
        args.ipm_dir, args.pump_freq_ghz, args.power_dbm, args.attenuation_db,
        args.z0_ohm, args.pump_current_jc_scale, args.pump_mode_count, args.nt,
        args.pump_port)
    N = M.shape[0]
    print(f"M: {N}x{N}  nnz={M.nnz} ({M.nnz/N:.1f}/row)", flush=True)
    rng = np.random.default_rng(0)
    b = M @ rng.standard_normal(N)

    def resid(x) -> float:
        return float(np.linalg.norm(M @ x - b) / np.linalg.norm(b))

    print("\n=== CPU SuperLU (scipy) ===", flush=True)
    Mcsc = M.tocsc()
    lu = spla.splu(Mcsc)
    f = timeit(lambda: spla.splu(Mcsc), n=3)
    s = timeit(lambda: lu.solve(b), n=10)
    print(f"factor {f*1e3:.1f} ms | solve {s*1e3:.1f} ms | resid {resid(lu.solve(b)):.1e}")

    print("\n=== CPU Pardiso phase-23 (numeric refactor + solve) ===", flush=True)
    try:
        from pypardiso import PyPardisoSolver
        ps = PyPardisoSolver()
        ps.set_statistical_info_off()
        ps.factorize(M)

        def numeric():
            ps._check_A(M); ps.set_phase(22); ps._call_pardiso(M, np.zeros(N))

        def solve():
            ps.set_phase(33); return ps._call_pardiso(M, b)

        print(f"numeric {timeit(numeric, 5)*1e3:.1f} ms | "
              f"solve {timeit(solve, 10)*1e3:.1f} ms | resid {resid(solve()):.1e}")
    except Exception as e:  # noqa: BLE001
        print("pardiso unavailable:", type(e).__name__, e)

    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as csl
    except Exception as e:  # noqa: BLE001
        print("\n[no cupy/GPU]:", type(e).__name__, e)
        return

    print("\n=== GPU cupyx.splu (CPU SuperLU factor + GPU trisolve) ===", flush=True)
    t = time.perf_counter()
    Mg = csp.csr_matrix(M); bg = cp.asarray(b); cp.cuda.Stream.null.synchronize()
    print(f"host->device transfer(M+b): {(time.perf_counter()-t)*1e3:.1f} ms")

    def gfactor():
        lug = csl.splu(Mg.tocsc()); cp.cuda.Stream.null.synchronize(); return lug

    lug = gfactor()

    def gsolve():
        x = lug.solve(bg); cp.cuda.Stream.null.synchronize(); return x

    rg = float(cp.linalg.norm(Mg @ lug.solve(bg) - bg) / cp.linalg.norm(bg))
    print(f"factor {timeit(gfactor,3)*1e3:.1f} ms | solve {timeit(gsolve,10)*1e3:.1f} ms"
          f" | resid {rg:.1e} | backend={type(lug).__name__}")

    print("\n=== GPU cuSOLVER csrlsvqr (spsolve, factor+solve fused) ===", flush=True)
    try:
        def gspsolve():
            x = csl.spsolve(Mg, bg); cp.cuda.Stream.null.synchronize(); return x

        rg = float(cp.linalg.norm(Mg @ gspsolve() - bg) / cp.linalg.norm(bg))
        print(f"factor+solve {timeit(gspsolve,5)*1e3:.1f} ms | resid {rg:.1e}")
    except Exception as e:  # noqa: BLE001
        print("spsolve failed:", type(e).__name__, e)


if __name__ == "__main__":
    main()
