# Pump solver — final verdict, bottlenecks, and matrix decomposition study

Authoritative record of where the IPM strong-pump (near-fold) solve stands after
exhausting the backend/preconditioner/decomposition search. Companion to the
full catalogue in [`pump_solver_catalog.md`](pump_solver_catalog.md).

Reference point: IPM design, `positive_odd_jc` (10 modes), nt=40, 7.0 GHz pump,
2× JC current, 35 dB attenuation, warm-started 0.5 dBm steps. "The fold" = the
−22 dBm harmonic-balance turning point where the Jacobian is near-singular.

---

## 1. Best solver — `schur_cpu_rcfast`

**Schur reduction over linear-internal nodes + the exact real-coupled
preconditioner factored with Pardiso (KLU-style symbolic reuse), plus the
power-axis secant predictor for the Newton count.**

`--inproc-pump-backend schur_cpu_mt --inproc-preconditioner real_coupled_fast
--inproc-fold-predictor secant` (needs `pypardiso`).

GMRES converges in **1 iteration/Newton** (the exact coupled factor incl. the
conjugate `k+q` term). At the −22 dBm fold: ~0.54 s, gain-identical to the
baseline (drift ~1e-8 dB), 5.4× over the original `full_real_coupled`.

### Per-Newton-step cost breakdown (uncontended, at the fold)

| stage | cost | GPU-friendly? | notes |
|---|---:|---|---|
| **assemble** `M.data = M_const + W @ src` | **~33 ms** | ✅ (plain spmv) | **largest single term** |
| **numeric LU factor** (Pardiso phase 22) | ~18–20 ms | ❌ (see §3) | at the decomposition floor (§3) |
| **triangular solve** (phase 33) | ~9 ms | ❌ (sequential) | |
| JVP matvec (GMRES=1) | ~2.8 ms | ✅ (4.6× on GPU) | only one, since GMRES=1 |

Outer loop: **Newton count** — 8 at the raw fold, **5 with the secant predictor**.

### Bottlenecks, ranked

1. **The assembly spmv (~33 ms) — not the LU — is the #1 per-step cost.** It is a
   single large sparse mat-vec (`W` has `M.nnz ≈ 3.0M` rows). This is the next
   real CPU optimization target (leaner scatter / smaller `W`), and unlike the
   factor it *is* GPU-friendly.
2. **The numeric LU factor (~20 ms).** Provably near-optimal for this matrix — no
   other decomposition is cheaper (§3). Do not spend effort here.
3. **The Newton count at the fold.** Addressed by the secant predictor (8→5).
4. **The triangular solve (~9 ms).** Sequential; the reason GPU can't help (§2).

---

## 2. Second-best solver — the GPU on-device pipeline (highest ceiling, blocked)

The one approach whose *ceiling* is above `schur_cpu_rcfast` — **if** its single
bottleneck were removed — is a **fully GPU-resident Newton loop**. Measured wins
and the blocker (branch `gpu-schur-jvp`, RTX 3060 Laptop, CuPy 14):

- The **batched Schur JVP matvec is 4.6× faster on GPU** (2.78 → 0.61 ms; 9× at
  nt=512). The assembly spmv (§1, the #1 CPU cost) is likewise a GPU-friendly
  mat-vec. So the residual/JVP/assembly half of the step wants to be on the GPU.
- **Blocking bottleneck: the preconditioner factorization + solve cannot run
  efficiently on this GPU.** There is *no* true GPU sparse-LU with symbolic reuse
  in the stack (`cupyx…splu`'s backend is literally CPU SuperLU; cuSOLVER
  `csrlsvqr` re-factors every call), the sparse triangular solve is sequential
  (833 ms), and just the **host→device transfer of `M` (~109 ms) already exceeds
  the entire CPU factor+solve (~27 ms)**. Measured LU per step:

  | backend | factor | solve | total | vs Pardiso |
  |---|---:|---:|---:|---:|
  | CPU Pardiso phase-23 | 17.8 ms | 9.2 ms | **~27 ms** | 1× |
  | CPU SuperLU | 143 ms | 6.2 ms | ~149 ms | 5.5× |
  | GPU `cupyx.splu` (SuperLU + GPU trisolve) | 221 ms | 833 ms | ~1054 ms | 39× |
  | GPU cuSOLVER `csrlsvqr` (`spsolve`) | — | — | 1184 ms | 44× |

**What "solving its bottleneck" would require:** a genuine GPU sparse-direct
factor with symbolic reuse (**cuDSS**, not installed here) **and** the entire
Newton loop resident on-device (assemble `M`, JVP, factor, solve all on GPU with
zero per-step transfer). Only then does the transfer cost vanish and the 4.6×
JVP + GPU assembly pay off. On this laptop GPU it is still uncertain — the matrix
is small and cheap to factor (§3), so there is little factor work to accelerate,
and 6 GB caps headroom. The payoff grows with harmonic content (larger
nt/sidebands), where the matvec/assembly dominate the (fixed-fill) factor.
Benchmark: `experiments/benchmark_gpu_lu.py`.

---

## 3. Decomposition study of the LU matrix `M`

Both the best and second-best solvers factor the same matrix: the real-packed
real-coupled Jacobian `M` (2H × 2H super-blocks of the retained node count,
here **50360 × 50360, 3.0M nnz, 59.8/row, 0.12% dense**). Is any decomposition
other than general sparse LU cheaper? Reproduce:
`python experiments/study_realcoupled_matrix.py`.

| property | measurement | consequence for decomposition |
|---|---|---|
| structural symmetry | **pattern 100% symmetric** | symmetric ordering (AMD/METIS) applies; Pardiso already uses it |
| numerical symmetry | **‖M−Mᵀ‖/‖M‖ = 7.2e-3** (nonsymmetric) | can't use a symmetric factor *exactly* |
| **source of asymmetry** | **entirely in the k==q blocks; harmonic coupling is exactly symmetric** | it's the conjugate `k+q` real-embedding (ri=`Pi−Li` vs ir=`Pi+Li`) — **intrinsic, not a numerical artifact** |
| symmetric-part spectrum | **indefinite** (eigenvalues both signs; small ones ≈ fold near-singularity) | **no Cholesky** |
| block structure (2H×2H) | **400/400 super-blocks populated, block-bw 19** | **dense in harmonic space → no block-tridiagonal** |
| reducibility | **1 strongly-connected component** | **irreducible → no block-triangular (Dulmage–Mendelsohn) reduction** |
| RCM scalar bandwidth | **135 / 50360** | narrow, but band storage (271/row) **> LU fill** below |
| **LU fill** | **L+U = 1.80× A** (5.4M vs 3.0M) | already minimal — the matrix is *cheap* to factor |

### Would a symmetric LDLᵀ win? No.

The pattern is symmetric and the asymmetry is only 0.7%, so the tempting move is
to factor the symmetrized `(M+Mᵀ)/2` with symmetric-indefinite **LDLᵀ** (~2×
cheaper factor). Measured preconditioner quality of the symmetrized matrix on the
*true* `M`: **residual 9.1e-3 after one apply → ~3.4 GMRES iterations** (vs 1 for
the exact factor). The ~10 ms factor saving is dwarfed by ~3 extra GMRES iters ×
(~9 ms solve + ~2.8 ms matvec) ≈ 36 ms → **net loss**. And the asymmetry it
discards *is* the conjugate coupling that collapses GMRES to one iteration —
removing it is exactly the forbidden physics change.

### Verdict

> **Sparse LU with symbolic reuse (Pardiso mtype 11) is at the practical floor for
> `M`.** Fill is already ~1.8×, the structurally-symmetric pattern is exploited by
> the ordering, and every alternative decomposition is provably worse:
> Cholesky (indefinite), LDLᵀ (kills the conjugate term → +GMRES), block-triangular
> (irreducible), banded (band storage > LU fill). The factor is *not* where the
> remaining time is — the **assembly spmv (§1)** is.

---

## Bottom line

- **Use `schur_cpu_rcfast` + secant predictor.** It is optimal on CPU: GMRES=1,
  LU at the decomposition floor, Newton count minimized.
- **Next CPU lever is the assembly spmv (~33 ms), not the LU.**
- **GPU only pays with a full on-device loop + cuDSS**, and only at larger
  harmonic content than the current IPM case. Do not port a naive GPU GMRES or
  retry `cupyx.splu`/`spsolve` as the preconditioner.

Scripts: `experiments/study_realcoupled_matrix.py` (decomposition study),
`experiments/benchmark_gpu_lu.py` (GPU vs CPU LU),
`experiments/exp10_full_ipm_pump_map_warmstart.py` (`--inproc-fold-predictor`).
