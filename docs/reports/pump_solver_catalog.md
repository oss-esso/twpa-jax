# Pump-solver catalog — everything tried at the strong-pump fold

Single-page index of every pump-solve backend / preconditioner / accelerator we
have prototyped for the IPM JTWPA harmonic-balance pump solve, with the verdict
for each. The task throughout: cut wall-time of the **near-fold** solves
(−23 → −22 dBm, where the HB Jacobian is near-singular) **without** changing the
physics (residual, analytic AFT JVP incl. the conjugate `k+q` term, source/port
convention, gain definition). Acceptance gate for any backend: exp09 gain drift
**< 0.01 dB** vs the mean_tangent baseline, and honest `VALID_CONVERGED` status.

Reference machine state: IPM design, `positive_odd_jc` basis (10 modes), nt=40,
signal `ws = wp − 100 MHz` per cell, 10 sidebands, 2× JC pump-current scale,
35 dB attenuation. Runtimes are one converged fold solve, warm-started from the
neighbouring power.

**Cross-checked against native JosephsonCircuits.jl** — see
[`runtime_vs_josephsoncircuits.md`](runtime_vs_josephsoncircuits.md): on one IPM
map tile (pump 6.765 GHz, 22 dB cell) `schur_cpu_rcfast` + secant is **~6×
faster end-to-end (~10× on the pump solve) at identical gain (22.07 dB)**.

## Headline numbers (one fold solve, warm-started)

| power | baseline `full_real_coupled` | `schur_cpu_mt` | **`schur_cpu_rcfast`** |
|---|---:|---:|---:|
| −23.0 dBm | 2.84 s | 0.76 s | **0.46 s** |
| −22.5 dBm | 3.39 s | 1.11 s | **0.60 s** |
| −22.0 dBm | 6.61 s | 2.27 s | **1.21 s** |

`schur_cpu_rcfast` is the winner: **5.4× over baseline at the −22 dBm fold**,
sub-1 s through −22.5, gain drift 1e-10, solution bit-identical (3e-15) to
mean_tangent.

## Legend

- ✅ **production** — accepted, in use.
- 🟡 **works, not best** — correct but slower than the winner; kept for reference.
- ❌ **dead end** — prototyped, does not beat the winner or does not converge.

---

## 1. Full-node backends (no reduction)

| # | backend | idea | fold runtime | verdict |
|---|---|---|---:|---|
| 1 | `full_mean_tangent` | block-diagonal (per-harmonic) mean-tangent preconditioner, matrix-free GMRES | slow at fold (GMRES blows up) | ❌ GMRES diverges at the fold; fine far from it |
| 2 | `full_spectral_coupled` | assemble mode-coupled (k−q) complex Jacobian, one LU/Newton | — | 🟡 cuts GMRES but LU/assembly cost cancels the win |
| 3 | `full_real_coupled` | **exact** real-packed Jacobian incl. conjugate (k+q); GMRES≈1/Newton | 6.61 s | 🟡 correct & robust but re-factors the full X-dependent Jacobian every Newton step (≈540 ms/step) → the baseline to beat |
| 4 | `full_adaptive` | adaptive λ-continuation ramp within one solve | — | 🟡 helps cold starts, not the warm near-fold case |

**Lesson:** at the fold you need the *exact coupled* factorization (GMRES=1);
the cheap block-diagonal preconditioner (mean_tangent) can't hold GMRES together.

## 2. Schur reduction over linear-internal nodes

Eliminate the 61 % of nodes (3928/6446) that are linear and internal; retain
JJ-incident + port nodes (2518). `D_k` is block-diagonal in harmonic and
**constant in X**, so factor `D_ee` **once per frequency** and assemble the
sparse Schur complement `S_k = D_nn − D_ne D_ee⁻¹ D_en` (~3 nnz/row) once.

| # | backend | precond on retained system | fold runtime | verdict |
|---|---|---|---:|---|
| 5 | ✅ `schur_cpu_mt` | assembled sparse `S_k` + mean_tangent | 2.27 s | ✅ 2.5–4.5× at fold, gain drift 1e-10; the robust default |
| 6 | 🟡 `schur_cpu_rc` | reduced real_coupled (re-factor each step) | slower than #5 | 🟡 loses to #5 (still re-factors every step) |
| 7 | ✅ **`schur_cpu_rcfast`** | **exact real_coupled + symbolic-factorization reuse** | **1.21 s** | ✅ **winner** — see §3 |
| 8 | ❌ `schur_mf_jfnk` | matrix-free back-sub apply (4.9 ms/matvec), no assembled `S_k` | — | ❌ loses to assembled `S_k`; matvec-only matters only for a future GPU banded solver |
| — | ❌ `D_nn`-only precond | drop the Schur correction term | — | ❌ GMRES never converges — the Schur correction is essential |

## 3. `schur_cpu_rcfast` — the winner (KLU-style reuse) ✅

The exact real-coupled preconditioner made cheap by removing the two costs that
scipy's `splu`+`bmat` re-pay every Newton step — the same costs
JosephsonCircuits.jl avoids with KLU:

- **Assembly reuse.** Real-coupled matrix pattern is constant; `.data` is linear
  in `gamma_hat`, so a precomputed batched Fourier projection plus sparse
  scatter map rebuilds `M.data = M_const + W @ khat_source` in ~19 ms
  (vs `bmat` ~280 ms).
- **Symbolic-factorization reuse.** MKL Pardiso (`pypardiso`) analyses once;
  each Newton step runs only the numeric phase (phase 23) ≈28 ms (vs SuperLU
  ~260 ms). Falls back to SuperLU if `pypardiso` absent (assembly reuse only).

Per Newton step at the fold: ~19 ms assemble refresh + ~28 ms numeric factor +
~13 ms solve = ~60 ms (vs legacy ~540 ms). GMRES = 1/Newton, so the residual cost
is now the numeric factor x Newton-count. **Remaining levers are the Newton count
(a power-axis tangent/arclength predictor) and the remaining `W @ khat_source`
scatter (~16 ms), not the linear solve.**  Use:
`--inproc-pump-backend schur_cpu_mt --inproc-preconditioner
real_coupled_fast` (needs `pypardiso`). Code: `pump_solvers/fast_coupled.py`.

## 4. LU-free / operator-only solvers ❌

Goal: `LU calls = 0`, GPU-friendly matvec-only iterations. Code:
`pump_solvers/lu_free.py`, bench `benchmark_lu_free_pump.py`.

| # | solver | idea | verdict |
|---|---|---|---|
| 9 | `pseudo_transient_mf` | ((1/τ)I + J)ΔX = −R, matrix-free GMRES + Jacobi precond, SER τ schedule | ❌ plateaus ~6–7 orders short of 1e-9, never `VALID_CONVERGED` |
| 10 | `anderson_relaxation` | Anderson-accelerated damped fixed point, complex-diagonal mean-tangent | ❌ same plateau |

**Lesson:** not a fold artifact — they stall from a zero start even at
well-conditioned −30 dBm. The HB pump solve *needs* a strong factorization-based
preconditioner; cheap diagonal/Neumann/Jacobi can't resolve the coupled
line+harmonic modes.

## 5. Preconditioner-reuse across Newton steps ❌

| # | idea | verdict |
|---|---|---|
| 11 | keep one factorization for several Newton steps (`precond_reuse`) | ❌ counterproductive at the fold — Jacobian changes too fast, GMRES blows up (123/Newton at −22). Kept as off-by-default machinery. |

## 6. Banded / block-tridiagonal "line" preconditioner ❌

Retained-node graph is quasi-1D (avg degree 3.0, RCM **bandwidth 5**), so the
real-coupled matrix is block-banded (scalar bw ~100) under node-major ordering.

| # | idea | verdict |
|---|---|---|
| 12 | exact banded `splu` (NATURAL order) | ❌ 112–178 ms — **slower** than Pardiso 28 ms (fill 5.4M, same as COLAMD) |
| 13 | approximate narrow band (±1/±2/±3 node-blocks) | ❌ pump GMRES fails (80/80/73 iters); only the full ±5 band (=exact M) gives GMRES=1 |

**Lesson:** the real-coupled couplings are **irreducible**; you can't cheapen the
band without breaking GMRES.

## 7. GPU (branch `gpu-schur-jvp`, RTX 3060, CUDA 12.8, CuPy) ❌ end-to-end

Native-Windows JAX has no GPU, so device path uses CuPy. Bench
`benchmark_jvp_device.py --device cupy`.

| kernel | CPU | GPU (CuPy) | ratio |
|---|---:|---:|---:|
| retained AFT/JVP matvec (batched block-diag `Sc`) | 2.78 ms | **0.61 ms** | **4.6× faster** |
| mean_tangent LU preconditioner *apply* (psolve) | 0.85 ms | 55.8 ms | **65× slower** |
| mean_tangent LU *factor* | 12 ms | 536 ms | 45× slower |

The **JVP is a real GPU win** (and scales — 9× at nt=512; batching the 10
per-harmonic applies into one block-diagonal spmv was essential). But the
sparse-direct preconditioner — the part that must stay resident to avoid
per-iteration CPU↔GPU transfer — is catastrophic on GPU (sequential triangular
solves), and GPU-friendly matvec-only preconditioners (Jacobi, Neumann-8) don't
converge at the fold. **Naive GPU port loses end-to-end.** GPU only pays at much
larger harmonic content (nt/sidebands) where the matvec dominates.

---

## 8. Newton-count reduction: power-axis secant predictor ✅

Orthogonal to the linear solve: instead of cheapening the factor, **take fewer
Newton steps** by giving each map cell a better initial guess. In the exp10 warm
pass, extrapolate the pump state along the **pump-current axis** from the last
two converged solutions (secant), rather than copying the previous solution:

    X_guess = X_prev + beta (X_prev - X_prevprev),
    beta    = (I - I_prev) / (I_prev - I_prevprev)   (I = injected pump current).

Only the *initial guess* changes — residual, JVP, preconditioner, convention are
untouched. A predicted solve that fails (fold overshoot) falls back once to the
plain warm start before the reseed. Flag: `--inproc-fold-predictor secant`.

Measured on the 7.0 GHz column (Schur + `real_coupled_fast`), warm-started
0.5 dBm steps:

| power | Newton (copy → secant) | pump s (copy → secant) | gain drift |
|---|---:|---:|---:|
| −23.0 dBm | 4 → **3** | 0.50 → **0.34** | 3.9e-8 dB |
| −22.5 dBm | 5 → **4** | 0.66 → **0.45** | 3.6e-9 dB |
| −22.0 dBm (fold) | **8 → 5** | **0.95 → 0.54** | 7.5e-8 dB |

At the −22 dBm fold: **−37 % Newton steps, −43 % runtime**, same
`VALID_CONVERGED`, gain drift ≪ 0.01 dB (≈1e-8). ~32 % faster over the whole warm
column. Stacks on top of `schur_cpu_rcfast`.

**Confirmed map-wide** (full 35×35 map, −30→−20 dBm × 6–8 GHz, corrected signal
`ws = wp − 100 MHz`, `none` vs `secant`):

| metric | copy (none) | secant |
|---|---:|---:|
| points converged | 918 / 1225 | **1039 / 1225 (+121)** |
| gain drift vs none (common) | — | **3.1e-7 dB** |
| Newton total (common PASS) | 5680 | **4803 (−15.4 %)** |
| pump runtime (common PASS) | 638 s | **554 s (−13.2 %)** |

The predictor also **extends the convergence frontier**: secant reaches a higher
pump power in 20/35 frequency columns (mean **+0.96 dBm**, max **+6.8 dBm**),
because the better initial guess lets Newton converge further up the gain ridge.
The 196 `secant_fallback` cells (overshoot → retried from the plain warm start)
introduced zero new failures. Code:
`exp10_full_ipm_pump_map_warmstart.py` (`secant_guess` + warm-pass loop); tests in
`tests/test_exp10_gate.py`. Optional next step if pushing *past* the fold: a
pseudo-arclength predictor/corrector (secant alone can overshoot the turning
point) — not needed while the map tops out at −22.

## Overarching conclusion

Across **every** backend above — mean_tangent, spectral/real coupled, banded,
Jacobi/Neumann, factor-reuse, pseudo-transient, Anderson, GPU — the same truth:

> **Near the fold the strong *exact* factorization is irreducibly required. The
> only winning lever is making that exact factor cheap via
> symbolic-factorization reuse (Pardiso / KLU) = `schur_cpu_rcfast`.**

Stop looking for a cheaper / approximate / LU-free *preconditioner*. The winning
levers are orthogonal to the solve — **do fewer / cheaper Newton steps**:

1. ✅ **Fewer Newton steps at the fold** — the power-axis secant predictor (§8):
   −37 % Newton / −43 % runtime at −22 dBm, gain-identical. Done.
2. **Faster gamma projection inside assembly** -- batched the `gamma_hat_ell`
   Fourier projection, cutting the fold assembly refresh from ~35 ms to ~19 ms.
   Remaining CPU micro-optimization target: the `W @ khat_source` scatter itself
   (~16 ms median).

## Where the code lives

- `experiments/pump_solvers/` — `schur_partition.py`, `schur_operators.py`,
  `fast_coupled.py` (winner), `jvp_backends.py`, `lu_free.py` (neg. result).
- Benchmarks — `benchmark_exp08_pump_schur_matrixfree.py`,
  `benchmark_lu_free_pump.py`, `benchmark_jvp_device.py`.
- Map integration — `exp10_full_ipm_pump_map_warmstart.py`
  (`--inproc-pump-backend schur_cpu_mt --inproc-preconditioner real_coupled_fast`).
- Tests — `tests/test_pump_solvers_schur.py`.
- Detailed write-ups — `pump_solver_backends_baseline.md`,
  `pump_solver_backends_schur_matrixfree.md`.

**Note:** the LU-free (#9,10), factor-reuse (#11), banded (#12,13), and GPU (§7)
experiments live on branch `gpu-schur-jvp`; only the production pieces
(`schur_cpu_mt`, `schur_cpu_rcfast`) are on `main`.
