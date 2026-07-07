"""Fast exact real-coupled preconditioner: assembly + symbolic-factorization reuse.

The exact ``real_coupled`` preconditioner makes GMRES converge in one iteration
near the fold, but the legacy path re-assembles the (50360x50360) real-packed
Jacobian with ``scipy.sparse.bmat`` (~280 ms) and re-factors it from scratch with
SuperLU (~260 ms) on *every* Newton step -- ~540 ms/step. Both costs are
unnecessary: the sparsity pattern is constant across power, frequency, and Newton
step (it depends only on the harmonic basis and the circuit graph). Only the
numerical values change.

This module exploits that, matching what makes JosephsonCircuits.jl fast (KLU
with symbolic reuse):

* **Assembly reuse.** ``khat_ell = Bphi diag(gamma_hat_ell) Bphi^T`` has a fixed
  pattern, and its ``.data`` is *linear* in ``gamma_hat_ell`` -- so a precomputed
  map turns each step's gamma harmonics straight into matrix values, scattered
  into a fixed ``M.data`` array. No ``bmat``, no pattern rediscovery.
* **Symbolic-factorization reuse.** MKL Pardiso (``pypardiso``) does the symbolic
  analysis once (phase 11/12); each Newton step runs only the numeric phase
  (phase 23), ~12x faster than SuperLU. Falls back to SuperLU if pypardiso is
  not installed (still benefits from assembly reuse).

The conjugate ``K_{k+q}`` coupling is kept exactly (it is what collapses GMRES).
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency
    threadpool_limits = None

try:
    from pypardiso import PyPardisoSolver

    _HAVE_PARDISO = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_PARDISO = False


def pardiso_available() -> bool:
    return _HAVE_PARDISO


class _KhatDataMap:
    """Linear map gamma_hat_ell (nb complex) -> khat_ell.data (fixed pattern)."""

    def __init__(self, Bphi_r: sp.csr_matrix, BphiT_r: sp.csr_matrix) -> None:
        Br = Bphi_r.tocsr()
        # Reference pattern from an all-ones diagonal.
        ones = np.ones(Br.shape[1])
        Kpat = (Br @ sp.diags(ones, 0, format="csr") @ BphiT_r).tocsr()
        Kpat.sort_indices()
        self.pattern = Kpat
        self.nnz = Kpat.nnz
        # khat[i,j] = sum_b Br[i,b] Br[j,b] gamma[b]; build the (nnz x nb) map.
        Bcoo = Br.tocoo()
        node_branches: dict[int, list[tuple[int, float]]] = {}
        for i, b, v in zip(Bcoo.row, Bcoo.col, Bcoo.data):
            node_branches.setdefault(int(i), []).append((int(b), float(v)))
        ent, brs, coeff = [], [], []
        Kc = Kpat.tocoo()
        for i, j in zip(Kc.row, Kc.col):
            di = dict(node_branches.get(int(i), []))
            dj = dict(node_branches.get(int(j), []))
            pos = Kpat.indptr[i] + np.searchsorted(
                Kpat.indices[Kpat.indptr[i]:Kpat.indptr[i + 1]], j)
            for b in set(di) & set(dj):
                ent.append(int(pos)); brs.append(b); coeff.append(di[b] * dj[b])
        self._map = sp.csr_matrix((coeff, (ent, brs)), shape=(Kpat.nnz, Br.shape[1]))

    def data(self, gamma_hat_ell: np.ndarray) -> np.ndarray:
        return self._map @ gamma_hat_ell


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


_LOGGED_FACTOR_BACKENDS: set[str] = set()


def _log_factor_backend_once(backend: str, detail: str = "") -> None:
    if not _env_true("TWPA_PARDISO_LOG"):
        return
    key = backend
    if key in _LOGGED_FACTOR_BACKENDS:
        return
    _LOGGED_FACTOR_BACKENDS.add(key)

    msg = f"[real_coupled_fast] factor_backend={backend}"
    if detail:
        msg += f" {detail}"
    print(msg)


def _pardiso_thread_context():
    """Limit MKL/PARDISO threads for stable sparse factorization."""
    if threadpool_limits is None:
        return nullcontext()
    raw = os.environ.get("TWPA_PARDISO_THREADS", "1").strip()
    try:
        limit = int(raw)
    except ValueError:
        limit = 1
    if limit <= 0:
        return nullcontext()
    return threadpool_limits(limits=limit, user_api="blas")


class FastCoupledPreconditioner:
    """Precompute the real-coupled assembly scatter + symbolic factorization once.

    Reused across every Newton step (and every power point at a fixed frequency,
    since the pattern is power-independent). Call :meth:`refactor` each step with
    the current spectral tangent, then :meth:`solve` per GMRES iteration.
    """

    def __init__(self, problem, *, use_pardiso: bool = True) -> None:
        self.problem = problem
        self.H = problem.H
        self.m = problem.n
        self.modes = [int(round(k)) for k in problem.grid.k]
        self.use_pardiso = bool(use_pardiso and _HAVE_PARDISO)
        self._khat_map = _KhatDataMap(problem.Bphi_r, problem.BphiT_r)
        # Precompute batched exp(-i ell theta) projectors for gamma_hat_ell.
        theta = problem.grid.omega * problem.grid.t
        self._ells = self._needed_ells()
        self._phase_matrix = (
            np.exp(-1j * np.asarray(self._ells)[:, None] * theta[None, :])
            / float(theta.size)
        )
        self._build_scatter()
        self._pardiso = None
        self._lu = None
        self._analyzed = False
        self.last_assembly_runtime_s = 0.0
        self.last_factor_runtime_s = 0.0
        self.last_pardiso_error = ""
        self.last_factor_backend = ""
        self.pardiso_strict = _env_true("TWPA_REQUIRE_PARDISO")

        if self.pardiso_strict and not self.use_pardiso:
            raise RuntimeError(
                "TWPA_REQUIRE_PARDISO=1, but pypardiso is not available "
                "or this preconditioner was constructed with use_pardiso=False."
            )

    def _gamma_hat_array(self, gamma_t: np.ndarray) -> np.ndarray:
        return self._phase_matrix @ gamma_t

    def _gamma_hat(self, gamma_t: np.ndarray) -> dict[int, np.ndarray]:
        gh = self._gamma_hat_array(gamma_t)
        return {ell: gh[i] for i, ell in enumerate(self._ells)}

    # ------------------------------------------------------------------ build
    def _build_scatter(self) -> None:
        """Assemble M once to fix its CSR pattern, and precompute, per harmonic
        super-block, the M.data target indices for each source contribution."""
        H, m, modes = self.H, self.m, self.modes
        kpat = self._khat_map.pattern
        kr, kc = kpat.tocoo().row, kpat.tocoo().col
        Sc = [self.problem.part.schur[h].tocsr() for h in range(H)]
        for s in Sc:
            s.sort_indices()

        zero = sp.csr_matrix((m, m), dtype=np.complex128)
        # Reference M: only the union pattern matters. Use DISTINCT generic
        # complex values per ell so that Pi - Li (and Lr - Pr) do not cancel to a
        # structural zero, which would drop those blocks from the pattern.
        rng = np.random.default_rng(12345)
        nb = self._khat_map._map.shape[1]
        khat_ref = {ell: self._khat_data_to_csr(
            self._khat_map.data(rng.standard_normal(nb) + 1j * rng.standard_normal(nb)))
            for ell in self._needed_ells()}

        jrr, jri, jir, jii = [], [], [], []
        for ki, k in enumerate(modes):
            rrr, rri, rir, rii = [], [], [], []
            for qi, q in enumerate(modes):
                L = khat_ref.get(k - q, zero)
                if ki == qi:
                    L = L + Sc[ki]
                P = khat_ref.get(k + q, zero)
                Lr, Li, Pr, Pi = L.real, L.imag, P.real, P.imag
                rrr.append((Lr + Pr).tocsr()); rri.append((Pi - Li).tocsr())
                rir.append((Li + Pi).tocsr()); rii.append((Lr - Pr).tocsr())
            jrr.append(rrr); jri.append(rri); jir.append(rir); jii.append(rii)
        top = sp.bmat([[sp.bmat(jrr), sp.bmat(jri)]])
        bot = sp.bmat([[sp.bmat(jir), sp.bmat(jii)]])
        M = sp.bmat([[top], [bot]], format="csr")
        M.sum_duplicates(); M.sort_indices()
        self.M = M
        self.M.data = M.data.astype(np.float64)
        # Global sorted keys row*N + col for vectorized M.data index lookup.
        self._N = M.shape[1]
        row_of = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
        self._M_keys = row_of.astype(np.int64) * self._N + M.indices.astype(np.int64)

        # For each (super-block a,b in 0..2H-1) the per-entry M.data index of a
        # source matrix with the khat pattern or the Sc_ki pattern.
        self._contribs = self._index_contributions(kr, kc, Sc)

    def _needed_ells(self) -> list[int]:
        modes = self.modes
        return sorted({k - q for k in modes for q in modes}
                      | {k + q for k in modes for q in modes})

    def _khat_data_to_csr(self, data: np.ndarray) -> sp.csr_matrix:
        kp = self._khat_map.pattern
        return sp.csr_matrix((data, kp.indices, kp.indptr), shape=kp.shape)

    def _block_target(self, M: sp.csr_matrix, a: int, b: int,
                      rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """M.data indices for entries (a*m + rows, b*m + cols), vectorized."""
        m = self.m
        keys = (a * m + rows).astype(np.int64) * self._N + (b * m + cols).astype(np.int64)
        return np.searchsorted(self._M_keys, keys)

    def _index_contributions(self, kr, kc, Sc):
        """Build M.data = M_const + W @ khat_source as one precomputed sparse map.

        khat_source concatenates [Re(khat_ell), Im(khat_ell)] for each ell; W
        scatters them (with +-1 coeffs) into M.data. The constant Sc blocks fold
        into M_const. One sparse matvec per Newton step instead of ~24M scatter
        adds."""
        H, modes, M = self.H, self.modes, self.M
        kr = np.asarray(kr); kc = np.asarray(kc); nnzk = kr.size
        self._nnzk = nnzk
        self._ell_index = {ell: i for i, ell in enumerate(self._ells)}
        nsrc = 2 * len(self._ells) * nnzk

        def src_seg(ell, part):  # part: 0=real, 1=imag
            base = (2 * self._ell_index[ell] + part) * nnzk
            return base + np.arange(nnzk)

        W_rows, W_cols, W_val = [], [], []

        def add(tgt, src, coeff):
            W_rows.append(tgt); W_cols.append(src); W_val.append(coeff)

        for ki, k in enumerate(modes):
            for qi, q in enumerate(modes):
                ed, es = k - q, k + q
                t_rr = self._block_target(M, ki, qi, kr, kc)
                t_ri = self._block_target(M, ki, qi + H, kr, kc)
                t_ir = self._block_target(M, ki + H, qi, kr, kc)
                t_ii = self._block_target(M, ki + H, qi + H, kr, kc)
                o = np.ones(nnzk)
                # L = khat_ed: Lr->(+rr,+ii), Li->(-ri,+ir)
                add(t_rr, src_seg(ed, 0), o); add(t_ii, src_seg(ed, 0), o)
                add(t_ri, src_seg(ed, 1), -o); add(t_ir, src_seg(ed, 1), o)
                # P = khat_es: Pr->(+rr,-ii), Pi->(+ri,+ir)
                add(t_rr, src_seg(es, 0), o); add(t_ii, src_seg(es, 0), -o)
                add(t_ri, src_seg(es, 1), o); add(t_ir, src_seg(es, 1), o)
        W = sp.csr_matrix(
            (np.concatenate(W_val),
             (np.concatenate(W_rows), np.concatenate(W_cols))),
            shape=(M.nnz, nsrc))
        self._W = W

        # Constant Sc contribution into M_const.
        Mconst = np.zeros(M.nnz)
        for ki in range(H):
            S = Sc[ki].tocoo()
            sr, sc = S.row, S.col
            t_rr = self._block_target(M, ki, ki, sr, sc)
            t_ri = self._block_target(M, ki, ki + H, sr, sc)
            t_ir = self._block_target(M, ki + H, ki, sr, sc)
            t_ii = self._block_target(M, ki + H, ki + H, sr, sc)
            np.add.at(Mconst, t_rr, S.data.real); np.add.at(Mconst, t_ii, S.data.real)
            np.add.at(Mconst, t_ri, -S.data.imag); np.add.at(Mconst, t_ir, S.data.imag)
        self._Mconst = Mconst
        return None

    # -------------------------------------------------------------- refactor
    def refactor(self, tangent) -> None:
        """Rebuild M.data from the current tangent gamma(t) and factor it.

        khat_ell.data is recomputed from gamma_hat_ell via the precomputed linear
        map -- guaranteeing the same pattern/order the scatter indices assume.
        """
        t0 = time.perf_counter()
        gh = self._gamma_hat_array(tangent.gamma_t)
        nnzk = self._nnzk
        src = np.empty(2 * len(self._ells) * nnzk)
        for i in range(len(self._ells)):
            d = self._khat_map.data(gh[i])
            src[2 * i * nnzk:(2 * i + 1) * nnzk] = d.real
            src[(2 * i + 1) * nnzk:(2 * i + 2) * nnzk] = d.imag
        data = self._W @ src
        data += self._Mconst
        self.M.data = data
        self.last_assembly_runtime_s = time.perf_counter() - t0
        self._factor()

    def _factor(self) -> None:
        t0 = time.perf_counter()

        if self.use_pardiso:
            A = self.M.tocsr()
            try:
                if self._pardiso is None:
                    # Real-packed Jacobian/preconditioner: real but generally
                    # nonsymmetric. Use MKL PARDISO's real-unsymmetric type.
                    self._pardiso = PyPardisoSolver(mtype=11)
                    self._pardiso.set_statistical_info_off()
                    with _pardiso_thread_context():
                        self._pardiso.factorize(A)  # analysis + numeric
                    self._analyzed = True
                else:
                    self._pardiso._check_A(A)
                    self._pardiso.set_phase(22)  # numeric only, reuse analysis
                    with _pardiso_thread_context():
                        self._pardiso._call_pardiso(A, np.zeros(A.shape[0]))

                self._lu = None
                self.last_pardiso_error = ""
                self.last_factor_backend = "pardiso"
                _log_factor_backend_once("pardiso")

            except Exception as exc:
                self.last_pardiso_error = repr(exc)
                self.last_factor_backend = "pardiso_failed"

                if self.pardiso_strict:
                    raise RuntimeError(
                        "PARDISO factorization failed while "
                        "TWPA_REQUIRE_PARDISO=1."
                    ) from exc

                self.use_pardiso = False
                self._pardiso = None
                self._analyzed = False
                self._lu = spla.splu(self.M.tocsc())
                self.last_factor_backend = "superlu_fallback"
                _log_factor_backend_once("superlu_fallback", f"error={self.last_pardiso_error}")
        else:
            self._lu = spla.splu(self.M.tocsc())
            self.last_factor_backend = "superlu"
            _log_factor_backend_once("superlu")
        self.last_factor_runtime_s = time.perf_counter() - t0

    def solve(self, b_real: np.ndarray) -> np.ndarray:
        if self.use_pardiso and self._pardiso is not None:
            try:
                self._pardiso.set_phase(33)
                with _pardiso_thread_context():
                    return self._pardiso._call_pardiso(self.M.tocsr(), b_real)
            except Exception as exc:
                self.last_pardiso_error = repr(exc)
                self.last_factor_backend = "pardiso_solve_failed"

                if self.pardiso_strict:
                    raise RuntimeError(
                        "PARDISO solve failed while TWPA_REQUIRE_PARDISO=1."
                    ) from exc

                self.use_pardiso = False
                self._pardiso = None
                self._analyzed = False
                self._lu = spla.splu(self.M.tocsc())
                self.last_factor_backend = "superlu_fallback"

        if self._lu is None:
            self._lu = spla.splu(self.M.tocsc())
            self.last_factor_backend = "superlu"
        return self._lu.solve(b_real)
