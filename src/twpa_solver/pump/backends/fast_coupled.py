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
from scipy.linalg.lapack import get_lapack_funcs

# A band this wide is not worth storing densely; refuse rather than allocate.
_BAND_BYTES_LIMIT = 4 * 1024**3
# Nonzeros per slice when building the band index, chosen so the working set
# stays a few MB regardless of matrix size.
_BAND_BUILD_CHUNK = 1 << 20

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
    """Limit MKL/PARDISO threads for stable sparse factorization.

    On the mixed OpenBLAS+MKL Windows environment used for production runs,
    MKL PARDISO intermittently fails the same Schur-reduced matrix during
    reordering (error -3) when it runs with its default multi-thread count. A
    single-threaded PARDISO call is deterministic here; expose an env override
    for explicit benchmarking on other MKL builds.
    """
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

    def __init__(
        self, problem, *, use_pardiso: bool = True, use_banded: bool | None = None
    ) -> None:
        self.problem = problem
        self.H = problem.H
        self.m = problem.n
        self.modes = list(problem.mode_keys)
        if use_banded is None:
            use_banded = _env_true("TWPA_BANDED_PRECOND")
        self.use_banded = bool(use_banded)
        self.use_pardiso = bool(use_pardiso and _HAVE_PARDISO and not self.use_banded)
        self._band_flat = None
        self._band_lu = None
        self._band_ipiv = None
        self._khat_map = _KhatDataMap(problem.Bphi_r, problem.BphiT_r)
        # Precompute batched exp(-i ell theta) projectors for gamma_hat_ell.
        self._ells = self._needed_ells()
        self._phase_matrix = problem.grid.phase_rows(self._ells)
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
        """Precompute, per (mode, mode) block, the M.data slots each khat entry
        feeds. The constant Sc blocks fold into M_const.

        The scatter used to be materialised as a sparse matrix ``W`` of shape
        ``(M.nnz, 2 * n_ells * nnzk)`` so assembly was one spmv. That matrix
        carried no information beyond these target indices: every stored value
        is +-1, and the source indices are contiguous per-ell ranges. At a
        31-tone multitone basis it cost ~630 MB -- the single largest term in a
        worker's footprint after the factors -- to hold 94 MB of indices. Here
        only the four target-index arrays per block are kept and assembly does
        four fancy-indexed adds per block instead. Within a block the khat
        pattern has unique (row, col) pairs and distinct blocks land in
        disjoint super-blocks of M, so no target index repeats and ``+=`` is
        exact rather than an unordered scatter.
        """
        H, modes, M = self.H, self.modes, self.M
        kr = np.asarray(kr); kc = np.asarray(kc); nnzk = kr.size
        self._nnzk = nnzk
        self._ell_index = {ell: i for i, ell in enumerate(self._ells)}
        self._nsrc = 2 * len(self._ells) * nnzk

        if M.nnz > np.iinfo(np.int32).max:
            raise ValueError(f"scatter map indices exceed int32: M.nnz={M.nnz}")

        n_modes = len(modes)
        # (block, quadrant, khat entry) -> index into M.data. Quadrant order is
        # (rr, ri, ir, ii), matching the real-packing of the reference assembly.
        targets = np.empty((n_modes, n_modes, 4, nnzk), dtype=np.int32)
        # Per block, which ell feeds the difference (k-q) and sum (k+q) terms.
        ell_diff = np.empty((n_modes, n_modes), dtype=np.int32)
        ell_sum = np.empty((n_modes, n_modes), dtype=np.int32)

        for ki, k in enumerate(modes):
            for qi, q in enumerate(modes):
                targets[ki, qi, 0] = self._block_target(M, ki, qi, kr, kc)
                targets[ki, qi, 1] = self._block_target(M, ki, qi + H, kr, kc)
                targets[ki, qi, 2] = self._block_target(M, ki + H, qi, kr, kc)
                targets[ki, qi, 3] = self._block_target(M, ki + H, qi + H, kr, kc)
                ell_diff[ki, qi] = self._ell_index[k - q]
                ell_sum[ki, qi] = self._ell_index[k + q]
        self._targets = targets
        self._ell_diff = ell_diff
        self._ell_sum = ell_sum
        self._value_buffer = np.empty(4 * nnzk, dtype=np.float64)

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
        n_ells = len(self._ells)
        khat = np.empty((n_ells, self._nnzk), dtype=np.complex128)
        for i in range(n_ells):
            khat[i] = self._khat_map.data(gh[i])

        data = self._Mconst.copy()
        targets = self._targets
        real, imag = khat.real, khat.imag
        nnzk = self._nnzk
        # One fancy-indexed add per block rather than one per quadrant: the four
        # quadrant target arrays are already contiguous in `targets`, so the
        # values can be staged into a single buffer and scattered together.
        buf = self._value_buffer
        q0, q1, q2, q3 = (slice(i * nnzk, (i + 1) * nnzk) for i in range(4))
        for ki in range(len(self.modes)):
            for qi in range(len(self.modes)):
                ed = self._ell_diff[ki, qi]
                es = self._ell_sum[ki, qi]
                # L = khat[k-q], P = khat[k+q]; the real packing sends
                # rr <- Lr + Pr, ri <- Pi - Li, ir <- Li + Pi, ii <- Lr - Pr.
                lr, li = real[ed], imag[ed]
                pr, pi = real[es], imag[es]
                np.add(lr, pr, out=buf[q0])
                np.subtract(pi, li, out=buf[q1])
                np.add(li, pi, out=buf[q2])
                np.subtract(lr, pr, out=buf[q3])
                data[targets[ki, qi].ravel()] += buf
        self.M.data = data
        self.last_assembly_runtime_s = time.perf_counter() - t0
        self._factor()

    # ------------------------------------------------------------------ band
    def _build_band(self) -> None:
        """Map M's nonzeros into LAPACK general-band storage.

        Packed tone-major, M is dense in tone index and spans the whole matrix.
        Reordered node-major -- index ``node * (2 * n_tones) + tone`` -- it
        collapses onto a band roughly three tone-blocks wide, because the device
        is a 1-D chain and only neighbouring nodes couple. That is a property of
        the circuit, not of the ordering search: reverse Cuthill-McKee finds
        nothing better (max bandwidth 147 against 89 on jtwpa S=2).

        Storing the factors in band form rather than as a general sparse LU
        removes the fill-in bookkeeping and the pivot search over the whole
        matrix, at the cost of holding the zeros inside the band. On jtwpa S=10
        that trades ~1.9 GB of sparse factors for ~0.7 GB of band.

        The bandwidth is measured from the assembled pattern rather than assumed
        to be ``3 * 2 * n_tones``: a circuit whose node numbering is not
        monotone along the chain would silently widen it, and a too-small band
        would drop entries and quietly degrade the preconditioner.
        """
        dim = self.M.shape[0]
        block = 2 * self.H
        # Packed index is ``super_block * m + node``; node-major is
        # ``node * (2 * n_tones) + super_block``. permutation[packed] = node_major.
        super_block, node = np.divmod(np.arange(dim), self.m)
        permutation = (node.astype(np.int64) * block + super_block).astype(np.int32)

        # Walk the CSR structure in chunks rather than materialising a COO view.
        # Permuted row and column copies plus the offset arithmetic would hold
        # several int64 arrays of M.nnz simultaneously, and on jtwpa S=10 that
        # transient is larger than the band it is building -- it set the peak
        # RSS, defeating the point of the backend.
        indptr, indices = self.M.indptr, self.M.indices
        row_of = np.repeat(np.arange(dim, dtype=np.int32), np.diff(indptr))

        kl = 0
        for start in range(0, self.M.nnz, _BAND_BUILD_CHUNK):
            stop = min(start + _BAND_BUILD_CHUNK, self.M.nnz)
            spread = (
                permutation[row_of[start:stop]].astype(np.int64)
                - permutation[indices[start:stop]]
            )
            kl = max(kl, int(np.abs(spread).max()))
        ku = kl
        leading = 2 * kl + ku + 1
        if leading * dim * 8 > _BAND_BYTES_LIMIT:
            raise ValueError(
                f"banded storage would need {leading * dim * 8 / 1024**3:.1f} GB "
                f"(bandwidth {kl} over dimension {dim}); this circuit is not "
                "banded enough for the banded backend"
            )

        # A[i, j] lives at ab[kl + ku + i - j, j]; ab is Fortran-ordered for
        # LAPACK, so the flat offset is column-major. M.data is in CSR order and
        # so is row_of, hence the mapping applies straight to the assembled data.
        if leading * dim > np.iinfo(np.int32).max:
            raise ValueError(
                f"band offsets exceed int32: leading={leading} dim={dim}"
            )
        flat = np.empty(self.M.nnz, dtype=np.int32)
        for start in range(0, self.M.nnz, _BAND_BUILD_CHUNK):
            stop = min(start + _BAND_BUILD_CHUNK, self.M.nnz)
            new_rows = permutation[row_of[start:stop]]
            new_cols = permutation[indices[start:stop]]
            flat[start:stop] = new_cols * leading + (kl + ku + new_rows - new_cols)

        # The band is the dominant allocation, so exactly one copy is kept and
        # reused: zeroed and refilled per step, then factored in place. Holding
        # a pristine template plus a working copy plus LAPACK's output would
        # cost three times the band and lose to the sparse backend outright.
        self._band_flat = flat
        self._band_permutation = permutation
        self._band_kl = kl
        self._band_ku = ku
        self._band_leading = leading
        self._band_dim = dim
        self._band_buffer = np.zeros(leading * dim, dtype=np.float64)
        self._gbtrf, self._gbtrs = get_lapack_funcs(("gbtrf", "gbtrs"),
                                                    (self._band_buffer,))

    def _factor_banded(self) -> None:
        if self._band_flat is None:
            self._build_band()
        buffer = self._band_buffer
        buffer.fill(0.0)
        buffer[self._band_flat] = self.M.data
        ab = buffer.reshape((self._band_leading, self._band_dim), order="F")
        lu, ipiv, info = self._gbtrf(
            ab, self._band_kl, self._band_ku, overwrite_ab=True
        )
        if info != 0:
            raise RuntimeError(
                f"banded factorization failed: dgbtrf info={info} "
                f"(info>0 means a zero pivot at that column)"
            )
        self._band_lu = lu
        self._band_ipiv = ipiv
        self.last_factor_backend = "banded"
        _log_factor_backend_once("banded", f"kl=ku={self._band_kl}")

    def _solve_banded(self, b_real: np.ndarray) -> np.ndarray:
        # permutation[packed] = node_major, so moving *into* node-major order is
        # a scatter and coming back out is a gather. Reversing the two silently
        # solves a differently-permuted system and still returns a finite vector.
        permutation = self._band_permutation
        permuted = np.empty_like(b_real)
        permuted[permutation] = b_real
        x, info = self._gbtrs(
            self._band_lu, self._band_kl, self._band_ku, permuted, self._band_ipiv
        )
        if info != 0:
            raise RuntimeError(f"banded solve failed: dgbtrs info={info}")
        return x[permutation]

    def _factor(self) -> None:
        t0 = time.perf_counter()

        if self.use_banded:
            self._factor_banded()
            self.last_factor_runtime_s = time.perf_counter() - t0
            return

        if self.use_pardiso:
            A = self.M.tocsr()
            try:
                if self._pardiso is None:
                    # Real-packed Jacobian/preconditioner: real but generally
                    # nonsymmetric. Use MKL PARDISO's real-unsymmetric type; the
                    # structurally-symmetric type (mtype=1) can fail during
                    # reordering on this Schur-reduced matrix with error -3.
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
                        "TWPA_REQUIRE_PARDISO=1. This means the run did not "
                        "use the required fastest backend."
                    ) from exc

                # Debug-only fallback.
                self.use_pardiso = False
                self._pardiso = None
                self._analyzed = False
                self._lu = spla.splu(self.M.tocsc())
                L_factor, U_factor = self._lu.L, self._lu.U
                self.last_factor_backend = "superlu_fallback"
                _log_factor_backend_once("superlu_fallback", f"error={self.last_pardiso_error}")

        else:
            self._lu = spla.splu(self.M.tocsc())
            L_factor, U_factor = self._lu.L, self._lu.U
            self.last_factor_backend = "superlu"
            _log_factor_backend_once("superlu")

        self.last_factor_runtime_s = time.perf_counter() - t0

    def solve(self, b_real: np.ndarray) -> np.ndarray:
        if self.use_banded:
            return self._solve_banded(b_real)
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
